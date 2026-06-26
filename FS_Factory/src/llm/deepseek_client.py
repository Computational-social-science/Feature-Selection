# src/llm/deepseek_client.py

"""
DeepSeek API 客户端
支持 deepseek-chat, deepseek-coder, deepseek-math 等模型
"""
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import time

from .base import BaseLLMClient, LLMResponse


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API 客户端
    
    支持的模型:
    - deepseek-chat: 通用对话模型
    - deepseek-coder: 代码专用模型
    - deepseek-math: 数学推理模型 (推荐用于特征选择公式生成)
    - deepseek-reasoner: 推理模型 (R1)
    
    API 文档: https://platform.deepseek.com/api-docs/
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "deepseek-chat",
        "deepseek-coder", 
        "deepseek-math",
        "deepseek-reasoner",
        "deepseek-r1",        # 别名
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-math",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        """
        初始化 DeepSeek 客户端
        
        Args:
            api_key: DeepSeek API Key (可从环境变量 DEEPSEEK_API_KEY 获取)
            model: 模型名称，推荐使用 deepseek-math 进行公式生成
            base_url: API 基础 URL
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
            timeout: 请求超时时间(秒)
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. "
                "Please set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # 验证模型名称
        if model not in self.SUPPORTED_MODELS:
            print(f"Warning: Model '{model}' may not be supported. "
                  f"Supported models: {self.SUPPORTED_MODELS}")
        
        # 初始化 OpenAI 兼容客户端
        self._init_client()
    
    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
        except ImportError:
            raise ImportError(
                "Please install openai package: pip install openai>=1.0.0"
            )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        """
        生成响应
        
        Args:
            prompt: 输入提示词
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling 参数
        
        Returns:
            LLMResponse 对象
        """
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, temperature, max_tokens, top_p, **kwargs)
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        """
        带系统提示的生成
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling 参数
        
        Returns:
            LLMResponse 对象
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages, temperature, max_tokens, top_p, **kwargs)
    
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        """
        带对话历史的生成
        
        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: nucleus sampling 参数
        
        Returns:
            LLMResponse 对象
        """
        return self._call_api(messages, temperature, max_tokens, top_p, **kwargs)
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> LLMResponse:
        """调用 API (带重试机制)"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=response.model,
                    usage={
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens,
                        'cost_usd': self._estimate_cost(response.usage)
                    },
                    finish_reason=response.choices[0].finish_reason
                )
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    print(f"DeepSeek API call failed (attempt {attempt + 1}), retrying... Error: {str(e)[:100]}")
        
        raise RuntimeError(f"DeepSeek API call failed after {self.max_retries} attempts: {last_error}")
    
    def _estimate_cost(self, usage) -> float:
        """
        估算 API 调用成本
        
        DeepSeek 定价 (2024):
        - deepseek-chat: $0.14/1M input, $0.28/1M output
        - deepseek-math: $0.14/1M input, $0.28/1M output
        - deepseek-reasoner: $0.55/1M input, $2.19/1M output
        """
        pricing = {
            "deepseek-chat": (0.14 / 1e6, 0.28 / 1e6),
            "deepseek-coder": (0.14 / 1e6, 0.28 / 1e6),
            "deepseek-math": (0.14 / 1e6, 0.28 / 1e6),
            "deepseek-reasoner": (0.55 / 1e6, 2.19 / 1e6),
        }
        
        input_price, output_price = pricing.get(self.model, (0.14 / 1e6, 0.28 / 1e6))
        
        cost = (
            usage.prompt_tokens * input_price +
            usage.completion_tokens * output_price
        )
        
        return round(cost, 6)
    
    def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        流式生成 (适用于长输出)
        
        Yields:
            str: 每个 chunk 的内容
        """
        messages = [{"role": "user", "content": prompt}]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class DeepSeekMathClient(DeepSeekClient):
    """
    DeepSeek Math 模型专用客户端
    
    针对数学公式生成优化的便捷封装
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化 DeepSeek Math 客户端
        
        Args:
            api_key: DeepSeek API Key
            **kwargs: 其他参数传递给 DeepSeekClient
        """
        super().__init__(api_key=api_key, model="deepseek-math", **kwargs)
    
    def generate_formula(
        self,
        description: str,
        available_operators: List[str] = None,
        constraints: List[str] = None
    ) -> Dict:
        """
        生成特征选择公式
        
        Args:
            description: 公式的自然语言描述
            available_operators: 可用算子列表
            constraints: 约束条件列表
        
        Returns:
            包含公式信息的字典
        """
        import json
        
        prompt = f"""
请根据以下描述生成一个特征选择公式。

描述: {description}

可用算子: {available_operators or ['mi', 'cmi', 'entropy', 'redundancy', 'nmi', 'su']}

约束条件:
{chr(10).join('- ' + c for c in (constraints or ['数值稳定', '避免除零']))}

请以 JSON 格式输出:
{{
    "method_name": "方法名称",
    "formula_dsl": "DSL公式",
    "formula_math": "LaTeX数学公式",
    "intuition": "设计直觉",
    "theoretical_advantage": "理论优势"
}}
"""
        
        response = self.generate(prompt, temperature=0.3)
        
        # 解析 JSON
        try:
            content = response.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except:
            pass
        
        return {
            'raw_content': response.content,
            'parse_error': True
        }


class DeepSeekReasonerClient(DeepSeekClient):
    """
    DeepSeek R1 推理模型客户端
    
    适用于需要深度推理的任务
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, model="deepseek-reasoner", **kwargs)
    
    def generate_with_reasoning(
        self,
        prompt: str,
        show_reasoning: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        带推理过程的生成
        
        DeepSeek R1 会在输出中包含 <think reasoning_content> 标签
        
        Args:
            prompt: 输入提示词
            show_reasoning: 是否显示推理过程
        
        Returns:
            LLMResponse 对象
        """
        response = self.generate(prompt, **kwargs)
        
        # 提取推理过程
        content = response.content
        
        if '<think' in content and '</think' in content:
            import re
            # 提取推理过程
            reasoning_match = re.search(
                r'<think[^>]*>(.*?)</think', 
                content, 
                re.DOTALL
            )
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # 提取最终答案 (推理之后的内容)
                final_answer = content.split('</think')[-1].strip()
                
                if show_reasoning:
                    print("=== 推理过程 ===")
                    print(reasoning)
                    print("=== 最终答案 ===")
                
                # 返回清理后的答案
                return LLMResponse(
                    content=final_answer,
                    model=response.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason
                )
        
        return response