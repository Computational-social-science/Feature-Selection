# src/llm/vllm_client.py

import os
import re
import json
from typing import Dict, List, Optional, Any, Union, Generator
from .base import BaseLLMClient, LLMResponse

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None

class VLLMClient(BaseLLMClient):
    """vLLM 本地部署客户端"""
    
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.7,
        trust_remote_code: bool = True,
        max_tokens: int = 2048,
        max_model_len: int = 2048,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        if LLM is None:
            raise ImportError("vLLM library is not installed.")
        
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"Loading vLLM model from: {model_path}...")
        print(f"Config: gpu_mem={gpu_memory_utilization}, enforce_eager=True")
        
        # === 关键修复：防止参数冲突 ===
        # 如果 kwargs 里已经有了 enforce_eager，先删掉它
        kwargs.pop('enforce_eager', None)
        
        # 初始化 vLLM 引擎
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=True,  # 强制启用
            **kwargs
        )

    def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # === 核心修复：注入停止词和复读惩罚 ===
        # 如果调用方没有传 stop，我们强制给它加上最严厉的停止标记
        stop_words = kwargs.get("stop", [
            "```\n\n",                 # 只要 JSON 一闭合，立刻停止！
            "}\n```",
            "}\n\n",
            "<｜end of sentence｜>", # DeepSeek 专属结束符
            "<|end_of_sentence|>",
            "</s>"
        ])
        sampling_params = SamplingParams(
            temperature=temp if temp > 0 else 0.8,
            top_p=top_p if top_p > 0 else 0.95,
            max_tokens=tokens,
            stop=stop_words,
            # repetition_penalty=1.15, # 惩罚复读机，防止大模型陷入死循环
            **{k: v for k, v in kwargs.items() if k not in ["stop", "repetition_penalty", "temperature", "top_p"]}
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        usage = {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            "cost_usd": 0.0
        }
        
        return LLMResponse(
            content=generated_text,
            model=self.model_path,
            usage=usage,
            finish_reason=output.outputs[0].finish_reason
        )

    def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs):
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        return self.generate(full_prompt, **kwargs)

    def generate_with_history(self, messages: List[Dict[str, str]], **kwargs):
        prompt = self._apply_chat_template(messages)
        return self.generate(prompt, **kwargs)

    def stream_generate(self, prompt: str, **kwargs):
        response = self.generate(prompt, **kwargs)
        yield response.content

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            if role == "System": formatted_prompt += f"{msg['content']}\n\n"
            elif role == "User": formatted_prompt += f"User: {msg['content']}\n"
            elif role == "Assistant": formatted_prompt += f"Assistant: {msg['content']}\n"
        formatted_prompt += "Assistant:"
        return formatted_prompt


class VLLMMathClient(VLLMClient):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
    
    def generate_formula(self, description: str, available_operators: List[str] = None, constraints: List[str] = None) -> Dict:
        # 简化版实现，防止之前的代码被截断
        prompt = f"生成公式: {description}"
        response = self.generate(prompt, temperature=0.9, top_p=0.95)
        
        # 解析 JSON
        import re
        try:
            content = response.content
            # 使用精准的非贪婪匹配，框定第一个完整的 JSON 块
            json_pattern = re.compile(r'(\{\s*"method_name"[\s\S]*?"theoretical_advantage"[\s\S]*?\})', re.IGNORECASE)
            match = json_pattern.search(content)
            
            if match:
                raw_json_str = match.group(1)
                # 补齐结尾的 '}' (保险起见)
                if not raw_json_str.strip().endswith('}'):
                    raw_json_str += '}'
                return json.loads(raw_json_str)
        except:
            pass
            
        return {'raw_content': response.content}

class VLLMReasonerClient(VLLMClient):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
    
    def generate_with_reasoning(self, prompt: str, **kwargs) -> LLMResponse:
        return self.generate(prompt, **kwargs)
