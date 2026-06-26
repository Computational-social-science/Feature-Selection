# src/llm/openai_client.py

"""
OpenAI 客户端
"""
import os
from typing import Dict, Optional
from .base import BaseLLMClient, LLMResponse

class OpenAIClient(BaseLLMClient):
    """OpenAI 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.8,
                max_tokens: int = 2000, top_p: float = 0.95, **kwargs) -> LLMResponse:
        """生成响应"""
        # 强制增加 stop 参数，防止复读
        stop_words = kwargs.get("stop", ["```\n\n", "}\n```", "}\n\n"])
        kwargs["stop"] = stop_words
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
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
                'total_tokens': response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason
        )
    
    def generate_with_system(self, system_prompt: str, user_prompt: str,
                            temperature: float = 0.8,
                            max_tokens: int = 2000, top_p: float = 0.95, **kwargs) -> LLMResponse:
        """带系统提示的生成"""
        # 强制增加 stop 参数，防止复读
        stop_words = kwargs.get("stop", ["```\n\n", "}\n```", "}\n\n"])
        kwargs["stop"] = stop_words
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
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
                'total_tokens': response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason
        )