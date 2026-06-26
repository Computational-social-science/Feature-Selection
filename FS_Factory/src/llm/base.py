# src/llm/base.py

"""
LLM 基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    usage: Dict
    finish_reason: str


class BaseLLMClient(ABC):
    """LLM 客户端基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def generate_with_system(self, system_prompt: str, user_prompt: str,
                            **kwargs) -> LLMResponse:
        """带系统提示的生成"""
        pass
    
    def parse_json_response(self, response: LLMResponse) -> Dict:
        """解析 JSON 响应"""
        import json
        import re
        
        content = response.content
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {'raw_content': content, 'parse_error': True}