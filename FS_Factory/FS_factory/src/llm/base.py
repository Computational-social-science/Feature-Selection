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
        
        # 尝试提取第一个完整的 JSON 块 (非贪婪匹配)
        json_pattern = re.compile(r'(\{\s*"method_name"[\s\S]*?"theoretical_advantage"[\s\S]*?\})', re.IGNORECASE)
        match = json_pattern.search(content)
        
        if match:
            raw_json_str = match.group(1)
            # 补齐结尾的 '}' (保险起见)
            if not raw_json_str.strip().endswith('}'):
                raw_json_str += '}'
            try:
                return json.loads(raw_json_str)
            except json.JSONDecodeError:
                pass
        
        # 兜底：更宽松但非贪婪的匹配
        fallback_match = re.search(r'(\{[\s\S]*?\})', content)
        if fallback_match:
            try:
                return json.loads(fallback_match.group(1))
            except:
                pass
        
        return {'raw_content': content, 'parse_error': True}