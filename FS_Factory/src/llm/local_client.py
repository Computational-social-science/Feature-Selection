# src/llm/local_client.py

"""
本地模型客户端 (模拟)
"""
from typing import Dict, Optional
from .base import BaseLLMClient, LLMResponse
import random

class LocalMockClient(BaseLLMClient):
    """本地模拟客户端 (用于测试)"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # 预定义的响应模板
        self.templates = [
            {
                "method_name": "EvolvedMI_alpha",
                "formula_dsl": "mi(X, Y) - 0.6 * redundancy(X, S) + 0.3 * sum([cmi(X, Y, s) for s in S])/max(len(S), 1)",
                "formula_math": r"$J = I(X;Y) - 0.6R(X,S) + 0.3\sum_s I(X;Y|s)/|S|$",
                "intuition": "平衡相关性、冗余性和条件互补性，引入可调权重",
                "theoretical_advantage": "相比 mRMR 增加了条件互信息项，可以捕捉特征间的协同效应"
            },
            {
                "method_name": "EvolvedNMI_beta",
                "formula_dsl": "nmi(X, Y) * (1 - safe_div(redundancy(X, S), entropy(X) + EPS))",
                "formula_math": r"$J = NMI(X;Y) \cdot (1 - R(X,S)/H(X))$",
                "intuition": "使用归一化互信息，并用熵来归一化冗余度",
                "theoretical_advantage": "归一化处理使公式对特征尺度不敏感"
            },
            {
                "method_name": "EvolvedIG_gamma",
                "formula_dsl": "mi(X, Y) + sum([max(0, cmi(X, Y, s) - mi(X, s)) for s in S])/max(len(S), 1)",
                "formula_math": r"$J = I(X;Y) + \sum_s \max(0, I(X;Y|s) - I(X;s))/|S|$",
                "intuition": "只保留正向的条件信息增益，忽略负交互",
                "theoretical_advantage": "相比 ICAP 更加激进地利用正向交互"
            }
        ]
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成模拟响应"""
        import json
        
        # 随机选择一个模板
        template = random.choice(self.templates)
        
        content = json.dumps(template, indent=2, ensure_ascii=False)
        
        return LLMResponse(
            content=content,
            model="local-mock",
            usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            finish_reason="stop"
        )
    
    def generate_with_system(self, system_prompt: str, user_prompt: str,
                            **kwargs) -> LLMResponse:
        """带系统提示的生成"""
        return self.generate(user_prompt, **kwargs)