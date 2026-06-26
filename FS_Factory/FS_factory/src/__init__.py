# src/__init__.py

"""
特征选择工厂
LLM 驱动的特征选择算子演化系统
"""
from .factory import FeatureSelectionFactory, EvolvedOperator
from .dsl import eval_dsl, DSLRegistry, get_registry
from .prompts import SOTA_METHODS, validate_formula, build_meta_prompt
from .sandbox import SandboxExecutor, EvaluationResult
from .router import RouterAgent, RoutingDecision
from .llm import BaseLLMClient, OpenAIClient, LocalMockClient

__version__ = "1.0.0"
__author__ = "Feature Selection Factory Team"

__all__ = [
    # 主工厂
    'FeatureSelectionFactory',
    'EvolvedOperator',
    
    # DSL
    'eval_dsl',
    'DSLRegistry',
    'get_registry',
    
    # 提示词
    'SOTA_METHODS',
    'validate_formula',
    'build_meta_prompt',
    
    # 沙箱
    'SandboxExecutor',
    'EvaluationResult',
    
    # 路由
    'RouterAgent',
    'RoutingDecision',
    
    # LLM
    'BaseLLMClient',
    'OpenAIClient',
    'LocalMockClient',
]