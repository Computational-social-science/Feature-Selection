# src/dsl/registry.py

"""
算子注册表
提供统一的 DSL 函数访问接口
"""
import numpy as np
from typing import Dict, Callable, Any, List, Optional
from .operators import BaseOperators, DSLOptions
from .advanced import AdvancedOperators, AggregationOperators

class DSLRegistry:
    """DSL 算子注册表"""
    
    _instance = None
    _operators: Optional[AdvancedOperators] = None
    _functions: Dict[str, Callable] = {}
    
    def __new__(cls, options: Optional[DSLOptions] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize(options)
        return cls._instance
    
    @classmethod
    def _initialize(cls, options: Optional[DSLOptions] = None):
        """初始化算子"""
        opts = options or DSLOptions()
        cls._operators = AdvancedOperators(opts)
        
        # 注册基础算子
        cls._functions = {
            # 信息论基础
            'entropy': cls._operators.entropy,
            'H': cls._operators.entropy,  # 别名
            'joint_entropy': cls._operators.joint_entropy,
            'cond_entropy': cls._operators.conditional_entropy,
            'mi': cls._operators.mutual_info,
            'I': cls._operators.mutual_info,  # 别名
            'cmi': cls._operators.conditional_mutual_info,
            
            # 归一化度量
            'nmi': cls._operators.normalized_mi,
            'su': cls._operators.symmetric_uncertainty,
            
            # 冗余度
            'redundancy': cls._operators.redundancy,
            'R': cls._operators.redundancy,
            'max_redundancy': cls._operators.max_redundancy,
            
            # 条件度量
            'cond_redundancy': cls._operators.conditional_redundancy,
            'ig': cls._operators.information_gain,
            'interaction': cls._operators.interaction_info,
            
            # 距离
            'feature_distance': cls._operators.feature_distance,
            'cond_distance': cls._operators.conditional_distance,
            
            # 相关性
            'corr': cls._operators.correlation,
            'spearman': cls._operators.spearman_correlation,
            
            # 聚合
            'mean': AggregationOperators.mean,
            'max': AggregationOperators.max_val,
            'min': AggregationOperators.min_val,
            'sum': AggregationOperators.sum_val,
            'weighted_sum': AggregationOperators.weighted_sum,
            
            # 数学函数
            'log': lambda x: np.log(np.abs(x) + 1e-10),
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'abs': np.abs,
            'pow': np.power,
            'exp': np.exp,
            
            # 安全操作
            'safe_div': cls._operators._safe_div,
            'clip': lambda x, lo, hi: np.clip(x, lo, hi),
            
            # 常量
            'EPS': 1e-10,
        }
    
    @classmethod
    def get_function(cls, name: str) -> Callable:
        """获取算子函数"""
        if name not in cls._functions:
            raise ValueError(f"Unknown DSL function: {name}")
        return cls._functions[name]
    
    @classmethod
    def get_all_functions(cls) -> Dict[str, Callable]:
        """获取所有算子"""
        return cls._functions.copy()
    
    @classmethod
    def register_function(cls, name: str, func: Callable):
        """注册自定义算子"""
        cls._functions[name] = func
    
    @classmethod
    def eval_formula(cls, formula: str, context: Dict[str, Any]) -> float:
        """
        安全评估 DSL 公式
        """
        # 如果还没加载算子，强制初始化
        if not cls._functions:
            cls._initialize()

        # 【一劳永逸的修复】：将所有函数和所有数据合并到同一个统一的命名空间中
        unified_namespace = {
            "__builtins__": {},  # 基础的安全沙箱保护
            **cls._functions,    # 注入所有的注册算子 (mi, mean, redundancy 等)
            **context,           # 注入所有的动态数据 (X, Y, S 等)
            'np': np,
            'len': len,
            'range': range,
            'list': list,
        }
        
        # 移除危险函数，防止 LLM 生成恶意代码
        dangerous = ['eval', 'exec', 'compile', 'open', 'import', '__']
        for d in dangerous:
            unified_namespace.pop(d, None)
        
        try:
            # 终极奥义：只传 unified_namespace 作为 globals (第二个参数)
            # 这样列表推导式无论需要函数还是数据，都能在全局作用域里一眼找到！
            result = eval(formula, unified_namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Formula evaluation failed: {e}")


# ==================== 便捷函数 ====================

def eval_dsl(formula: str, X: np.ndarray = None, Y: np.ndarray = None,
             S: List[np.ndarray] = None, **kwargs) -> float:
    """
    便捷的 DSL 评估函数
    
    Example:
        >>> score = eval_dsl("mi(X, Y) - 0.5 * redundancy(X, S)", 
        ...                  X=candidate, Y=target, S=selected)
    """
    context = kwargs.copy()
    
    if X is not None:
        context['X'] = X
    if Y is not None:
        context['Y'] = Y
    if S is not None:
        context['S'] = S
    
    return DSLRegistry.eval_formula(formula, context)


# ==================== 初始化 ====================

def get_registry(options: DSLOptions = None) -> DSLRegistry:
    """获取 DSL 注册表实例"""
    return DSLRegistry(options)