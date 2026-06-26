# # # src/llm/__init__.py

# # """
# # LLM 接口模块
# # """
# # from .base import BaseLLMClient, LLMResponse
# # from .openai_client import OpenAIClient
# # from .local_client import LocalMockClient

# # __all__ = [
# #     'BaseLLMClient',
# #     'LLMResponse',
# #     'OpenAIClient',
# #     'LocalMockClient',
# # ]
# # src/llm/__init__.py (更新版)

# """
# LLM 接口模块
# 支持多种 LLM 后端
# """
# from typing import Optional, Dict, Any

# from .base import BaseLLMClient, LLMResponse
# # from .openai_client import OpenAIClient
# from .local_client import LocalMockClient
# # from .deepseek_client import DeepSeekClient, DeepSeekMathClient, DeepSeekReasonerClient

# # 尝试导入可选客户端
# try:
#     from .claude_client import ClaudeClient
# except ImportError:
#     ClaudeClient = None

# try:
#     from .vllm_client import VLLMClient
# except ImportError:
#     VLLMClient = None


# # 客户端注册表
# CLIENT_REGISTRY = {
#     'vllm': VLLMClient,
#     # 'openai': OpenAIClient,
#     'mock': LocalMockClient,
#     # 'deepseek': DeepSeekClient,
#     # 'deepseek-math': DeepSeekMathClient,
#     # 'deepseek-reasoner': DeepSeekReasonerClient,
#     'claude': ClaudeClient,
    
# }


# def create_client(
#     provider: str,
#     api_key: Optional[str] = None,
#     model: Optional[str] = None,
#     **kwargs
# ) -> BaseLLMClient:
#     """
#     工厂函数：创建 LLM 客户端
    
#     Args:
#         provider: 提供商名称
#         api_key: API Key
#         model: 模型名称
#         **kwargs: 其他参数
    
#     Returns:
#         LLM 客户端实例
    
#     Examples:
#         >>> # 创建 DeepSeek Math 客户端
#         >>> client = create_client('deepseek-math', api_key='sk-xxx')
        
#         >>> # 创建 OpenAI 客户端
#         >>> client = create_client('openai', model='gpt-4')
        
#         >>> # 创建本地模型客户端
#         >>> client = create_client('vllm', model_path='/path/to/model')
#     """
#     provider_lower = provider.lower()
    
#     if provider_lower not in CLIENT_REGISTRY:
#         available = [k for k, v in CLIENT_REGISTRY.items() if v is not None]
#         raise ValueError(
#             f"Unknown provider: '{provider}'. "
#             f"Available providers: {available}"
#         )
    
#     client_class = CLIENT_REGISTRY[provider_lower]
    
#     if client_class is None:
#         raise ImportError(
#             f"Client for '{provider}' is not installed. "
#             f"Please install the required dependencies."
#         )
    
#     # 根据不同客户端类型构建参数
#     if provider_lower == 'deepseek-math':
#         return client_class(api_key=api_key, **kwargs)
#     elif provider_lower == 'deepseek-reasoner':
#         return client_class(api_key=api_key, **kwargs)
#     elif provider_lower == 'deepseek':
#         return client_class(api_key=api_key, model=model or 'deepseek-chat', **kwargs)
#     elif provider_lower == 'openai':
#         return client_class(api_key=api_key, model=model or 'gpt-4', **kwargs)
#     elif provider_lower == 'vllm':
#         # vLLM 必须要有 model_path
#         if 'model_path' not in kwargs and 'model' in kwargs:
#              kwargs['model_path'] = kwargs.pop('model')
#         if 'model_path' not in kwargs:
#             raise ValueError("VLLM client requires 'model_path' parameter")
#         return client_class(**kwargs)
#     elif provider_lower == 'mock':
#         return client_class(**kwargs)
#     else:
#         return client_class(api_key=api_key, model=model, **kwargs)


# def get_available_providers() -> Dict[str, bool]:
#     """获取所有可用的提供商"""
#     return {
#         name: cls is not None 
#         for name, cls in CLIENT_REGISTRY.items()
#     }


# __all__ = [
#     # 基类
#     'BaseLLMClient',
#     'LLMResponse',
    
#     # 客户端
#     'OpenAIClient',
#     'LocalMockClient',
#     'DeepSeekClient',
#     'DeepSeekMathClient',
#     'DeepSeekReasonerClient',
#     'ClaudeClient',
#     'VLLMClient',
    
#     # 工厂函数
#     'create_client',
#     'get_available_providers',
    
#     # 注册表
#     'CLIENT_REGISTRY',
# ]
# src/llm/__init__.py

"""
LLM 接口模块 - 纯本地环境兼容版
"""
from typing import Optional, Dict, Any
import logging

# 1. 基础模块 (必须存在)
from .base import BaseLLMClient, LLMResponse
from .local_client import LocalMockClient

logger = logging.getLogger(__name__)

# =========================================================
# 2. 外部 API 客户端 (OpenAI/DeepSeek API/Claude) - 设为可选
# =========================================================

# OpenAI
OpenAIClient = None
try:
    from .openai_client import OpenAIClient
except ImportError:
    pass # 用户没装 openai 库，忽略

# DeepSeek API (通常依赖 openai 库)
DeepSeekClient = None
DeepSeekMathClient = None
DeepSeekReasonerClient = None
try:
    from .deepseek_client import DeepSeekClient, DeepSeekMathClient, DeepSeekReasonerClient
except ImportError:
    pass # 用户没装相关依赖，忽略

# Claude
ClaudeClient = None
try:
    from .claude_client import ClaudeClient
except ImportError:
    pass

# =========================================================
# 3. 本地 vLLM 客户端 (核心)
# =========================================================
VLLMClient = None
try:
    from .vllm_client import VLLMClient
except ImportError as e:
    # 如果是为了跑本地模型，vllm 必须装好。如果没装，这里打印个警告，但不要崩。
    logger.warning(f"Warning: vLLM not installed. Local inference will not work. Error: {e}")


# =========================================================
# 4. 注册与工厂函数
# =========================================================

# 注册表：值为 None 的客户端表示不可用
CLIENT_REGISTRY = {
    'openai': OpenAIClient,
    'mock': LocalMockClient,
    'deepseek': DeepSeekClient,
    'deepseek-math': DeepSeekMathClient,
    'deepseek-reasoner': DeepSeekReasonerClient,
    'claude': ClaudeClient,
    'vllm': VLLMClient,
}

def create_client(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """工厂函数"""
    provider_lower = provider.lower()
    
    # 1. 检查 provider 是否在注册表中
    if provider_lower not in CLIENT_REGISTRY:
        available = [k for k in CLIENT_REGISTRY.keys()]
        raise ValueError(f"Unknown provider: '{provider}'. Known providers: {available}")
    
    client_class = CLIENT_REGISTRY[provider_lower]
    
    # 2. 检查依赖是否已安装 (即 client_class 是否为 None)
    if client_class is None:
        error_msg = f"Client for '{provider}' is not available in this environment."
        if provider == 'openai' or 'deepseek' in provider:
            error_msg += " (Hint: 'pip install openai' is missing)"
        elif provider == 'vllm':
            error_msg += " (Hint: 'pip install vllm' is missing)"
        raise ImportError(error_msg)
    
    # 3. 实例化客户端
    # 针对 vLLM 的特殊处理
    if provider_lower == 'vllm':
        # 兼容处理：有时用户把路径传给 model 参数
        if 'model_path' not in kwargs and model:
             kwargs['model_path'] = model
        
        if 'model_path' not in kwargs:
            # 尝试从 settings 读取，或者报错
            raise ValueError("VLLM client requires 'model_path' parameter")
        
        return client_class(**kwargs)
        
    # 针对 API 类的通用处理
    elif provider_lower in ['deepseek', 'deepseek-math', 'deepseek-reasoner', 'openai', 'claude']:
        # 本地模式下，API Key 可以是随意的字符串，防止报错
        if not api_key: 
            api_key = "dummy_key_for_local_env"
        return client_class(api_key=api_key, model=model, **kwargs)
    
    else:
        return client_class(**kwargs)

def get_available_providers() -> Dict[str, bool]:
    return {k: v is not None for k, v in CLIENT_REGISTRY.items()}

__all__ = [
    'BaseLLMClient', 'LLMResponse', 'LocalMockClient',
    'OpenAIClient', 'DeepSeekClient', 'DeepSeekMathClient', 'DeepSeekReasonerClient',
    'ClaudeClient', 'VLLMClient',
    'create_client', 'get_available_providers', 'CLIENT_REGISTRY',
]