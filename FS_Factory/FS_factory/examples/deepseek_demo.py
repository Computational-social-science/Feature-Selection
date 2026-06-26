# examples/deepseek_demo.py

"""
DeepSeek Math 模型使用示例
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import DeepSeekClient, DeepSeekMathClient, create_client
from src import FeatureSelectionFactory

# ============================================
# 方式 1: 直接使用 DeepSeekMathClient
# ============================================

# 设置 API Key (或通过环境变量 DEEPSEEK_API_KEY 设置)
api_key = "sk-your-deepseek-api-key"

client = DeepSeekMathClient(api_key=api_key)

# 测试连接
response = client.generate("请用一句话解释什么是互信息。")
print("响应:", response.content)
print(f"Token 使用: {response.usage}")


# ============================================
# 方式 2: 使用工厂函数创建
# ============================================

client = create_client('deepseek-math', api_key=api_key)


# ============================================
# 方式 3: 集成到特征选择工厂
# ============================================

factory = FeatureSelectionFactory(
    llm_client=DeepSeekMathClient(api_key=api_key),
    output_dir="./outputs"
)

# 运行演化
best_operator = factory.evolve_operators(n_iterations=3, verbose=True)


# ============================================
# 方式 4: 使用 DeepSeek R1 推理模型
# ============================================

reasoner = create_client('deepseek-reasoner', api_key=api_key)

response = reasoner.generate_with_reasoning(
    "设计一个新的特征选择准则函数，需要考虑特征间的协同效应。",
    show_reasoning=True
)
print("最终答案:", response.content)


# ============================================
# 方式 5: 流式输出 (适用于长输出)
# ============================================

print("\n流式输出演示:")
for chunk in client.stream_generate("请详细解释 mRMR 算法的原理。"):
    print(chunk, end='', flush=True)
print()