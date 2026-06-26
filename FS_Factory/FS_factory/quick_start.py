# quick_start.py

"""
快速开始示例
"""
import numpy as np
from sklearn.datasets import make_classification

# 导入工厂
from src import FeatureSelectionFactory, LocalMockClient, eval_dsl

# 1. 创建测试数据
X, y = make_classification(
    n_samples=1000, 
    n_features=100, 
    n_informative=20,
    random_state=42
)

print("数据集:", X.shape)

# 2. 创建工厂 (使用模拟客户端)
factory = FeatureSelectionFactory(
    llm_client=LocalMockClient(),
    output_dir="./outputs"
)

# 3. 演化算子
print("\n" + "="*50)
print("开始演化...")
best_operator = factory.evolve_operators(n_iterations=3, verbose=True)

# 4. 使用路由选择特征
print("\n" + "="*50)
print("使用路由进行特征选择...")
selected, meta = factory.select_features(X, y, method='auto', k=20)

print(f"选中了 {len(selected)} 个特征")
print(f"路由信息: {meta['routing']}")

# 5. 运行基准测试
print("\n" + "="*50)
print("运行基准测试...")
benchmark_results = factory.run_benchmark(verbose=False)

print(f"\n结果摘要:")
print(benchmark_results['results'].groupby('method')['mean_accuracy'].mean().sort_values(ascending=False))

# 6. 直接使用 DSL 公式
print("\n" + "="*50)
print("直接使用 DSL 公式评估...")

# 使用 mRMR 公式
score = eval_dsl(
    "mi(X, Y) - redundancy(X, S)",
    X=X[:, 0],  # 候选特征
    Y=y,        # 目标
    S=[]        # 已选特征 (空)
)
print(f"mRMR 得分 (第一个特征): {score:.4f}")

print("\n快速开始完成！")