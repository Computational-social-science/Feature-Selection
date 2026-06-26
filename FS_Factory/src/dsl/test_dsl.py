import numpy as np
from dsl.registry import get_registry, eval_dsl

def test_mrmr_formula():
    # 1. 初始化注册表 [cite: 1, 16]
    get_registry()
    
    # 2. 构造模拟数据（离散化后的数据）
    x = np.array([1, 0, 1, 1, 0])
    y = np.array([1, 0, 1, 0, 0])
    s = [np.array([0, 1, 0, 1, 1])] # S 必须是包含数组的列表 [cite: 15, 16]
    
    # 3. 测试公式
    formula = "mi(X, Y) - redundancy(X, S)"
    try:
        # eval_dsl 会调用 DSLRegistry.eval_formula [cite: 14, 16]
        score = eval_dsl(formula, X=x, Y=y, S=s)
        print(f"✅ 测试成功！公式计算结果为: {score}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_mrmr_formula()