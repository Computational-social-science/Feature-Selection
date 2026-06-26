# scripts/run_evolution.py

"""
运行算子演化
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factory import FeatureSelectionFactory
from src.llm import OpenAIClient, LocalMockClient

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征选择算子演化')
    parser.add_argument('--iterations', type=int, default=5, help='演化迭代次数')
    parser.add_argument('--output', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--use-openai', action='store_true', help='使用 OpenAI API')
    parser.add_argument('--model', type=str, default='gpt-4', help='模型名称')
    
    args = parser.parse_args()
    
    # 创建 LLM 客户端
    if args.use_openai:
        try:
            llm_client = OpenAIClient(model=args.model)
            print(f"使用 OpenAI 模型: {args.model}")
        except Exception as e:
            print(f"OpenAI 客户端初始化失败: {e}")
            print("使用本地模拟客户端")
            llm_client = LocalMockClient()
    else:
        print("使用本地模拟客户端")
        llm_client = LocalMockClient()
    
    # 创建工厂
    factory = FeatureSelectionFactory(
        llm_client=llm_client,
        output_dir=args.output
    )
    
    # 运行演化
    best_operator = factory.evolve_operators(
        n_iterations=args.iterations,
        verbose=True
    )
    
    if best_operator:
        print("\n" + "="*60)
        print("演化完成！最佳算子:")
        print(f"  名称: {best_operator.name}")
        print(f"  公式: {best_operator.formula_dsl}")
        print(f"  平均准确率: {best_operator.evaluation_result.avg_accuracy:.4f}")


if __name__ == '__main__':
    main()