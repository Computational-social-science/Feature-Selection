# scripts/run_benchmark.py

"""
运行基准测试
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factory import FeatureSelectionFactory
from src.prompts import SOTA_METHODS
from src.validation import ResultVisualizer

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征选择基准测试')
    parser.add_argument('--output', type=str, default='./outputs', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建工厂
    factory = FeatureSelectionFactory(output_dir=args.output)
    
    # 准备方法
    methods = {}
    
    # 添加 SOTA 方法
    for method in SOTA_METHODS:
        methods[method.name] = method.formula_dsl
    
    # 如果有演化历史，也添加
    for op in factory.evolution_history:
        methods[op.name] = op.formula_dsl
    
    # 运行基准测试
    print("运行基准测试...")
    print(f"方法数量: {len(methods)}")
    print(f"方法列表: {list(methods.keys())}")
    
    results = factory.run_benchmark(
        methods=methods,
        verbose=True
    )
    
    # 可视化
    visualizer = ResultVisualizer()
    visualizer.plot_comparison(
        results['results'],
        save_path=os.path.join(args.output, 'comparison.png')
    )
    
    print("\n" + "="*60)
    print("基准测试完成！")
    print(f"结果已保存到: {args.output}")


if __name__ == '__main__':
    main()