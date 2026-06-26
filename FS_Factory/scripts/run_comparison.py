# scripts/run_comparison.py

"""
运行人类 vs Agent 对比实验
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.factory import FeatureSelectionFactory
from src.prompts import SOTA_METHODS
from src.sandbox import DatasetManager
from src.validation import ExperimentRunner, run_statistical_analysis, ResultVisualizer, generate_report

def main():
    """主函数"""
    
    print("="*60)
    print("人类设计算子 vs Agent 演化算子 对比实验")
    print("="*60)
    
    # 准备数据集
    datasets = DatasetManager.get_all_datasets()
    dataset_dict = {ds.name: (ds.X, ds.y) for ds in datasets}
    # datasets = DatasetGenerator.create_standard_benchmark()[:3]
    # dataset_dict = {ds.name: (ds.X, ds.y) for ds in datasets}

    # 准备方法
    methods = {}
    
    # 人类设计的方法
    for method in SOTA_METHODS[:4]:  # 选择前4个
        methods[method.name] = method.formula_dsl
    
    # Agent "演化"的方法 (这里用几个示例)
    agent_methods = {
        'Agent_Evolved_1': 'mi(X, Y) - 0.7 * redundancy(X, S) + 0.4 * sum([cmi(X, Y, s) for s in S])/max(len(S), 1)',
        'Agent_Evolved_2': 'nmi(X, Y) * (1 - safe_div(redundancy(X, S), entropy(X) + EPS))',
        'Agent_Evolved_3': 'mi(X, Y) + sum([max(0, cmi(X, Y, s) - mi(X, s)) for s in S])/max(len(S), 1)',
    }
    
    methods.update(agent_methods)
    
    print(f"\n参与比较的方法:")
    print("  人类设计:", list(SOTA_METHODS[i].name for i in range(4)))
    print("  Agent演化:", list(agent_methods.keys()))
    
    # 运行实验
    runner = ExperimentRunner()
    results = runner.run_comparison(methods, dataset_dict, verbose=True)
    
    # 统计分析
    print("\n" + "="*60)
    print("统计分析")
    print("="*60)
    
    analysis = run_statistical_analysis(results, baseline_method='mRMR')
    
    print(f"\nFriedman 检验:")
    friedman = analysis['friedman']
    print(f"  χ² 统计量: {friedman.statistic:.4f}")
    print(f"  p 值: {friedman.p_value:.4e}")
    print(f"  最佳方法: {friedman.winner}")
    
    # 生成报告
    report = generate_report(results, analysis, list(agent_methods.keys()))
    
    # 保存
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write(report)
    
    # 可视化
    visualizer = ResultVisualizer()
    visualizer.plot_comparison(results, save_path=os.path.join(output_dir, 'comparison_plot.png'))
    
    print("\n" + "="*60)
    print("实验完成！")
    print(f"结果已保存到: {output_dir}")
    print("\n" + report)


if __name__ == '__main__':
    main()