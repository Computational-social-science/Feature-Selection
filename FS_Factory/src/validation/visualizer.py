# src/validation/visualizer.py

"""
可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os

class ResultVisualizer:
    """结果可视化"""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        self.colors = sns.color_palette("husl", 10)
    
    def plot_comparison(self, results: pd.DataFrame,
                       save_path: Optional[str] = None,
                       figsize: tuple = (14, 6)) -> plt.Figure:
        """绘制对比图"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图: 各数据集性能
        df_pivot = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        
        df_pivot.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Accuracy by Dataset and Method', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Dataset', fontsize=10)
        axes[0].set_ylabel('Accuracy', fontsize=10)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 1])
        
        # 右图: 平均性能
        avg_perf = results.groupby('method')['mean_accuracy'].mean().sort_values(ascending=True)
        
        colors = ['green' if i >= len(avg_perf) - 3 else 'steelblue' 
                 for i in range(len(avg_perf))]
        
        bars = axes[1].barh(range(len(avg_perf)), avg_perf.values, color=colors)
        axes[1].set_yticks(range(len(avg_perf)))
        axes[1].set_yticklabels(avg_perf.index)
        axes[1].set_xlabel('Mean Accuracy', fontsize=10)
        axes[1].set_title('Average Performance', fontsize=12, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, avg_perf.values)):
            axes[1].text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ranking(self, results: pd.DataFrame,
                    save_path: Optional[str] = None) -> plt.Figure:
        """绘制排名图"""
        # 计算每个数据集上的排名
        df_pivot = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        ranks = df_pivot.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("RdYlGn_r", len(avg_ranks))
        
        bars = ax.bar(range(len(avg_ranks)), avg_ranks.values, color=colors)
        ax.set_xticks(range(len(avg_ranks)))
        ax.set_xticklabels(avg_ranks.index, rotation=45, ha='right')
        ax.set_ylabel('Average Rank', fontsize=11)
        ax.set_title('Method Ranking (Lower is Better)', fontsize=12, fontweight='bold')
        
        # 添加参考线
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Best possible')
        ax.axhline(y=len(avg_ranks)/2, color='gray', linestyle='--', alpha=0.5, label='Random')
        
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(self, results: pd.DataFrame,
                    save_path: Optional[str] = None) -> plt.Figure:
        """绘制热力图"""
        df_pivot = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(df_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax, vmin=0.5, vmax=1.0)
        
        ax.set_title('Accuracy Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def generate_report(results: pd.DataFrame,
                   statistical_analysis: Dict,
                   agent_methods: List[str] = None) -> str:
    """生成实验报告"""
    
    # 汇总统计
    summary = results.groupby('method').agg({
        'mean_accuracy': ['mean', 'std', 'max', 'min']
    }).round(4)
    
    summary.columns = ['Mean', 'Std', 'Max', 'Min']
    summary = summary.sort_values('Mean', ascending=False)
    
    # Friedman 检验结果
    friedman = statistical_analysis.get('friedman', {})
    
    report = f"""
# 特征选择算子演化实验报告

## 1. 实验配置

- 重复次数: 10
- 交叉验证折数: 5
- 选择特征数: 30

## 2. 性能汇总

| 排名 | 方法 | 平均准确率 | 标准差 | 最大值 | 最小值 |
|------|------|-----------|--------|--------|--------|
"""
    
    for rank, (method, row) in enumerate(summary.iterrows(), 1):
        method_type = "🤖 Agent" if agent_methods and method in agent_methods else "👤 Human"
        report += f"| {rank} | {method} {method_type} | {row['Mean']:.4f} | {row['Std']:.4f} | {row['Max']:.4f} | {row['Min']:.4f} |\n"
    
    report += f"""
## 3. 统计检验

### Friedman 检验
- χ² 统计量: {friedman.get('statistic', 0):.4f}
- p 值: {friedman.get('p_value', 1):.4e}
- 结论: {'存在显著差异' if friedman.get('significant', False) else '不存在显著差异'}

### 最佳方法
{friedman.get('winner', 'N/A')}

## 4. 结论

"""
    
    best_method = summary.index[0]
    best_acc = summary.loc[best_method, 'Mean']
    
    if agent_methods and best_method in agent_methods:
        report += f"""
**🎉 Agent 演化的方法 `{best_method}` 在实验中取得了最佳性能 ({best_acc:.4f})！**

这证明了 LLM 驱动的算子演化方法的有效性。
"""
    else:
        report += f"""
最佳方法为人类设计的 `{best_method}` ({best_acc:.4f})。

Agent 演化的方法尚未超越人类设计，需要进一步优化。
"""
    
    return report