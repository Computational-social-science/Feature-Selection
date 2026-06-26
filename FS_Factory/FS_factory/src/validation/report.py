# src/validation/report.py

"""
报告生成
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

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
