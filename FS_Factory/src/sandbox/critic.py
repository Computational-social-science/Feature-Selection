# # src/sandbox/critic.py

# """
# 批评者 Agent
# """
# from typing import Dict, List, Optional
# from dataclasses import dataclass
# import numpy as np

# from ..prompts import SOTA_METHODS, get_best_method
# from .executor import EvaluationResult

# @dataclass
# class Criticism:
#     """批评结果"""
#     overall_verdict: str  # success, needs_improvement, failed
#     beat_sota: bool
#     best_dataset: Optional[str]
#     worst_dataset: Optional[str]
#     suggestions: List[str]
#     detailed_analysis: Dict


# class CriticAgent:
#     """批评者 Agent"""
    
#     @staticmethod
#     def analyze(result: EvaluationResult) -> Criticism:
#         """分析评估结果"""
        
#         if not result.success:
#             return Criticism(
#                 overall_verdict='failed',
#                 beat_sota=False,
#                 best_dataset=None,
#                 worst_dataset=None,
#                 suggestions=["公式执行失败，请检查语法和数值稳定性"],
#                 detailed_analysis={'error': result.error_message}
#             )
        
#         # 获取 SOTA 基准
#         best_sota = get_best_method()
#         sota_avg = np.mean(list(best_sota.performance.values()))
        
#         # 判断是否超越 SOTA
#         beat_sota = result.avg_accuracy > sota_avg
        
#         # 找最佳和最差数据集
#         valid_results = {
#             k: v for k, v in result.dataset_results.items()
#             if 'accuracy' in v and v['accuracy'] > 0
#         }
        
#         best_dataset = None
#         worst_dataset = None
        
#         if valid_results:
#             sorted_datasets = sorted(
#                 valid_results.items(),
#                 key=lambda x: x[1]['accuracy'],
#                 reverse=True
#             )
#             best_dataset = sorted_datasets[0][0]
#             worst_dataset = sorted_datasets[-1][0]
        
#         # 生成建议
#         suggestions = []
        
#         if beat_sota:
#             overall_verdict = 'success'
#             suggestions.append(f"🎉 恭喜！公式超越 SOTA，平均准确率 {result.avg_accuracy:.4f} > {sota_avg:.4f}")
#         else:
#             overall_verdict = 'needs_improvement'
#             delta = sota_avg - result.avg_accuracy
#             suggestions.append(f"当前落后 SOTA {delta:.4f}，需要改进")
        
#         # 基于表现模式的建议
#         if worst_dataset:
#             worst_acc = valid_results.get(worst_dataset, {}).get('accuracy', 0)
#             suggestions.append(f"在 {worst_dataset} 上表现最差 ({worst_acc:.4f})，考虑针对性优化")
        
#         # 基于数值问题的建议
#         if result.validation_issues:
#             suggestions.append(f"数值稳定性警告: {', '.join(result.validation_issues[:2])}")
        
#         # 基于数据集特性的建议
#         if valid_results:
#             acc_std = np.std([v['accuracy'] for v in valid_results.values()])
#             if acc_std > 0.1:
#                 suggestions.append("不同数据集表现差异大，考虑增强泛化能力")
        
#         # 详细分析
#         detailed_analysis = {
#             'avg_accuracy': result.avg_accuracy,
#             'sota_avg_accuracy': sota_avg,
#             'delta': result.avg_accuracy - sota_avg,
#             'n_datasets': len(valid_results),
#             'accuracy_std': np.std([v['accuracy'] for v in valid_results.values()]) if valid_results else 0,
#             'per_dataset': {
#                 k: v.get('accuracy', 0) for k, v in result.dataset_results.items()
#             }
#         }
        
#         return Criticism(
#             overall_verdict=overall_verdict,
#             beat_sota=beat_sota,
#             best_dataset=best_dataset,
#             worst_dataset=worst_dataset,
#             suggestions=suggestions,
#             detailed_analysis=detailed_analysis
#         )
    
#     @staticmethod
#     def generate_improvement_prompt(original_formula: str, 
#                                     result: EvaluationResult,
#                                     criticism: Criticism) -> str:
#         """生成改进提示词"""
#         prompt = f"""
# # 📝 公式改进任务

# 你的上一个公式表现不够理想，请根据反馈进行改进。

# ## 原始公式
# {original_formula}

# ## 评估结果
# - 平均准确率: {result.avg_accuracy:.4f}
# - 最佳数据集: {criticism.best_dataset}
# - 最差数据集: {criticism.worst_dataset}

# ## 各数据集表现
# """
        
#         for dataset_name, dataset_result in result.dataset_results.items():
#             acc = dataset_result.get('accuracy', 'N/A')
#             if isinstance(acc, float):
#                 prompt += f"- {dataset_name}: {acc:.4f}\n"
#             else:
#                 prompt += f"- {dataset_name}: {acc}\n"
        
#         prompt += f"""
# ## 批评意见
# {chr(10).join('- ' + s for s in criticism.suggestions)}

# ## 改进方向建议
# 1. 考虑调整相关性和冗余性的权重平衡
# 2. 尝试引入条件互信息项来捕捉特征交互
# 3. 增加对弱信号数据集的处理能力
# 4. 检查数值稳定性，使用 safe_div 和 EPS

# 请输出改进后的公式 (JSON 格式)。
# """
#         return prompt

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from ..prompts import SOTA_METHODS, get_best_method
from .executor import EvaluationResult

@dataclass
class Criticism:
    """批评结果"""
    overall_verdict: str  # success, needs_improvement, failed
    beat_sota: bool
    best_dataset: Optional[str]
    worst_dataset: Optional[str]
    suggestions: List[str]
    detailed_analysis: Dict

class CriticAgent:
    """批评者 Agent"""
    
    @staticmethod
    def analyze(result: EvaluationResult) -> Criticism:
        """分析评估结果，生成带有针对性算子建议的批评意见"""
        
        # 1. 处理执行失败
        if not result.success:
            return Criticism(
                overall_verdict='failed',
                beat_sota=False,
                best_dataset=None,
                worst_dataset=None,
                suggestions=[
                    f"🚨 公式执行失败。错误信息: {result.error_message}",
                    "👉 必须确保当已选集 S 为空时公式可计算（例如使用 if/else 逻辑或 max(len(S), 1)）。",
                    "👉 请检查是否遗漏了安全算子，强制使用 safe_div(a, b) 和 log(abs(x) + EPS)。"
                ],
                detailed_analysis={'error': result.error_message}
            )
        
        # 2. 提取基准和当前表现
        best_sota = get_best_method()
        sota_values = list(best_sota.performance.values())
        sota_avg = np.mean(sota_values) if sota_values else 0.0
        beat_sota = result.avg_accuracy > sota_avg
        
        valid_results = {
            k: v for k, v in result.dataset_results.items()
            if isinstance(v.get('accuracy'), (int, float)) and v['accuracy'] > 0
        }
        
        best_dataset, worst_dataset = None, None
        suggestions = []
        
        # 3. 数据集表现诊断
        if valid_results:
            sorted_datasets = sorted(
                valid_results.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            best_dataset = sorted_datasets[0][0]
            worst_dataset = sorted_datasets[-1][0]
            
            acc_values = [v['accuracy'] for v in valid_results.values()]
            acc_std = np.std(acc_values)
            
            worst_acc = valid_results[worst_dataset]['accuracy']
            suggestions.append(f"📉 在 {worst_dataset} 上表现最差 (准确率: {worst_acc:.4f})。")
            
            # 结合算子库给出建议
            if acc_std > 0.1:
                suggestions.append("⚠️ 跨数据集方差大。建议降低高阶互信息项的权重，或者使用 `nmi(X, Y)` 替代 `mi(X, Y)` 以统一度量尺度。")
        
        # 4. SOTA 对比诊断
        if beat_sota:
            overall_verdict = 'success'
            suggestions.append(f"🎉 表现优异！平均准确率 ({result.avg_accuracy:.4f}) 超越当前 SOTA ({sota_avg:.4f})。")
        else:
            overall_verdict = 'needs_improvement'
            delta = sota_avg - result.avg_accuracy
            suggestions.append(f"📊 平均准确率落后 SOTA {delta:.4f}。建议：尝试使用 `cmi(X, Y, s)` 来捕捉特征与已选集之间的条件互补性，而不仅仅是简单的 `redundancy(X, S)` 惩罚。")

        # 5. 数值稳定性诊断
        if result.validation_issues:
            issues_str = ', '.join(result.validation_issues[:2])
            suggestions.append(f"⚡ 数值警告: {issues_str}。请严格包裹 `safe_div` 或 `log(abs(x) + EPS)`。")

        detailed_analysis = {
            'avg_accuracy': result.avg_accuracy,
            'sota_avg_accuracy': sota_avg,
            'delta': result.avg_accuracy - sota_avg,
            'n_datasets': len(valid_results),
            'accuracy_std': acc_std if valid_results else 0,
            'per_dataset': {k: v.get('accuracy', 0) for k, v in result.dataset_results.items()}
        }
        
        return Criticism(overall_verdict, beat_sota, best_dataset, worst_dataset, suggestions, detailed_analysis)
    
    @staticmethod
    def generate_improvement_prompt(original_formula: str, 
                                    result: EvaluationResult,
                                    criticism: Criticism) -> str:
        """生成严格贴合创新任务要求的改进提示词"""
        
        # 格式化各数据集表现
        dataset_performance = "\n".join([
            f"  - {name}: {res.get('accuracy', 'N/A')}{'' if isinstance(res.get('accuracy'), str) else ':.4f'}"
            for name, res in result.dataset_results.items()
        ])
        
        # 格式化批评建议
        suggestions_list = "\n".join(f"- {s}" for s in criticism.suggestions)

        # 构建最终 Prompt
        prompt = f"""
# 🔄 算子迭代与优化任务

你上一次设计的特征选择准则函数已经完成评估。虽然有一定效果，但仍需结合反馈进行迭代优化。

## 🔬 上次迭代回顾
**你提交的公式:** `{original_formula}`

**整体表现:**
- 平均准确率: {result.avg_accuracy:.4f}
- 最佳数据集: {criticism.best_dataset or 'N/A'}
- 最差数据集: {criticism.worst_dataset or 'N/A'}

**各数据集明细:**
{dataset_performance}

## 🎯 诊断意见与优化方向
{suggestions_list}

## 🛠️ 改进指南 (结合算子库)
1. **相关性与互补性**: 如果你的公式只有 `mi(X, Y)`，尝试引入 `mean([cmi(X, Y, s) for s in S])` 来挖掘条件增益。
2. **冗余度控制**: 简单的 `redundancy(X, S)` 可能不够，可以尝试非线性组合，比如 `max_redundancy(X, S)`，或者利用 `entropy(X)` 进行归一化。
3. **边界与稳定**: 必须处理 $S=\emptyset$ 的情况。严禁裸露的 `/`，务必替换为 `safe_div(..., ...)`。

## 📝 输出要求
请基于上述反馈，设计一个全新的或大幅优化的公式，并严格遵循以下 JSON 格式输出：

```json
{{
    "method_name": "方法名称 (如 RobustCMI_Selector)",
    "formula_dsl": "DSL 公式 (仅使用提供的算子)",
    "formula_math": "LaTeX 数学公式",
    "intuition": "设计直觉 (中文，50字以内，利用信息论相关内容进行解释)",
    "theoretical_advantage": "与上一个公式相比的改进点 (中文，100字以内)"
}}
"""
        return prompt