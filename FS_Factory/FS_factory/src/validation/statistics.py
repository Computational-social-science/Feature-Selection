# src/validation/statistics.py

"""
统计检验
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    winner: Optional[str] = None
    details: Dict = None


class StatisticalValidator:
    """统计检验器"""
    
    @staticmethod
    def paired_t_test(results: pd.DataFrame,
                     method1: str, method2: str,
                     alpha: float = 0.05) -> StatisticalTestResult:
        """配对 t 检验"""
        df = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        
        if method1 not in df.columns or method2 not in df.columns:
            return StatisticalTestResult(
                test_name='paired_t_test',
                statistic=0, p_value=1, significant=False,
                details={'error': 'Method not found'}
            )
        
        scores1 = df[method1].values
        scores2 = df[method2].values
        
        # 移除 NaN
        valid = ~(np.isnan(scores1) | np.isnan(scores2))
        scores1 = scores1[valid]
        scores2 = scores2[valid]
        
        if len(scores1) < 3:
            return StatisticalTestResult(
                test_name='paired_t_test',
                statistic=0, p_value=1, significant=False,
                details={'error': 'Insufficient data'}
            )
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        mean_diff = np.mean(scores1) - np.mean(scores2)
        winner = method1 if mean_diff > 0 else method2
        
        return StatisticalTestResult(
            test_name='paired_t_test',
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < alpha,
            winner=winner,
            details={
                'mean_diff': mean_diff,
                'n_pairs': len(scores1)
            }
        )
    
    @staticmethod
    def wilcoxon_test(results: pd.DataFrame,
                     method1: str, method2: str,
                     alpha: float = 0.05) -> StatisticalTestResult:
        """Wilcoxon 符号秩检验"""
        df = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        
        scores1 = df[method1].values
        scores2 = df[method2].values
        
        valid = ~(np.isnan(scores1) | np.isnan(scores2))
        scores1 = scores1[valid]
        scores2 = scores2[valid]
        
        if len(scores1) < 10:
            return StatisticalTestResult(
                test_name='wilcoxon',
                statistic=0, p_value=1, significant=False,
                details={'error': 'Need at least 10 pairs'}
            )
        
        # 检查是否有足够的非零差异
        diffs = scores1 - scores2
        if np.all(diffs == 0):
            return StatisticalTestResult(
                test_name='wilcoxon',
                statistic=0, p_value=1, significant=False,
                details={'error': 'All differences are zero'}
            )
        
        try:
            stat, p_value = stats.wilcoxon(scores1, scores2)
        except ValueError as e:
            return StatisticalTestResult(
                test_name='wilcoxon',
                statistic=0, p_value=1, significant=False,
                details={'error': str(e)}
            )
        
        winner = method1 if np.mean(scores1) > np.mean(scores2) else method2
        
        return StatisticalTestResult(
            test_name='wilcoxon',
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
            winner=winner
        )
    
    @staticmethod
    def friedman_test(results: pd.DataFrame,
                     alpha: float = 0.05) -> StatisticalTestResult:
        """Friedman 检验 (多方法比较)"""
        df = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        
        # 移除有 NaN 的行
        df = df.dropna()
        
        if len(df) < 3:
            return StatisticalTestResult(
                test_name='friedman',
                statistic=0, p_value=1, significant=False,
                details={'error': 'Insufficient datasets'}
            )
        
        n_datasets, n_methods = df.shape
        
        # 计算每行的排名
        ranks = df.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean()
        
        # Friedman 统计量
        chi2 = (12 * n_datasets / (n_methods * (n_methods + 1))) * \
               (np.sum(avg_ranks**2) - n_methods * (n_methods + 1)**2 / 4)
        
        p_value = 1 - stats.chi2.cdf(chi2, n_methods - 1)
        
        best_method = avg_ranks.idxmin()
        
        return StatisticalTestResult(
            test_name='friedman',
            statistic=chi2,
            p_value=p_value,
            significant=p_value < alpha,
            winner=best_method,
            details={
                'average_ranks': avg_ranks.to_dict(),
                'n_datasets': n_datasets,
                'n_methods': n_methods
            }
        )
    
    @staticmethod
    def nemenyi_posthoc(results: pd.DataFrame,
                        alpha: float = 0.05) -> pd.DataFrame:
        """Nemenyi 事后检验"""
        df = results.pivot(index='dataset', columns='method', values='mean_accuracy')
        df = df.dropna()
        
        n_datasets, n_methods = df.shape
        
        # 平均排名
        ranks = df.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().sort_values()
        
        # 临界差
        q_alpha = {0.05: 2.343, 0.10: 2.160}  # for n_methods=5, 需要根据实际调整
        q = q_alpha.get(alpha, 2.343)
        cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
        
        # 计算成对差异
        methods = avg_ranks.index.tolist()
        diff_matrix = pd.DataFrame(index=methods, columns=methods)
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i != j:
                    rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
                    diff_matrix.loc[m1, m2] = rank_diff
        
        return diff_matrix


def run_statistical_analysis(results: pd.DataFrame,
                            baseline_method: str = 'mRMR') -> Dict:
    """运行完整统计分析"""
    methods = results['method'].unique()
    
    analysis = {
        'friedman': StatisticalValidator.friedman_test(results),
        'pairwise': {},
        'summary': {}
    }
    
    # 与基线的成对比较
    for method in methods:
        if method != baseline_method:
            analysis['pairwise'][f'{method}_vs_{baseline_method}'] = \
                StatisticalValidator.paired_t_test(results, method, baseline_method)
    
    # 汇总
    summary = results.groupby('method')['mean_accuracy'].agg(['mean', 'std'])
    summary['rank'] = summary['mean'].rank(ascending=False)
    analysis['summary'] = summary.sort_values('rank')
    
    return analysis