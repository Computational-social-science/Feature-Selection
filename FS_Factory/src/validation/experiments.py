# src/validation/experiments.py

"""
实验运行器
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..dsl import eval_dsl
from ..sandbox.datasets import TestDataset

@dataclass
class ExperimentConfig:
    """实验配置"""
    n_repeats: int = 10
    n_folds: int = 5
    n_features_to_select: int = 30
    random_state: int = 42


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results_df: Optional[pd.DataFrame] = None
    
    def run_comparison(self,
                       methods: Dict[str, str],
                       datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       verbose: bool = True) -> pd.DataFrame:
        """
        运行对比实验
        
        Args:
            methods: 方法字典 {name: formula}
            datasets: 数据集字典 {name: (X, y)}
            verbose: 是否打印进度
        
        Returns:
            结果 DataFrame
        """
        results = []
        
        for dataset_name, (X, y) in datasets.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            for method_name, formula in methods.items():
                if verbose:
                    print(f"  Testing {method_name}...", end=" ")
                
                repeat_scores = []
                repeat_times = []
                
                for repeat in range(self.config.n_repeats):
                    try:
                        # 特征选择
                        start_time = time.time()
                        selected = self._select_features(X, y, formula)
                        select_time = time.time() - start_time
                        
                        if not selected:
                            repeat_scores.append(np.nan)
                            continue
                        
                        # 评估
                        score = self._evaluate_features(X, y, selected)
                        repeat_scores.append(score)
                        repeat_times.append(select_time)
                        
                    except Exception as e:
                        if verbose:
                            print(f"Error in repeat {repeat}: {str(e)[:50]}")
                        repeat_scores.append(np.nan)
                
                # 统计
                valid_scores = [s for s in repeat_scores if np.isfinite(s)]
                
                if valid_scores:
                    mean_acc = np.mean(valid_scores)
                    std_acc = np.std(valid_scores)
                    mean_time = np.mean(repeat_times) if repeat_times else 0
                else:
                    mean_acc = 0.0
                    std_acc = 0.0
                    mean_time = 0.0
                
                results.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'mean_accuracy': mean_acc,
                    'std_accuracy': std_acc,
                    'mean_time': mean_time,
                    'n_valid': len(valid_scores)
                })
                
                if verbose:
                    print(f"{mean_acc:.4f} ± {std_acc:.4f}")
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def _select_features(self, X: np.ndarray, y: np.ndarray,
                        formula: str) -> List[int]:
        """特征选择"""
        n_features = X.shape[1]
        k = min(self.config.n_features_to_select, n_features)
        selected = []
        remaining = list(range(n_features))
        
        for _ in range(k):
            best_score = -np.inf
            best_feat = None
            
            for feat_idx in remaining:
                try:
                    S_features = [X[:, i] for i in selected] if selected else []
                    score = eval_dsl(formula, X=X[:, feat_idx], Y=y, S=S_features)
                    
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_feat = feat_idx
                except:
                    continue
            
            if best_feat is not None:
                selected.append(best_feat)
                remaining.remove(best_feat)
            else:
                break
        
        return selected
    
    def _evaluate_features(self, X: np.ndarray, y: np.ndarray,
                          selected: List[int]) -> float:
        """评估特征"""
        X_sel = X[:, selected]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        return scores.mean()
    
    def get_summary(self) -> pd.DataFrame:
        """获取汇总"""
        if self.results_df is None:
            return pd.DataFrame()
        
        summary = self.results_df.groupby('method').agg({
            'mean_accuracy': ['mean', 'std'],
            'mean_time': 'mean'
        }).round(4)
        
        summary.columns = ['avg_accuracy', 'std_accuracy', 'avg_time']
        summary = summary.sort_values('avg_accuracy', ascending=False)
        
        return summary