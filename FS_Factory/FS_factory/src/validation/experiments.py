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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# 尝试导入 cuML 进行 GPU 加速
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    HAS_CUML = True
    print("✅ cuML (GPU) 已成功导入，可以使用 GPU 加速评估。")
except ImportError:
    HAS_CUML = False
    # 这里不打印，由启动脚本决定是否报警

from ..dsl import eval_dsl
from ..sandbox.datasets import TestDataset

@dataclass
class ExperimentConfig:
    """实验配置"""
    n_repeats: int = 10
    n_folds: int = 5
    n_features_to_select: int = 30
    random_state: int = 42
    use_gpu: bool = False
    n_jobs: int = -1  # 用于并行特征选择


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.results_df: Optional[pd.DataFrame] = None
    
    def run_comparison(self,
                       methods: Dict[str, str],
                       datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       classifiers: Optional[List[str]] = None,
                       verbose: bool = True) -> pd.DataFrame:
        """
        运行对比实验
        
        Args:
            methods: 方法字典 {name: formula}
            datasets: 数据集字典 {name: (X, y)}
            classifiers: 分类器列表 ['bayes', 'rf', 'svm', 'knn']
            verbose: 是否打印进度
        
        Returns:
            结果 DataFrame
        """
        if classifiers is None:
            classifiers = ['rf']
            
        results = []
        
        for dataset_name, (X, y) in datasets.items():
            dataset_results = self.run_single_dataset(dataset_name, X, y, methods, classifiers, verbose)
            results.extend(dataset_results)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def run_single_dataset(self,
                           dataset_name: str,
                           X: np.ndarray,
                           y: np.ndarray,
                           methods: Dict[str, str],
                           classifiers: List[str],
                           verbose: bool = True) -> List[Dict]:
        """
        运行单个数据集的对比实验
        
        Args:
            dataset_name: 数据集名称
            X: 特征矩阵
            y: 标签向量
            methods: 方法字典 {name: formula}
            classifiers: 分类器列表
            verbose: 是否打印进度
            
        Returns:
            结果字典列表
        """
        results = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
        
        for method_name, formula in methods.items():
            if verbose:
                print(f"  Testing {method_name}...", end=" ")
            
            # 特征选择
            try:
                start_time = time.time()
                selected = self._select_features(X, y, formula)
                select_time = time.time() - start_time
                
                if not selected:
                    if verbose:
                        print("No features selected.")
                    continue
                
                # 对每个分类器进行评估
                method_results = {
                    'dataset': dataset_name,
                    'method': method_name,
                    'selection_time': select_time
                }
                
                for clf_name in classifiers:
                    scores = []
                    # 减少重复次数以加快速度
                    n_reps = max(1, self.config.n_repeats // 2)
                    for repeat in range(n_reps): 
                        score = self._evaluate_features(X, y, selected, classifier=clf_name)
                        scores.append(score)
                    
                    mean_acc = np.mean(scores)
                    std_acc = np.std(scores)
                    method_results[f'{clf_name}_accuracy'] = mean_acc
                    method_results[f'{clf_name}_std'] = std_acc
                
                results.append(method_results)
                
                if verbose:
                    acc_str = ", ".join([f"{c}: {method_results[f'{c}_accuracy']:.4f}" for c in classifiers])
                    print(f"Done. {acc_str}")
                    
            except Exception as e:
                if verbose:
                    print(f"Error testing {method_name}: {str(e)}")
        
        return results
    
    def _select_features(self, X: np.ndarray, y: np.ndarray,
                        formula: str) -> List[int]:
        """特征选择：使用 Joblib 并行加速评估"""
        n_features = X.shape[1]
        k = min(self.config.n_features_to_select, n_features)
        selected = []
        remaining = list(range(n_features))
        
        # 预先准备 S 列表，但在每一轮迭代中，S 包含的特征数会增加
        for _ in range(k):
            # 获取当前已选特征的列表 (S 列表)
            S_features = [X[:, i] for i in selected] if selected else []
            
            # 使用并行加速对所有剩余特征进行评估
            def evaluate_single_feature(feat_idx):
                try:
                    score = eval_dsl(formula, X=X[:, feat_idx], Y=y, S=S_features)
                    return feat_idx, score if np.isfinite(score) else -np.inf
                except:
                    return feat_idx, -np.inf

            # 并行执行
            results = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(evaluate_single_feature)(feat_idx) for feat_idx in remaining
            )
            
            # 找到得分最高的特征
            best_feat, best_score = max(results, key=lambda x: x[1])
            
            if best_score > -np.inf:
                selected.append(best_feat)
                remaining.remove(best_feat)
            else:
                break
        
        return selected
    
    def _evaluate_features(self, X: np.ndarray, y: np.ndarray,
                          selected: List[int],
                          classifier: str = 'rf') -> float:
        """评估特征：支持 GPU (cuML) 加速"""
        X_sel = X[:, selected]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        
        # 尝试使用 GPU 加速
        use_gpu = self.config.use_gpu and HAS_CUML
        
        if classifier == 'rf':
            if use_gpu:
                clf = cuRF(
                    n_estimators=100,
                    random_state=self.config.random_state
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
        elif classifier == 'bayes':
            clf = GaussianNB() # Bayes 目前 cuML 不常用，使用 CPU 版本
        elif classifier == 'svm':
            if use_gpu:
                clf = cuSVC(probability=True, random_state=self.config.random_state)
            else:
                clf = SVC(probability=True, random_state=self.config.random_state)
        elif classifier == 'knn':
            if use_gpu:
                clf = cuKNN(n_neighbors=5)
            else:
                clf = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # 注意：cuML 的 cross_val_score 可能不完全兼容 sklearn
        # 对于 cuML，通常建议在 GPU 上直接处理，这里我们保持 sklearn 的 CV 逻辑
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