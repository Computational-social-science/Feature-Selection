# src/sandbox/executor.py

"""
沙箱执行器
"""
import numpy as np
import time
import traceback
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

# Scikit-learn and cuML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

from ..dsl import eval_dsl, DSLRegistry
from ..prompts import validate_formula
from .datasets import TestDataset, DatasetManager


CLASSIFIERS = {
    'bayes': GaussianNB,
    'rf': RandomForestClassifier,
    'svm': SVC,
    'knn': KNeighborsClassifier
}

CUML_CLASSIFIERS = {
    'rf': cuRF if HAS_CUML else None,
    'svm': cuSVC if HAS_CUML else None,
    'knn': cuKNN if HAS_CUML else None
}

@dataclass
class EvaluationResult:
    """评估结果"""
    success: bool
    method_name: str
    formula: str
    avg_accuracy: float
    dataset_results: Dict[str, dict]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    validation_issues: List[str] = field(default_factory=list)


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, formula: str):
        self.formula = formula
        self.validation_result = validate_formula(formula)
    
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       k: int, verbose: bool = False) -> List[int]:
        """
        前向特征选择
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量
            k: 选择的特征数
            verbose: 是否打印详细信息
        
        Returns:
            选中的特征索引列表
        """
        if not self.validation_result.is_valid:
            raise ValueError(f"Invalid formula: {self.validation_result.issues}")
        
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))
        
        for step in range(min(k, n_features)):
            best_score = -np.inf
            best_feature = None
            
            scores = []
            for feat_idx in remaining:
                try:
                    # 构建上下文
                    S_features = [X[:, i] for i in selected] if selected else []
                    
                    score = eval_dsl(
                        self.formula,
                        X=X[:, feat_idx],
                        Y=y,
                        S=S_features
                    )
                    
                    if np.isfinite(score):
                        scores.append((score, feat_idx))
                    else:
                        if verbose:
                            print(f"  Feature {feat_idx}: non-finite score")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Feature {feat_idx}: error - {str(e)[:50]}")
            
            if scores:
                # 选择得分最高的
                scores.sort(reverse=True)
                best_score, best_feature = scores[0]
                
                if verbose and step % 5 == 0:
                    print(f"  Step {step+1}: selected feature {best_feature}, score={best_score:.4f}")
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break
        
        return selected


class SandboxExecutor:
    """沙箱执行器"""
    
    def __init__(self, dataset_manager: Optional[DatasetManager] = None,
                 n_features_to_select: int = 30,
                 cv_folds: int = 5,
                 use_gpu: bool = False):
        self.dataset_manager = dataset_manager or DatasetManager()
        self.n_features_to_select = n_features_to_select
        self.cv_folds = cv_folds
        self.use_gpu = use_gpu and HAS_CUML
        
    def _get_classifier(self, name: str):
        """获取分类器实例"""
        if self.use_gpu and name in CUML_CLASSIFIERS and CUML_CLASSIFIERS[name] is not None:
            if name == 'rf':
                return CUML_CLASSIFIERS[name](n_estimators=100, random_state=42)
            elif name == 'svm':
                return CUML_CLASSIFIERS[name](probability=True, random_state=42)
            elif name == 'knn':
                return CUML_CLASSIFIERS[name](n_neighbors=5)
        
        # Fallback to CPU
        if name == 'rf':
            return CLASSIFIERS[name](n_estimators=100, random_state=42, n_jobs=-1)
        elif name == 'svm':
            return CLASSIFIERS[name](probability=True, random_state=42)
        elif name == 'knn':
            return CLASSIFIERS[name](n_neighbors=5)
        elif name == 'bayes':
            return CLASSIFIERS[name]()
        
        raise ValueError(f"Unknown classifier: {name}")
    
    def evaluate_formula(self, formula: str, method_name: str,
                        datasets: Optional[List[TestDataset]] = None,
                        verbose: bool = False) -> EvaluationResult:
        """
        评估公式
        
        Args:
            formula: DSL 公式
            method_name: 方法名称
            datasets: 测试数据集列表
            verbose: 是否打印详细信息
        
        Returns:
            EvaluationResult
        """
        start_time = time.time()
        
        # 验证公式
        validation = validate_formula(formula)
        
        if not validation.is_valid:
            return EvaluationResult(
                success=False,
                method_name=method_name,
                formula=formula,
                avg_accuracy=0.0,
                dataset_results={},
                error_message=f"Validation failed: {validation.issues}",
                validation_issues=validation.issues
            )
        
        # 获取数据集
        if datasets is None:
            datasets = self.dataset_manager.get_all_datasets()
        
        # 创建特征选择器
        try:
            selector = FeatureSelector(formula)
        except Exception as e:
            return EvaluationResult(
                success=False,
                method_name=method_name,
                formula=formula,
                avg_accuracy=0.0,
                dataset_results={},
                error_message=f"Selector creation failed: {str(e)}"
            )
        
        # 在每个数据集上评估
        dataset_results = {}
        accuracies = []
        
        for dataset in datasets:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Dataset: {dataset.name}")
                print(f"  Samples: {dataset.n_samples}, Features: {dataset.n_features}")
            
            try:
                # 特征选择
                select_start = time.time()
                selected = selector.select_features(
                    dataset.X, dataset.y,
                    k=min(self.n_features_to_select, dataset.n_features),
                    verbose=verbose
                )
                select_time = time.time() - select_start
                
                if verbose:
                    print(f"  Selected {len(selected)} features in {select_time:.2f}s")
                
                if not selected:
                    dataset_results[dataset.name] = {
                        'error': 'No features selected',
                        'accuracy': 0.0
                    }
                    continue
                
                # 评估
                X_selected = dataset.X[:, selected]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_selected)
                
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                
                # 多分类器评估
                clf_names = ['bayes', 'rf', 'svm', 'knn']
                clf_results = {}
                
                for clf_name in clf_names:
                    clf = self._get_classifier(clf_name)
                    scores = cross_val_score(
                        clf, X_scaled, dataset.y,
                        cv=cv, scoring='accuracy', n_jobs=-1 if not self.use_gpu else None
                    )
                    clf_results[f'{clf_name}_accuracy'] = scores.mean()
                    clf_results[f'{clf_name}_std'] = scores.std()
                
                # 综合准确率 (取平均)
                accuracy = np.mean([clf_results[f'{c}_accuracy'] for c in clf_names])
                std = np.mean([clf_results[f'{c}_std'] for c in clf_names])
                
                accuracies.append(accuracy)
                dataset_results[dataset.name] = {
                    'accuracy': accuracy,
                    'std': std,
                    'n_selected': len(selected),
                    'selection_time': select_time,
                    'selected_features': selected,
                    **clf_results
                }
                
                if verbose:
                    acc_str = ", ".join([f"{c}: {clf_results[f'{c}_accuracy']:.4f}" for c in clf_names])
                    print(f"  Accuracy: {accuracy:.4f} (Avg) | {acc_str}")
                
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                dataset_results[dataset.name] = {
                    'error': error_msg,
                    'accuracy': 0.0
                }
                if verbose:
                    print(f"  Error: {str(e)[:100]}")
        
        # 汇总结果
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            success=len(accuracies) > 0,
            method_name=method_name,
            formula=formula,
            avg_accuracy=avg_accuracy,
            dataset_results=dataset_results,
            execution_time=execution_time,
            validation_issues=validation.warnings
        )
    
    def compare_with_baseline(self, result: EvaluationResult,
                              baseline_results: Dict[str, EvaluationResult]) -> Dict:
        """与基线比较"""
        comparison = {
            'method': result.method_name,
            'avg_accuracy': result.avg_accuracy,
            'vs_baseline': {}
        }
        
        for baseline_name, baseline_result in baseline_results.items():
            delta = result.avg_accuracy - baseline_result.avg_accuracy
            comparison['vs_baseline'][baseline_name] = {
                'baseline_acc': baseline_result.avg_accuracy,
                'delta': delta,
                'win': delta > 0
            }
        
        return comparison