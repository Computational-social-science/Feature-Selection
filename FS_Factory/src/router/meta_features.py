# src/router/meta_features.py

"""
元特征提取
"""
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional

@dataclass
class MetaFeatures:
    """数据集元特征"""
    # 基本维度
    n_samples: int
    n_features: int
    n_classes: int
    
    # 比例
    samples_per_feature: float
    samples_per_class: float
    feature_to_sample_ratio: float
    
    # 分布特征
    mean_skewness: float
    mean_kurtosis: float
    mean_sparsity: float
    
    # 相关性
    mean_feature_corr: float
    max_feature_corr: float
    mean_target_corr: float
    max_target_corr: float
    
    # 信息论特征 (可选)
    mean_entropy: Optional[float] = None
    target_entropy: Optional[float] = None
    
    @classmethod
    def to_vector(cls, mf: 'MetaFeatures') -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            np.log1p(mf.n_samples),
            np.log1p(mf.n_features),
            mf.n_classes,
            mf.samples_per_feature,
            mf.samples_per_class,
            mf.feature_to_sample_ratio,
            mf.mean_skewness,
            mf.mean_kurtosis,
            mf.mean_sparsity,
            mf.mean_feature_corr,
            mf.max_feature_corr,
            mf.mean_target_corr,
            mf.max_target_corr,
        ])


class MetaFeatureExtractor:
    """元特征提取器"""
    
    def __init__(self, max_features_to_sample: int = 500):
        self.max_features_to_sample = max_features_to_sample
    
    def extract(self, X: np.ndarray, y: np.ndarray) -> MetaFeatures:
        """提取元特征"""
        
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 基本比例
        samples_per_feature = n_samples / n_features
        samples_per_class = n_samples / n_classes
        feature_to_sample_ratio = n_features / n_samples
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 采样特征 (避免大矩阵)
        if n_features > self.max_features_to_sample:
            sample_idx = np.random.choice(n_features, self.max_features_to_sample, replace=False)
            X_sample = X_scaled[:, sample_idx]
        else:
            X_sample = X_scaled
        
        # 分布特征
        skewness_list = []
        kurtosis_list = []
        sparsity_list = []
        
        for i in range(X_sample.shape[1]):
            col = X_sample[:, i]
            skewness_list.append(abs(stats.skew(col)))
            kurtosis_list.append(stats.kurtosis(col))
            
            # 稀疏度: 接近零的值比例
            threshold = np.std(col) * 0.01
            sparsity_list.append(np.mean(np.abs(col) < threshold))
        
        mean_skewness = np.mean(skewness_list)
        mean_kurtosis = np.mean(kurtosis_list)
        mean_sparsity = np.mean(sparsity_list)
        
        # 特征间相关性
        if X_sample.shape[1] <= 500:
            corr_matrix = np.abs(np.corrcoef(X_sample.T))
            np.fill_diagonal(corr_matrix, 0)
            mean_feature_corr = np.mean(corr_matrix)
            max_feature_corr = np.max(corr_matrix)
        else:
            # 采样估计
            n_pairs = min(1000, X_sample.shape[1] * (X_sample.shape[1] - 1) // 2)
            corrs = []
            for _ in range(n_pairs):
                i, j = np.random.choice(X_sample.shape[1], 2, replace=False)
                c = np.abs(np.corrcoef(X_sample[:, i], X_sample[:, j])[0, 1])
                corrs.append(c)
            mean_feature_corr = np.mean(corrs)
            max_feature_corr = np.max(corrs)
        
        # 特征-目标相关性
        f_scores, _ = f_classif(X_sample, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 归一化 F 分数
        if np.max(f_scores) > 0:
            f_scores_norm = f_scores / np.max(f_scores)
        else:
            f_scores_norm = np.zeros_like(f_scores)
        
        mean_target_corr = np.mean(f_scores_norm)
        max_target_corr = np.max(f_scores_norm)
        
        return MetaFeatures(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            samples_per_feature=samples_per_feature,
            samples_per_class=samples_per_class,
            feature_to_sample_ratio=feature_to_sample_ratio,
            mean_skewness=mean_skewness,
            mean_kurtosis=mean_kurtosis,
            mean_sparsity=mean_sparsity,
            mean_feature_corr=mean_feature_corr,
            max_feature_corr=max_feature_corr,
            mean_target_corr=mean_target_corr,
            max_target_corr=max_target_corr
        )
    
    def compute_similarity(self, mf1: MetaFeatures, mf2: MetaFeatures) -> float:
        """计算两个数据集的相似度"""
        v1 = MetaFeatures.to_vector(mf1)
        v2 = MetaFeatures.to_vector(mf2)
        
        # 加权欧氏距离
        weights = np.array([1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.3, 0.3, 0.3, 1.0, 0.5, 1.0, 0.5])
        
        # 归一化差异
        max_vals = np.maximum(np.abs(v1), np.abs(v2), 1e-10)
        normalized_diff = (v1 - v2) / max_vals
        
        weighted_distance = np.sqrt(np.sum(weights * normalized_diff**2) / np.sum(weights))
        
        # 转换为相似度
        similarity = 1.0 / (1.0 + weighted_distance)
        
        return similarity