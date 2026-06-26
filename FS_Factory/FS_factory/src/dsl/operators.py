# src/dsl/operators.py

"""
特征选择算子 DSL 库 - 基础算子
提供信息论基础函数，确保数值稳定性
"""
import numpy as np
from scipy import stats
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Optional, Tuple
import warnings
from dataclasses import dataclass

# 数值稳定性常量
EPS = 1e-10
LOG_EPS = np.log(EPS)

@dataclass
class DSLOptions:
    """DSL 配置选项"""
    eps: float = 1e-10
    log_base: float = 2.0
    knn_k: int = 5
    knn_algorithm: str = 'auto'

class BaseOperators:
    """基础信息论算子"""
    
    def __init__(self, options: Optional[DSLOptions] = None):
        self.options = options or DSLOptions()
        self.eps = self.options.eps
        self.log_base = self.options.log_base
    
    # ==================== 工具函数 ====================
    
    def _safe_log(self, x: np.ndarray) -> np.ndarray:
        """安全对数"""
        return np.log(np.maximum(x, self.eps))
    
    def _safe_div(self, numerator: float, denominator: float, 
                  default: float = 0.0) -> float:
        """安全除法"""
        if np.abs(denominator) < self.eps:
            return default
        return numerator / denominator
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """标准化"""
        mean = np.mean(x)
        std = np.std(x)
        if std < self.eps:
            return x - mean
        return (x - mean) / std
    
    # ==================== 熵估计 ====================
    
    def entropy(self, X: np.ndarray) -> float:
        """
        计算熵 H(X)
        
        公式: $H(X) = -\sum_{x} p(x) \log p(x)$
        
        对于连续变量使用 kNN 估计
        """
        X = np.asarray(X).flatten().reshape(-1, 1)
        n = len(X)
        
        if n == 0:
            return 0.0
        
        # 判断离散还是连续
        n_unique = len(np.unique(X))
        
        if n_unique < max(10, n * 0.05):
            # 离散型：直方图估计
            _, counts = np.unique(X, return_counts=True)
            probs = counts / n
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log(probs))
        else:
            # 连续型：kNN 估计
            h = self._knn_entropy(X)
        
        # 转换对数底
        if self.log_base != np.e:
            h = h / np.log(self.log_base)
        
        return max(0.0, h)
    
    def _knn_entropy(self, X: np.ndarray) -> float:
        """基于 kNN 的熵估计"""
        n, d = X.shape
        
        if n <= self.options.knn_k + 1:
            return 0.0
        
        # 标准化
        X = self._normalize_data(X)
        
        # kNN 搜索
        k = min(self.options.knn_k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=self.options.knn_algorithm)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # 第 k 个邻居的距离
        r = distances[:, -1] + self.eps
        
        # Kraskov 熵估计
        h = (digamma(n) - digamma(k) + 
             d * np.mean(self._safe_log(2 * np.pi * r * r)) + 
             d / 2 * np.log(np.pi) - 
             np.log(gamma(d / 2 + 1)))
        
        return h
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """数据标准化"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + self.eps
        return (X - mean) / std
    
    # ==================== 联合熵 ====================
    
    def joint_entropy(self, *arrays: np.ndarray) -> float:
        """
        联合熵 H(X1, X2, ..., Xn)
        
        公式: $H(X_1, ..., X_n) = -\sum_{x_1,...,x_n} p(x_1,...,x_n) \log p(x_1,...,x_n)$
        """
        if len(arrays) == 0:
            return 0.0
        
        if len(arrays) == 1:
            return self.entropy(arrays[0])
        
        # 合并为矩阵
        X = np.column_stack([a.flatten().reshape(-1, 1) for a in arrays])
        h = self._knn_entropy(X)
        
        if self.log_base != np.e:
            h = h / np.log(self.log_base)
        
        return max(0.0, h)
    
    # ==================== 条件熵 ====================
    
    def conditional_entropy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        条件熵 H(X|Y)
        
        公式: $H(X|Y) = H(X,Y) - H(Y)$
        """
        h_xy = self.joint_entropy(X, Y)
        h_y = self.entropy(Y)
        return max(0.0, h_xy - h_y)
    
    # ==================== 互信息 ====================
    
    def mutual_info(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        互信息 I(X;Y)
        
        公式: $I(X;Y) = H(X) + H(Y) - H(X,Y)$
        """
        h_x = self.entropy(X)
        h_y = self.entropy(Y)
        h_xy = self.joint_entropy(X, Y)
        
        mi = h_x + h_y - h_xy
        return max(0.0, mi)
    
    def conditional_mutual_info(self, X: np.ndarray, Y: np.ndarray, 
                                Z: np.ndarray) -> float:
        """
        条件互信息 I(X;Y|Z)
        
        公式: $I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)$
        
        展开: $I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)$
        """
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        
        h_xz = self.joint_entropy(X, Z)
        h_yz = self.joint_entropy(Y, Z)
        h_xyz = self.joint_entropy(X, Y, Z)
        h_z = self.entropy(Z)
        
        cmi = h_xz + h_yz - h_xyz - h_z
        return max(0.0, cmi)
    
    # ==================== 相关性 ====================
    
    def correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Pearson 相关系数
        """
        X = X.flatten()
        Y = Y.flatten()
        
        if np.std(X) < self.eps or np.std(Y) < self.eps:
            return 0.0
        
        corr = np.corrcoef(X, Y)[0, 1]
        
        if np.isnan(corr):
            return 0.0
        
        return corr
    
    def spearman_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Spearman 秩相关系数"""
        X = X.flatten()
        Y = Y.flatten()
        
        corr, _ = stats.spearmanr(X, Y)
        
        if np.isnan(corr):
            return 0.0
        
        return corr