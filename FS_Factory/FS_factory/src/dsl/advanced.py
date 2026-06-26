# src/dsl/advanced.py

"""
高级特征选择算子
"""
import numpy as np
from typing import List, Optional, Union
from .operators import BaseOperators, DSLOptions

class AdvancedOperators(BaseOperators):
    """高级特征选择算子"""
    
    def __init__(self, options: Optional[DSLOptions] = None):
        super().__init__(options)
    
    # ==================== 归一化互信息 ====================
    
    def normalized_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        归一化互信息 NMI(X,Y)
        
        公式: $NMI(X;Y) = \frac{I(X;Y)}{\sqrt{H(X) \cdot H(Y)}}$
        """
        mi = self.mutual_info(X, Y)
        h_x = self.entropy(X)
        h_y = self.entropy(Y)
        
        denominator = np.sqrt(h_x * h_y) + self.eps
        return self._safe_div(mi, denominator, 0.0)
    
    def symmetric_uncertainty(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        对称不确定性 SU(X,Y)
        
        公式: $SU(X;Y) = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)}$
        """
        mi = self.mutual_info(X, Y)
        h_x = self.entropy(X)
        h_y = self.entropy(Y)
        
        denominator = h_x + h_y + self.eps
        return self._safe_div(2 * mi, denominator, 0.0)
    
    # ==================== 冗余度 ====================
    
    def redundancy(self, X: np.ndarray, S: List[np.ndarray]) -> float:
        """
        冗余度 R(X, S)
        
        公式: $R(X, S) = \frac{1}{|S|} \sum_{s \in S} I(X; s)$
        """
        if not S:
            return 0.0
        
        total_mi = sum(self.mutual_info(X, s) for s in S)
        return total_mi / len(S)
    
    def max_redundancy(self, X: np.ndarray, S: List[np.ndarray]) -> float:
        """
        最大冗余度
        
        公式: $R_{max}(X, S) = \max_{s \in S} I(X; s)$
        """
        if not S:
            return 0.0
        
        return max(self.mutual_info(X, s) for s in S)
    
    # ==================== 条件信息增益 ====================
    
    def conditional_redundancy(self, X: np.ndarray, Y: np.ndarray,
                              S: List[np.ndarray]) -> float:
        """
        条件冗余度
        
        公式: $CR(X, Y, S) = \frac{1}{|S|} \sum_{s \in S} I(X; Y | s)$
        """
        if not S:
            return self.mutual_info(X, Y)
        
        total_cmi = sum(self.conditional_mutual_info(X, Y, s) for s in S)
        return total_cmi / len(S)
    
    def information_gain(self, X: np.ndarray, Y: np.ndarray,
                        S: List[np.ndarray]) -> float:
        """
        信息增益 IG(X, Y, S)
        
        公式: $IG(X, Y, S) = I(X; Y) - R(X, S) + CR(X, Y, S)$
        """
        mi = self.mutual_info(X, Y)
        red = self.redundancy(X, S)
        cond_red = self.conditional_redundancy(X, Y, S)
        
        return mi - red + cond_red
    
    # ==================== 交互信息 ====================
    
    def interaction_info(self, X: np.ndarray, Y: np.ndarray,
                        Z: np.ndarray) -> float:
        """
        交互信息 (多变量)
        
        公式: $I(X;Y;Z) = I(X;Y) - I(X;Y|Z)$
        """
        mi = self.mutual_info(X, Y)
        cmi = self.conditional_mutual_info(X, Y, Z)
        
        return mi - cmi
    
    # ==================== 特征间距离 ====================
    
    def feature_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        特征距离 (基于互信息)
        
        公式: $D(X, Y) = 1 - NMI(X, Y)$
        """
        nmi = self.normalized_mi(X, Y)
        return 1.0 - nmi
    
    def conditional_distance(self, X: np.ndarray, Y: np.ndarray,
                            Z: np.ndarray) -> float:
        """
        条件距离
        
        公式: $D(X, Y | Z) = 1 - \frac{I(X;Y|Z)}{\sqrt{H(X|Z) \cdot H(Y|Z)}}$
        """
        cmi = self.conditional_mutual_info(X, Y, Z)
        h_x_z = self.conditional_entropy(X, Z)
        h_y_z = self.conditional_entropy(Y, Z)
        
        denominator = np.sqrt(h_x_z * h_y_z) + self.eps
        ncmi = self._safe_div(cmi, denominator, 0.0)
        
        return 1.0 - ncmi


# ==================== 聚合算子 ====================

class AggregationOperators:
    """聚合算子"""
    
    @staticmethod
    def mean(values: List[float]) -> float:
        """平均值"""
        if not values:
            return 0.0
        return np.mean(values)
    
    @staticmethod
    def max_val(values: List[float]) -> float:
        """最大值"""
        if not values:
            return 0.0
        return np.max(values)
    
    @staticmethod
    def min_val(values: List[float]) -> float:
        """最小值"""
        if not values:
            return 0.0
        return np.min(values)
    
    @staticmethod
    def sum_val(values: List[float]) -> float:
        """求和"""
        if not values:
            return 0.0
        return np.sum(values)
    
    @staticmethod
    def weighted_sum(values: List[float], weights: List[float]) -> float:
        """加权和"""
        if not values or not weights:
            return 0.0
        return np.sum(np.array(values) * np.array(weights[:len(values)]))