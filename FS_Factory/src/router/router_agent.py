# src/router/router_agent.py

"""
路由智能体
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from .meta_features import MetaFeatures, MetaFeatureExtractor
from .performance_db import PerformanceDatabase, PerformanceRecord

@dataclass
class RoutingDecision:
    """路由决策"""
    selected_method: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]]
    meta_features: MetaFeatures


class RouterAgent:
    """路由智能体"""
    
    def __init__(self, perf_db: Optional[PerformanceDatabase] = None):
        self.extractor = MetaFeatureExtractor()
        self.perf_db = perf_db or PerformanceDatabase()
        
        # 已注册的方法
        self.methods: Dict[str, Callable] = {}
        
        # 规则库
        self.rules = self._init_rules()
    
    def _init_rules(self) -> List[Dict]:
        """初始化规则库"""
        return [
            {
                'name': 'high_dim_small_sample',
                'condition': lambda mf: mf.samples_per_feature < 10,
                'method': 'CMIM',
                'reason': '高维小样本，使用条件互信息方法'
            },
            {
                'name': 'high_redundancy',
                'condition': lambda mf: mf.mean_feature_corr > 0.5,
                'method': 'mRMR',
                'reason': '特征间高冗余，使用mRMR去冗余'
            },
            {
                'name': 'multi_class',
                'condition': lambda mf: mf.n_classes > 5,
                'method': 'JMI',
                'reason': '多分类任务，使用联合互信息'
            },
            {
                'name': 'weak_signal',
                'condition': lambda mf: mf.max_target_corr < 0.3,
                'method': 'CMIM',
                'reason': '弱信号特征，需要挖掘条件关系'
            },
            {
                'name': 'balanced',
                'condition': lambda mf: 0.3 <= mf.mean_feature_corr <= 0.5,
                'method': 'JMI',
                'reason': '中等冗余，使用平衡方法'
            },
        ]
    
    def register_method(self, name: str, func: Callable):
        """注册方法"""
        self.methods[name] = func
    
    def route(self, X: np.ndarray, y: np.ndarray,
              use_history: bool = True) -> RoutingDecision:
        """
        路由决策
        
        Args:
            X: 特征矩阵
            y: 目标变量
            use_history: 是否使用历史数据
        
        Returns:
            RoutingDecision
        """
        # 提取元特征
        meta_features = self.extractor.extract(X, y)
        
        # 基于历史的推荐
        history_recommendations = []
        if use_history:
            history_recommendations = self.perf_db.get_best_method_for_features(
                meta_features, top_k=5
            )
        
        # 基于规则的推荐
        rule_recommendations = self._apply_rules(meta_features)
        
        # 综合决策
        if history_recommendations:
            # 历史优先
            best_method = history_recommendations[0][0]
            confidence = min(0.9, 0.5 + len(history_recommendations) * 0.1)
            reasoning = f"基于 {len(history_recommendations)} 条相似历史记录推荐"
            alternatives = history_recommendations[1:4]
        
        elif rule_recommendations:
            # 规则兜底
            best_rule = rule_recommendations[0]
            best_method = best_rule['method']
            confidence = 0.6
            reasoning = best_rule['reason']
            alternatives = [(r['method'], 0.5) for r in rule_recommendations[1:4]]
        
        else:
            # 默认方法
            best_method = 'mRMR'
            confidence = 0.4
            reasoning = "无历史数据且无匹配规则，使用默认方法"
            alternatives = [('JMI', 0.3), ('CMIM', 0.3)]
        
        return RoutingDecision(
            selected_method=best_method,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            meta_features=meta_features
        )
    
    def _apply_rules(self, mf: MetaFeatures) -> List[Dict]:
        """应用规则"""
        matched_rules = []
        
        for rule in self.rules:
            try:
                if rule['condition'](mf):
                    matched_rules.append(rule)
            except Exception:
                pass
        
        return matched_rules
    
    def update_with_result(self, X: np.ndarray, y: np.ndarray,
                          method_name: str, accuracy: float,
                          selection_time: float):
        """更新性能数据库"""
        meta_features = self.extractor.extract(X, y)
        
        record = PerformanceRecord(
            id=f"{method_name}_{len(self.perf_db.records)}",
            method_name=method_name,
            meta_features=meta_features,
            accuracy=accuracy,
            selection_time=selection_time
        )
        
        self.perf_db.add_record(record)
    
    def get_meta_feature_description(self, X: np.ndarray, y: np.ndarray) -> str:
        """获取元特征描述"""
        mf = self.extractor.extract(X, y)
        
        desc = f"""
## 数据集元特征

### 维度
- 样本数: {mf.n_samples}
- 特征数: {mf.n_features}
- 类别数: {mf.n_classes}
- 样本/特征比: {mf.samples_per_feature:.2f}

### 分布
- 平均偏度: {mf.mean_skewness:.4f}
- 平均峰度: {mf.mean_kurtosis:.4f}
- 平均稀疏度: {mf.mean_sparsity:.4f}

### 相关性
- 特征间平均相关: {mf.mean_feature_corr:.4f}
- 特征-目标相关(均值): {mf.mean_target_corr:.4f}
- 特征-目标相关(最大): {mf.max_target_corr:.4f}
"""
        return desc