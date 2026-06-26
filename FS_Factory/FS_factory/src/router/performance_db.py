# src/router/performance_db.py

"""
性能数据库
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
from datetime import datetime

from .meta_features import MetaFeatures

@dataclass
class PerformanceRecord:
    """性能记录"""
    id: str
    method_name: str
    meta_features: MetaFeatures
    accuracy: float
    selection_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'method_name': self.method_name,
            'meta_features': self.meta_features.__dict__,
            'accuracy': self.accuracy,
            'selection_time': self.selection_time,
            'timestamp': self.timestamp
        }


class PerformanceDatabase:
    """性能数据库"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "./outputs/performance_db.json"
        self.records: List[PerformanceRecord] = []
        self._load()
    
    def _load(self):
        """从文件加载"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                
                for item in data.get('records', []):
                    mf = MetaFeatures(**item['meta_features'])
                    record = PerformanceRecord(
                        id=item['id'],
                        method_name=item['method_name'],
                        meta_features=mf,
                        accuracy=item['accuracy'],
                        selection_time=item['selection_time'],
                        timestamp=item.get('timestamp', '')
                    )
                    self.records.append(record)
            except Exception as e:
                print(f"Warning: Failed to load performance db: {e}")
    
    def _save(self):
        """保存到文件"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        data = {
            'records': [r.to_dict() for r in self.records],
            'updated_at': datetime.now().isoformat()
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_record(self, record: PerformanceRecord):
        """添加记录"""
        self.records.append(record)
        self._save()
    
    def get_similar_records(self, meta_features: MetaFeatures,
                           top_k: int = 10,
                           similarity_threshold: float = 0.5) -> List[Tuple[float, PerformanceRecord]]:
        """获取相似记录"""
        from .meta_features import MetaFeatureExtractor
        extractor = MetaFeatureExtractor()
        
        similarities = []
        for record in self.records:
            sim = extractor.compute_similarity(meta_features, record.meta_features)
            if sim >= similarity_threshold:
                similarities.append((sim, record))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def get_best_method_for_features(self, meta_features: MetaFeatures,
                                     top_k: int = 10) -> List[Tuple[str, float]]:
        """获取最佳方法推荐"""
        similar_records = self.get_similar_records(meta_features, top_k)
        
        if not similar_records:
            return []
        
        # 聚合每个方法的得分
        method_scores: Dict[str, List[Tuple[float, float]]] = {}
        
        for sim, record in similar_records:
            if record.method_name not in method_scores:
                method_scores[record.method_name] = []
            method_scores[record.method_name].append((sim, record.accuracy))
        
        # 计算加权平均
        results = []
        for method, scores in method_scores.items():
            total_sim = sum(s for s, _ in scores)
            weighted_acc = sum(s * a for s, a in scores) / total_sim
            results.append((method, weighted_acc))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_method_stats(self, method_name: str) -> Dict:
        """获取方法统计信息"""
        method_records = [r for r in self.records if r.method_name == method_name]
        
        if not method_records:
            return {}
        
        accuracies = [r.accuracy for r in method_records]
        times = [r.selection_time for r in method_records]
        
        return {
            'n_runs': len(method_records),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'mean_time': np.mean(times)
        }