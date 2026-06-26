# src/llm/cost_tracker.py

"""
LLM 调用成本追踪
"""
import os
import json
from datetime import datetime, date
from typing import Dict, Optional
from dataclasses import dataclass, field
import threading


@dataclass
class CostRecord:
    """成本记录"""
    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class CostTracker:
    """成本追踪器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, log_dir: str = "./outputs/cost_logs"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(log_dir)
        return cls._instance
    
    def _initialize(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.records: Dict[date, list] = {}
        self._load_today()
    
    def _load_today(self):
        """加载今日记录"""
        today = date.today()
        log_file = self._get_log_file(today)
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    self.records[today] = [
                        CostRecord(**r) for r in data.get('records', [])
                    ]
            except:
                self.records[today] = []
        else:
            self.records[today] = []
    
    def _get_log_file(self, d: date) -> str:
        """获取日志文件路径"""
        return os.path.join(self.log_dir, f"cost_{d.isoformat()}.json")
    
    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        """记录一次调用"""
        today = date.today()
        
        record = CostRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd
        )
        
        if today not in self.records:
            self.records[today] = []
        
        self.records[today].append(record)
        
        # 保存
        self._save(today)
    
    def _save(self, d: date):
        """保存记录"""
        log_file = self._get_log_file(d)
        
        data = {
            'date': d.isoformat(),
            'records': [
                {
                    'timestamp': r.timestamp,
                    'provider': r.provider,
                    'model': r.model,
                    'prompt_tokens': r.prompt_tokens,
                    'completion_tokens': r.completion_tokens,
                    'cost_usd': r.cost_usd
                }
                for r in self.records.get(d, [])
            ]
        }
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_daily_cost(self, d: date = None) -> float:
        """获取每日成本"""
        d = d or date.today()
        return sum(r.cost_usd for r in self.records.get(d, []))
    
    def get_monthly_cost(self, year: int = None, month: int = None) -> float:
        """获取月度成本"""
        today = date.today()
        year = year or today.year
        month = month or today.month
        
        total = 0.0
        for d, records in self.records.items():
            if d.year == year and d.month == month:
                total += sum(r.cost_usd for r in records)
        
        return total
    
    def get_summary(self) -> Dict:
        """获取成本汇总"""
        today = date.today()
        
        return {
            'today': self.get_daily_cost(today),
            'this_month': self.get_monthly_cost(),
            'total_records': sum(len(r) for r in self.records.values())
        }
    
    def print_summary(self):
        """打印成本汇总"""
        summary = self.get_summary()
        
        print("\n" + "=" * 40)
        print("LLM 调用成本汇总")
        print("=" * 40)
        print(f"今日花费: ${summary['today']:.4f}")
        print(f"本月花费: ${summary['this_month']:.4f}")
        print(f"总调用次数: {summary['total_records']}")
        print("=" * 40)


# 全局追踪器
_tracker: Optional[CostTracker] = None

def get_cost_tracker() -> CostTracker:
    """获取全局成本追踪器"""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker