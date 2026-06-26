# src/validation/__init__.py

"""
验证与消融模块
"""
from .experiments import ExperimentRunner, ExperimentConfig
from .statistics import StatisticalValidator, StatisticalTestResult, run_statistical_analysis
from .report import generate_report
# from .visualizer import ResultVisualizer

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'StatisticalValidator',
    'StatisticalTestResult',
    'run_statistical_analysis',
    # 'ResultVisualizer',
    'generate_report',
]