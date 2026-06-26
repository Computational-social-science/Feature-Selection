# src/validation/__init__.py

"""
验证与消融模块
"""
from .experiments import ExperimentRunner, ExperimentConfig
from .statistics import StatisticalValidator, StatisticalTestResult, run_statistical_analysis
from .visualizer import ResultVisualizer, generate_report

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'StatisticalValidator',
    'StatisticalTestResult',
    'run_statistical_analysis',
    'ResultVisualizer',
    'generate_report',
]