# src/prompts/validator.py

"""
公式验证器
"""
import re
import ast
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    formula: str
    
    def __bool__(self):
        return self.is_valid


class FormulaValidator:
    """公式验证器"""
    
    # 危险关键字
    DANGEROUS_KEYWORDS = [
        'import', 'exec', 'eval', 'compile', 'open', 'file',
        '__import__', '__builtins__', 'globals', 'locals',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'os.', 'sys.', 'subprocess', 'pickle', 'marshal',
    ]
    
    # 允许的 DSL 函数
    ALLOWED_FUNCTIONS = {
        # 信息论
        'entropy', 'H', 'joint_entropy', 'cond_entropy',
        'mi', 'I', 'cmi', 'nmi', 'su',
        # 冗余
        'redundancy', 'R', 'max_redundancy', 'cond_redundancy',
        # 增益
        'ig', 'interaction',
        # 距离
        'feature_distance', 'cond_distance',
        # 相关性
        'corr', 'spearman',
        # 聚合
        'sum', 'mean', 'max', 'min', 'weighted_sum',
        # 数学
        'log', 'sqrt', 'abs', 'pow', 'exp',
        # 安全
        'safe_div', 'clip',
        # 内置
        'len', 'range', 'list', 'float', 'int',
    }
    
    # 允许的变量
    ALLOWED_VARIABLES = {'X', 'Y', 'S', 's', 'EPS', 'np'}
    
    @classmethod
    def validate(cls, formula: str) -> ValidationResult:
        """
        完整验证公式
        
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # 1. 语法检查
        syntax_issues = cls._check_syntax(formula)
        issues.extend(syntax_issues)
        
        if issues:  # 语法错误则不再继续
            return ValidationResult(False, issues, warnings, formula)
        
        # 2. 安全检查
        security_issues = cls._check_security(formula)
        issues.extend(security_issues)
        
        # 3. 数值稳定性检查
        stability_issues, stability_warnings = cls._check_numerical_stability(formula)
        issues.extend(stability_issues)
        warnings.extend(stability_warnings)
        
        # 4. DSL 函数检查
        dsl_issues, dsl_warnings = cls._check_dsl_functions(formula)
        issues.extend(dsl_issues)
        warnings.extend(dsl_warnings)
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, formula)
    
    @classmethod
    def _check_syntax(cls, formula: str) -> List[str]:
        """检查语法"""
        issues = []
        
        try:
            ast.parse(formula)
        except SyntaxError as e:
            issues.append(f"语法错误: {str(e)}")
        
        # 检查括号匹配
        if formula.count('(') != formula.count(')'):
            issues.append("括号不匹配")
        
        if formula.count('[') != formula.count(']'):
            issues.append("方括号不匹配")
        
        if formula.count('{') != formula.count('}'):
            issues.append("花括号不匹配")
        
        return issues
    
    @classmethod
    def _check_security(cls, formula: str) -> List[str]:
        """检查安全性"""
        issues = []
        
        formula_lower = formula.lower()
        
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword.lower() in formula_lower:
                issues.append(f"检测到危险关键字: {keyword}")
        
        # 检查字符串操作
        if '"' in formula or "'" in formula:
            # 允许在注释中，但不允许在主要代码中
            pass  # 可以添加更严格的检查
        
        return issues
    
    @classmethod
    def _check_numerical_stability(cls, formula: str) -> Tuple[List[str], List[str]]:
        """检查数值稳定性"""
        issues = []
        warnings = []
        
        # 检查裸除法
        if '/' in formula:
            # 检查是否使用了 safe_div
            if 'safe_div' not in formula:
                # 检查是否是明显的安全除法 (如 /max(len(S), 1))
                if not re.search(r'/\s*(max|len)', formula):
                    warnings.append("建议使用 safe_div(a, b) 替代普通除法，避免除零错误")
        
        # 检查对数
        log_calls = re.findall(r'log\s*\([^)]+\)', formula)
        for call in log_calls:
            if 'EPS' not in call and 'abs' not in call and '1e' not in call:
                warnings.append(f"对数调用 {call} 缺少数值保护")
        
        # 检查 sqrt
        sqrt_calls = re.findall(r'sqrt\s*\([^)]+\)', formula)
        for call in sqrt_calls:
            if 'abs' not in call:
                warnings.append(f"开方调用 {call} 建议使用 sqrt(abs(x))")
        
        # 检查 max(len(S), 1) 模式
        if 'len(S)' in formula:
            if 'max(len(S)' not in formula and '/len(S)' in formula:
                issues.append("使用 len(S) 作为除数时必须使用 max(len(S), 1) 或类似保护")
        
        return issues, warnings
    
    @classmethod
    def _check_dsl_functions(cls, formula: str) -> Tuple[List[str], List[str]]:
        """检查 DSL 函数使用"""
        issues = []
        warnings = []
        
        # 提取所有函数调用
        func_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(func_pattern, formula)
        
        for func in functions:
            if func not in cls.ALLOWED_FUNCTIONS:
                # 检查是否是变量
                if func not in cls.ALLOWED_VARIABLES:
                    warnings.append(f"未知函数或变量: {func}")
        
        return issues, warnings
    
    @classmethod
    def quick_validate(cls, formula: str) -> bool:
        """快速验证（只检查关键问题）"""
        result = cls.validate(formula)
        return result.is_valid


def validate_formula(formula: str) -> ValidationResult:
    """便捷验证函数"""
    return FormulaValidator.validate(formula)