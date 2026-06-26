import sympy as sp
from sympy import Function
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import inspect
import os


# ============================================================
# ① 支持：自定义符号、自定义函数
# ============================================================

import sympy as sp
from sympy import Function, Symbol, symbols


# ======================
# 基础：熵 H()
# ======================
class H(Function):
    """
    熵函数 H(X), H(X,Y), H(X|Y) 等。
    注意：这是符号函数，不在 symbolic 阶段计算。
    """
    nargs = (1, 2)  # H(X) 或 H(X|Y)

    @classmethod
    def eval(cls, *args):
        return None  # 保持符号形式，不做计算


# ======================
# 基础互信息 I(X;Y)
# ======================
class I(Function):
    """
    互信息:
        I(X;Y)
        I(X;Y|Z)
        I(X1,...,Xn;Y)
        I(X;Y;Z)（交互信息）
    根据参数数量和形式自动解释
    """
    @classmethod
    def eval(cls, *args):
        return None


# 注册在表达式解析命名空间
custom_namespace = {
    "H": H,
    "I": I,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos
}



# ============================================================
# ② 将 SymPy 表达式编译为可执行函数
# ============================================================
import numpy as np
from collections import Counter
from math import log


def entropy(X):
    """H(X)"""
    n = len(X)
    counts = Counter(X)
    return -sum((c/n) * log(c/n) for c in counts.values())


def joint_entropy(*vars):
    """H(X,Y,Z,...)"""
    n = len(vars[0])
    joint = list(zip(*vars))
    counts = Counter(joint)
    return -sum((c/n) * log(c/n) for c in counts.values())


def conditional_entropy(X, Y):
    """H(X|Y) = H(X,Y) - H(Y)"""
    return joint_entropy(X, Y) - entropy(Y)

def mutual_information(X, Y):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    return entropy(X) + entropy(Y) - joint_entropy(X, Y)

def conditional_mutual_information(X, Y, Z):
    """
    I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    """
    return (
        conditional_entropy(X, Z)
        + conditional_entropy(Y, Z)
        - conditional_entropy(list(zip(X, Y)), Z)
    )

def multi_variable_mutual_information(X_list, Y):
    """
    I(X1,...,Xn ; Y)
    = H(X1,...,Xn) + H(Y) - H(X1,...,Xn, Y)
    """
    X_tuple = list(zip(*X_list))
    return entropy(X_tuple) + entropy(Y) - joint_entropy(X_tuple, Y)

def interaction_information(X, Y, Z):
    """
    I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
    """
    return mutual_information(X, Y) - conditional_mutual_information(X, Y, Z)


def compile_expression(expr_str):
    expr = sp.sympify(expr_str, locals=custom_namespace)
    vars_sorted = sorted(expr.free_symbols, key=lambda s: s.name)

    f = sp.lambdify(vars_sorted, expr, "numpy")

    def func(data_dict):
        args = [data_dict[str(v)] for v in vars_sorted]
        return f(*args)

    return func, expr, vars_sorted

runtime_namespace = {
    "H": entropy,
    "joint_H": joint_entropy,
    "cond_H": conditional_entropy,
    "I_basic": mutual_information,
    "I_cond": conditional_mutual_information,
    "I_multi": multi_variable_mutual_information,
    "I_inter": interaction_information
}

# ============================================================
# ③ 自动生成 Python 程序保存到文件
# ============================================================

def save_expression_as_python(expr, vars_sorted, output_path="generated_expression.py"):
    expr_code = sp.ccode(expr)

    args_str = ", ".join([str(v) for v in vars_sorted])
    python_code = f"""
import numpy as np

def compute({args_str}):
    return {expr_code}
"""

    with open(output_path, "w", encoding="utf8") as f:
        f.write(python_code)

    print(f"[OK] Python 函数已生成：{output_path}")


# ============================================================
# ④ 多进程批量计算
# ============================================================

def parallel_compute(func, data_list, n_jobs=4):
    """
    func: 单条数据的计算函数
    data_list: 列表，每个元素是 dict，例如 {"x":1, "y":2}
    """
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, data_list))
    return results


# ============================================================
# ============      演示：完整流程   ===========================
# ============================================================

if __name__ == "__main__":

    # ===== 1. 输入 SymPy 表达式（支持自定义符号）=====
    expr_str = "I(x, y) + H(z) + x**2 - 2*y + sin(z)"

    # ===== 2. 编译为可执行函数 =====
    func, expr, vars_sorted = compile_expression(expr_str)

    print("表达式解析为：", expr)
    print("变量：", vars_sorted)

    # ===== 3. 保存为 Python 文件 =====
    save_expression_as_python(expr, vars_sorted, "my_generated_expr.py")

    # ===== 4. 批量输入数据 =====
    data_list = [
        {"x": 1, "y": 2, "z": 0.5},
        {"x": 3, "y": 1, "z": 1.0},
        {"x": 2, "y": 5, "z": 0.3}
    ]

    # ===== 5. 多进程批量计算 =====
    results = parallel_compute(func, data_list, n_jobs=4)

    print("批量计算结果：", results)
