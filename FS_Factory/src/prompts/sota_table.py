# src/prompts/sota_table.py

"""
SOTA 方法库
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class SOTAMethod:
    """SOTA 方法记录"""
    name: str
    formula_dsl: str
    formula_math: str
    year: int
    authors: str
    description: str
    performance: Dict[str, float]  # dataset -> accuracy
    
    def to_markdown_row(self) -> str:
        """转换为 Markdown 表格行"""
        avg_perf = sum(self.performance.values()) / len(self.performance) if self.performance else 0
        return f"| {self.name} | {self.year} | {self.formula_math} | {avg_perf:.4f} |"


# 人类设计的经典特征选择方法
BASE_SOTA_METHODS: List[SOTAMethod] = [
    SOTAMethod(
        name="mRMR",
        formula_dsl="mi(X, Y) - redundancy(X, S)",
        formula_math=r"$I(X;Y) - \frac{1}{|S|}\sum_{s \in S} I(X;s)$",
        year=2005,
        authors="Peng et al.",
        description="最小冗余最大相关，平衡相关性和冗余性",
        performance={"madelon": 0.65, "gisette": 0.72, "arcene": 0.78}
    ),
    SOTAMethod(
        name="JMI",
        formula_dsl="mi(X, Y) - redundancy(X, S) + sum([cmi(X, Y, s) for s in S])/max(len(S), 1)",
        formula_math=r"$I(X;Y) - \sum_{s \in S} I(X;s) + \sum_{s \in S} I(X;Y|s)$",
        year=2012,
        authors="Brown et al.",
        description="联合互信息，考虑特征间的互补效应",
        performance={"madelon": 0.68, "gisette": 0.74, "arcene": 0.80}
    ),
    SOTAMethod(
        name="CMIM",
        formula_dsl="mi(X, Y) - max([mi(X, s) - cmi(X, Y, s) for s in S] + [0])",
        formula_math=r"$I(X;Y) - \max_{s \in S}[I(X;s) - I(X;Y|s)]$",
        year=2004,
        authors="Fleuret",
        description="条件互信息最大化，处理复杂特征交互",
        performance={"madelon": 0.66, "gisette": 0.71, "arcene": 0.77}
    ),
    SOTAMethod(
        name="CIFE",
        formula_dsl="mi(X, Y) - 2 * redundancy(X, S)",
        formula_math=r"$I(X;Y) - 2\sum_{s \in S} I(X;s)$",
        year=2006,
        authors="Lin & Tang",
        description="条件信息增益，强调去冗余",
        performance={"madelon": 0.62, "gisette": 0.70, "arcene": 0.75}
    ),
    SOTAMethod(
        name="DISR",
        formula_dsl="sum([nmi(X, Y) * su(X, s) for s in S])/max(len(S), 1)",
        formula_math=r"$\frac{1}{|S|}\sum_{s \in S} NMI(X;Y) \cdot SU(X;s)$",
        year=2008,
        authors="Peng et al.",
        description="双输入对称相关性，归一化处理",
        performance={"madelon": 0.64, "gisette": 0.73, "arcene": 0.76}
    ),
    SOTAMethod(
        name="ICAP",
        formula_dsl="mi(X, Y) - sum([max(0, mi(X, s) - cmi(X, Y, s)) for s in S])",
        formula_math=r"$I(X;Y) - \sum_{s \in S} \max(0, I(X;s) - I(X;Y|s))$",
        year=2005,
        authors="Jakulin",
        description="交互信息覆盖，保守处理负交互",
        performance={"madelon": 0.67, "gisette": 0.72, "arcene": 0.79}
    ),
    SOTAMethod(
        name="MIM",
        formula_dsl="mi(X, Y)",
        formula_math=r"$I(X;Y)$",
        year=1960,
        authors="Lewis",
        description="最简单的互信息筛选，不考虑冗余",
        performance={"madelon": 0.58, "gisette": 0.65, "arcene": 0.72}
    ),
]
# 2. 定义全局运行时的 SOTA 列表和本地存储路径
SOTA_METHODS: List[SOTAMethod] = []
# 将 AI 演化的 SOTA 存在同目录下的 ai_sota_methods.json 里
SOTA_FILE_PATH = os.path.join(os.path.dirname(__file__), "ai_sota_methods.json")

def load_sota_methods():
    """从本地文件加载所有 SOTA 方法"""
    global SOTA_METHODS
    SOTA_METHODS = BASE_SOTA_METHODS.copy()  # 先装载人类经典方法
    
    if os.path.exists(SOTA_FILE_PATH):
        try:
            with open(SOTA_FILE_PATH, 'r', encoding='utf-8') as f:
                custom_methods = json.load(f)
                for m_dict in custom_methods:
                    SOTA_METHODS.append(SOTAMethod(**m_dict))
            print(f"📦 成功从本地加载了 {len(custom_methods)} 个 AI 演化的 SOTA 算子！")
        except Exception as e:
            print(f"⚠️ 加载 AI SOTA 文件失败: {e}")

def save_new_sota(method: SOTAMethod):
    """将新的神级算子持久化保存到本地文件"""
    global SOTA_METHODS
    SOTA_METHODS.append(method) # 1. 更新内存
    
    # 2. 读取现有自定义方法
    custom_methods = []
    if os.path.exists(SOTA_FILE_PATH):
        try:
            with open(SOTA_FILE_PATH, 'r', encoding='utf-8') as f:
                custom_methods = json.load(f)
        except:
            pass
            
    # 3. 追加新方法并写回硬盘
    method_dict = {
        "name": method.name,
        "formula_dsl": method.formula_dsl,
        "formula_math": method.formula_math,
        "year": method.year,
        "authors": method.authors,
        "description": method.description,
        "performance": method.performance
    }
    custom_methods.append(method_dict)
    
    with open(SOTA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(custom_methods, f, indent=4, ensure_ascii=False)

# 3. 在模块被导入时，自动执行一次加载
load_sota_methods()

def get_best_method() -> SOTAMethod:
    """获取当前最佳方法"""
    return max(SOTA_METHODS, key=lambda m: sum(m.performance.values()) / len(m.performance))


def get_method_by_name(name: str) -> Optional[SOTAMethod]:
    """按名称获取方法"""
    for method in SOTA_METHODS:
        if method.name.lower() == name.lower():
            return method
    return None


def get_sota_table_markdown() -> str:
    """获取 SOTA 表格的 Markdown 格式"""
    header = "| 方法 | 年份 | 公式 | 平均准确率 |\n|------|------|------|----------|\n"
    rows = "\n".join(m.to_markdown_row() for m in SOTA_METHODS)
    return header + rows