"""
元提示词模板 (已优化：防重复、防幻觉、严格JSON格式、纯文本列表)
"""
from typing import List, Optional, Dict
from .sota_table import SOTA_METHODS, get_best_method, get_sota_table_markdown

class MetaPromptBuilder:
    """元提示词构建器"""
    
    @staticmethod
    def build_evolution_prompt(
        context: str = "",
        constraints: Optional[List[str]] = None,
        examples: Optional[List[Dict]] = None,
        history_formulas: Optional[List[str]] = None,
        performance_context: str = "",
        language: str = "chinese"
    ) -> str:
        """
        构建算子演化提示词
        
        Args:
            context: 额外上下文信息
            constraints: 额外约束
            examples: 示例公式
            history_formulas: 历史已经生成的公式黑名单
            performance_context: 历史算子的性能表现
            language: 输出语言
        """
        
        if language == "chinese":
            return MetaPromptBuilder._build_chinese_prompt(context, constraints, examples, history_formulas, performance_context)
        else:
            return MetaPromptBuilder._build_english_prompt(context, constraints, examples, history_formulas, performance_context)
    
    @staticmethod
    def _build_chinese_prompt(context: str, constraints: Optional[List[str]], 
                              examples: Optional[List[Dict]], history_formulas: Optional[List[str]],
                              performance_context: str) -> str:
        """构建中文提示词"""
        
        best_method = get_best_method()
        sota_table = get_sota_table_markdown()
        
        constraints_text = ""
        if constraints:
            constraints_text = "\n### 额外约束\n" + "\n".join(f"- {c}" for c in constraints)
        
        examples_text = ""
        if examples:
            examples_text = "\n### 参考示例\n```json\n"
            for ex in examples[:2]:
                examples_text += f"{ex}\n"
            examples_text += "```\n"

        history_text = ""
        if history_formulas:
            history_text = "\n### 🛑 历史算子黑名单 (严禁重复)\n以下公式已经被生成过或测试过，**绝对禁止**生成与下列公式在数学等价或形式上高度相似的算子：\n"
            history_text += "\n".join(f"- {f}" for f in history_formulas) + "\n"
            
        perf_text = ""
        if performance_context:
            perf_text = f"\n### 📊 历史算子性能反馈\n以下是之前演化出的算子在多个数据集和分类器（Bayes, RF, SVM, KNN）上的真实表现。请仔细分析哪些算子表现好，哪些表现差，并据此进行**有针对性的迭代提升**：\n{performance_context}\n"
        
        prompt = f"""
# 🎯 特征选择算子演化任务

你是一位顶尖的机器学习特征选择算法专家。你的任务是**设计一个新的特征选择准则函数**。

🚨 **最高优先级规则 (CRITICAL REQUIREMENTS) 🚨**
1. **绝对禁止自定义函数**: 严禁在输出中写任何 `def my_method(X, Y, S):` 或包含 Python 原生代码块。
2. **纯净 DSL 语法**: 你的 `formula_dsl` 必须是可执行的伪代码/Python风格代码。严禁在 DSL 中包含任何 \\sum, \\frac 等 LaTeX 语法！
3. **JSON 字符转义**: 你的输出必须是合法的 JSON 格式。严禁在 JSON 字符串中出现未转义的控制字符（如 \\f, \\n）。LaTeX 中的反斜杠必须双写转义（例如使用 \\\\frac）。
4. **禁止幻觉**: 在描述理论优势时，严禁捏造虚假的数据集表现或准确率指标，只能基于信息论或马尔可夫毯等理论解释优势。
5. **闭嘴规则**: 输出完必需的 JSON 结构后，立即停止输出，不要重复你的答案，不要解释你的代码。

## 背景知识

### 前向特征选择过程
1. 从空集 S = {{}} 开始
2. 每次选择使准则函数 J(X, Y, S) 最大化的特征 X
3. 将 X 加入已选集 S
4. 重复直到选择足够多的特征

### 设计原则
- **相关性**: 特征 X 与目标 Y 的相关性要高
- **冗余性**: 特征 X 与已选特征 S 的冗余要低  
- **互补性**: 特征 X 与 S 组合后对预测 Y 的增益要大

## 当前最先进方法 (SOTA)

{sota_table}

**当前最佳方法**: {best_method.name}，平均准确率: {sum(best_method.performance.values())/len(best_method.performance):.4f}

**你的目标**: 设计一个新方法，在理论机制上超越 {best_method.name}
{history_text}
{perf_text}
## 可用算子库


### 信息论算子
- `mi(X, Y)`: 互信息 I(X;Y)，度量 X 与 Y 的相关性
- `cmi(X, Y, s)`: 条件互信息 I(X;Y|s)，给定 s 时 X 与 Y 的相关性
- `entropy(X)`: 熵 H(X)，X 的不确定性
- `nmi(X, Y)`: 归一化互信息，范围 [0, 1]
- `su(X, Y)`: 对称不确定性，范围 [0, 1]

### 冗余度算子
- `redundancy(X, S)`: 平均冗余，\\\\frac{{1}}{{|S|}}\\\\sum_s I(X;s)
- `max_redundancy(X, S)`: 最大冗余，\\\\max_s I(X;s)

### 聚合算子
- `sum([...])`: 列表求和
- `mean([...])`: 列表平均
- `max([...])`: 列表最大值
- `min([...])`: 列表最小值

### 安全算子
- `safe_div(a, b)`: 安全除法，避免除零
- `log(abs(x) + EPS)`: 安全对数
- `sqrt(abs(x))`: 安全开方

### 变量
- `X`: 候选特征 (一维数组)
- `Y`: 目标变量 (一维数组)
- `S`: 已选特征列表 (列表，每个元素是一维数组)
- `s`: S 中的单个特征 (用于迭代)

## 约束条件

1. **数值稳定性**:
   - 除法必须使用 `safe_div(a, b)` 或 `max(len(S), 1)` 保护
   - 对数必须使用 `log(abs(x) + EPS)` 或类似保护
   
2. **边界条件**:
   - 当 S 为空时公式必须可计算
   - 必须返回有限浮点数

3. **复杂度**:
   - 避免嵌套超过 3 层的 cmi 计算
   - 优先使用可向量化的公式结构
{constraints_text}
{examples_text}
{context}

## 输出格式

请严格按照以下 JSON 模板输出，注意理解各项的“要求”，并且输出完 JSON 后不要有任何额外文字:

```json
{{
    "method_name": "要求：必须是 3 到 5 个大写英文字母组成的全新算法缩写（例如：CMIFS, JMI）。严禁输出中文。",
    "formula_dsl": "要求：可执行的 Python 伪代码表达式。🚨绝对禁止直接抄袭已实现算子，必须发挥创造力生成新结构！严禁包含任何 \\sum, \\frac 等 LaTeX 语法！",
    "formula_math": "要求：标准 LaTeX 数学公式。注意 JSON 中反斜杠必须双写转义（例如：$J = I(X;Y) - \\\\frac{{1}}{{|S|}}\\\\sum_{{s \\\\in S}} NMI(X;s) \\\\cdot SU(X;s)$）。",
    "intuition": "要求：简明扼要地解释该公式背后的信息论直觉（中文，50字以内）。",
    "theoretical_advantage": "要求：纯粹从理论机制解释其优越性（中文，100字以内）。绝对严禁编造准确率或数据集表现！"
}}
请立刻开始你的设计，必须直接以 ```json 开头：
"""
        return prompt.strip()

    @staticmethod
    def _build_english_prompt(context: str, constraints: Optional[List[str]], 
                            examples: Optional[List[Dict]], history_formulas: Optional[List[str]],
                            performance_context: str) -> str:
        """Build the English version of the evolution prompt"""
        
        best_method = get_best_method()
        sota_table = get_sota_table_markdown()
        
        constraints_text = ""
        if constraints:
            constraints_text = "\n### Extra Constraints\n" + "\n".join(f"- {c}" for c in constraints)
        
        examples_text = ""
        if examples:
            examples_text = "\n### Reference Examples\n```json\n"
            for ex in examples[:2]:
                examples_text += f"{ex}\n"
            examples_text += "```\n"
            
        history_text = ""
        if history_formulas:
            history_text = "\n### 🛑 Historical Formula Blacklist\nThe following formulas have already been evaluated. You are **STRICTLY FORBIDDEN** from generating mathematically equivalent or highly similar formulas:\n"
            history_text += "\n".join(f"- {f}" for f in history_formulas) + "\n"
            
        perf_text = ""
        if performance_context:
            perf_text = f"\n### 📊 Historical Operator Performance Feedback\nBelow is the actual performance of previously evolved operators across multiple datasets and classifiers (Bayes, RF, SVM, KNN). Please carefully analyze which operators performed well and which did poorly, and use this to make **targeted iterative improvements**:\n{performance_context}\n"
        
        perf_values = list(best_method.performance.values())
        avg_perf = sum(perf_values) / len(perf_values) if perf_values else 0.0

        prompt = f"""
    #Feature Selection Operator Evolution Task
    You are a world-class expert in feature selection algorithms. Your task is to design a novel feature selection criterion function (objective function).

    #CRITICAL REQUIREMENTS#

    NO Custom Functions: Do not output def my_method(X, Y, S): or any native Python code blocks.

    Pure DSL Syntax: Your formula_dsl MUST be executable pseudo-code/Python-style code. STRICTLY NO LaTeX syntax like \sum or \frac inside the DSL!

    JSON Escaping: Your output must be a valid JSON. Unescaped control characters are forbidden. LaTeX backslashes MUST be escaped in JSON (e.g., use \\frac).

    NO Hallucinations: Do NOT invent fake dataset performance or accuracy metrics in the theoretical_advantage. Base your claims purely on theoretical mechanisms.

    Shut Up Rule: Stop generating immediately after the required JSON block. No extra explanations.

    Background Knowledge
    Forward Feature Selection Process
    Start with an empty set S = {{}}

    In each step, select a feature X that maximizes the criterion function J(X, Y, S)

    Add the selected X to the set S

    Repeat until the desired number of features is reached

    Design Principles
    Relevance: High correlation between feature X and target Y.

    Redundancy: Low overlap/redundancy between feature X and already selected set S.

    Complementarity: Maximize the information gain of Y when X is combined with S.

    State-of-the-Art (SOTA) Methods
    {sota_table}

    Current Best Method: {best_method.name}, Average Accuracy: {avg_perf:.4f}

    Your Goal: Design a new method that theoretically outperforms {best_method.name}.
    {history_text}
    {perf_text}

    Available Operator Library
    Information Theory Operators
    mi(X, Y): Mutual Information I(X;Y), Measures relevance between X and Y

    cmi(X, Y, s): Conditional Mutual Information I(X;Y|s), Relevance of X and Y given s

    entropy(X): Entropy H(X), Uncertainty of X

    nmi(X, Y): Normalized Mutual Information, Range [0, 1]

    su(X, Y): Symmetric Uncertainty, Range [0, 1]

    Redundancy Operators
    redundancy(X, S): Average Redundancy, \\frac{{1}}{{|S|}}\\sum_{{s \\in S}} I(X;s)

    max_redundancy(X, S): Maximum Redundancy, \\max_{{s \\in S}} I(X;s)

    Aggregation & Transformation
    sum([...]): Sum of a list

    mean([...]): Average of a list

    max([...]): Maximum value of a list

    min([...]): Minimum value of a list

    abs(x): Absolute value

    Safety Operators (Mandatory for stability)
    safe_div(a, b): Division with zero-check

    log(abs(x) + EPS): Numerical stable logarithm

    sqrt(abs(x)): Safe square root

    Variables
    X: Candidate feature (1D array)

    Y: Target variable (1D array)

    S: Selected features list (List of 1D arrays)

    s: An individual feature in S (Used for iterations/comprehensions)

    Constraints & Requirements
    Numerical Stability:

    Use safe_div(a, b) or max(len(S), 1) to prevent division by zero.

    Always protect log/sqrt operations using the provided safety operators.

    Boundary Conditions:

    The formula MUST be computable when S is empty (e.g., use conditional logic or robust aggregators).

    Must return a finite float value.

    Efficiency:

    Avoid nesting cmi more than 3 levels deep.

    Prefer vectorized logic over complex Python loops where possible.

    {constraints_text}
    {examples_text}
    {context}

    Output Format
    You must return the result strictly in the following JSON format. Follow the instructions in the values:
    {{
        "method_name": "Requirement: 3-5 uppercase letters acronym ONLY (e.g., CMIFS).",
        "formula_dsl": "Requirement: Executable Python expression  NO LaTeX syntax allowed!",
        "formula_math": "Requirement: Standard LaTeX formula. Remember to double-escape backslashes in JSON (e.g., $J = I(X;Y) - \\\\frac{{1}}{{|S|}}$).",
        "intuition": "Requirement: Brief info-theory intuition (under 50 words).",
        "theoretical_advantage": "Requirement: Purely theoretical advantage (under 100 words). DO NOT invent fake accuracy metrics!"
    }}
    Now, start designing your innovative feature selection operator. Must start with ```json immediately:
    """
        return prompt.strip()
    @staticmethod
    def build_refinement_prompt(original_formula: str, 
                                evaluation_result: Dict,
                                criticism: str,
                                history_formulas: Optional[List[str]] = None) -> str:
        
        history_text = ""
        if history_formulas:
            history_text = "\n🛑 历史算子黑名单 (严禁重复)\n请确保你改进后的公式**不等于**以下任何历史公式：\n"
            history_text += "\n".join(f"- {f}" for f in history_formulas) + "\n"

        prompt = f"""
    📝 公式改进任务
    你的上一个公式表现不够理想，或者出现了语法/格式错误，请根据反馈进行改进。

    【原始公式】
    {original_formula}

    【评估结果】
    平均准确率: {evaluation_result.get('avg_accuracy', 'N/A')}
    各数据集表现: {evaluation_result.get('results', {})}

    【批评与错误日志】
    {criticism}
    {history_text}
    【改进方向要求】

    解决上述日志中提到的所有语法或数值错误。

    尝试调整相关性和冗余性的权重平衡。

    尝试引入条件互信息项来捕捉特征交互。

    务必遵守严格的 JSON 格式要求，注意 LaTeX 转义，且 formula_dsl 中绝对不能包含 LaTeX 代码。

    不要编造不存在的提升数据。

    请直接输出改进后的 JSON，格式与要求完全遵循系统主提示词。必须直接以 ```json 开头：
    """
        return prompt.strip()

def build_meta_prompt(**kwargs) -> str:
  return MetaPromptBuilder.build_evolution_prompt(**kwargs)