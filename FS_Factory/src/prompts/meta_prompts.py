"""
元提示词模板
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
        language: str = "chinese"
    ) -> str:
        """
        构建算子演化提示词
        
        Args:
            context: 额外上下文信息
            constraints: 额外约束
            examples: 示例公式
            language: 输出语言
        """
        
        if language == "chinese":
            return MetaPromptBuilder._build_chinese_prompt(context, constraints, examples)
        else:
            return MetaPromptBuilder._build_english_prompt(context, constraints, examples)
    
    @staticmethod
    def _build_chinese_prompt(context: str, constraints: Optional[List[str]], 
                              examples: Optional[List[Dict]]) -> str:
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
        
        prompt = f"""
# 🎯 特征选择算子演化任务

你是一位特征选择算法专家。你的任务是**设计一个新的特征选择准则函数**。

🚨 **最高优先级规则 (CRITICAL REQUIREMENTS) 🚨**
1. **绝对禁止自定义函数**: 严禁在输出中写任何 `def my_method(X, Y, S):` 或包含 Python 原生代码块。
2. **仅限基础算子**: 你的 `formula_dsl` 必须且只能是由下方【可用算子库】中的基础算子组合而成的一行表达式。例如 `mi(X, Y) - mean([mi(X, s) for s in S])` 是合法的；但 `my_new_method(X, Y, S)` 是绝对非法的！
3. **闭嘴规则**: 输出完必需的 JSON 结构后，立即停止输出，不要重复你的答案，不要解释你的代码。

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

**你的目标**: 设计一个新方法，在至少 2 个数据集上超越 {best_method.name}

## 可用算子库

### 信息论算子
| 算子 | 数学含义 | 用法 |
|------|----------|------|
| `mi(X, Y)` | 互信息 $I(X;Y)$ | 度量 X 与 Y 的相关性 |
| `cmi(X, Y, s)` | 条件互信息 $I(X;Y|s)$ | 给定 s 时 X 与 Y 的相关性 |
| `entropy(X)` | 熵 $H(X)$ | X 的不确定性 |
| `nmi(X, Y)` | 归一化互信息 | 范围 [0, 1] |
| `su(X, Y)` | 对称不确定性 | 范围 [0, 1] |

### 冗余度算子
| 算子 | 含义 | 用法 |
|------|------|------|
| `redundancy(X, S)` | 平均冗余 | $\frac{{1}}{{|S|}}\sum_s I(X;s)$ |
| `max_redundancy(X, S)` | 最大冗余 | $\max_s I(X;s)$ |

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

请严格按照以下 JSON 格式输出，并且输出完 JSON 后不要有任何额外文字:

```json
{{
    "method_name": "方法名称 (英文，如 MyNewMethod)",
    "formula_dsl": "DSL 公式 (必须由提供的基础算子组成，如 mi(X, Y) - 0.5 * redundancy(X, S))",
    "formula_math": "LaTeX 数学公式 (如 $J = I(X;Y) - 0.5 \cdot R(X,S)$)",
    "intuition": "设计直觉 (中文，50字以内，利用信息论相关内容进行解释)",
    "theoretical_advantage": "与现有方法相比的理论优势 (中文，100字以内，可以在表现数据上进行形容)"
}}
请立刻开始你的设计，必须直接以 ```json 开头：
"""
        return prompt.strip()


    @staticmethod
    def _build_english_prompt(self, context: str, constraints: Optional[List[str]], 
                            examples: Optional[List[Dict]]) -> str:
        """Build the English version of the evolution prompt"""
        
        from .prompts import get_best_method, get_sota_table_markdown
        
        best_method = get_best_method()
        sota_table = get_sota_table_markdown()
        
        # Handle constraints
        constraints_text = ""
        if constraints:
            constraints_text = "\n### Extra Constraints\n" + "\n".join(f"- {c}" for c in constraints)
        
        # Handle examples
        examples_text = ""
        if examples:
            examples_text = "\n### Reference Examples\n```json\n"
            for ex in examples[:2]:
                examples_text += f"{ex}\n"
            examples_text += "```\n"
        
        # Calculate current best performance
        perf_values = list(best_method.performance.values())
        avg_perf = sum(perf_values) / len(perf_values) if perf_values else 0.0

        prompt = f"""
    # 🎯 Feature Selection Operator Evolution Task

    You are a world-class expert in feature selection algorithms. Your task is to **design a novel feature selection criterion function (objective function)**.

    ## Background Knowledge

    ### Forward Feature Selection Process
    1. Start with an empty set S = {{}}
    2. In each step, select a feature X that maximizes the criterion function J(X, Y, S)
    3. Add the selected X to the set S
    4. Repeat until the desired number of features is reached

    ### Design Principles
    - **Relevance**: High correlation between feature X and target Y.
    - **Redundancy**: Low overlap/redundancy between feature X and already selected set S.
    - **Complementarity**: Maximize the information gain of Y when X is combined with S.

    ## State-of-the-Art (SOTA) Methods

    {sota_table}

    **Current Best Method**: {best_method.name}, Average Accuracy: {avg_perf:.4f}

    **Your Goal**: Design a new method that outperforms {best_method.name} on at least 2 datasets.

    ## Available Operator Library

    ### Information Theory Operators
    | Operator | Mathematical Meaning | Usage |
    |----------|----------------------|-------|
    | `mi(X, Y)` | Mutual Information $I(X;Y)$ | Measures relevance between X and Y |
    | `cmi(X, Y, s)` | Conditional Mutual Information $I(X;Y|s)$ | Relevance of X and Y given s |
    | `entropy(X)` | Entropy $H(X)$ | Uncertainty of X |
    | `nmi(X, Y)` | Normalized Mutual Information | Range [0, 1] |
    | `su(X, Y)` | Symmetric Uncertainty | Range [0, 1] |

    ### Redundancy Operators
    | Operator | Meaning | Formula |
    |----------|---------|---------|
    | `redundancy(X, S)` | Average Redundancy | $\\frac{{1}}{{|S|}}\\sum_{{s \\in S}} I(X;s)$ |
    | `max_redundancy(X, S)` | Maximum Redundancy | $\\max_{{s \\in S}} I(X;s)$ |

    ### Aggregation & Transformation
    - `sum([...])`: Sum of a list
    - `mean([...])`: Average of a list
    - `max([...])`: Maximum value of a list
    - `min([...])`: Minimum value of a list
    - `abs(x)`: Absolute value

    ### Safety Operators (Mandatory for stability)
    - `safe_div(a, b)`: Division with zero-check
    - `log(abs(x) + EPS)`: Numerical stable logarithm
    - `sqrt(abs(x))`: Safe square root

    ### Variables
    - `X`: Candidate feature (1D array)
    - `Y`: Target variable (1D array)
    - `S`: Selected features list (List of 1D arrays)
    - `s`: An individual feature in S (Used for iterations/comprehensions)

    ## Constraints & Requirements

    1. **Numerical Stability**:
    - Use `safe_div(a, b)` or `max(len(S), 1)` to prevent division by zero.
    - Always protect log/sqrt operations using the provided safety operators.
    
    2. **Boundary Conditions**:
    - The formula MUST be computable when S is empty (e.g., use conditional logic or robust aggregators).
    - Must return a finite float value.

    3. **Efficiency**:
    - Avoid nesting `cmi` more than 3 levels deep.
    - Prefer vectorized logic over complex Python loops where possible.
    {constraints_text}
    {examples_text}
    {context}

    ## Output Format

    You must return the result strictly in the following JSON format:

    ```json
    {{
        "method_name": "Unique and descriptive name (e.g., AdaptiveGain, RelRedPenalty)",
        "formula_dsl": "DSL Formula (e.g., mi(X, Y) - 0.5 * redundancy(X, S))",
        "formula_math": "LaTeX Formula (e.g., $J = I(X;Y) - 0.5 \\cdot R(X,S)$)",
        "intuition": "Brief explanation of the design logic (under 50 words, using info theory concepts)",
        "theoretical_advantage": "Detailed theoretical advantage over SOTA (under 100 words, explaining why it performs better)"
    }}
    Now, start designing your innovative feature selection operator!
    """
        return prompt.strip()    


    @staticmethod
    def build_refinement_prompt(original_formula: str, 
                                evaluation_result: Dict,
                                criticism: str) -> str:
        #构建提示词
        prompt = f"""
        📝 公式改进任务
        你的上一个公式表现不够理想，请根据反馈进行改进。

        原始公式

        {original_formula}
        评估结果
        平均准确率: {evaluation_result.get('avg_accuracy', 'N/A')}
        各数据集表现: {evaluation_result.get('results', {})}
        批评意见
        {criticism}

        改进建议
        考虑调整相关性和冗余性的权重平衡
        尝试引入条件互信息项来捕捉特征交互
        检查是否存在数值不稳定的情况
        请输出改进后的公式 (JSON 格式同上)。
        """
        return prompt.strip()

def build_meta_prompt(**kwargs) -> str:

        return MetaPromptBuilder.build_evolution_prompt(**kwargs)
