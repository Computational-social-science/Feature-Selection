# 项目使用指南与完整 README

---

## 一、项目核心概念解析

### 1.1 项目目标

这个项目实现了一个**LLM 驱动的特征选择算子演化系统**，核心目标是：

```
┌─────────────────────────────────────────────────────────────┐
│                    输入: 数据集 (X, y)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: DSL 算子库提供基础运算 (mi, entropy, cmi...)        │
│  Step 2: LLM 基于元提示词生成新的特征选择公式                 │
│  Step 3: 沙箱评估公式在多个数据集上的表现                     │
│  Step 4: Router 根据数据集特征选择最佳算法                    │
│  Step 5: 统计检验验证 Agent 算子 vs 人类算子                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              输出: 最优特征子集 + 新演化算子                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 两种使用模式

| 模式 | 用途 | 输入 | 输出 |
|------|------|------|------|
| **演化模式** | 让 LLM 发明新的特征选择算法 | 元提示词 + SOTA 对比 | 新的特征选择公式 |
| **应用模式** | 对你的数据进行特征选择 | 数据集 | 选中的特征索引 |

---

## 二、需要你自己补充和改动的地方

### 2.1 必须补充的内容

| 位置 | 内容 | 原因 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 你的 DeepSeek API Key | 调用 LLM 必需 |
| `config/settings.yaml` | 可调整参数配置 | 根据你的需求定制 |
| `data/raw/` | 你的自定义数据集 | 扩展测试数据 |
| `src/prompts/sota_table.py` | 新增 SOTA 方法 | 扩展基准对比 |

### 2.2 可选改动的内容

| 位置 | 内容 | 场景 |
|------|------|------|
| `src/dsl/operators.py` | 添加新算子 | 需要新的数学运算 |
| `src/sandbox/datasets.py` | 添加新数据集类型 | 特殊数据格式 |
| `src/router/router_agent.py` | 自定义路由规则 | 特定领域知识 |
| `src/prompts/meta_prompts.py` | 修改提示词模板 | 优化 LLM 输出 |

### 2.3 具体修改指南

#### ① 配置 API Key

```bash
# 方式一：环境变量 (推荐)
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"

# 方式二：写入 .env 文件
echo 'DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"' > .env

# 方式三：代码中直接传入
client = DeepSeekMathClient(api_key="sk-xxxxxxxxxxxxxxxx")
```

#### ② 添加自定义数据集

```python
# src/sandbox/datasets.py 中添加

@staticmethod
def load_custom_dataset(csv_path: str, target_column: str) -> TestDataset:
    """加载自定义 CSV 数据集"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    
    return TestDataset(
        name=os.path.basename(csv_path),
        X=X,
        y=y,
        metadata={'source': 'custom', 'target_column': target_column}
    )
```

#### ③ 添加自定义 DSL 算子

```python
# src/dsl/advanced.py 中添加

def my_custom_score(self, X: np.ndarray, Y: np.ndarray, 
                    S: List[np.ndarray]) -> float:
    """
    自定义评分函数
    
    公式: 你的数学公式
    """
    # 你的实现
    mi_xy = self.mutual_info(X, Y)
    red = self.redundancy(X, S)
    
    # 自定义组合
    score = mi_xy ** 2 / (red + self.eps + 1)
    
    return score

# 然后在 src/dsl/registry.py 中注册
cls._functions['my_custom_score'] = cls._operators.my_custom_score
```

---

## 三、完整 README 文档

```markdown
# Feature Selection Factory

<div align="center">

**LLM 驱动的特征选择算子演化系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english-documentation) | [中文文档](#中文文档)

</div>

---

## 中文文档

### 📖 项目简介

Feature Selection Factory 是一个创新的特征选择框架，利用大语言模型 (LLM) 自动演化出新的特征选择算法。系统能够：

- 🧬 **自动演化**：让 LLM 设计新的特征选择准则函数
- 📊 **自动评估**：在多个数据集上自动测试生成的公式
- 🔄 **智能路由**：根据数据集特征自动选择最佳算法
- 📈 **超越 SOTA**：目标是超越人工设计的特征选择方法

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Feature Selection Factory                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  Step 1      │    │  Step 2      │    │  Step 3      │              │
│   │  DSL 算子库  │───▶│  Prompt 工程 │───▶│  评估沙箱    │              │
│   │              │    │              │    │              │              │
│   │  mi, cmi,    │    │  元提示词    │    │  自动测试    │              │
│   │  entropy...  │    │  SOTA 对比   │    │  批评反馈    │              │
│   └──────────────┘    └──────────────┘    └──────────────┘              │
│          │                   │                   │                       │
│          └───────────────────┴───────────────────┘                       │
│                              ↓                                           │
│   ┌──────────────────────────────────────────────────────┐              │
│   │                    Step 4: Router Agent               │              │
│   │         元特征提取 → 相似匹配 → 算法路由              │              │
│   └──────────────────────────────────────────────────────┘              │
│                              ↓                                           │
│   ┌──────────────────────────────────────────────────────┐              │
│   │                 Step 5: 验证与消融                    │              │
│   │        人类算子 vs Agent算子 → 统计检验              │              │
│   └──────────────────────────────────────────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🚀 快速开始

#### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/feature-selection-factory.git
cd feature-selection-factory

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 2. 配置 API Key

```bash
# 设置 DeepSeek API Key (推荐使用 DeepSeek Math)
export DEEPSEEK_API_KEY="sk-your-api-key-here"

# 或使用 OpenAI
# export OPENAI_API_KEY="sk-your-openai-key"
```

> 💡 **获取 API Key**: 访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并获取

#### 3. 运行演化

```bash
# 使用 DeepSeek Math 模型运行算子演化
python scripts/run_evolution_deepseek.py --iterations 5 --model deepseek-math

# 查看结果
ls outputs/operators/
```

#### 4. 使用特征选择

```python
from src import FeatureSelectionFactory
from src.llm import DeepSeekMathClient
import numpy as np

# 创建工厂
factory = FeatureSelectionFactory(
    llm_client=DeepSeekMathClient(api_key="your-api-key")
)

# 对数据进行特征选择
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=100, random_state=42)

# 自动路由选择最佳方法
selected_features, info = factory.select_features(X, y, method='auto', k=20)
print(f"选中的特征: {selected_features}")
print(f"路由信息: {info}")
```

### 📚 详细使用指南

#### 模式一：算子演化模式

让 LLM 自动发明新的特征选择算法：

```python
from src import FeatureSelectionFactory
from src.llm import DeepSeekMathClient

# 初始化
factory = FeatureSelectionFactory(
    llm_client=DeepSeekMathClient(),
    output_dir="./my_outputs"
)

# 运行演化 (5 次迭代)
best_operator = factory.evolve_operators(
    n_iterations=5,
    verbose=True
)

# 查看最佳算子
print(f"名称: {best_operator.name}")
print(f"公式: {best_operator.formula_dsl}")
print(f"准确率: {best_operator.evaluation_result.avg_accuracy:.4f}")
```

**命令行方式：**

```bash
# 基础运行
python scripts/run_evolution_deepseek.py --iterations 5

# 使用 R1 推理模型 (更深度的推理)
python scripts/run_evolution_deepseek.py --model deepseek-reasoner --iterations 3

# 指定输出目录
python scripts/run_evolution_deepseek.py --output ./results --iterations 5
```

#### 模式二：特征选择应用模式

对你的数据进行特征选择：

```python
import pandas as pd
from src import FeatureSelectionFactory

# 加载你的数据
df = pd.read_csv("your_data.csv")
X = df.drop(columns=['target']).values
y = df['target'].values

# 创建工厂 (不需要 LLM，使用已有方法)
factory = FeatureSelectionFactory()

# 方式 1: 自动路由 (推荐)
selected, info = factory.select_features(X, y, method='auto', k=30)
print(f"Router 选择方法: {info['routing']['selected_method']}")

# 方式 2: 指定方法
selected, _ = factory.select_features(X, y, method='mRMR', k=30)
selected, _ = factory.select_features(X, y, method='JMI', k=30)

# 使用选中的特征
X_selected = X[:, selected]
```

#### 模式三：基准测试模式

对比不同方法的性能：

```bash
# 运行完整基准测试
python scripts/run_benchmark.py

# 运行人类 vs Agent 对比
python scripts/run_comparison.py
```

```python
from src import FeatureSelectionFactory

factory = FeatureSelectionFactory()

# 自定义方法对比
methods = {
    'mRMR': 'mi(X, Y) - redundancy(X, S)',
    'MyMethod': 'mi(X, Y) - 0.5 * redundancy(X, S) + 0.3 * sum([cmi(X, Y, s) for s in S])/max(len(S), 1)'
}

results = factory.run_benchmark(methods=methods, verbose=True)
print(results['report'])
```

### 🔧 配置说明

#### 配置文件

```yaml
# config/settings.yaml

# DSL 配置
dsl:
  eps: 1.0e-10          # 数值稳定性常数
  log_base: 2           # 对数底数 (2=bits, e=nats)
  knn:
    k: 5                # kNN 熵估计的 k 值

# 提示词配置
prompts:
  model: "deepseek-math"
  temperature: 0.7
  max_tokens: 4096
  n_candidates: 3       # 每次生成的候选数

# 沙箱配置
sandbox:
  timeout: 300          # 超时时间(秒)
  max_features_to_select: 30
  n_repeats: 3
  cv_folds: 5

# 路由配置
router:
  similarity_top_k: 10
  default_method: "mRMR"
```

#### 支持的 LLM 模型

| 提供商 | 模型 | 用途 | 成本 |
|--------|------|------|------|
| DeepSeek | `deepseek-math` | 数学公式生成 (推荐) | $0.14/1M tokens |
| DeepSeek | `deepseek-reasoner` | 深度推理 | $0.55/1M tokens |
| DeepSeek | `deepseek-chat` | 通用对话 | $0.14/1M tokens |
| OpenAI | `gpt-4` | 高质量生成 | $30/1M tokens |
| OpenAI | `gpt-3.5-turbo` | 成本优先 | $0.5/1M tokens |

### 📁 项目结构

```
feature-selection-factory/
├── config/
│   └── settings.yaml          # 配置文件
├── data/
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
├── src/
│   ├── dsl/                   # DSL 算子库
│   │   ├── operators.py       # 基础信息论算子
│   │   ├── advanced.py        # 高级算子
│   │   └── registry.py        # 算子注册表
│   ├── prompts/               # 提示词工程
│   │   ├── meta_prompts.py    # 元提示词
│   │   ├── sota_table.py      # SOTA 方法库
│   │   └── validator.py       # 公式验证
│   ├── sandbox/               # 评估沙箱
│   │   ├── executor.py        # 沙箱执行器
│   │   ├── datasets.py        # 测试数据集
│   │   └── critic.py          # 批评者 Agent
│   ├── router/                # 路由智能体
│   │   ├── meta_features.py   # 元特征提取
│   │   ├── router_agent.py    # 路由决策
│   │   └── performance_db.py  # 性能数据库
│   ├── validation/            # 验证消融
│   │   ├── experiments.py     # 实验运行
│   │   ├── statistics.py      # 统计检验
│   │   └── visualizer.py      # 可视化
│   ├── llm/                   # LLM 接口
│   │   ├── base.py            # 基类
│   │   ├── deepseek_client.py # DeepSeek 客户端
│   │   └── openai_client.py   # OpenAI 客户端
│   └── factory.py             # 主工厂类
├── scripts/
│   ├── run_evolution_deepseek.py  # 演化脚本
│   ├── run_benchmark.py           # 基准测试
│   └── run_comparison.py          # 对比实验
├── tests/                     # 单元测试
├── outputs/                   # 输出目录
│   ├── operators/             # 演化的算子
│   ├── results/               # 实验结果
│   └── figures/               # 图表
├── requirements.txt
├── setup.py
└── README.md
```

### 🧪 DSL 算子参考

#### 基础算子

| 算子 | 数学公式 | 说明 |
|------|----------|------|
| `mi(X, Y)` | $I(X;Y) = H(X) + H(Y) - H(X,Y)$ | 互信息 |
| `cmi(X, Y, Z)` | $I(X;Y\|Z) = H(X\|Z) - H(X\|Y,Z)$ | 条件互信息 |
| `entropy(X)` | $H(X) = -\sum p(x)\log p(x)$ | 熵 |
| `nmi(X, Y)` | $NMI = I(X;Y)/\sqrt{H(X)H(Y)}$ | 归一化互信息 |
| `su(X, Y)` | $SU = 2I(X;Y)/(H(X)+H(Y))$ | 对称不确定性 |

#### 高级算子

| 算子 | 数学公式 | 说明 |
|------|----------|------|
| `redundancy(X, S)` | $R = \frac{1}{\|S\|}\sum_s I(X;s)$ | 冗余度 |
| `ig(X, Y, S)` | $I(X;Y) - R(X,S) + CR(X,Y,S)$ | 信息增益 |

#### 使用示例

```python
from src.dsl import eval_dsl

# 使用 mRMR 公式评估特征
score = eval_dsl(
    "mi(X, Y) - redundancy(X, S)",
    X=candidate_feature,
    Y=target,
    S=selected_features_list
)

# 使用自定义公式
custom_formula = """
    nmi(X, Y) * (1 - safe_div(redundancy(X, S), entropy(X) + EPS))
"""
score = eval_dsl(custom_formula, X=X, Y=Y, S=S)
```

### 📊 内置 SOTA 方法

| 方法 | 公式 | 年份 | 特点 |
|------|------|------|------|
| mRMR | $I(X;Y) - \frac{1}{\|S\|}\sum_s I(X;s)$ | 2005 | 平衡相关性与冗余性 |
| JMI | $I(X;Y) - \sum I(X;s) + \sum I(X;Y\|s)$ | 2012 | 考虑特征协同 |
| CMIM | $I(X;Y) - \max[I(X;s) - I(X;Y\|s)]$ | 2004 | 处理复杂交互 |
| CIFE | $I(X;Y) - 2\sum I(X;s)$ | 2006 | 强调去冗余 |
| DISR | $\sum NMI(X;Y) \cdot SU(X;s)$ | 2008 | 归一化处理 |
| ICAP | $I(X;Y) - \sum \max(0, I(X;s) - I(X;Y\|s))$ | 2005 | 保守处理负交互 |

### 🔌 扩展开发

#### 添加新的 DSL 算子

```python
# src/dsl/advanced.py

def my_operator(self, X, Y, S):
    """
    自定义算子
    
    Args:
        X: 候选特征
        Y: 目标变量
        S: 已选特征列表
    
    Returns:
        特征得分
    """
    # 你的实现
    mi = self.mutual_info(X, Y)
    red = self.redundancy(X, S)
    
    # 自定义逻辑
    score = mi / (red + 1)
    
    return score

# 在 src/dsl/registry.py 中注册
cls._functions['my_operator'] = cls._operators.my_operator
```

#### 添加新的 LLM 客户端

```python
# src/llm/my_client.py

from .base import BaseLLMClient, LLMResponse

class MyLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        # 初始化你的客户端
        pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # 实现生成逻辑
        content = "..."  # 你的 LLM 响应
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage={'prompt_tokens': 0, 'completion_tokens': 0},
            finish_reason='stop'
        )

# 在 src/llm/__init__.py 中注册
CLIENT_REGISTRY['my_llm'] = MyLLMClient
```

#### 添加自定义数据集

```python
# 在你的脚本中

from src.sandbox import TestDataset
import pandas as pd

def load_my_dataset(path: str) -> TestDataset:
    df = pd.read_csv(path)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    return TestDataset(
        name="my_dataset",
        X=X,
        y=y,
        metadata={'source': 'custom'}
    )

# 使用
my_data = load_my_dataset("data/my_data.csv")
factory.sandbox.evaluate_formula(formula, "test", datasets=[my_data])
```

### ❓ 常见问题

#### Q1: API Key 配置问题

```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY

# 如果为空，设置环境变量
export DEEPSEEK_API_KEY="sk-your-key"
```

#### Q2: 数值不稳定错误

```python
# 确保公式中使用了安全算子
# ❌ 错误
formula = "mi(X, Y) / redundancy(X, S)"

# ✅ 正确
formula = "safe_div(mi(X, Y), redundancy(X, S) + EPS)"
```

#### Q3: 内存不足

```yaml
# 在 config/settings.yaml 中调整
sandbox:
  max_features_to_select: 15  # 减少选择数量
  
dsl:
  knn:
    k: 3  # 减少邻居数
```

#### Q4: 演化效果不佳

1. 增加迭代次数: `--iterations 10`
2. 尝试不同模型: `--model deepseek-reasoner`
3. 调整 temperature: 在配置文件中设置 `temperature: 0.5`

### 📈 性能优化建议

1. **大规模数据集**: 先降采样再进行演化
2. **并行化**: 使用 `n_jobs=-1` 启用多核
3. **缓存**: 相同数据集的结果会被缓存
4. **早停**: 设置 `early_stop=True` 在超越 SOTA 后停止

### 📄 License

MIT License

### 🙏 致谢

- 特征选择经典方法论文作者
- DeepSeek 团队提供的优质 API
- OpenAI 提供的 API 接口设计参考

---

## English Documentation

### 📖 Introduction

Feature Selection Factory is an innovative framework that leverages Large Language Models (LLMs) to automatically evolve new feature selection algorithms.

### 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Set API Key
export DEEPSEEK_API_KEY="your-api-key"

# Run evolution
python scripts/run_evolution_deepseek.py --iterations 5
```

### 📚 Usage

```python
from src import FeatureSelectionFactory
from src.llm import DeepSeekMathClient

# Initialize
factory = FeatureSelectionFactory(
    llm_client=DeepSeekMathClient()
)

# Evolve new operators
best = factory.evolve_operators(n_iterations=5)

# Select features
selected, info = factory.select_features(X, y, method='auto', k=20)
```

### 📁 Project Structure

See Chinese documentation for detailed structure.

### 🔧 Configuration

Edit `config/settings.yaml` to customize parameters.

### 📄 License

MIT License
```

---

## 四、快速上手脚本

创建一个完整的一键运行脚本：

```python
# quick_start.py

"""
一键快速开始脚本
运行: python quick_start.py
"""
import os
import sys

def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("Step 1: 检查环境配置")
    print("=" * 60)
    
    # 检查 Python 版本
    py_version = sys.version_info
    print(f"  Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version < (3, 8):
        print("  ❌ 需要 Python 3.8+")
        return False
    
    # 检查依赖
    required = ['numpy', 'scipy', 'sklearn', 'pandas', 'openai']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} 未安装")
            missing.append(pkg)
    
    if missing:
        print(f"\n请安装缺失的依赖: pip install {' '.join(missing)}")
        return False
    
    # 检查 API Key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if api_key:
        print(f"  ✅ DEEPSEEK_API_KEY 已设置 ({api_key[:10]}...)")
    else:
        print("  ⚠️  DEEPSEEK_API_KEY 未设置")
        print("      请设置: export DEEPSEEK_API_KEY='your-key'")
    
    return True


def run_demo():
    """运行演示"""
    print("\n" + "=" * 60)
    print("Step 2: 运行演示")
    print("=" * 60)
    
    # 添加项目路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from src import FeatureSelectionFactory
    from src.llm import LocalMockClient
    from sklearn.datasets import make_classification
    
    # 创建测试数据
    print("\n创建测试数据集...")
    X, y = make_classification(
        n_samples=500,
        n_features=50,
        n_informative=10,
        random_state=42
    )
    print(f"  数据形状: X={X.shape}, y={y.shape}")
    
    # 创建工厂 (使用模拟客户端进行演示)
    print("\n初始化特征选择工厂...")
    factory = FeatureSelectionFactory(
        llm_client=LocalMockClient(),  # 演示用模拟客户端
        output_dir="./demo_outputs"
    )
    
    # 测试特征选择
    print("\n执行特征选择 (使用 mRMR 方法)...")
    selected, info = factory.select_features(X, y, method='mRMR', k=10)
    print(f"  选中的特征索引: {selected}")
    
    # 测试 DSL
    print("\n测试 DSL 算子...")
    from src.dsl import eval_dsl
    
    score = eval_dsl("mi(X, Y)", X=X[:, 0], Y=y)
    print(f"  第一个特征的互信息: {score:.4f}")
    
    score = eval_dsl("mi(X, Y) - redundancy(X, S)", X=X[:, 0], Y=y, S=[])
    print(f"  第一个特征的 mRMR 得分: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          Feature Selection Factory - 快速开始                  ║
║                                                               ║
║  这个脚本将:                                                   ║
║  1. 检查你的环境配置                                           ║
║  2. 运行一个简单的演示                                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    if not check_environment():
        print("\n❌ 环境检查失败，请先解决上述问题")
        return
    
    print("\n✅ 环境检查通过!")
    
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ 演示运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                        下一步                                  ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. 设置 DeepSeek API Key:                                    ║
║     export DEEPSEEK_API_KEY="your-api-key"                    ║
║                                                               ║
║  2. 运行算子演化:                                              ║
║     python scripts/run_evolution_deepseek.py --iterations 5   ║
║                                                               ║
║  3. 运行基准测试:                                              ║
║     python scripts/run_benchmark.py                           ║
║                                                               ║
║  4. 查看文档:                                                  ║
║     阅读 README.md 了解更多功能                                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
```

---

## 五、完整运行流程总结

```bash
# ============================================
# 完整使用流程
# ============================================

# 1. 安装依赖
pip install -r requirements.txt
pip install -e .

# 2. 配置 API Key
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key"

# 3. 快速验证环境
python quick_start.py

# 4. 运行算子演化 (使用 DeepSeek Math)
python scripts/run_evolution_deepseek.py --iterations 5 --model deepseek-math

# 5. 查看演化结果
cat outputs/operators/*.json

# 6. 运行基准测试
python scripts/run_benchmark.py

# 7. 查看对比报告
cat outputs/benchmark_report.md

# 8. 在你的数据上使用
python -c "
from src import FeatureSelectionFactory
import pandas as pd

df = pd.read_csv('your_data.csv')
X = df.drop(columns=['target']).values
y = df['target'].values

factory = FeatureSelectionFactory()
selected, info = factory.select_features(X, y, method='auto', k=20)
print('Selected features:', selected)
"
```

---

## 六、注意事项清单

| 序号 | 事项 | 说明 |
|------|------|------|
| 1 | ✅ 安装所有依赖 | `pip install -r requirements.txt` |
| 2 | ✅ 设置 API Key | `export DEEPSEEK_API_KEY="..."` |
| 3 | ✅ 运行快速测试 | `python quick_start.py` |
| 4 | ⚠️ 检查输出目录 | 默认 `./outputs`，可通过参数修改 |
| 5 | ⚠️ 监控 API 成本 | DeepSeek 约 $0.14/百万 token |
| 6 | ⚠️ 数据预处理 | 确保数据无缺失值、已标准化 |
| 7 | ⚠️ 特征数量 | 建议小于 10000，否则会很慢 |
| 8 | ⚠️ 迭代次数 | 建议先 3 次，效果好再加到 5-10 次 |

如果遇到任何问题，请检查：
1. Python 版本是否 >= 3.8
2. 所有依赖是否正确安装
3. API Key 是否有效
4. 数据格式是否正确