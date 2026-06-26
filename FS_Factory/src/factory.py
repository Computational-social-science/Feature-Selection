# src/factory.py

"""
特征选择工厂 - 主类
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from datetime import datetime
import re

from .dsl import DSLRegistry, eval_dsl
from .prompts import (
    SOTA_METHODS, get_best_method, 
    MetaPromptBuilder, validate_formula
)
from .sandbox import (
    SandboxExecutor, EvaluationResult, 
    CriticAgent, Criticism,
    DatasetManager
)
from .router import RouterAgent, RoutingDecision
from .validation import ExperimentRunner, run_statistical_analysis, generate_report
from .llm import BaseLLMClient, LocalMockClient

@dataclass
class EvolvedOperator:
    """演化的算子"""
    name: str
    formula_dsl: str
    formula_math: str
    intuition: str
    theoretical_advantage: str
    evaluation_result: Optional[EvaluationResult] = None
    creation_time: str = ""
    
    def __post_init__(self):
        if not self.creation_time:
            self.creation_time = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'formula_dsl': self.formula_dsl,
            'formula_math': self.formula_math,
            'intuition': self.intuition,
            'theoretical_advantage': self.theoretical_advantage,
            'avg_accuracy': self.evaluation_result.avg_accuracy if self.evaluation_result else 0,
            'creation_time': self.creation_time
        }


class FeatureSelectionFactory:
    """特征选择工厂"""
    
    def __init__(self, 
                 llm_client: Optional[BaseLLMClient] = None,
                 output_dir: str = "./outputs"):
        """
        初始化工厂
        
        Args:
            llm_client: LLM 客户端 (None 则使用模拟客户端)
            output_dir: 输出目录
        """
        self.llm_client = llm_client or LocalMockClient()
        self.output_dir = output_dir
        
        # 初始化组件
        self.sandbox = SandboxExecutor()
        self.router = RouterAgent()
        self.critic = CriticAgent()
        self.dataset_manager = DatasetManager()
        
        # 演化历史
        self.evolution_history: List[EvolvedOperator] = []
        self.best_operator: Optional[EvolvedOperator] = None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "operators"), exist_ok=True)
 


    def evolve_operators(self, 
                        n_iterations: int = 5,
                        target_datasets: Optional[List[str]] = None,
                        verbose: bool = True) -> EvolvedOperator:
        """
        演化特征选择算子
        """
        if verbose:
            print("="*60)
            print("特征选择算子演化开始")
            print("="*60)
        
        best_score = 0.0
        
        # 【新增 1】：初始化一个集合，用于记录所有已经生成过的公式，防止大模型做无用功
        history_formulas = {method.formula_dsl for method in SOTA_METHODS}
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"迭代 {iteration + 1}/{n_iterations}")
            
            # 【新增 2】：动态构建约束条件 (Constraints) 
            dynamic_constraints = []
            if history_formulas:
                avoid_list = "\n- ".join(list(history_formulas))
                dynamic_constraints.append(f"绝对禁止生成以下已经尝试过的公式：\n- {avoid_list}")

            # Step 1: 生成元提示词 (带入防重复约束)
            meta_prompt = MetaPromptBuilder.build_evolution_prompt(
                language="chinese",
                constraints=dynamic_constraints  # 【修改】：将黑名单传给提示词
            )
            
            # Step 2: 调用 LLM 生成算子
            response = self.llm_client.generate(meta_prompt, temperature=0.7)
            print('@'*60)
            print(response)
            print('@'*60)
            operator_info = self._parse_llm_response(response.content)

            # --- 优化点 1: 解决 "Unknown" 命名问题 ---
            method_name = operator_info.get('method_name', 'Unknown')
            if method_name == 'Unknown' or not method_name:
                method_name = f"Evolved_Iter{iteration+1}_{datetime.now().strftime('%m%d%H%M')}"
            
            # 【新增 3】：将刚刚生成的公式立刻加入黑名单
            current_formula = operator_info.get('formula_dsl', '')
            if current_formula and current_formula != 'mi(X, Y)': # 排除兜底公式
                history_formulas.add(current_formula)
            
            if verbose:
                print(f"\n生成算子: {method_name}")
                print(f"公式: {current_formula[:80]}...")
            
            # Step 3: 验证公式
            validation = validate_formula(current_formula)
            
            if not validation.is_valid:
                if verbose:
                    print(f"验证失败: {validation.issues}")
                continue
            
            # Step 4: 沙箱评估数据集获取
            datasets = None
            if target_datasets:
                datasets = [self.dataset_manager.get_dataset(name) for name in target_datasets]
            else:
                datasets = self.dataset_manager.get_all_datasets()
            
            if not datasets:
                print("⚠️ 错误: 未找到任何本地可用数据集进行评估！生成标准数据集！")
                datasets = [self.dataset_manager.load_or_generate(name) for name in target_datasets]
            
            eval_result = self.sandbox.evaluate_formula(
                current_formula,
                method_name,
                datasets=datasets,
                verbose=verbose
            )
            
            # Step 5: 批评者分析
            criticism = self.critic.analyze(eval_result)
            
            if verbose:
                print(f"\n评估结果: {eval_result.avg_accuracy:.4f}")
                print(f"是否超越SOTA: {criticism.beat_sota}")
                for suggestion in criticism.suggestions[:3]:
                    print(f"  - {suggestion}")
            
            # 创建算子对象
            operator = EvolvedOperator(
                name=method_name,
                formula_dsl=current_formula,
                formula_math=operator_info.get('formula_math', ''),
                intuition=operator_info.get('intuition', ''),
                theoretical_advantage=operator_info.get('theoretical_advantage', ''),
                evaluation_result=eval_result
            )
            
            # 记录历史
            self.evolution_history.append(operator)
            
            # 更新最佳
            if eval_result.avg_accuracy > best_score:
                best_score = eval_result.avg_accuracy
                self.best_operator = operator
                
                if verbose:
                    print(f"\n🎉 发现新的全局最佳算子！")
            
            # 【新增 4】：如果表现超越 SOTA，将其动态注入到 SOTA_METHODS 内存中！
            if criticism.beat_sota:
                if verbose:
                    print(f"🏆 算子表现卓越，已将其动态升格并存入 SOTA 表中！")
                
                # 尝试将其动态追加到 SOTA 列表中，下一次迭代大模型就会看到它
                try:
                    # 使用反射机制，动态创建一个跟当前 SOTA 列表里相同的对象
                    sota_class = type(SOTA_METHODS[0]) 
                    
                    # 构造 mock 的 performance 字典 (适配你的代码逻辑)
                    dataset_perf = {ds: eval_result.avg_accuracy for ds in [d.name for d in datasets]}
                    
                    # 【核心修复】：严格按照 SOTAMethod 的数据结构传入参数
                    new_sota = sota_class(
                        name=operator.name,
                        formula_dsl=operator.formula_dsl,
                        formula_math=operator.formula_math,
                        year=datetime.now().year,               # 自动填入当前年份
                        authors="AutoFS Agent",                 # 署名 AI 智能体
                        description=operator.intuition,         # 将 intuition 映射到 description
                        performance=dataset_perf
                    )
                    SOTA_METHODS.append(new_sota)
                except Exception as e:
                    print(f"⚠️ 动态加入 SOTA 列表时发生错误: {e}")
            
            # 保存算子 (存为 JSON 文件)
            self._save_operator(operator)
        
        if verbose:
            print("\n" + "="*60)
            print("演化完成！")
            if self.best_operator:
                print(f"最佳算子: {self.best_operator.formula_dsl}")
                print(f"平均准确率: {best_score:.4f}")
        
        return self.best_operator




    def select_features(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       method: str = 'auto',
                       k: int = 30,
                       verbose: bool = False) -> Tuple[List[int], Dict]:
        """
        特征选择入口
        
        Args:
            X: 特征矩阵
            y: 目标变量
            method: 方法名 ('auto' 或指定方法)
            k: 选择特征数
            verbose: 是否打印详细信息
        
        Returns:
            (选中的特征索引列表, 元信息)
        """
        meta_info = {}
        
        if method == 'auto':
            # 使用路由智能体
            routing = self.router.route(X, y)
            method = routing.selected_method
            meta_info['routing'] = {
                'selected_method': method,
                'confidence': routing.confidence,
                'reasoning': routing.reasoning
            }
            
            if verbose:
                print(f"Router 选择方法: {method} (置信度: {routing.confidence:.2f})")
        
        # 获取公式
        formula = self._get_method_formula(method)
        
        if verbose:
            print(f"使用公式: {formula[:60]}...")
        
        # 执行特征选择
        selected = self._forward_selection(X, y, formula, k, verbose)
        
        # 更新路由历史
        if 'routing' in meta_info:
            # 这里可以添加评估后更新路由逻辑
            pass
        
        return selected, meta_info
    
    def run_benchmark(self,
                     methods: Optional[Dict[str, str]] = None,
                     datasets: Optional[List[str]] = None,
                     verbose: bool = True) -> Dict:
        """
        运行基准测试
        
        Args:
            methods: 方法字典 {name: formula}
            datasets: 数据集名称列表
            verbose: 是否打印详细信息
        
        Returns:
            测试结果
        """
        if methods is None:
            # 使用 SOTA 方法 + 演化的方法
            methods = {m.name: m.formula_dsl for m in SOTA_METHODS}
            
            for op in self.evolution_history:
                methods[op.name] = op.formula_dsl
        
        if datasets is None:
            datasets = ['madelon_like', 'gisette_like', 'high_corr']
        
        # 准备数据集
        dataset_dict = {}
        for name in datasets:
            ds = self.dataset_manager.load_or_generate(name)
            dataset_dict[name] = (ds.X, ds.y)
        
        # 运行实验
        runner = ExperimentRunner()
        results = runner.run_comparison(methods, dataset_dict, verbose=verbose)
        
        # 统计分析
        analysis = run_statistical_analysis(results)
        
        # 生成报告
        agent_methods = [op.name for op in self.evolution_history]
        report = generate_report(results, analysis, agent_methods)
        
        # 保存结果
        results.to_csv(os.path.join(self.output_dir, 'benchmark_results.csv'), index=False)
        with open(os.path.join(self.output_dir, 'benchmark_report.md'), 'w') as f:
            f.write(report)
        
        return {
            'results': results,
            'analysis': analysis,
            'report': report
        }

    def _parse_llm_response(self, content: str) -> Dict:
        """解析 LLM 响应"""
        
        def clean_json_string(s: str) -> str:
            # 修复 LLM 生成 LaTeX 时忘记转义反斜杠的问题 (例如将 \cdot 修复为 \\cdot)
            # 正则逻辑：匹配单反斜杠 \，如果它后面跟着的不是合法的 JSON 转义符 (", \, /, b, f, n, r, t, u)，则将其翻倍。
            return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)

        def sanitize_parsed_json(parsed_data: Dict) -> Dict:
            """清洗大模型生成的脏数据"""
            if 'formula_dsl' in parsed_data:
                dsl = parsed_data['formula_dsl']
                # 终极斩头术：如果大模型写了 "J = mi(X, Y)"，直接取等号右边的部分
                if '=' in dsl:
                    dsl = dsl.split('=', 1)[1].strip()
                parsed_data['formula_dsl'] = dsl
            return parsed_data

        # 策略 1：精准提取 ```json ... ``` 包裹的 {...} 块
        json_pattern = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
        match = json_pattern.search(content)
        
        if match:
            raw_json_str = match.group(1)
            cleaned_json_str = clean_json_string(raw_json_str) # 清洗非法反斜杠
            try:
                parsed = json.loads(cleaned_json_str)
                return sanitize_parsed_json(parsed) # <--- 使用清洗函数
            except json.JSONDecodeError as e:
                print(f"Debug: 代码块内 JSON 解析失败 - {e}")
                
        # 策略 2（兜底）：大模型有时会忘记写 ```，直接匹配完整 {...} 结构
        fallback_pattern = re.compile(r'(\{[\s\S]*?"method_name"[\s\S]*?"theoretical_advantage"[\s\S]*?\})', re.DOTALL)
        fallback_match = fallback_pattern.search(content)
        
        if fallback_match:
            raw_json_str = fallback_match.group(1)
            # 补齐大括号防截断
            if not raw_json_str.strip().endswith('}'):
                raw_json_str += '\n}'
            cleaned_json_str = clean_json_string(raw_json_str) # 清洗非法反斜杠
            try:
                parsed = json.loads(cleaned_json_str)
                return sanitize_parsed_json(parsed) # <--- 使用清洗函数
            except json.JSONDecodeError as e:
                print(f"Debug: 兜底 JSON 解析失败 - {e}")

        # 如果所有提取尝试都失败，记录原始文本并返回默认值
        print(f"--- 解析彻底失败，原始返回内容如下 ---\n{content}\n----------------------------------")
        
        return {
            'method_name': 'Unknown',
            'formula_dsl': 'mi(X, Y)',
            'formula_math': '$J = I(X;Y)$',
            'intuition': 'Failed to parse response',
            'theoretical_advantage': 'Unknown'
        }

    def _get_method_formula(self, method_name: str) -> str:
        """获取方法公式"""
        # 检查 SOTA 方法
        for method in SOTA_METHODS:
            if method.name.lower() == method_name.lower():
                return method.formula_dsl
        
        # 检查演化的方法
        for op in self.evolution_history:
            if op.name.lower() == method_name.lower():
                return op.formula_dsl
        
        # 默认 mRMR
        return "mi(X, Y) - redundancy(X, S)"
    
    def _forward_selection(self, X: np.ndarray, y: np.ndarray,
                          formula: str, k: int, verbose: bool) -> List[int]:
        """前向特征选择"""
        n_features = X.shape[1]
        k = min(k, n_features)
        selected = []
        remaining = list(range(n_features))
        
        for step in range(k):
            best_score = -np.inf
            best_feat = None
            
            for feat_idx in remaining:
                try:
                    S_features = [X[:, i] for i in selected] if selected else []
                    score = eval_dsl(formula, X=X[:, feat_idx], Y=y, S=S_features)
                    
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_feat = feat_idx
                except:
                    continue
            
            if best_feat is not None:
                selected.append(best_feat)
                remaining.remove(best_feat)
                
                if verbose and step % 5 == 0:
                    print(f"Step {step+1}: selected feature {best_feat}, score={best_score:.4f}")
            else:
                break
        
        return selected
    
    def _save_operator(self, operator: EvolvedOperator):
        """保存算子"""
        path = os.path.join(self.output_dir, "operators", f"{operator.name}.json")
        
        with open(path, 'w') as f:
            json.dump(operator.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load_operator(self, name: str) -> Optional[EvolvedOperator]:
        """加载算子"""
        path = os.path.join(self.output_dir, "operators", f"{name}.json")
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            return EvolvedOperator(**data)
        
        return None
    
    def get_available_methods(self) -> List[str]:
        """获取所有可用方法"""
        methods = [m.name for m in SOTA_METHODS]
        methods.extend(op.name for op in self.evolution_history)
        return methods