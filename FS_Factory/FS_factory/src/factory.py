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
        data = {
            'name': self.name,
            'formula_dsl': self.formula_dsl,
            'formula_math': self.formula_math,
            'intuition': self.intuition,
            'theoretical_advantage': self.theoretical_advantage,
            'avg_accuracy': self.evaluation_result.avg_accuracy if self.evaluation_result else 0,
            'creation_time': self.creation_time
        }
        
        # 如果有详细的多分类器结果，也保存下来
        if self.evaluation_result and self.evaluation_result.dataset_results:
            # 提取第一个数据集的结果来获取分类器名称
            first_ds_res = next(iter(self.evaluation_result.dataset_results.values()))
            for clf in ['bayes', 'rf', 'svm', 'knn']:
                if f'{clf}_accuracy' in first_ds_res:
                    # 计算该分类器在所有数据集上的平均准确率
                    clf_accs = [res.get(f'{clf}_accuracy', 0) for res in self.evaluation_result.dataset_results.values() if 'error' not in res]
                    if clf_accs:
                        data[f'{clf}_avg_accuracy'] = sum(clf_accs) / len(clf_accs)
                        
        return data


class FeatureSelectionFactory:
    """特征选择工厂"""
    
    def __init__(self, 
                 llm_client: Optional[BaseLLMClient] = None,
                 output_dir: str = "./outputs",
                 performance_context: str = "",
                 use_gpu_for_evaluation: bool = False):
        """
        初始化工厂
        
        Args:
            llm_client: LLM 客户端 (None 则使用模拟客户端)
            output_dir: 输出目录
            performance_context: 历史算子性能，用于 Prompt
            use_gpu_for_evaluation: 评估时是否使用 GPU
        """
        self.llm_client = llm_client or LocalMockClient()
        self.output_dir = output_dir
        self.performance_context = performance_context
        
        # 初始化组件
        self.sandbox = SandboxExecutor(use_gpu=use_gpu_for_evaluation)
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
        
        # 辅助函数：去除所有空白字符用于强力比对
        def normalize_formula(f: str) -> str:
            return "".join(f.split())

        # 【关键修复】：初始化历史公式记录器，确保查重机制生效
        # 将历史记录全部标准化存储，防止空格差异导致失效
        history_formulas = set()
        for method in SOTA_METHODS:
            history_formulas.add(normalize_formula(method.formula_dsl))
        
        # 记录已尝试过的算子名，防止模型一直用同一个名字
        history_names = set()
        
        # 动态反馈缓存：用于打破死循环的负反馈信息
        dynamic_negative_feedback = ""
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"迭代 {iteration + 1}/{n_iterations}")
            
            # 【优化】：动态构建黑名单约束
            dynamic_constraints = []
            if history_formulas:
                avoid_list = "\n- ".join(list(history_formulas))
                dynamic_constraints.append(f"🚨 绝对禁止生成以下已经尝试过的公式（严禁重复）：\n- {avoid_list}")

            # Step 1: 生成元提示词 (整合历史性能、黑名单以及最新的负反馈)
            current_perf_context = self.performance_context
            if dynamic_negative_feedback:
                current_perf_context = f"{current_perf_context}\n\n⚠️ 【最新评估警告】：\n{dynamic_negative_feedback}"

            meta_prompt = MetaPromptBuilder.build_evolution_prompt(
                language="chinese",
                constraints=dynamic_constraints,
                performance_context=current_perf_context
            )
            
            # Step 2: 调用 LLM 生成算子 (增加随机性，防止模式崩溃)
            response = self.llm_client.generate(
                meta_prompt, 
                temperature=0.9,  # 提高随机性
                top_p=0.95,       # 增加采样广度
                stop=["}\n```", "}\n\n", "}\n"] # 强制切断生成，防止复读
            )
            
            if verbose:
                print(f"--- LLM Response Received ---")
                
            operator_info = self._parse_llm_response(response.content)

            # 解析算子信息
            method_name = operator_info.get('method_name', 'Unknown')
            current_formula = operator_info.get('formula_dsl', '').strip()
            
            # 命名容错与唯一化
            if method_name == 'Unknown' or not method_name or method_name in history_names:
                method_name = f"Evolved_Iter{iteration+1}_{datetime.now().strftime('%m%d%H%M')}"
            history_names.add(method_name)

            # 【核心修改】：引入强制查重与硬拦截机制 (Hard Interception)
            normalized_formula = normalize_formula(current_formula)
            if not current_formula or current_formula == 'mi(X, Y)':
                if verbose:
                    print("⚠️ 模型输出了无效或兜底公式，跳过评估。")
                continue

            # 立即保存初始算子 JSON (保证迭代被记录)
            operator = EvolvedOperator(
                name=method_name,
                formula_dsl=current_formula,
                formula_math=operator_info.get('formula_math', ''),
                intuition=operator_info.get('intuition', ''),
                theoretical_advantage=operator_info.get('theoretical_advantage', ''),
                evaluation_result=None # 初始状态无评估结果
            )
            self._save_operator(operator)
            
            # 强力查重校验：如果是已经测试过的公式，立即硬拦截
            if normalized_formula in history_formulas:
                # ！！！硬拦截：伪造负反馈并进入下一轮！！！
                criticism = (
                        f"🚨 严重警告：你刚刚生成的公式 `{current_formula}` 与历史完全重复，或者是直接抄袭了提示词中的例子！\n"
                        f"请立即停止偷懒！下一轮你必须尝试引入之前没用过的算子（比如 entropy, cmi 或 max_redundancy），"
                        f"或者改变聚合方式（比如用 max 代替 mean）。如果再次重复，任务将判定为彻底失败！"
                    )
                if verbose:
                    print(f"🚫 【硬拦截】检测到公式重复（标准化对比）！")
                    print(f"  公式: {current_formula[:80]}...")        
                
                # 将强烈警告作为负反馈，强制改变下一轮的 Prompt
                dynamic_negative_feedback = criticism
                
                # 【关键】即使被拦截了，也要确保它被加入当前运行实例的历史记录中
                self.evolution_history.append(operator)
                continue 
            
            # 🚨 加入监控探头：打印出来看看到底哪里不一样！
            if verbose:
                print(f"============== 查重探头 ==============")
                print(f"当前模型起名: {method_name}")
                print(f"当前提取公式: {normalized_formula}")
                print(f"当前历史库中共有 {len(history_formulas)} 条记录")
                print(f"====================================")

            # 记录此公式到黑名单 (存储标准化后的公式)
            history_formulas.add(normalized_formula)
            # 清空之前的负反馈，如果当前生成是全新的
            dynamic_negative_feedback = ""

            if verbose:
                print(f"生成算子: {method_name}")
                print(f"公式: {current_formula[:80]}...")
            
            # Step 3: 验证公式语法
            validation = validate_formula(current_formula)
            if not validation.is_valid:
                if verbose:
                    print(f"验证失败: {validation.issues}")
                dynamic_negative_feedback = f"你生成的公式 `{current_formula}` 存在语法错误或 DSL 不兼容问题：{validation.issues}。请修正后再提交。"
                # 即使验证失败，也要加入历史记录，防止模型重复犯错
                self.evolution_history.append(operator)
                continue
                
    # ... 下面继续执行你的 Colon.csv 评估代码 ...
            # Step 4: 执行评估 (仅对通过查重和验证的算子进行昂贵评估)
            datasets = None
            if target_datasets:
                datasets = [self.dataset_manager.get_dataset(name) for name in target_datasets]
            else:
                datasets = self.dataset_manager.get_all_datasets()
            
            if not datasets:
                if verbose:
                    print("⚠️ 错误: 未找到任何可用数据集，生成标准数据集。")
                datasets = [self.dataset_manager.load_or_generate(name) for name in (target_datasets or ["madelon_like"])]
            
            eval_result = self.sandbox.evaluate_formula(
                current_formula,
                method_name,
                datasets=datasets,
                verbose=verbose
            )
            
            # 更新算子对象的评估结果并再次保存 (补全分类数据)
            operator.evaluation_result = eval_result
            self._save_operator(operator)
            
            # Step 5: 批评者分析与反馈循环
            criticism = self.critic.analyze(eval_result)
            
            if verbose:
                print(f"\n评估结果 (Avg Acc): {eval_result.avg_accuracy:.4f}")
                print(f"是否超越 SOTA: {criticism.beat_sota}")
            
            # 记录反馈供下一轮迭代参考
            dynamic_negative_feedback = f"上一次生成的算子 `{method_name}` 表现为 {eval_result.avg_accuracy:.4f}。建议：{'; '.join(criticism.suggestions[:2])}"
            
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
        """解析 LLM 响应，针对含有 LaTeX 的 JSON 进行强力纠错"""
        
        def clean_json_string(s: str) -> str:
            # 终极反斜杠修复大法：
            # 为了避免 \frac (被误认为 \f) 等问题，我们采用“保护-全转义-恢复”策略
            
            # 1. 保护大模型输出的合法且必要的转义（换行和引号）
            s = s.replace('\\n', '___NEWLINE___')
            s = s.replace('\\"', '___QUOTE___')
            
            # 2. 将剩下的所有单反斜杠全部翻倍，解决所有 LaTeX 符号 (\sum, \cdot, \frac 等)
            s = s.replace('\\', '\\\\')
            
            # 3. 恢复之前的换行和引号
            s = s.replace('___NEWLINE___', '\\n')
            s = s.replace('___QUOTE___', '\\"')
            return s

        def sanitize_parsed_json(parsed_data: Dict) -> Dict:
            """清洗大模型生成的脏数据"""
            if 'formula_dsl' in parsed_data:
                dsl = str(parsed_data['formula_dsl'])
                # 终极斩头术：如果大模型写了 "J = mi(X, Y)"，直接取等号右边的部分
                if '=' in dsl:
                    dsl = dsl.split('=', 1)[1].strip()
                # 强行剔除 DSL 中可能残留的 LaTeX 痕迹（比如模型没听话，悄悄写了 \sum）
                dsl = dsl.replace('\\', '')
                parsed_data['formula_dsl'] = dsl.strip()
            return parsed_data

        # 核心提取逻辑：使用精准的非贪婪匹配，框定第一个完整的 JSON 块
        # 核心提取逻辑：容忍缺失的右大括号
        # [^}]* 会匹配 'theoretical_advantage' 之后所有非 } 的字符（即直到字符串末尾）
        json_pattern = re.compile(r'(\{\s*"method_name"[\s\S]*?"theoretical_advantage"[^}]*)', re.IGNORECASE)
        match = json_pattern.search(content)
        
        if match:
            raw_json_str = match.group(1).strip()
            # 【关键修复】如果字符串不是以 } 结尾，强行给它补上闭合括号！
            if not raw_json_str.endswith('}'):
                raw_json_str += '\n}'
                
            cleaned_json_str = clean_json_string(raw_json_str)
            
            try:
                parsed = json.loads(cleaned_json_str)
                return sanitize_parsed_json(parsed)
            except json.JSONDecodeError as e:
                print(f"Debug: JSON 解析依然失败 - {e}")

        # 如果提取失败或解析彻底失败，打印日志并返回安全的默认基线算子
        print(f"--- 解析彻底失败，原始返回内容如下 ---\n{content}\n----------------------------------")
        
        return {
            'method_name': 'MRMR_FALLBACK',
            'formula_dsl': 'mi(X, Y) - redundancy(X, S)',
            'formula_math': '$J = I(X;Y) - \frac{1}{|S|}\sum_{s \in S} I(X;s)$',
            'intuition': '解析失败，系统自动回退至经典 mRMR 算子。',
            'theoretical_advantage': '作为系统容错的兜底基线。'
        }

    def _get_method_formula(self, method_name: str) -> str:
        """获取方法公式"""
        # 增加 strip() 和统一大小写，防止大模型生成的 method_name 带有空格或大小写乱跳
        target_name = method_name.strip().lower()
        
        # 1. 检查 SOTA 方法
        for method in SOTA_METHODS:
            if method.name.strip().lower() == target_name:
                return method.formula_dsl
        
        # 2. 检查演化的方法
        for op in self.evolution_history:
            if op.name.strip().lower() == target_name:
                return op.formula_dsl
        
        # 3. 默认兜底方案 (如果在记录中找不到，给予警告并返回安全算子)
        print(f"Warning: 算子记录中未找到 [{method_name}] 的定义，已使用默认 mRMR 兜底。")
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
        """保存算子，如果重名则另存为副本"""
        # 如果该 operator 实例已经分配了保存路径，则直接使用，确保单次迭代内的更新覆盖同一个文件
        if hasattr(operator, '_saved_path'):
            path = operator._saved_path
        else:
            # 否则，寻找一个不冲突的路径
            original_name = operator.name
            # 清理文件名中可能的非法字符
            base_filename = re.sub(r'[\\/*?:"<>|]', '_', original_name)
            
            base_path = os.path.join(self.output_dir, "operators", base_filename)
            path = f"{base_path}.json"
            
            counter = 1
            while os.path.exists(path):
                # 更新 operator 的名字以匹配副本名
                new_name = f"{original_name}_v{counter}"
                new_filename = re.sub(r'[\\/*?:"<>|]', '_', new_name)
                path = os.path.join(self.output_dir, "operators", f"{new_filename}.json")
                operator.name = new_name
                counter += 1
            
            # 记录分配到的路径
            operator._saved_path = path
            
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