import sys
import os
import argparse
import yaml
import pandas as pd
import glob
import random
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factory import FeatureSelectionFactory
from src.llm import create_client, VLLMClient

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def load_and_summarize_benchmarks(benchmark_dir: str) -> str:
    """加载并汇总 benchmark 结果，生成性能报告"""
    abs_benchmark_dir = os.path.abspath(benchmark_dir)
    print(f"🔍 Searching for results in: {abs_benchmark_dir}")
    csv_files = glob.glob(os.path.join(abs_benchmark_dir, "*_results.csv"))
    
    if not csv_files:
        print(f"⚠️ No benchmark files found in {abs_benchmark_dir}")
        return "No benchmark results found."
    
    print(f"📄 Found {len(csv_files)} benchmark files.")
    all_results = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                print(f"⚠️ File {os.path.basename(f)} is empty.")
                continue
            df['method'] = os.path.basename(f).replace("_results.csv", "")
            all_results.append(df)
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
        
    if not all_results:
        return "No valid benchmark data found."
        
    full_df = pd.concat(all_results, ignore_index=True)
    
    # 确保列是数值类型
    accuracy_cols = [col for col in full_df.columns if 'accuracy' in col]
    for col in accuracy_cols + ['selection_time']:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    # 计算每个方法在所有数据集上的平均性能
    perf_summary = full_df.groupby('method').mean(numeric_only=True)
    
    if perf_summary.empty:
        print("⚠️ No numeric performance data could be aggregated.")
        return "Performance summary is empty after processing."
        
    accuracy_cols = [col for col in perf_summary.columns if 'accuracy' in col]
    if not accuracy_cols:
        print(f"⚠️ No accuracy columns found. Available columns: {list(perf_summary.columns)}")
        return "No accuracy columns found in benchmark results."
        
    perf_summary['avg_accuracy'] = perf_summary[accuracy_cols].mean(axis=1)
    perf_summary = perf_summary.sort_values('avg_accuracy', ascending=False)
    
    # 生成 Markdown 格式的性能报告
    report = "| Method | Avg Accuracy | Bayes Acc | RF Acc | SVM Acc | KNN Acc | Selection Time |\n"
    report += "|:---|:---|:---|:---|:---|:---|:---|\n"
    
    for method, row in perf_summary.head(10).iterrows(): # 只展示前10名
        report += f"| {method} | **{row.get('avg_accuracy', 0):.4f}** | {row.get('bayes_accuracy', 0):.4f} | {row.get('rf_accuracy', 0):.4f} | {row.get('svm_accuracy', 0):.4f} | {row.get('knn_accuracy', 0):.4f} | {row.get('selection_time', 0):.2f}s |\n"
        
    return report

def get_available_datasets(data_dir: str) -> List[str]:
    """获取所有可用的数据集文件名"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    return [os.path.basename(f) for f in csv_files]

def main():
    parser = argparse.ArgumentParser(description='Run Evolution with Performance Feedback')
    parser.add_argument('--iterations', type=int, default=20, help='迭代次数')
    parser.add_argument('--output', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--benchmark_dir', type=str, default='./outputs/benchmark_results', help='Benchmark 结果目录')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='数据集目录')
    parser.add_argument('--n_eval_datasets', type=int, default=2, help='每次演化评估使用的数据集数量')
    parser.add_argument('--backend', type=str, default='local', choices=['local', 'nvidia'], help='LLM 后端选择: local (vLLM) 或 nvidia (NVIDIA API)')
    parser.add_argument('--model-path', type=str, default=None, help='本地模型路径 (仅当 --backend=local 时有效)')
    parser.add_argument('--nvidia_api_key', type=str, default=None, help='NVIDIA API Key (仅当 --backend=nvidia 时有效)')
    parser.add_argument('--nvidia_model', type=str, default='meta/llama-3.1-70b-instruct', help='NVIDIA API 模型名称 (仅当 --backend=nvidia 时有效)')
    parser.add_argument('--use_gpu', action='store_true', help='是否在评估时使用 GPU')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='vLLM 张量并行大小 (GPU 数量，仅当 --backend=local 时有效)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='vLLM GPU 显存占用率 (仅当 --backend=local 时有效)')
    parser.add_argument('--log_file', type=str, default=None, help='将所有输出保存到指定文件')
    parser.add_argument('--config_file', type=str, default=None, help='从 YAML 配置文件加载参数')
    args = parser.parse_args()
    
    print("="*60)
    print(" 🚀 Feature Selection Factory - Evolution with Feedback")
    print("="*60)

    # === 日志文件重定向 ===
    original_stdout = sys.stdout
    log_file_handle = None
    if args.log_file:
        try:
            log_file_handle = open(args.log_file, 'w', encoding='utf-8')
            sys.stdout = log_file_handle
            print(f"所有输出将重定向到文件: {args.log_file}")
        except Exception as e:
            print(f"⚠️ 无法打开日志文件 {args.log_file} 进行写入: {e}")
            sys.stdout = original_stdout # 恢复 stdout
            log_file_handle = None # 确保不再尝试关闭无效句柄

    try:
        # ==========================================
        # 1. 客户端初始化模块
        # ==========================================
        client = None
        if args.backend == 'local':
            model_path = args.model_path
            if not model_path:
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')
                config = load_config(config_path)
                model_path = config.get('llm', {}).get('vllm', {}).get('model_path')

            if not model_path:
                print("❌ 错误: 未找到本地模型路径！请通过 --model-path 参数指定或在 config/settings.yaml 中配置。")
                return
            
            print(f"Loading local vLLM model from: {model_path}")
            try:
                client = create_client(
                    provider='vllm',
                    model_path=model_path,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True,
                    max_model_len=8192, 
                    max_tokens=4096,
                    enforce_eager=True,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
                print("✅ 本地 vLLM 模型加载成功！")
            except Exception as e:
                print(f"\n❌ 本地 vLLM 模型加载失败: {e}")
                return
                
        elif args.backend == 'nvidia':
            if not args.nvidia_api_key:
                print("❌ 错误: 使用 NVIDIA API 需要提供 --nvidia_api_key。")
                return
                
            print(f"Using NVIDIA API model: {args.nvidia_model}")
            try:
                client = create_client(
                    provider='openai', 
                    api_key=args.nvidia_api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    model=args.nvidia_model
                )
                print("✅ NVIDIA API 客户端初始化成功！")
            except Exception as e:
                print(f"\n❌ NVIDIA API 客户端初始化失败: {e}")
                return
                
        else:
            print(f"❌ 错误: 未知的后端类型 '{args.backend}'。")
            return

        if client is None:
            print("❌ 错误: LLM 客户端未能成功初始化。")
            return

        # ==========================================
        # 2. 演化准备与主循环模块 (必须保持在 try 块内缩进)
        # ==========================================
        # 加载历史性能数据作为上下文
        print(f"\n🔍 Loading performance context from: {args.benchmark_dir}")
        performance_context = load_and_summarize_benchmarks(args.benchmark_dir)
        print("--- Performance Context for Prompt ---")
        print(performance_context)
        print("-------------------------------------")

        # 选择用于评估的数据集子集，提升迭代效率
        all_datasets = get_available_datasets(args.data_dir)
        if not all_datasets:
            print(f"❌ 错误: 在 {args.data_dir} 中未找到数据集！")
            return
        
        eval_datasets = random.sample(all_datasets, min(len(all_datasets), args.n_eval_datasets))
        print(f"\n🎯 演化过程中将使用以下 {len(eval_datasets)} 个数据集进行快速评估: {eval_datasets}")

        # 创建工厂，并传入性能上下文
        factory = FeatureSelectionFactory(
            llm_client=client, 
            output_dir=args.output,
            performance_context=performance_context,
            use_gpu_for_evaluation=args.use_gpu
        )
        
        print(f"\n开始演化 (迭代次数: {args.iterations})...")
        # 传入选定的数据集子集
        best_operator = factory.evolve_operators(
            n_iterations=args.iterations, 
            target_datasets=eval_datasets,
            verbose=True
        )
        
        if best_operator:
            print(f"\n🎉 最佳算子: {best_operator.name}, Avg Acc: {best_operator.evaluation_result.avg_accuracy:.4f}")

    finally:
        # ==========================================
        # 3. 资源清理模块 (与 try 平齐)
        # ==========================================
        # 恢复标准输出并关闭日志文件
        if log_file_handle:
            log_file_handle.close()
        sys.stdout = original_stdout
        if args.log_file:
            print(f"所有输出已保存到 {args.log_file}")
if __name__ == '__main__':
    main()
