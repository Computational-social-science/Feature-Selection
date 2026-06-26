# scripts/run_benchmark.py

"""
运行基准测试：评估所有演化出的方法在不同数据集上的性能。
支持增量保存和按数据集特征数量排序运行。
"""
import sys
import os
import json
import glob
import pandas as pd
import numpy as np
import time

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sandbox.datasets import DatasetManager
from src.validation.experiments import ExperimentRunner, ExperimentConfig, HAS_CUML
from src.prompts.sota_table import SOTA_METHODS

def check_gpu_status(use_gpu_flag):
    """检查 GPU 与 cuML 是否可用"""
    if not use_gpu_flag:
        return
    
    import subprocess
    try:
        # 1. 检查 nvidia-smi 是否有响应
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if smi.returncode != 0:
            print("⚠️ [警告] nvidia-smi 运行失败或未检测到驱动程序。")
            print(f"  错误信息: {smi.stderr.strip()[:100]}...")
        else:
            print("✅ GPU 驱动程序正常。")
    except Exception:
        print("⚠️ [警告] 未在系统中找到 nvidia-smi，无法确认 GPU 状态。")
    
    # 2. 检查 cuML 是否导入成功
    if not HAS_CUML:
        print("❌ [严重] 您请求使用 GPU 加速，但未检测到 cuML 库！")
        print("   系统将自动回退到 CPU 模式。")
        print("   安装建议: conda install -c rapidsai -c nvidia -c conda-forge cuml")
    else:
        print("🚀 cuML 加速库已就绪，GPU 加速已启用。")

def load_operators(operators_dir):
    """加载 operators 文件夹中的所有 JSON 方法"""
    methods = {}
    
    # 获取所有 .json 文件
    json_files = glob.glob(os.path.join(operators_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                method_name = data.get('method_name') or os.path.splitext(os.path.basename(file_path))[0]
                formula_dsl = data.get('formula_dsl')
                
                if formula_dsl:
                    methods[method_name] = formula_dsl
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return methods

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='全面特征选择基准测试')
    parser.add_argument('--operators_dir', type=str, default='./outputs/operators', help='方法 JSON 所在的目录')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='数据集所在的目录')
    parser.add_argument('--output_dir', type=str, default='./outputs/benchmark_results', help='结果保存目录')
    parser.add_argument('--repeats', type=int, default=1, help='每个实验的重复次数')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用 GPU (cuML) 加速分类器评估')
    parser.add_argument('--n_jobs', type=int, default=-1, help='特征选择评估的并行进程数')
    parser.add_argument('--n_features', type=int, default=20, help='选择特征的数量')
    
    args = parser.parse_args()
    
    # 0. 检查 GPU 状态
    check_gpu_status(args.use_gpu)
    
    # 1. 准备目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载所有方法
    print(f"正在从 {args.operators_dir} 加载方法...")
    methods = load_operators(args.operators_dir)
    print(f"加载了 {len(methods)} 个方法: {list(methods.keys())}")
    
    # 3. 加载所有数据集并按特征数量排序
    print(f"正在从 {args.data_dir} 准备数据集...")
    dm = DatasetManager(raw_dir=args.data_dir)
    
    dataset_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    datasets_metadata = []
    
    for ds_path in dataset_files:
        ds_name = os.path.basename(ds_path)
        try:
            ds = dm.get_dataset(ds_name)
            datasets_metadata.append({
                'name': ds_name,
                'X': ds.X,
                'y': ds.y,
                'n_features': ds.X.shape[1],
                'n_samples': ds.X.shape[0]
            })
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {e}")
            
    # 按特征数量升序排序
    datasets_metadata.sort(key=lambda x: x['n_features'])
    
    print(f"准备了 {len(datasets_metadata)} 个数据集 (已按特征数排序):")
    for d in datasets_metadata:
        print(f"  - {d['name']}: {d['n_features']} features, {d['n_samples']} samples")
    
    if not methods or not datasets_metadata:
        print("未找到方法或数据集，退出。")
        return

    # 4. 配置实验
    config = ExperimentConfig(
        n_repeats=args.repeats, 
        n_folds=5, 
        n_features_to_select=args.n_features,
        use_gpu=args.use_gpu,
        n_jobs=args.n_jobs
    )
    runner = ExperimentRunner(config=config)
    classifiers = ['bayes', 'rf', 'svm', 'knn']
    
    # 5. 逐个数据集运行并实时保存
    print("\n开始运行基准测试...")
    
    for ds_info in datasets_metadata:
        ds_name = ds_info['name']
        X, y = ds_info['X'], ds_info['y']
        
        # 运行单个数据集
        dataset_start_time = time.time()
        dataset_results = runner.run_single_dataset(
            dataset_name=ds_name,
            X=X,
            y=y,
            methods=methods,
            classifiers=classifiers,
            verbose=True
        )
        dataset_duration = time.time() - dataset_start_time
        print(f"数据集 {ds_name} 运行完成，耗时: {dataset_duration:.2f}s")
        
        # 实时保存结果
        for res in dataset_results:
            method_name = res['method']
            safe_name = "".join([c if c.isalnum() else "_" for c in method_name])
            save_path = os.path.join(args.output_dir, f"{safe_name}_results.csv")
            
            # 构建当前结果的 DataFrame
            res_df = pd.DataFrame([res])
            
            # 选择输出列
            output_cols = ['dataset'] + [f'{c}_accuracy' for c in classifiers] + ['selection_time']
            available_cols = [c for c in output_cols if c in res_df.columns]
            res_df = res_df[available_cols]
            
            # 如果文件已存在，则追加；否则创建并写入表头
            if os.path.exists(save_path):
                # 检查是否已经存在该数据集的记录（防止重复运行导致的重复写入）
                existing_df = pd.read_csv(save_path)
                if ds_name in existing_df['dataset'].values:
                    # 如果已存在，更新该行
                    existing_df.loc[existing_df['dataset'] == ds_name, available_cols] = res_df.values
                    existing_df.to_csv(save_path, index=False)
                else:
                    # 如果不存在，追加
                    res_df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                res_df.to_csv(save_path, index=False)

    print("\n" + "="*60)
    print("所有基准测试任务已完成！")
    print(f"结果文件夹: {args.output_dir}")

if __name__ == '__main__':
    main()
