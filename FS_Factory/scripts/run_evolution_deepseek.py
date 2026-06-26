# scripts/run_evolution_deepseek.py
import sys
import os
import argparse
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factory import FeatureSelectionFactory
from src.llm import create_client, VLLMClient

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description='Run Evolution')
    parser.add_argument('--iterations', type=int, default=3)#迭代次数
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()
    
    print("="*60)
    print(" 🚀 Feature Selection Factory - Local DeepSeek Edition")
    print("="*60)

    model_path = args.model_path
    if not model_path:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')
        config = load_config(config_path)
        model_path = config.get('llm', {}).get('vllm', {}).get('model_path')

    if args.mode == 'local':
        if not model_path:
            print("❌ 错误: 未找到模型路径！")
            return
        print(f"Loading local model from: {model_path}")
        try:
            client = create_client(
                provider='vllm',
                model_path=model_path,
                gpu_memory_utilization=0.9, # 降至 0.7 GPU利用率
                trust_remote_code=True,
                max_model_len=4096,
                max_tokens=4096,
                enforce_eager=True         # 关键参数
            )
            print("✅ 本地模型加载成功！")
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            return
    else:
        print("API 模式暂不支持")
        return

    factory = FeatureSelectionFactory(llm_client=client, output_dir=args.output)
    print(f"\n开始演化 (迭代次数: {args.iterations})...")
    best_operator = factory.evolve_operators(n_iterations=args.iterations, verbose=True)
    if best_operator:
        print(f"\n🎉 最佳算子: {best_operator.name}, Acc: {best_operator.evaluation_result.avg_accuracy:.4f}")

if __name__ == '__main__':
    main()
