命令	用途
python scripts/run_evolution.py --iterations 5	运行算子演化
python scripts/run_evolution.py --use-openai --model gpt-4	使用 GPT-4 演化
python scripts/run_benchmark.py	运行基准测试
python scripts/run_comparison.py	人类 vs Agent 对比
这个完整的项目框架提供了：

模块化设计: DSL、Prompt、Sandbox、Router、Validation 各司其职
可扩展性: 易于添加新算子、新方法、新 LLM 后端
完整工作流: 从算子演化到评估对比的完整流程
生产就绪: 包含错误处理、日志、持久化等


完整的命令行运行示例
# 设置环境变量
export DEEPSEEK_API_KEY="sk-your-api-key"

# 使用 DeepSeek Math 模型运行演化
python scripts/run_evolution_deepseek.py --model deepseek-math --iterations 5

# 使用 DeepSeek R1 推理模型 (更深度推理)
python scripts/run_evolution_deepseek.py --model deepseek-reasoner --iterations 3

# 指定输出目录
python scripts/run_evolution_deepseek.py --output ./my_results --iterations 5


组件	说明
DeepSeekMathClient	DeepSeek Math 模型专用，适合公式生成
DeepSeekReasonerClient	DeepSeek R1 推理模型，适合深度推理任务
DeepSeekClient	通用客户端，支持所有 DeepSeek 模型
create_client('deepseek-math')	工厂函数创建客户端
CostTracker	成本监控工具