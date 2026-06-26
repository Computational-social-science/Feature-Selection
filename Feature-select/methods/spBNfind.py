from pgmpy.readwrite import BIFReader
import json
from joblib import Parallel, delayed
# single_thread_bifreader.py
import os
from pgmpy.readwrite import BIFReader

# 禁用 joblib 多进程，确保 Windows 下不会报 _winapi 错误
os.environ["JOBLIB_MULTIPROCESSING"] = "0"


# 1. 读取 BIF 文件
# 指定你的 BIF 文件路径
bif_file = "D:/111/Feature/dataset/bnnet"
datasets = ["alarm"]
for dataset in datasets:
    file = bif_file + "/" + dataset + ".bif"
    # 初始化 BIFReader
    reader = BIFReader(file)
    cpt = reader.get_values()
    file_path = "output.txt"  # 可以改为你想要的路径和文件名
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(cpt))
    parents = reader.get_parents()
    states = reader.get_states()
    print(parents,'\n',states)

    # model = reader.get_model()
    #
    # # 2. 获取所有节点及其取值空间
    # states_dict = {node: model.get_cpds(node).state_names[node] for node in model.nodes()}
    #
    # # 3. 获取马尔可夫毯（MB）
    # mb_dict = {}
    # for node in model.nodes():
    #     mb_dict[node] = list(model.get_markov_blanket(node))
    #
    # # 4. 进一步细分：针对节点每个取值，找到条件化的 MB 子集
    # #    这里的 S_a 是一种 context-specific Markov Blanket (CSI)
    # S_a = {}
    # for node, states in states_dict.items():
    #     S_a[node] = {}
    #     for state in states:
    #         # 初始候选是 MB
    #         candidates = mb_dict[node]
    #         subset = []
    #         for y in candidates:
    #             # 检查在 node=state 时，y 是否仍影响 node 或其 children
    #             # 方法：对比 node 的 CPT 在 node=state 时是否依赖 y
    #             cpd = model.get_cpds(node)
    #             relevant_vars = set(cpd.variables) - {node}
    #             if y in relevant_vars:
    #                 subset.append(y)
    #             else:
    #                 # 再看子节点的 CPT 是否依赖 y
    #                 children = model.get_children(node)
    #                 dependent = False
    #                 for child in children:
    #                     cpd_child = model.get_cpds(child)
    #                     if y in (set(cpd_child.variables) - {child}):
    #                         dependent = True
    #                         break
    #                 if dependent:
    #                     subset.append(y)
    #         S_a[node][state] = subset
    #
    # # 5. 保存为 JSON
    # with open(dataset+"_spresult.json", "w") as f:
    #     json.dump(S_a, f, indent=4)
