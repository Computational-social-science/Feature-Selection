import json
import os
import re
from collections import OrderedDict
from pprint import pprint
#
#
# def process_json_files_individually(input_folder, output_folder):
#     """
#     独立处理每个JSON文件并单独保存
#
#     参数:
#         input_folder: 原始JSON文件所在文件夹路径
#         output_folder: 处理后的JSON文件保存路径
#     """
#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 遍历输入文件夹中的所有JSON文件
#     for filename in os.listdir(input_folder):
#         if not filename.endswith('.json'):
#             continue
#
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)  # 保持相同文件名
#
#         try:
#
#             # 读取JSON文件（保持原始顺序）
#             with open(input_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f, object_pairs_hook=OrderedDict)
#
#             # 获取所有唯一键（保持顺序）
#             all_keys = list(data.keys())
#
#             # 验证索引范围
#             max_index = max(max(indices) for indices in data.values() if indices)
#             if max_index >= len(all_keys):
#                 raise ValueError(f"最大索引 {max_index} 超出键数量 {len(all_keys)}")
#
#             # 处理每个键值对
#             processed_data = OrderedDict()
#             for key, indices in data.items():
#                 mapped_keys = [all_keys[idx] for idx in indices]
#                 processed_data[key] = mapped_keys
#                 print(f"转换: {key}: {indices} -> {mapped_keys}")
#
#             # 保存处理后的数据（保持相同文件名）
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 json.dump(processed_data, f, indent=2, ensure_ascii=False)
#
#             print(f"✓ 成功保存到: {output_path}")
#
#         except json.JSONDecodeError:
#             print(f"× 错误: 文件 {filename} 不是有效的JSON格式")
#         except ValueError as ve:
#             print(f"× 错误: {str(ve)}")
#         except Exception as e:
#             print(f"× 处理文件 {filename} 时出错: {str(e)}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 配置路径（请修改为您的实际路径）
#     input_dir = os.path.join(os.getcwd(), "../BNresult")
#     output_dir = os.path.join(os.getcwd(), "../BNresult1")
#
#     print("=== JSON文件独立处理程序 ===")
#     print(f"输入目录: {input_dir}")
#     print(f"输出目录: {output_dir}")
#
#     # 执行处理
#     process_json_files_individually(input_dir, output_dir)
#
#     print("\n处理完成！输出文件保存在:", os.path.abspath(output_dir))
import json
from collections import defaultdict
def parse_input_file(file_path):
    """解析输入文件，返回初步的结构化数据"""
    data = defaultdict(dict)
    current_key = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检查是否是键行 (如 "BirthAsphyxia: {1: {'Disease'}, 2: {'Disease'}}")
            if ':' in line and '{' in line:
                key_part, value_part = line.split(':', 1)
                current_key = key_part.strip()

                # 使用正则表达式提取所有数字和对应的集合
                matches = re.finditer(r'(\d+):\s*\{([^}]*)\}', value_part)
                for match in matches:
                    num = int(match.group(1))
                    items = {x.strip("' ") for x in match.group(2).split(',') if x.strip()}
                    data[current_key][num] = items

    return data
def merge_data(parsed_data):
    """合并数据，相同键值取并集"""
    result = {}

    for key, num_dict in parsed_data.items():
        if not num_dict:
            continue

        # 检查所有数字对应的值是否相同
        first_values = next(iter(num_dict.values()))
        all_same = all(values == first_values for values in num_dict.values())

        if all_same:
            result[key] = first_values
        else:
            # 不相同则取并集
            merged = set()
            for values in num_dict.values():
                merged.update(values)
            result[key] = merged

    return result

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.readwrite import BIFReader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def load_model(bif_path):
    reader = BIFReader(bif_path)
    return reader.get_model()

def get_model_mbs(model):
    mb_dict = {}
    for node in model.nodes():
        mb_dict[node] = set(model.get_markov_blanket(node))
    return mb_dict


def calculate_accuracy_metrics(correct_arr, test_arr):
    """
    计算B数组相对于A数组的准确度指标
    A数组为正确数组，B数组为测试数组
    """
    # 转换为集合便于比较
    correct_set = set(correct_arr)
    test_set = set(test_arr)

    # 计算各种指标
    true_positives = len(correct_set & test_set)  # 正确匹配的元素
    false_positives = len(test_set - correct_set)  # B中有但A中没有的（误报）
    false_negatives = len(correct_set - test_set)  # A中有但B中没有的（漏报）

    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(correct_set) if len(correct_set) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }
def compute_mb_scores(true_mbs: list, pred_mbs: list, all_nodes: list):
    precision_list, recall_list, f1_list = [], [], []

    for node in true_mbs:
        true_set = true_mbs[node]
        pred_set = pred_mbs.get(node, set())

        all_other_nodes = set(all_nodes) - {node}
        y_true = [1 if x in true_set else 0 for x in all_other_nodes]
        y_pred = [1 if x in pred_set else 0 for x in all_other_nodes]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # print(f"{node}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    print("\n--- 平均结果 ---")
    print(f"Avg Precision: {np.mean(precision_list):.3f}")
    print(f"Avg Recall   : {np.mean(recall_list):.3f}")
    print(f"Avg F1 Score : {np.mean(f1_list):.3f}")
    return np.mean(precision_list),np.mean(recall_list),np.mean(f1_list)
# 示例用法
if __name__ == '__main__':
    # 加载 BIF 模型
    # model = load_model('alarm.bif')"pigs"
    datanames = ["child1000", "alarm1000", "hepar21000", "insurance1000", "child5000", "alarm5000", "hepar25000",
                 "insurance5000", ]
    datasets = ["hepar2", "alarm", "insurance", "child"]  # ,
    methods = ['FSMB', 'GSMB', 'FBED', 'LRH', 'IAMB', "EAMB", "Fast_IAMB", "Inter_IAMB",'EAMB_sp' ]  #
    base_url = r"./dataset/bnnet/"
    r = pd.DataFrame(index=range(len(datasets) * 2), columns=range(len(methods)))
    r.columns = methods
    r.index = datanames
    p = pd.DataFrame(index=range(len(datasets) * 2), columns=range(len(methods)))
    p.columns = methods
    p.index = datanames
    s = pd.DataFrame(index=range(len(datasets) * 2), columns=range(len(methods)))
    s.columns = methods
    s.index = datanames
    a = pd.DataFrame(index=range(len(datasets) * 2), columns=range(len(methods)))
    a.columns = methods
    a.index = datanames
    for dataname in datasets:
        print(dataname)
        reader = BIFReader(base_url + dataname + '.bif')
        model = reader.get_model()
        model1 = load_model(base_url + dataname + '.bif')
        # 提取所有节点的 Markov Blanket 子集
        true_mb_dict = {node: list(model.get_markov_blanket(node)) for node in model.nodes()}
        # 找到MB最大的节点
        max_mb_node = None
        max_mb_size = -1

        for node, mb_nodes in true_mb_dict.items():
            mb_size = len(mb_nodes)
            if mb_size > max_mb_size:
                max_mb_size = mb_size
                max_mb_node = node

        print(f"MB最大的节点是: {max_mb_node}")
        print(f"它的MB大小为: {max_mb_size}")
        print(f"它的MB包含的节点: {true_mb_dict[max_mb_node]}")
        for method in methods:
            print(method + "1000")
            input_path = "./BNresult1/" + method + "_" + dataname + '1000.json'

                # 获得模型预测的 MB 子集
            if method == "EAMB_sp":
                parsed_data = parse_input_file(input_path)
                data = merge_data(parsed_data)
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f, object_pairs_hook=OrderedDict)
            pred_mb_dict = data[max_mb_node]
            result = calculate_accuracy_metrics(true_mb_dict[max_mb_node], pred_mb_dict)
            p[method][dataname + '1000'] = result['precision']
            r[method][dataname + '1000'] = result['recall']
            s[method][dataname + '1000'] = result['f1_score']
            a[method][dataname + '1000'] = result['accuracy']
            print("---------")
            print(method + "5000")
            input_path = "./BNresult1/" + method + "_" + dataname + '5000.json'
            # 获得模型预测的 MB 子集
            if method == "EAMB_sp":
                parsed_data = parse_input_file(input_path)
                data2 = merge_data(parsed_data)
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data2 = json.load(f, object_pairs_hook=OrderedDict)
            pred_mb_dict2 = data2[max_mb_node]
            result2 = calculate_accuracy_metrics(true_mb_dict[max_mb_node], pred_mb_dict2)
            p[method][dataname + '5000'] = result2['precision']
            r[method][dataname + '5000'] = result2['recall']
            s[method][dataname + '5000'] = result2['f1_score']
            a[method][dataname + '5000'] = result2['accuracy']
            p.to_excel('tables/' + 'bntestp.xlsx', index=True)
            r.to_excel('tables/' + 'bntestr.xlsx', index=True)
            s.to_excel('tables/' + 'bntests.xlsx', index=True)
            a.to_excel('tables/' + 'bntesta.xlsx', index=True)
