import os
import re
import json
import pandas as pd
import numpy as np

methods = ['EAMB-sp']

datasets = [
    'Arcene','Authorship','CLL_SUB_111','Colon','Dermatology','dna',
    'Factors','GLIOMA','madelon','Movement_libras','Musk1','Orlraws10P',
    'Pixel','Prostate_GE','Sorlie','Su','SRBCT','spambase','splice',
    'Synthetic_control','TOX_171','Waveform','Wdbc','WarpPIE10P',
    'WarpAR10P','Yeoh'
]

BASE_DIR = "FeatureSelect"
OUTPUT_FILE = "num11.xlsx"

# 匹配所有数字（整数/小数/科学计数法）
number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

result_matrix = pd.DataFrame(index=datasets, columns=methods)

# =====================================
# 主统计逻辑
# =====================================
for method in methods:
    method_path = os.path.join(BASE_DIR, method)

    for dataset in datasets:
        fold_feature_nums = []

        for fold in range(10):
            filename = f"{dataset}_sp_result{fold}.txt"
            file_path = os.path.join(method_path, filename)

            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    continue

                # =====================================
                # ① 尝试按 JSON 解析
                # =====================================
                try:
                    data = json.loads(content)

                    if isinstance(data, dict):
                        class_counts = []

                        for key in data:
                            if isinstance(data[key], list):
                                class_counts.append(len(data[key]))

                        if class_counts:
                            file_feature_num = np.mean(class_counts)
                        else:
                            file_feature_num = np.nan
                    else:
                        file_feature_num = np.nan

                # =====================================
                # ② 普通文本情况
                # =====================================
                except:
                    numbers = number_pattern.findall(content)
                    file_feature_num = len(numbers)

                if not np.isnan(file_feature_num):
                    fold_feature_nums.append(file_feature_num)

            except Exception as e:
                print(f"读取失败: {file_path}, 错误: {e}")

        # 对 10-fold 求平均
        if fold_feature_nums:
            dataset_avg = np.mean(fold_feature_nums)
            result_matrix.loc[dataset, method] = round(dataset_avg, 4)
        else:
            result_matrix.loc[dataset, method] = np.nan

# =====================================
# 保存 Excel
# =====================================
result_matrix.to_excel(OUTPUT_FILE)

print("统计完成，结果已保存为:", OUTPUT_FILE)
