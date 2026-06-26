import os
import json
from collections import OrderedDict


    # 定义文件夹路径
folder_path = r'D:/111/Feature/final_result'

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 构建文件的完整路径
    old_file_path = os.path.join(folder_path, file_name)
    # 读取文件
    with open(old_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ⚠️ 如果 key 是数字字符串，先转成 int 排序
    ordered = OrderedDict(sorted(data.items(), key=lambda x: int(x[0])))

    # 保存回文件（美化格式）
    with open(old_file_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=4)
    # # 检查是否为文件
    # if os.path.isfile(old_file_path):
    #     # 替换文件名中的 "CSF" 为 "CFS"
    #     new_file_name = file_name.replace('CSF', 'CFS')
    #
    #     # 构建新的文件完整路径
    #     new_file_path = os.path.join(folder_path, new_file_name)
    #
    #     # 重命名文件
    #     os.rename(old_file_path, new_file_path)
    #
    #     print(f"Renamed: {file_name} -> {new_file_name}")

print("All files processed.")

import json
from collections import OrderedDict


# bpredict_labels_squeezed = np.squeeze(predict_lable['b'], axis=1)
# kpredict_labels_squeezed = np.squeeze(predict_lable['k'], axis=1)
# spredict_labels_squeezed = np.squeeze(predict_lable['s'], axis=1)
# rpredict_labels_squeezed = np.squeeze(predict_lable['r'], axis=1)
# bpredict_probabity_squeezed = np.squeeze(predict_lable['b'], axis=1)
# kpredict_probabity_squeezed = np.squeeze(predict_lable['k'], axis=1)
# spredict_probabity_squeezed = np.squeeze(predict_lable['s'], axis=1)
# rpredict_probabity_squeezed = np.squeeze(predict_lable['r'], axis=1)