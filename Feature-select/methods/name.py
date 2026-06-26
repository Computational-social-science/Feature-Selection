# import os
#
# # 指定目标文件夹路径
# # folds = ["Fast-IAMB","IAMB","Inter-IAMB","GSMB","FBED"]
# # for fold in folds:
# folder_path = r"E:\Feature\FeatureSelect\EAMB"  # 请替换为实际路径
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 生成文件的完整路径
#     file_path = os.path.join(folder_path, filename)
#
#     # 只处理文件而不是子文件夹
#     if os.path.isfile(file_path):
#         # print(file_path)
#         new_filename = filename
#         # 去除开头的 '0'
#         if new_filename.startswith('0'):
#             new_filename = new_filename[1:]
#         # 去除 'result-' 后的 '-'
#         if '_result' in new_filename:
#             print(new_filename,1111111111)
#             result_index = new_filename.index('_result')
#             # 检查 'result-' 后是否有 '-'，如果有则去掉它
#             print(new_filename[result_index-1])
#             if new_filename[result_index-1] in ['0','1','2','3','4','5','6','7','8','9']:
#                 # print(new_filename[:result_index-1] + '_result'+new_filename[result_index-1]+new_filename[-4:])
#                 new_filename = new_filename[:result_index-1] + '_result'+new_filename[result_index-1]+new_filename[-4:]
#         # 获取新的文件完整路径
#         new_file_path = os.path.join(folder_path, new_filename)
#         # 重命名文件
#         if new_filename != filename:
#             os.rename(file_path, new_file_path)
#             print(f"重命名: {filename} -> {new_filename}")
# print("所有文件重命名完成！")











# import json
#
# result_dict_fixed = {0: [617, 71], 1: [617, 458], 2: [617, 388], 3: [617, 470], 4: [617, 466], 5: [617, 415], 6: [617, 362], 7: [617, 418], 8: [617, 9], 9: [617, 426], 10: [395, 617], 11: [617, 215], 12: [617, 435], 13: [617, 371], 14: [617, 549], 15: [617, 394], 16: [617, 392], 17: [471, 617], 18: [617, 412], 19: [617, 360], 20: [617, 332], 21: [617, 581], 22: [617, 325], 23: [617, 418], 24: [617, 20], 25: [617, 580]}
# file_name = r"./feature_select_result/EAMB" + '/' + 'Isolet' + "_sp_result9" + ".txt"
# with open(file_name, 'w') as f:
#     json.dump(result_dict_fixed, f, indent=4)
#



import json
import os
from collections import OrderedDict
path = r"E:/Feature/feature_select_result/EAMB11/feature_select_result/EAMB11/"
for filename in os.listdir(path):
    # 1. 读取文件内容
    # 假设你已经把 txt 文件内容读成了字符串变量 data_str
    with open(path+filename, "r") as f:
        data_str = f.read()

    # 2. 解析 JSON 为字典
    data = json.loads(data_str)

    # 3. 按键的数值顺序排序（键是字符串，需要转成 int 排序）
    sorted_data = OrderedDict(sorted(data.items(), key=lambda item: int(item[0])))

    # 4. 将排序结果写回原文件（覆盖写入）
    with open(path+filename, "w") as f:
        json.dump(sorted_data, f, indent=4)

    print("排序并保存成功！")
