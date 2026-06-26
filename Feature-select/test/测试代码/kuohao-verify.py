# import os
#
#
# def read_and_process_files(directory):
#     # 新的保存文件夹路径
#     newfilename = r'.\feature_select_result'
#     os.makedirs(newfilename, exist_ok=True)  # 确保输出目录存在
#
#     for file_name in os.listdir(directory):
#         file_path = os.path.join(directory, file_name)
#         if os.path.isfile(file_path):
#             with open(file_path, 'r') as f:
#                 content = f.read().strip()
#
#                 # 去除 'class' 字符串
#                 content = content.replace('\'', '')
#                 content = content.replace('class', '')
#
#
#                 print(content)
#
#                 # 处理内容
#                 lines = content.split('\n')
#                 result = []
#                 for line in lines:
#                     if line:
#                         numbers = list(map(int, line.strip().split(', ')))
#                         result.append(numbers)
#
#                 # 将其加上方括号和逗号并保存
#                 formatted_result = ['[' + ', '.join(map(str, row)) + ']' for row in result]
#                 formatted_content = '\n'.join(formatted_result)
#
#                 # 新的文件路径
#                 new_file_path = os.path.join(newfilename, file_name)
#
#                 with open(new_file_path, 'w') as f:
#                     f.write(formatted_content)
#     print(f"Processed and saved file {file_name}.")
#
# # 调用函数并传入需要处理的目录路径
# read_and_process_files(r'C:\path\to\your\input\directory')
#
#
#
# # 指定文件夹路径
# directory = r'C:\Users\rzdin\Desktop\Feature\feature_select_result\CFS-BF/'
# read_and_process_files(directory)
import os
import ast

def read_and_process_files(directory):
    # newfilename = r'D:\111study\Feature\feature_select_result\CFS-GEN'
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                content = f.read().strip()
                content = content.replace('\'', '')
                content = content.replace('class', '')
                # content = content.replace('[[', '[')
                # content = content.replace(']]', ']')
                # content = content.replace('], [', ']\n[')
                # content = content.replace('\n', ',')
                with open(file_path, 'w') as f:
                        f.write(str(content))
                # print(content)
                # 检查内容是否包含方括号
                if '[' in content or ']' in content:
                    print(f"Skipping file {file_name} as it contains brackets.")
                    continue
                else:
                    # 读取所有数字
                    lines = content.split('\n')
                    result = []
                    for line in lines:
                        if line:
                            numbers = list(map(int, line.strip().split(' ')))
                            result.append(numbers)



                    #将其加上方括/号和逗号并保存/
                    formatted_result = ['[' + ', '.join(map(str, row)) + ']' for row in result]

                    with open(file_path, 'w') as f:
                            f.write(str(formatted_result))
                # print(content)


                print(f"Processed and saved file {file_name}.")

# 指定文件夹路径
directorys = ['CFS-BF',]#'CFS-GEN',
for a in directorys:
    directory = '../feature_select_result/' + a
    read_and_process_files(directory)


import os
import ast

# def process_file(file_path):
#     # 读取文件内容
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # 检查文件内容是否符合要求
#     if not all(line.strip().startswith('[') for line in lines):
#         print(f"Skipping file {file_path}: Not in expected format")
#         return
#
#     # 处理每行数据
#     data = []
#     for line in lines:
#         try:
#             data_list = ast.literal_eval(line.strip())
#             data.append(data_list)
#         except (ValueError, SyntaxError):
#             print(f"Skipping invalid line in {file_path}: {line.strip()}")
#
#     # 检查是否有两个一维数组
#     if len(data) != 2 or not all(isinstance(i, list) for i in data):
#         print(f"Skipping file {file_path}: Expected two 1D arrays")
#         return
#
#     # 拆分为两个一维数组
#     array1, array2 = data
#
#     # 写入文件
#     with open(file_path, 'w') as file:
#         file.write(','.join(map(str, array1)) + '\n')
#         file.write(','.join(map(str, array2)) + '\n')
#
#     print(f"Processed file {file_path}")
#
# def process_files_in_folder(folder_path):
#     # 遍历文件夹中的文件
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         # 检查文件是否是文本文件
#         if os.path.isfile(file_path) and file_path.endswith('.txt'):
#             process_file(file_path)
#
# # 指定要处理的文件夹路径
# folder_path = r'D:\111study\Feature\feature_select_result\CON-GEN'
# process_files_in_folder(folder_path)
