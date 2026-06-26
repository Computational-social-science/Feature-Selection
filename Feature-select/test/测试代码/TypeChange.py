# import string
# from scipy.io import arff
# import numpy as np
# from scipy.io import savemat
# import scipy.io as scio
# import re
# from rpy2.robjects import r
# import pandas as pd
# from rpy2.robjects import pandas2ri
# import rpy2.robjects as ro
# # from sklearn.preprocessing import LabelEncoder
# #
# # datanames = ['Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene','Dermatology','Waveform','Wdbc','Synthetic_control','Authorship','Factors','Pixel','Isolet','ORL','WarpPIE10P','Su','Prostate_GE','TOX_171','Orlraws10P','CLL_SUB_111','Yeoh']
#
#
# # datanames = ['Waveform','Factors','Synthetic_control','Authorship','Pixel']#,
# from sklearn.preprocessing import LabelEncoder
#
# # datanames = ['Colon','SRBCT']
# # base_url = r"./dataset/testdatasets/"
# # for dataname in datanames:
# #     data_url = dataname +'.arff'
# #     url = base_url + data_url # 读取数据文件
#     # url = base_url + data_url  # 读取数据文件
#     # data = scio.loadmat('./dataset/' + dataname + '.mat')
#     # print(data)
#     # X1 = pd.DataFrame(data['X'])
#     # y1 = pd.DataFrame(data['Y'])
#     # # 将数据转换回 DataFrame
#     # X1 = pd.DataFrame(data['X'])
#     # y1 = pd.DataFrame(data['Y'], columns=['class'])
#     # label_encoder = LabelEncoder()
#     # y_encoded = label_encoder.fit_transform(X1[33])
#     # pd.set_option('display.max_rows', None)
#     # pd.set_option('display.max_columns', None)
#     # print(dataname,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # print("Loaded DataFrame:")
#     # print(X1)
#     # print(y1)
#     # print(X1[33])
#     # s_tuples = X1[33].apply(tuple)
#     # print(s_tuples.unique())
#     # import pandas as pd
#     # 将数组转换为基本类型
#     # def extract_value(x):
#     #     print(x)
#     #     print(type(x))
#     #     if isinstance(x, (int, float, complex)):
#     #         return x
#     #     else:
#     #         return extract_value(x[0])
#     # def has_decimal(number):
#     #     # 将数字转换为字符串
#     #     number_str = str(number)
#     #     # 检查字符串中是否包含小数点
#     #     return '.' in number_str
#     # def changetosinge(x):
#     #     if bool(re.match(r'^\d+$', x)):
#     #         if has_decimal(x):
#     #             return float(x)
#     #         else:
#     #             return int(x)
#     #     else:
#     #         return x
#     # def changetosinge(x):
#     #     return float(x)
#     # # 应用函数到 DataFrame 的每个元素
#     # X1 = X1.applymap(changetosinge)
#     # y1 = y1.applymap(changetosinge)
#     # print("result===========")
#     # print(X1.head())
#     # print(y1.head())
# #---------------------------------------------r语言相关转化----------------
# # r['load'](url)
# # r_data_frame = r['sorlie']
# # with (ro.default_converter + pandas2ri.converter).context():
# #   pd_from_r_df = ro.conversion.get_conversion().rpy2py(r_data_frame)
# # # 将R中的数据框转换为Python中的DataFrame
# # data = pd_from_r_df
# # X = data['x']
# # columns = range(0,len(X[0]))
# # X = pd.DataFrame(X, columns=columns)
# # y = data['y']
# # y = np.ravel(y)
# # Class = set(y)
# # y = pd.DataFrame(y)
# # X = pd.DataFrame()
# # data_url = '/'+dataname+'.data'
# #---------------------------------------------r语言相关转化----------------
# # # data_url = r'\musk1.data'
#  # # dataname = "Musk1"
#  # # data_url = r'\sorlie.Rdata'
#  # # dataname = "Sorlie"
#  # url = base_url + data_url # 读取数据文件
#  # data = pd.read_csv(url, header=None)
#  # print(data)
#  # new_columns = data[0].str.split(expand=True)
#  # # 将新列添加到原始 DataFrame 中
#  # data[new_columns.columns] = new_columns
#  # # 删除原始的文本列
#  # data.drop(columns=[0], inplace=True)
#  # data.loc[:99, 'label'] = 1
#  # data.loc[100:199, 'label'] = 2
#  # data.loc[200:299, 'label'] = 3
#  # data.loc[300:399, 'label'] = 4
#  # data.loc[400:499, 'label'] = 5
#  # data.loc[500:599, 'label'] = 6
#  # X0 = data.iloc[:,:-1]
#  # # X0 = data.iloc[:,2:-1]
#  # y = data.iloc[:,-1]
#  # def unpack_element(x):
#  #     if isinstance(x, (list, np.ndarray)):
#  #         return x[0]
#  #     return x
#  #
#  # data = data.applymap(unpack_element)
#  # X = data.iloc[:,2:]
#  # X = data.iloc[:,2:-1]
#  # y = data.iloc[:,1]
#  # label_encoder = LabelEncoder()
#  # y_encoded = label_encoder.fit_transform(y)
#  # y=pd.DataFrame(y)
#  # pd.set_option('display.max_rows', None)
#  # pd.set_option('display.max_columns', None)
#  # print(len(set(y)))
#  # 保存DataFrame为MAT文件
#     # print(X.to_numpy())
#     # 分离特征和标签
#     # X0 = pd.DataFrame(data.iloc[:, :-1]) # 特征数据
#     # y = pd.DataFrame(data.iloc[:, -1])   # 标签数据
#     # print(X0,y)
#
#
#
# #-----------------------------arff转化为mat------------------------------------------------
# import pandas as pd
# datanames = ['Colon','SRBCT']
# base_url = r"../dataset/testdatasets/"
# for dataname in datanames:
#     data_url = base_url+dataname +'.arff'
#     # data = read_arrf(data_url)
#     data = arff.loadarff(data_url)
#     df = pd.DataFrame(data[0])
#     label_encoder = LabelEncoder()
#     if dataname == 'Colon':
#         y = label_encoder.fit_transform(df["class"])
#     else:
#         y = label_encoder.fit_transform(df["CLASS"])
#     new_columns = list(range(df.shape[1]))
#     df = df.rename(columns=dict(zip(df.columns, new_columns)))
#     X = df.iloc[:, :-1]
#     y = pd.DataFrame(y)
#     # print(df.iloc[:, :-1])
#     savemat('../dataset/testdatasets/'+dataname+'.mat', {'X': X.values.reshape(X.shape[0], X.shape[1]), 'Y': y.values.reshape(-1, 1)})
# # -----------------------------arff转化为mat------------------------------------------------
#     # dataname = "GLIOMA"
# # print(type(data))
# # print("----------------------------------------------------------------------------------------")
# # 定义一个函数来去除括号并转换为数值
# #     data1 = data.applymap(lambda x: x[0][0] if isinstance(x, list) else x)
# #     print(df)
# # dataname = "GLIOMA"
# # url = base_url + data_url # 读取数据文件
# # data = scio.loadmat(data_url)
# # # X = data.iloc[:,2:]
# # X0 = data.iloc[:-3,:-1]
# # y = data.iloc[:-3:,-1]
# # columns = range(0,X0.shape[1])
# # # print(X0)
# # X0.columns = columns
# # y = pd.DataFrame(y)
# # # print(X0,y)
# # # 保存为 .mat 文件
# # savemat('./dataset/GLIOMA.mat', {'X': X0.values, 'Y': y.values.reshape(-1, 1)})
# # from scipy.io import savemat
#
# # print(X0.values)
#
# # # train_index=[1,2,3,4,5]
# # # test_index=[11,23,5,23,12,3]
# # # X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
# # # y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
# # #
# # # time = 3
# # # bnacc,knacc,rnacc,snacc = othertest(time, [1,2,3], X_test, y_test, X_train, y_train)
# # print(X_train, X_test)
# # print("=========================================================================================result")
# # print(X1,y1)
# # # print(len(set(y[0])))
# #
# #
# # import pandas as pd
# # import numpy as np
# # from scipy.io import savemat
# #
# # # 示例 DataFrame
# # data = {
# #     0: [2, 3, 2, 2, 2],
# #     1: [2, 3, 1, 2, 3],
# #     2: [0, 3, 2, 2, 2],
# #     3: [3, 2, 3, 0, 2],
# #     4: [0, 1, 1, 0, 2],
# #     33: [55, 8, 26, 40, 45],
# #     'class': [2, 1, 3, 1, 3]
# # }
# #
# # df = pd.DataFrame(data)
# #
# # # 将 DataFrame 的值保存为 .mat 文件
# # X0 = df.iloc[:, :-1]  # 特征数据
# # y = df.iloc[:, -1]    # 标签数据
# #
# # # 保存为 .mat 文件
# # savemat('./dataset/111.mat', {'X': X0.values, 'Y': y.values.reshape(-1, 1)})
# # from scipy.io import loadmat
# #
# # # 读取 .mat 文件
# # data = loadmat('./dataset/111.mat')
# #
# # # 打印读取的数据结构
# # print("Loaded .mat file keys:")
# # print(data.keys())
# #
# # # 将数据转换回 DataFrame
# # X1 = pd.DataFrame(data['X'])
# # y1 = pd.DataFrame(data['Y'], columns=['class'])
# #
# # print("Loaded DataFrame:")
# # print(X1.head())
# # print(y1.head())
# from sklearn.preprocessing import LabelEncoder
# # # import pandas as pd
# # from weka.core.converters import Loader,Saver
# # from weka.core.dataset import Instances
# # import weka.core.jvm as jvm
# # from weka.filters import AttributeSelection,ASEvaluation,ASSearch
# # import scipy.io as scio
# # from sklearn.model_selection import KFold
#
# import numpy as np
# # datanames = ['Waveform','Factors','Synthetic_control','Authorship','Pixel','Wdbc','Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene','Isolet','ORL','WarpPIE10P','Su','Prostate_GE','TOX_171','Orlraws10P','CLL_SUB_111','Yeoh','Pixel']#
# # datanames = ['Dermatology',]#
# # base_url = r"./dataset/"
# # for dataname in datanames:
# #     print(dataname)
# #     data_url = dataname + '.mat'
# #     url = base_url + data_url #数据路径
# #     data0 = scio.loadmat(url)# 读取数据文件
# #     X0 = pd.DataFrame(data0['X'])#读取训练数据
# #     y0 = pd.DataFrame(data0['Y'])#读取标签
# #     Special = X0.iloc[:, -1]
# #     # print(Special.values)
# #     X0 = X0.iloc[:, :-1]  # 读取训练数据
# #     a = np.array([item[0] for item in Special])
# #     label_encoder = LabelEncoder()
# #     a33 = label_encoder.fit_transform(a)
# #     X0[33] = a33
# #     # label_encoder = LabelEncoder()
# #     # y_encoded = label_encoder.fit_transform(X0[33])
# #     # data = {33:y_encoded}
# #     # newf = pd.DataFrame(data)
# #     # X0[33]=newf[33]
# #     def changetosinge(x):
# #         return float(x)
# #     # 应用函数到 DataFrame 的每个元素
# #     X0 = X0.applymap(changetosinge)
# #     y0 = y0.applymap(changetosinge)
# #     X = pd.DataFrame()
# #     for col in X0.columns:
# #         X[col] = pd.cut(X0[col], bins=5, labels=False)
# #     # 使用rename函数重新命名列名，将x列名控制在0-n
# #     new_columns = list(range(X.shape[1]+1))
# #     X = X.rename(columns=dict(zip(X.columns, new_columns)))
# #     newdata = X
# #     newdata['class'] = y0
# #     # print(newdata)
# #     #==========================================================读取文件
# #     kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
# #     for fold, (train_index, test_index) in enumerate(kfold.split(X)):  # fold赋值之后就是0-9
# #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
# #         # y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
# #         # result = []  # 特征提取结果数组
# #         # print(X_train)
# #     # print(X0,y0)
# #         newfile = base_url +'arff/' + dataname + str(fold) + '.arff'
# #         # 将数据保存为ARFF文件
# #         with open(newfile, 'w') as f:
# #             # 写入ARFF文件头部信息
# #             f.write('@relation MyDataset\n\n')
# #
# #             # 写入属性信息
# #             for column in newdata.columns:
# #                 f.write('@attribute ' + str(column) + ' numeric\n')
# #             f.write('\n@data\n')
# #
# #             # 写入数据
# #             for index, row in newdata.iterrows():
# #                 f.write(','.join(map(str, row.values)) + '\n')
# #         #==============================================================将mat转化为arff
import scipy.io as sio
import pandas as pd
import numpy as np
import os
from pathlib import Path


def mat_to_csv(mat_file_path, csv_file_path=None, data_key=None):
    """
    将单个 MATLAB .mat 文件转换为 CSV 格式，列名格式：0,1,2...target

    参数:
        mat_file_path: str, .mat 文件的路径（必填）
        csv_file_path: str, 输出 CSV 文件的路径（可选，默认与 mat 文件同目录同名称）
        data_key: str, 指定要提取的数据键名（可选，自动识别主要数据）
    返回:
        bool: 转换成功返回 True，失败返回 False
    """
    # 设置默认输出路径
    if csv_file_path is None:
        csv_file_path = os.path.splitext(mat_file_path)[0] + '.csv'

    try:
        # 1. 读取 MAT 文件
        mat_data = sio.loadmat(mat_file_path)

        # 2. 筛选有用数据（排除 MATLAB 自带的元数据）
        valid_keys = [key for key in mat_data.keys() if not key.startswith('__')]

        # 3. 确定要导出的数据
        if data_key:
            if data_key not in valid_keys:
                raise ValueError(f"指定的键 {data_key} 不存在，可用键: {valid_keys}")
            selected_key = data_key
        else:
            if len(valid_keys) == 0:
                raise ValueError("MAT 文件中未找到有效数据")
            elif len(valid_keys) > 1:
                selected_key = valid_keys[0]
            selected_key = valid_keys[0]

        # 4. 提取并处理数据
        data = mat_data[selected_key]

        # 处理不同维度的数据
        if len(data.shape) == 1:
            # 一维数据：只有1列，列名直接设为 target
            df = pd.DataFrame(data, columns=['target'])
        elif len(data.shape) == 2:
            # 二维数据：前n-1列用0,1,2...，最后1列用target
            col_count = data.shape[1]
            columns = [str(i) for i in range(col_count - 1)] + ['target']
            df = pd.DataFrame(data, columns=columns)
        else:
            # 高维数据：展平后按二维规则命名
            flattened_data = data.reshape(-1, data.shape[-1])
            col_count = flattened_data.shape[1]
            columns = [str(i) for i in range(col_count - 1)] + ['target']
            df = pd.DataFrame(flattened_data, columns=columns)

        # 5. 保存为 CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        return True

    except Exception as e:
        print(f"  转换失败: {str(e)}")
        return False


def batch_mat_to_csv(folder_path, data_key=None, recursive=False):
    """
    批量转换文件夹中的所有 .mat 文件为 CSV 格式，列名格式：0,1,2...target

    参数:
        folder_path: str, 目标文件夹路径
        data_key: str, 指定要提取的数据键名（可选）
        recursive: bool, 是否递归处理子文件夹（默认 False）
    """
    # 验证文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return

    # 统计变量
    total_files = 0
    success_files = 0
    failed_files = []

    # 遍历文件夹
    print(f"开始批量转换，目标文件夹: {folder_path}")
    print(f"是否递归处理子文件夹: {recursive}")
    print("-" * 50)

    # 选择遍历方式：递归/非递归
    if recursive:
        file_pattern = Path(folder_path).rglob("*.mat")
    else:
        file_pattern = Path(folder_path).glob("*.mat")

    # 处理每个 .mat 文件
    for mat_file in file_pattern:
        mat_file_str = str(mat_file)
        total_files += 1
        print(f"正在处理 [{total_files}]: {mat_file_str}")

        # 生成对应的 CSV 路径
        csv_file_str = os.path.splitext(mat_file_str)[0] + '.csv'

        # 转换文件
        if mat_to_csv(mat_file_str, csv_file_str, data_key):
            success_files += 1
            print(f"  转换成功 -> {csv_file_str}")
        else:
            failed_files.append(mat_file_str)

    # 输出统计结果
    print("-" * 50)
    print(f"批量转换完成！")
    print(f"总计找到 .mat 文件: {total_files} 个")
    print(f"转换成功: {success_files} 个")
    print(f"转换失败: {len(failed_files)} 个")

    if failed_files:
        print("失败文件列表:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 配置参数
    # 替换为你的目标文件夹路径
    TARGET_FOLDER = r"D:\111\Feature\dataset"
    # 是否递归处理子文件夹（True/False）
    RECURSIVE_PROCESS = False
    # 指定MAT文件中的数据键名（可选，设为None则自动识别）
    DATA_KEY = None

    # 执行批量转换
    batch_mat_to_csv(
        folder_path=TARGET_FOLDER,
        recursive=RECURSIVE_PROCESS,
        data_key=DATA_KEY
    )
