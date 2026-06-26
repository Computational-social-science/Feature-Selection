import json
import math
import warnings
import pandas as pd
import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
# from OtherWay import *
from WCFSnew import LearnALL
# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#'alarm1000','Waveform','Wdbc', 'Synthetic_control','Authorship',
datanames = [ 'Factors', 'Pixel', 'Isolet', 'ORL', 'WarpPIE10P', 'Su', ]#
          #   'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111', 'Yeoh''Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',
base_url = r"./dataset/"


#BN数据集获取
# from pgmpy.readwrite import BIFReader
# import pandas as pd
#
# # 读取 BIF 文件
# reader = BIFReader("alarm.bif")  # 替换为你的文件
# model = reader.get_model()
#
# # 获取所有变量（节点）
# nodes = model.nodes()
# print("Nodes:", nodes)
#
# # 获取所有边（因果关系）
# edges = model.edges()
# print("Edges:", edges)
#
# # 生成 DataFrame (可选，若想存储网络结构)
# df_edges = pd.DataFrame(edges, columns=["Parent", "Child"])
# print(df_edges)
# from pgmpy.sampling import BayesianModelSampling
#
# # 生成样本数据（采样 1000 行）
# inference = BayesianModelSampling(model)
# sampled_data = inference.forward_sample(size=1000)
#
# # 转换为 DataFrame
# df = pd.DataFrame(sampled_data)
from sklearn.preprocessing import LabelEncoder
data_url = 'alarm1000' + '.pkl'
# # # dataname = "movement_libras"
url = base_url + data_url  # 数据路径
data = pd.read_csv(url)
# # # print(data)
encoder = LabelEncoder()
# # # 遍历所有列，对类别数据进行整数编码
df_encoded = data.apply(encoder.fit_transform)
print(df_encoded.columns)
# print(df_encoded)
# X0 = pd.DataFrame(data['X'])  # 读取训练数据
#
# Special = X0.iloc[:, -1]
# print(Special.values)
# X0 = X0.iloc[:, :-1]  # 读取训练数据
# y0 = pd.DataFrame(data['Y'])  # 读取标签
# a = np.array([item[0] for item in Special])
# label_encoder = LabelEncoder()
# a33 = label_encoder.fit_transform(a)
# X0[33] = a33
# print(a,"!!!!!!!!!!!!!!!!!!!!!!!")
# print(data)
# print(X0,y0)
#完成数据处理
print("====================================================================================================")
def convert_set(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_set(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_set(i) for i in obj]
    return obj
#
#
# # 创建LabelEncoder实例
result_dict = {}
for i in df_encoded.columns:
    print(i)
    result = LearnALL(df_encoded,i)
    dictresult = dict()
    result_dict[i] = set(result)
    # dictresult[i] =
    # print(len(result),set(result),len(set(result)))
print(result_dict)
data_serializable = {key: value for key, value in result_dict.items()}
print(data_serializable)
# # 保存字典到 .txt 文件
with open("data_dict_alarm.txt", "w") as file:
    data_serializable1 = convert_set(data_serializable)
    json.dump(data_serializable1, file, indent=4)

# for dataname in datanames:
#     data_url = dataname + '.mat'
# #     # dataname = "movement_libras"
#     url = base_url + data_url  # 数据路径
# #     # data = pd.read_csv(url)
#     data = scio.loadmat(url)
#     X0 = pd.DataFrame(data['X'])
#     y0 = pd.DataFrame(data['Y'])
#     # print(data)
#     if dataname == 'Dermatology':
#         Special = X0.iloc[:, -1]
#         X0 = X0.iloc[:, :-1]  # 读取训练数据
#         a = np.array([item[0] for item in Special])
#         label_encoder = LabelEncoder()
#         a33 = label_encoder.fit_transform(a)
#         X0[33] = a33
#     # encoder = LabelEncoder()
#     # 遍历所有列，对类别数据进行整数编码
#     # df_encoded = data.apply(encoder.fit_transform)
#     # print(df_encoded.columns)
#     # 将y标签控制在0-n
#     # print(X0,y0)
#     def changetosinge(x):
#         return float(x)
#     # 应用函数到 DataFrame 的每个元素
#     X0 = X0.applymap(changetosinge)
#     y0 = y0.applymap(changetosinge)
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y0)
#     Class = set(y_encoded)
#     y = pd.DataFrame(y_encoded)
#     # 数据离散化
#     X = pd.DataFrame()
#     # label_encoder = LabelEncoder()
#     # y_encoded = label_encoder.fit_transform(X0[33])
#     # data = {33:y_encoded}
#     # newf = pd.DataFrame(data)
#     # X0[33]=newf[33]
#
#     # print(X0,y)
#     for col in X0.columns:
#         X[col] = pd.cut(X0[col], bins=5, labels=False)
#     # 使用rename函数重新命名列名，将x列名控制在0-n
#     new_columns = list(range(X.shape[1] + 1))
#     X = X.rename(columns=dict(zip(X.columns, new_columns)))
#     y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
#     # print(y)
#
#     data_processed = pd.concat([X, y], axis=1)
#     # print(X, y)
#     # ====================================数据导入=====================================
#     # methods1 = [csmdccmr]
#     # methods2 = [MRMR.mrmr, CIFE.cife]
#     # methods3 = [jmim, mri, cfr, dcsf, mrmd, iwfs, ucrfs]
#     # for n in methods3:
#     #     print(n.__name__)
#     #     method = n.__name__
#         # 创建一个十折交叉验证对象
#     kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
#
#     classnum = len(Class)  # 类别数，分为几类
#     samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
#     # accmax = []  # 准确度记录数组，为了画折线图
#     Fall = X.shape[1]  # 特征数
#     # 选取的特征数量
#     if Fall < 50:
#         time = Fall
#     else:
#         time = 50
#     # 对每一个9折，进行特征选择
#     baverage, kaverage, saverage, raverage = 0, 0, 0, 0
#     bnacc, knacc, snacc, rnacc = [], [], [], []
#     result = LearnALL(data_processed, list(range(data_processed.shape[1]))[-1])
#     print(result)
    # 十折交叉验证
    # for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
    #     print("当前处理到第", fold, "折\n")
    #     bacc, kacc, sacc, racc = [], [], [], []
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
    #     result = []  # 特征提取结果数组
    #     # -----------------cmdccmr
    #     # for ck in Class:# 为每一个类别进行特征选择      #X为纯数组，y为series
    #     #     cpick = csmdccmr(X_train.values, np.ravel(y_train), ck, n_selected_features=time)
    #     #     result.append(cpick)
    #     # -----------------cmdccmr
    #     # result,a,b = n(X_train.values, np.ravel(y_train), n_selected_features=time)#只提取一次特征
    #     result = LearnALL(data_processed, list(range(data_processed.shape[1]))[-1])
    #     # result = n(X_train.values, np.ravel(y_train), n_selected_features=time)  # 只提取一次特征
    #     # -----------------otherway
    #     file_name = r"feature_select_result/" + 'bntest' + '/' + dataname + str(fold) + "_result.txt"
    #     np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
    #     # result = np.loadtxt(file_name)#加载特征提取结果
    #     result = np.array(result)
    #     # 对每一个数量的特征选择做测试
    #     # bnacc,knacc,rnacc,snacc = csmdccmrtest(time, result, X_test, y_test, X_train, y_train)
    #     bnacc, knacc, rnacc, snacc = othertest(time, result, X_test, y_test, X_train, y_train)
    # calculate(time)
#
#         # =======================================画图==========================================
#         import matplotlib.pyplot as plt
#
#         # 数据读取
#         y_values = np.loadtxt("accuracy/" + dataname + '_' + method + "_accuracy.txt")
#         x_values = np.arange(1, time + 1)
#         # 创建折线图
#         plt.figure(figsize=(8, 4))
#         # plt.ylim(0,1)
#         # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#         plt.plot(x_values, y_values, color='black')  # 黑色
#         # 添加标题和标签
#         plt.title(dataname + ' ' + method)
#         plt.xlabel('Number of selected features')
#         plt.ylabel('Classification accuracy')
#         # 显示图形
#         plt.show()
