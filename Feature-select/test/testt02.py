import argparse
import time
import numpy as np
import scipy.io as scio
import pandas as pd
from numpy import ravel
from pgmpy.estimators import ParameterEstimator
from skfeature.utility.entropy_estimators import midd
from sklearn.preprocessing import LabelEncoder
from skfeature.utility.entropy_estimators import *
import math

from eamb import EAMB

# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)  # 显示所有列
def changetosinge(x):
    return float(x)

import numpy as np
from scipy.special import gammaincc

# 假设的 alpha 值，用于判断依赖显著性
alpha = 0.05
N_CASES_PER_DF = 10  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy

import math
import warnings

import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
#
# # 忽略特定类型的警告
# warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# # ====================================数据导入=====================================
# #
datasets = [
    'Prostate_GE'
]#

base_url = r"../dataset/"
# def changetosinge(x):
#     return float(x)

# for dataname in datasets:
#     data_url = dataname + '.mat'
#     url = base_url + data_url  # 数据路径
#     data = scio.loadmat(url)  # 读取数据文件
#     X0 = pd.DataFrame(data['X'])  # 读取训练数据
#     y0 = pd.DataFrame(data['Y'])  # 读取标签
#     print(dataname)
#     print("====================================================================================================")
#     #=======================Dermatology==================
#     if dataname == 'Dermatology':
#         Special = X0.iloc[:,-1]
#         # print(Special.values)
#         X0 = X0.iloc[:,:-1]  # 读取训练数据
#         a = np.array([item[0] for item in Special])
#         label_encoder = LabelEncoder()
#         a33 = label_encoder.fit_transform(a)
#         X0[33] = a33
#     #=======================Dercomatoly==================
#     # 将y标签控制在0-n
#     # 应用函数到 DataFrame 的每个元素
#     X0 = X0.applymap(changetosinge)
#     y0 = y0.applymap(changetosinge)
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y0)
#     Class = set(y_encoded)
#     y = pd.DataFrame(y_encoded)
#     # 数据离散化
#     X = pd.DataFrame()
#     # print(X0,y)
#     #离散化
#     for col in X0.columns:
#         X[col] = pd.cut(X0[col], bins=5, labels=False)
#     # 使用rename函数重新命名列名，将x列名控制在0-n
#     new_columns = list(range(X.shape[1] + 1))
#     X = X.rename(columns=dict(zip(X.columns, new_columns)))
#     y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
#     data_processed = pd.concat([X, y], axis=1)
#     result = EAMB( list(range(data_processed.shape[1]))[-1],data_processed,30)
#     print('123321\n',result)










# from sklearn.preprocessing import LabelEncoder
# data_url = 'alarm1000' + '.pkl'
# # # # dataname = "movement_libras"
# url = base_url + data_url  # 数据路径
# data = pd.read_csv(url)
# # # # print(data)
# encoder = LabelEncoder()
# # # # 遍历所有列，对类别数据进行整数编码
# df_encoded = data.apply(encoder.fit_transform)
# #完成数据处理
# print("====================================================================================================")
# def convert_set(obj):
#     if isinstance(obj, set):
#         return list(obj)
#     elif isinstance(obj, dict):
#         return {k: convert_set(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_set(i) for i in obj]
#     return obj
# #
# #
# # # 创建LabelEncoder实例
# result_dict = {}
# for i in df_encoded.columns:
#     print(i)
#     result = EAMB(i,df_encoded,30)
#     dictresult = dict()
#     result_dict[i] = set(result)
# print(result_dict)


for dataname in datasets:
    data_url = dataname + '.mat'
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    print(dataname)
    print("====================================================================================================")
    #=======================Dermatology==================
    if dataname == 'Dermatology':
        Special = X0.iloc[:,-1]
        # print(Special.values)
        X0 = X0.iloc[:,:-1]  # 读取训练数据
        a = np.array([item[0] for item in Special])
        label_encoder = LabelEncoder()
        a33 = label_encoder.fit_transform(a)
        X0[33] = a33
    #=======================Dercomatoly==================
    # 将y标签控制在0-n
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    y = pd.DataFrame(y_encoded)
    # 数据离散化
    X = pd.DataFrame()
    # print(X0,y)
    #离散化
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    # 将列名命名为字符串
    new_columns = [str(i) for i in range(X.shape[1] + 1)]  # 将列名转换为字符串
    X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))  # 重命名 X 的列名
    y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))  # 重命名 y 的列名
    data_processed = pd.concat([X, y], axis=1)
    # new_columns = list(range(X.shape[1] + 1))
    # X = X.rename(columns=dict(zip(X.columns, new_columns)))
    # y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
    # data_processed = pd.concat([X, y], axis=1)
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    classnum = len(Class)  # 类别数，分为几类
    classes = list(Class)
    samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
    Fall = X.shape[1]  # 特征数
    # 选取的特征数量
    # if Fall < 50:
    #     time = Fall
    # else:
    #     time = 50
    # 对每一个9折，进行特征选择
    # 十折交叉验证
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
        print("当前是第"+ str(fold) +"折\n")
        if fold in [0,1,2,3,4,5,6,7]:
            continue
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]#提取训练集条目
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]#提取测试集条目
        # result_dict = {}
        data_trained = pd.concat([X_train, y_train], axis=1)
        # print("Class:",Class)
        # for nowy in Class:
        #     print("nowy:",nowy)
        result = EAMB(str(X.shape[1]), data_trained, 0.1)
        result_dict = set(result)
        # print(dataname)
        # print(result_dict)
        result_list = [int(x) for x in result_dict]
        result_array = np.array(result_list)
        print(result_array)
        file_name = r"../feature_select_result/" + 'eamb0' + '/New/' + dataname + str(fold) + "_result.txt"
        np.savetxt(file_name, result_array, fmt='%d')  # 使用savetxt()函数保存数组到文件
    # import json
    #
    # # 将集合转换为列表
    # result_dict_serializable = {k: list(v) for k, v in result_dict.items()}
    #
    # # 存储到文件
    # with open(dataname+'_result_dict.json', 'w') as f:
    #     json.dump(result_dict_serializable, f, indent=4)














    # print(X, y)
    # ====================================数据导入=====================================
    # kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    # classnum = len(Class)  # 类别数，分为几类
    # samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
    #
    # methods2 = [IAMB]
    # for n in methods2:
    #     print(n.__name__)
    #     method = n.__name__
    #     # 创建一个十折交叉验证对象
    #     Fall = X.shape[1]  # 特征数
    #     # 选取的特征数量
    #     if Fall < 50:
    #         time = Fall
    #     else:
    #         time = 50
    #     # 十折交叉验证
    #     # print(f"alpha: {alpha}, n_cases: {n_cases}, n_vars: {n_vars}")
    #     for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
    #         print("\n当前处理到第", fold, "折")
    #         bacc, kacc, sacc, racc = [], [], [], []
    #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
    #         y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
    #         result = []  # 特征提取结果数组
    #         # result,a,b = n(X_train.values, np.ravel(y_train), n_selected_features=time)#只提取一次特征
    #         result = n(X_train, y_train)  # 只提取一次特征
    #         print('result:',result)
    #         file_name = r"feature_select_result/" + method + '/' + dataname + str(fold) + "_result.txt"
    #         np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
    #         result = np.loadtxt(file_name)#加载特征提取结果
    #         # othertest(len(result), result, X_test, y_test, X_train, y_train)
    #         # calculate(time)
    #         result = np.array(result)
    #         # print(result)
    #         # 对每一个数量的特征选择做测试

        # =======================================画图==========================================
        # import matplotlib.pyplot as plt
        #
        # # 数据读取
        # y_values = np.loadtxt("accuracy/1/" + dataname + '_' + method + "_accuracy.txt")
        # x_values = np.arange(1, time + 1)
        # # 创建折线图
        # plt.figure(figsize=(8, 4))
        # # plt.ylim(0,1)
        # # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        # plt.plot(x_values, y_values, color='black')  # 黑色
        # # 添加标题和标签
        # plt.title(dataname + ' ' + method)
        # plt.xlabel('Number of selected features')
        # plt.ylabel('Classification accuracy')
        # # 显示图形
        # plt.show()
#==================================数据预测完毕，进行预测结果的处理=======================================================================
#处理得到结果计算出的F1和AUC
# T = '1'#用于区别储存
# methods = ['IAMB']#'cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs',
# #定义八个表格用于存储最后的不同方法和不同数据集的F1和AUC结果
# bf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# bf1table.columns = methods
# bf1table.index = datasets
# sf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# sf1table.columns = methods
# sf1table.index = datasets
# kf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# kf1table.columns = methods
# kf1table.index = datasets
# rf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# rf1table.columns = methods
# rf1table.index = datasets
#
# bacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# bacctable.columns = methods
# bacctable.index = datasets
# sacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# sacctable.columns = methods
# sacctable.index = datasets
# kacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# kacctable.columns = methods
# kacctable.index = datasets
# racctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# racctable.columns = methods
# racctable.index = datasets
#
#
# bauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# bauctable.columns = methods
# bauctable.index = datasets
# sauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# sauctable.columns = methods
# sauctable.index = datasets
# kauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# kauctable.columns = methods
# kauctable.index = datasets
# rauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# rauctable.columns = methods
# rauctable.index = datasets
#

#设置最大化展示
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
# for method in methods:#对每个方法进行处理
#     print(method,"=============================================================================")
#     #存储不同方法所有数据集23个的结果
#     for dataname in datasets:#对于每个数据集进行处理
#         iposition = datasets.index(dataname)#定方法的位置
#         #读取数据及其真实值
#         data_url = dataname + '.mat'
#         url = base_url + data_url  # 数据路径
#         data = scio.loadmat(url)  # 读取数据文件
#         print(dataname)
#         #读取文件中的真实标签
#         y0 = pd.DataFrame(data['Y'])  # 读取标签
#         # 应用函数到 DataFrame 的每个元素
#         y0 = y0.applymap(changetosinge)
#         #将标签控制在0-n
#         label_encoder = LabelEncoder()
#         y_encoded = label_encoder.fit_transform(y0)
#         Class = set(y_encoded)
#         y = pd.DataFrame(y_encoded)
#         classes = list(Class)
#
#         #当前数据集的十折数据所有样本结果存储
#         # 十折交叉验证
#         allfold_b_f1 = []
#         allfold_k_f1 = []
#         allfold_s_f1 = []
#         allfold_r_f1 = []
#         allfold_b_auc = []
#         allfold_k_auc = []
#         allfold_s_auc = []
#         allfold_r_auc = []
#         bpredict_labels_all = pd.DataFrame()
#         kpredict_labels_all = pd.DataFrame()
#         spredict_labels_all = pd.DataFrame()
#         rpredict_labels_all = pd.DataFrame()
#         bprobabily_all = []
#         kprobabily_all = []
#         sprobabily_all = []
#         rprobabily_all = []
#         bpredict_probabity_list = []
#         kpredict_probabity_list = []
#         spredict_probabity_list = []
#         rpredict_probabity_list = []
#         bpredict_labels_list = []
#         kpredict_labels_list = []
#         spredict_labels_list = []
#         rpredict_labels_list = []
#         y_all = pd.DataFrame()  # 真实值的所有样本结果
#         for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
#             print("当前是第",fold)
#             fold_b_f1 = []
#             fold_k_f1 = []
#             fold_s_f1 = []
#             fold_r_f1 = []
#             fold_b_acc = []
#             fold_k_acc = []
#             fold_s_acc = []
#             fold_r_acc = []
#             fold_b_auc = []
#             fold_k_auc = []
#             fold_s_auc = []
#             fold_r_auc = []
#             b_all_auc = []
#             k_all_auc = []
#             s_all_auc = []
#             r_all_auc = []
#             y_test = y.iloc[test_index]  # 提取当前折测试集条目
#             #读取预测结果
#             file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
#             file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
#             predict_pro = scio.loadmat(file_name1)  # 读取数据文件
#             predict_lable = scio.loadmat(file_name2)  # 读取数据文件
#             lenth = len(y_test)#当前折的测试集数量
#             #对于每一折的数据都读取对应数量
#             bpredict_probabity = np.array(predict_pro['b'])[:, :lenth, :]#time个结果，每个结果有class个分类器，每个分类器有n个样本和class种可能
#             kpredict_probabity = np.array(predict_pro['k'])[:, :lenth, :]
#             spredict_probabity = np.array(predict_pro['s'])[:, :lenth, :]
#             rpredict_probabity = np.array(predict_pro['r'])[:, :lenth, :]
#             spredict_labels = pd.DataFrame(predict_lable['s']).iloc[:, :lenth]
#             bpredict_labels = pd.DataFrame(predict_lable['b']).iloc[:, :lenth]#time个结果，每个结果n个样本，
#             kpredict_labels = pd.DataFrame(predict_lable['k']).iloc[:, :lenth]
#             rpredict_labels = pd.DataFrame(predict_lable['r']).iloc[:, :lenth]
#             # print(bpredict_probabity,bpredict_labels)
#             bpredict_probabity_list.append(bpredict_probabity)
#             kpredict_probabity_list.append(kpredict_probabity)
#             spredict_probabity_list.append(spredict_probabity)
#             rpredict_probabity_list.append(rpredict_probabity)
#             bpredict_labels_list.append(bpredict_labels)
#             kpredict_labels_list.append(kpredict_labels)
#             spredict_labels_list.append(spredict_labels)
#             rpredict_labels_list.append(rpredict_labels)
#
#             bpredict_labels_all = pd.concat([bpredict_labels_all, bpredict_labels], axis=1)
#             kpredict_labels_all = pd.concat([kpredict_labels_all, kpredict_labels], axis=1)
#             spredict_labels_all = pd.concat([spredict_labels_all, spredict_labels], axis=1)
#             rpredict_labels_all = pd.concat([rpredict_labels_all, rpredict_labels], axis=1)
#             y_all = pd.concat([y_all, y_test], axis=0)
#         # print(bpredict_labels_all,y_all)
#         bprobabily_all = np.concatenate(bpredict_probabity_list, axis=1)
#         kprobabily_all = np.concatenate(kpredict_probabity_list, axis=1)
#         sprobabily_all = np.concatenate(spredict_probabity_list, axis=1)
#         rprobabily_all = np.concatenate(rpredict_probabity_list, axis=1)
#         # print(bprobabily_all.shape,bpredict_labels_all.shape)
#         # 转置矩阵，使行为样本列为特征数量
#         sample_num = bpredict_labels_all.shape[1]
#         feature_num = bpredict_labels_all.shape[0]
#         new_column_names = range(sample_num)
#         bpredict_labels_all.columns = new_column_names[:sample_num]
#         bpredict_labels_all = bpredict_labels_all.T
#         kpredict_labels_all.columns = new_column_names[:sample_num]
#         kpredict_labels_all = kpredict_labels_all.T
#         spredict_labels_all.columns = new_column_names[:sample_num]
#         spredict_labels_all = spredict_labels_all.T
#         rpredict_labels_all.columns = new_column_names[:sample_num]
#         rpredict_labels_all = rpredict_labels_all.T
#         # print(rpredict_labels_all,y_all)
#         # for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
#             # print(y_true_binarized,bprobabily_all[i])
#             # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
#         # print(bpredict_labels_all)
#         time_bf1 = f1_score(bpredict_labels_all[0], y_all, average='micro')
#         time_kf1 = f1_score(kpredict_labels_all[0], y_all, average='micro')
#         time_sf1 = f1_score(spredict_labels_all[0], y_all, average='micro')
#         time_rf1 = f1_score(rpredict_labels_all[0], y_all, average='micro')
#         # print(time_bf1, time_kf1, time_sf1, time_rf1)
#         acc_b = calculate_accuracy(bpredict_labels_all[0], y_all[0],)
#         acc_k = calculate_accuracy(bpredict_labels_all[0], y_all[0], )
#         acc_s = calculate_accuracy(bpredict_labels_all[0], y_all[0], )
#         acc_r = calculate_accuracy(bpredict_labels_all[0], y_all[0], )
#         # print(acc_b, acc_k, acc_s, acc_r)
#         # 种类个数特殊处理，为了多分类的auc进行处理
#         if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
#             y_true_binarized = pd.get_dummies(y_all[0])
#             y_true_binarized_array = y_true_binarized.values
#         else:  # 多类别可以直接调用函数
#             y_true_binarized = label_binarize(y_all, classes=classes)
#         # for j in range(0, len(classes)):  # 不同分类器结果进行分别处理
#         # print(y_true_binarized,bprobabily_all)
#         b_all_auc = roc_auc_score(y_true_binarized, bprobabily_all[0], average='macro', multi_class='ovr')
#         k_all_auc = roc_auc_score(y_true_binarized, kprobabily_all[0], average='macro', multi_class='ovr')
#         s_all_auc = roc_auc_score(y_true_binarized, sprobabily_all[0], average='macro', multi_class='ovr')
#         r_all_auc = roc_auc_score(y_true_binarized, rprobabily_all[0], average='macro', multi_class='ovr')
#         # print(b_all_auc, k_all_auc, s_all_auc, r_all_auc)
#         # time_bauc = np.mean(b_all_auc)
#         # time_kauc = np.mean(k_all_auc)
#         # time_sauc = np.mean(s_all_auc)
#         # time_rauc = np.mean(r_all_auc)
#         fold_b_f1.append(time_bf1)
#         fold_k_f1.append(time_kf1)
#         fold_s_f1.append(time_sf1)
#         fold_r_f1.append(time_rf1)
#         fold_b_acc.append(acc_b)
#         fold_k_acc.append(acc_k)
#         fold_s_acc.append(acc_s)
#         fold_r_acc.append(acc_r)
#         fold_b_auc.append(b_all_auc)
#         fold_k_auc.append(k_all_auc)
#         fold_s_auc.append(s_all_auc)
#         fold_r_auc.append(r_all_auc)
#         # allfold_b_f1.append(fold_b_f1)
#         # allfold_k_f1.append(fold_k_f1)
#         # allfold_s_f1.append(fold_s_f1)
#         # allfold_r_f1.append(fold_r_f1)
#         # allfold_b_auc.append(fold_b_auc)
#         # allfold_k_auc.append(fold_k_auc)
#         # allfold_s_auc.append(fold_s_auc)
#         # allfold_r_auc.append(fold_r_auc)
#         #当前数据集的结果为所有特征数量结果的平均数
#         #处理完后存入表格记录
#         bacctable[method][iposition] = np.mean(fold_b_acc)
#         sacctable[method][iposition] = np.mean(fold_k_acc)
#         kacctable[method][iposition] = np.mean(fold_s_acc)
#         racctable[method][iposition] = np.mean(fold_r_acc)
#         bf1table[method][iposition]  = np.mean(fold_b_f1)
#         sf1table[method][iposition]  = np.mean(fold_s_f1)
#         kf1table[method][iposition]  = np.mean(fold_k_f1)
#         rf1table[method][iposition]  = np.mean(fold_r_f1)
#         bauctable[method][iposition] = np.mean(fold_b_auc)
#         sauctable[method][iposition] = np.mean(fold_s_auc)
#         kauctable[method][iposition] = np.mean(fold_k_auc)
#         rauctable[method][iposition] = np.mean(fold_r_auc)

# tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
# # 计算每列的平均值
# for i, table in enumerate(tables):
#     average_row = table.mean(axis=0)
#     table.loc['average'] = average_row
#
# #对每个表格求平均值并按照大小进行升序排序
# bf1table = bf1table.sort_values(by='average', axis=1)
# sf1table = sf1table.sort_values(by='average', axis=1)
# kf1table = kf1table.sort_values(by='average', axis=1)
# rf1table = rf1table.sort_values(by='average', axis=1)
# bauctable = bauctable.sort_values(by='average', axis=1)
# sauctable = sauctable.sort_values(by='average', axis=1)
# kauctable = kauctable.sort_values(by='average', axis=1)
# rauctable = rauctable.sort_values(by='average', axis=1)
# print(bacctable,'\n',kacctable,'\n',sacctable,'\n',racctable,'\n')
# print(bf1table,'\n',kf1table,'\n',sf1table,'\n',rf1table,'\n')
# print(bauctable,'\n',sauctable,'\n',kauctable,'\n',rauctable,'\n')
#存储
# bf1table.to_excel( 'tables/'+T+'/ten-bf1table.xlsx', index=True)
# sf1table.to_excel( 'tables/'+T+'/ten-sf1table.xlsx', index=True)
# kf1table.to_excel( 'tables/'+T+'/ten-kf1table.xlsx', index=True)
# rf1table.to_excel( 'tables/'+T+'/ten-rf1table.xlsx', index=True)
# bauctable.to_excel('tables/'+T+'/ten-bauctable.xlsx', index=True)
# sauctable.to_excel('tables/'+T+'/ten-sauctable.xlsx', index=True)
# kauctable.to_excel('tables/'+T+'/ten-kauctable.xlsx', index=True)
# rauctable.to_excel('tables/'+T+'/ten-rauctable.xlsx', index=True)
#绘制箱线图---------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# import numpy as np
# bf1table = pd.read_excel( 'tables/'+T+'/ten-bf1table.xlsx', index_col=0)
# sf1table = pd.read_excel( 'tables/'+T+'/ten-sf1table.xlsx', index_col=0)
# kf1table = pd.read_excel( 'tables/'+T+'/ten-kf1table.xlsx', index_col=0)
# rf1table = pd.read_excel( 'tables/'+T+'/ten-rf1table.xlsx', index_col=0)
# bauctable = pd.read_excel('tables/'+T+'/ten-bauctable.xlsx', index_col=0)
# sauctable = pd.read_excel('tables/'+T+'/ten-sauctable.xlsx', index_col=0)
# kauctable = pd.read_excel('tables/'+T+'/ten-kauctable.xlsx', index_col=0)
# rauctable = pd.read_excel('tables/'+T+'/ten-rauctable.xlsx', index_col=0)
# tables = [bf1table,sf1table,kf1table,rf1table,bauctable,sauctable,kauctable,rauctable]
# for table in tables:
#     # 计算每列的平均值
#     average_row = table.mean(axis=0)
#     table.loc['average'] = average_row
#     df_sorted = table.sort_values(by='average', axis=1)
#     print(table)
#
# print(bf1table)
# 创建一些示例数据
# f1datas = [bf1table,sf1table,kf1table,rf1table]
# aucdatas = [bauctable,sauctable,kauctable,rauctable]
# names = ['Naive Bayes','3NN','SVM','RandomForest']
# # 创建箱线图
# i=-1
# for data in f1datas:
#     i+=1
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(data, labels=data.columns)
#     # plt.ylim(0.3, 1)
#     # 添加标题和标签
#     plt.title(names[i])
#     plt.ylabel('F1')
#     plt.grid(False)
#     # 显示图形
#     plt.grid(True)
#     # plt.savefig("picture/" + names[i] + T +"_ten_F1.png", dpi=300, bbox_inches='tight')
#     plt.show()
# j=-1
# for data in aucdatas:
#     j+=1
#     plt.figure(figsize=(8, 6))
#     plt.boxplot(data, labels=data.columns)
#     # plt.ylim(0.3, 1)
#     # 添加标题和标签
#     plt.title(names[j])
#     plt.ylabel('AUC')
#     plt.grid(False)
#     # 显示图形
#     plt.grid(True)
#     # plt.savefig("picture/" + names[j] + T +"_ten_AUC.png", dpi=300, bbox_inches='tight')
#     plt.show()







