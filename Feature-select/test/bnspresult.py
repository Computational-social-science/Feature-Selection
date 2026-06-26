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

from eamb_cs import EAMB_sp, ifdep_sp


# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)  # 显示所有列
def changetosinge(x):
    return float(x)

import numpy as np
from scipy.special import gammaincc

# 假设的 alpha 值，用于判断依赖显著性
alpha = 0.1
N_CASES_PER_DF = 10  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy

#EAMB Algorithm
from scipy.stats import chi2_contingency
from skfeature.utility.entropy_estimators import cmidd

from csmdccmr import middspecial, redundancy, entropydpro, entropyed
import pandas as pd
from skfeature.utility.entropy_estimators import cmidd
from csmdccmr import middspecial, redundancy
from eamb import ESMB

def cmiddspecial_fixed(x, y, z, fixed_var, fixed_value):
    """
    计算 I(X = a; Y | Z), I(X; Y = a | Z) 或 I(X; Y | Z = a)

    参数：
        x, y, z: 数据列表
        fixed_var: 要固定的变量 (只能是 'X', 'Y' 或 'Z')
        fixed_value: 该变量的固定值
    返回：
        计算后的条件互信息值
    """
    # 将 X, Y, Z 组合成 (x, y, z) 元组
    xyz = list(zip(x, y, z))

    # 确定固定变量的位置
    var_index = {'X': 0, 'Y': 1, 'Z': 2}

    if fixed_var not in var_index:
        raise ValueError("fixed_var 必须是 'X', 'Y' 或 'Z'")

    # 筛选符合 fixed_var == fixed_value 的样本
    filtered_xyz = [sample for sample in xyz if sample[var_index[fixed_var]] == fixed_value]

    if not filtered_xyz:
        return 0  # 没有符合条件的样本，返回 0

    # 重新构造 X, Y, Z
    filtered_x = [s[0] for s in filtered_xyz]
    filtered_y = [s[1] for s in filtered_xyz]
    filtered_z = [s[2] for s in filtered_xyz]

    # 计算条件互信息 I(X; Y | Z)
    return (entropydpro(list(zip(filtered_x, filtered_y, filtered_z)), fixed_value) +
            entropyed(list(zip(filtered_x, filtered_z)), fixed_value) -
            entropyed(list(zip(filtered_x, filtered_y, filtered_z)), fixed_value) -
            entropydpro(list(zip(filtered_x, filtered_z)), fixed_value))

def ifdep(dataframe, C, Y, X, alpha=0.15):
    """
    检验条件独立性 C ⊥⊥ Y | X，其中 X 可以是多列条件集合。

    参数:
    - dataframe: pandas.DataFrame，包含变量 C, Y, X 的数据框。
    - C: str，变量 C 的列名。
    - Y: str，变量 Y 的列名。
    - X: list of str，条件变量集合的列名列表。
    - alpha: float，显著性水平，默认为 0.05。

    返回:
    - bool: 如果 C ⊥⊥ Y | X 成立（独立），返回 True；否则返回 False。
    """
    # print(C,Y,X)
    # 如果 X 为空，直接检验 C 和 Y 的独立性
    ifdep.call_out += 1
    if not X:
        contingency_table = pd.crosstab(dataframe[C], dataframe[Y])
        _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return p_value >= alpha

    # 获取条件变量 X 的所有唯一组合
    grouped = dataframe.groupby(X)

    # 遍历每个条件组合
    for name, group in grouped:
        # 构建列联表
        contingency_table = pd.crosstab(group[C], group[Y])

        # 如果列联表为空或只有一行/一列，跳过
        if contingency_table.size == 0 or contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
            continue

        # 使用 G² 检验（似然比检验）检验 C 和 Y 是否独立
        _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")

        # 如果在某个条件下 p 值小于 alpha，则 C 和 Y 不独立
        if p_value < alpha:
            return False

    # 如果所有条件下 p 值都大于 alpha，则 C 和 Y 独立
    return True

import pandas as pd
from scipy.stats import chi2_contingency

def ifdep_sp(dataframe, C, Y, X, y_value, alpha=0.2):
    """
    检验在 Y = y_value 的条件下，C 和 Y 是否独立，即 C ⊥⊥ Y = y_value | X。

    参数:
    - dataframe: pandas.DataFrame，包含变量 C, Y, X 的数据框。
    - C: str，变量 C 的列名。
    - Y: str，变量 Y 的列名。
    - X: list of str，条件变量集合的列名列表。
    - y_value: Y 的特定值，即 Y = y_value。
    - alpha: float，显著性水平，默认为 0.05。

    返回:
    - bool: 如果 C ⊥⊥ Y = y_value | X 成立（独立），返回 True；否则返回 False。
    """
    # 筛选出 Y = y_value 的数据
    filtered_data = dataframe[dataframe[Y] == y_value]

    # 如果 X 为空，直接检验 C 和 Y 的独立性
    if not X:
        contingency_table = pd.crosstab(filtered_data[C], filtered_data[Y])
        _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return p_value >= alpha

    # 获取条件变量 X 的所有唯一组合
    grouped = filtered_data.groupby(X)

    # 遍历每个条件组合
    for name, group in grouped:
        # 构建列联表
        contingency_table = pd.crosstab(group[C], group[Y])

        # 如果列联表为空或只有一行/一列，跳过
        if contingency_table.size == 0 or contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
            continue

        # 使用 G² 检验（似然比检验）检验 C 和 Y 是否独立
        _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")

        # 如果在某个条件下 p 值小于 alpha，则 C 和 Y 不独立
        if p_value < alpha:
            return False

    # 如果所有条件下 p 值都大于 alpha，则 C 和 Y 独立
    return True
# def ifdep(dataframe, C, Y, X=None, alpha=0.05):
#     """
#     检验条件独立性 C ⊥⊥ Y | X 是否成立。如果 X 为空，则检验 C 和 Y 是否独立。
#
#     参数:
#     - dataframe: pandas.DataFrame，包含变量 C, Y, X 的数据框。
#     - C: str，变量 C 的列名。
#     - Y: str，变量 Y 的列名。
#     - X: str 或 None，条件变量 X 的列名。如果为 None，则直接检验 C 和 Y 的独立性。
#     - alpha: float，显著性水平，默认为 0.05。
#
#     返回:
#     - bool: 如果 C ⊥⊥ Y | X 成立（独立），返回 True；否则返回 False。
#     """
#     if X is None:
#         # 如果 X 为空，直接检验 C 和 Y 的独立性
#         contingency_table = pd.crosstab(dataframe[C], dataframe[Y])
#         _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
#         return p_value >= alpha
#     else:
#         # 如果 X 不为空，检验条件独立性 C ⊥⊥ Y | X
#         conditions = dataframe[X].unique()
#         for condition in conditions:
#             subset = dataframe[dataframe[X] == condition]
#             contingency_table = pd.crosstab(subset[C], subset[Y])
#             if contingency_table.size == 0 or contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
#                 continue
#             _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
#             if p_value < alpha:
#                 return False
#         return True
# def ifdep(x, y, s):#是否独立      #这个有很大问题
#     """判断x和y在条件s下是否独立，s为条件变量列表"""
#     # print(x,y,s)
#     return cmidd(x, y, s) == 0


# def conditional_g2_test(dataframe, C, Y, X, alpha=0.05):
#     """
#     检验条件独立性 C ⊥⊥ Y | X 是否成立。
#
#     参数:
#     - dataframe: pandas.DataFrame，包含变量 C, Y, X 的数据框。
#     - C: str，变量 C 的列名。
#     - Y: str，变量 Y 的列名。
#     - X: str，条件变量 X 的列名。
#     - alpha: float，显著性水平，默认为 0.05。
#
#     返回:
#     - bool: 如果 C ⊥⊥ Y | X 成立（独立），返回 True；否则返回 False。
#     """
#     # 获取条件变量 X 的所有唯一值
#     conditions = dataframe[X].unique()
#
#     # 遍历每个条件
#     for condition in conditions:
#         # 筛选出当前条件下的数据
#         subset = dataframe[dataframe[X] == condition]
#
#         # 构建列联表
#         contingency_table = pd.crosstab(subset[C], subset[Y])
#
#         # 如果列联表为空或只有一行/一列，跳过
#         if contingency_table.size == 0 or contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
#             continue
#
#         # 使用 G² 检验（似然比检验）检验 C 和 Y 是否独立
#         _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
#
#         # 如果在某个条件下 p 值小于 alpha，则 C 和 Y 不独立
#         if p_value < alpha:
#             return False
#
#     # 如果所有条件下 p 值都大于 alpha，则 C 和 Y 独立
#     return True

# def ifdep(dataframe, col1, col2, alpha=0.05):
#     """
#     使用 G² 检验（似然比检验）检验两个分类变量的独立性，并根据 p 值返回 True 或 False。
#
#     参数:
#     - dataframe: pandas.DataFrame，包含分类变量的数据框。
#     - col1: str，第一个分类变量的列名。
#     - col2: str，第二个分类变量的列名。
#     - alpha: float，显著性水平，默认为 0.05。
#
#     返回:
#     - bool: 如果 p 值小于 alpha，返回 True（变量不独立）；否则返回 False（变量独立）。
#     """
#     # 构建列联表
#     contingency_table = pd.crosstab(dataframe[col1], dataframe[col2])
#
#     # 使用 scipy 的 chi2_contingency 函数计算 G² 统计量
#     _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
#
#     # 根据 p 值和 alpha 判断是否独立
#     return p_value < alpha


def compinfo(x, y, z=None, s=None):
    info = 0
    if z is None and s is None:#x为计算特征，y为目标特征
        # 计算I(x, y)
        unique_y = y.unique()
        for ay in unique_y:
            p_ay = y.value_counts(normalize=True)[ay]
            w = 1 - p_ay
            info += w * middspecial(x, y, ay)
    elif z is not None:#xy都为计算特征，z为目标特征
        # 计算Iz(x, y)
        unique_z = z.unique()
        for az in unique_z:
            p_az = z.value_counts(normalize=True)[az]
            w = 1 - p_az
            info += w * redundancy(x, y, z, az)
    else:
        # 计算I(x, y | s)
        unique_y = y.unique()
        for ay_val in unique_y:
            p_ay = y.value_counts(normalize=True)[ay_val]
            w = 1 - p_ay
            if s.empty:
                info += w * middspecial(x, y, ay_val)
                # info +=  middspecial(x, y, ay_val)
            else:
                info += w * cmiddspecial_fixed(x, y, s, 'Y', ay_val)
                # info += cmiddspecial_fixed(x, y, s, 'Y', ay_val)
    return info


def sp_compinfo(x, y, sp, s=None):
    info = 0
    # print('y:',y.name)
    # print("\ny：" ,y, '\n nowsp:',sp)
    p_ay = y.value_counts(normalize=True)[sp]
    w = 1 - p_ay
    if s is None:#x为计算特征，y为目标特征
        # 计算I(x, y)
        # unique_y = y.unique()
        # for ay in unique_y:
        info += w * middspecial(x, y, sp)
    # unique_y = y.unique()
    # for ay_val in unique_y:

    # print(x, y, sp, s)
    elif s.empty:
        info += w * middspecial(x, y, sp)
        # info +=  middspecial(x, y, sp)
    else:
        info += w * cmiddspecial_fixed(x, y, s, 'Y', sp)
        # info += cmiddspecial_fixed(x, y, s, 'Y', sp)
    return info


def findm(data,Cfeature,target,sp,CMB,find):#输入为候选特征们、目标特征和当前MB子集
    nowf = None
    # print(11111111111111111,data[target],target)
    if find == 'max':
        maxinfo = 0
        for f in Cfeature:
            info = sp_compinfo(data[f],data[target],sp = sp,s=data[CMB])
            if info >= maxinfo:
                maxinfo = info
                nowf = f
    else:
        mininfo = 99999999
        for f in Cfeature:
            if f in CMB:
                nowfind = list(set(CMB) - set(f))
            else:
                nowfind = list(CMB)
            info = sp_compinfo(data[f], data[target],sp = sp, s=data[nowfind])
            if info <= mininfo:
                mininfo = info
                nowf = f
    return nowf

def searchk(features,CQ,target,k,sp):
    column_values = {col: sp_compinfo(features[col],features[target],sp= sp) for col in CQ}
    # 选出前k个最大的数据，并将列名降序排序
    num = int(len(CQ) * k * 0.01)
    num = max(1, num)  # 至少保留一列
    Q = sorted(column_values.keys(), key=lambda x: column_values[x], reverse=True)[:num]
    return Q

def ESMB_sp( target,features,sp):
    CMB = []
    data = features
    featuresname = data.columns.tolist()
    # print(target)
    # print(target,features,sp,)
    CF = featuresname.copy()
    # print(CF,target)
    # CF.remove(str(target))
    oldCMB = ['fss']
    first = True
    while oldCMB != CMB and not CF==[]:
        print(CF)
        oldCMB = CMB.copy()
        # print('target:',target)
        best = findm(features,CF,target,sp,CMB,'max')#输入为候选特征们、目标特征和当前MB子集
        # print("CMB:",CMB,"\nCF:",CF)
        if not ifdep(features,best,target,CMB):
            CMB.append(best)
            CF.remove(best)
        if oldCMB != CMB or first:
            first = False
            for f in CF.copy():  # 创建副本用于遍历
                if ifdep(features, f, target, CMB):
                    CF.remove(f)  # 修改原列表
            worst = findm(features,CMB,target,sp,CMB,'min')
            # print(worst,CMB)
            if worst != None:
                nowmb = CMB.copy().remove(worst)
                if ifdep(features, worst, target, nowmb):
                    CMB.remove(worst)
    CF = list(set(featuresname)-set(CMB))
    first = True
    while oldCMB != CMB and not CF==[]:
        print(CF)
        oldCMB = CMB.copy()
        # print('target:',target)
        best = findm(features,CF,target,sp,CMB,'max')#输入为候选特征们、目标特征和当前MB子集
        # print("CMB:",CMB,"\nCF:",CF)
        if not ifdep(features,best,target,CMB):
            CMB.append(best)
            CF.remove(best)
        if oldCMB != CMB or first:
            first = False
            for f in CF.copy():  # 创建副本用于遍历
                if ifdep(features, f, target, CMB):
                    CF.remove(f)  # 修改原列表
            worst = findm(features,CMB,target,sp,CMB,'min')
            # print(worst,CMB)
            if worst != None:
                nowmb = CMB.copy().remove(worst)
                if ifdep(features, worst, target, nowmb):
                    CMB.remove(worst)
    return CMB

def SRMB_sp(target,features, CMB,k,sp):
    clist = list(set(features.columns.tolist())-set(CMB))
    Queue = searchk(features,clist,target,k,sp)
    MB = CMB
    for f in Queue:
        CMB = ESMB(f,features,)
        if target in CMB:
            MB.append(f)
    return MB

def EAMB_sp(target,features,k,sp):
    CMB = ESMB_sp(target,features,sp)
    MB = SRMB_sp(target,features, CMB,k,sp)
    return MB


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
# #'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh',
# datasets = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
datasets = ['dna', 'Factors','GLIOMA','Isolet','madelon','Movement_libras','Musk1','ORL','Orlraws10P','Pixel',  'Prostate_GE', 'Sorlie', 'Su','SRBCT','spambase','splice','Synthetic_control', 'TOX_171', 'Waveform','Wdbc',  'WarpPIE10P','WarpAR10P', 'Yeoh', 'Yale', ]#
#  #,'Wdbc','Synthetic_control',
base_url = r"./dataset/"
# def changetosinge(x):
#     return float(x)
#
def calculate_accuracy(predicted_labels, true_labels):
    # 确保预测结果和真实标签的长度相同
    if len(predicted_labels) != len(true_labels):
        raise ValueError("预测结果和真实标签的长度不一致。")
    predicted_labels = [int(x) for x in predicted_labels]
    true_labels = [int(x) for x in true_labels]
    # print(predicted_labels,'\n',true_labels)
    # 计算预测正确的数量
    correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)

    # 计算准确度
    accuracy = correct_count / len(true_labels)

    return accuracy
def othertest(time, result, X_test, y_test, X_train, y_train):
    global b_time_predict_probabity, k_time_predict_probabity, s_time_predict_probabity, r_time_predict_probabity, r_time_predict_labels, s_time_predict_labels, k_time_predict_labels, b_time_predict_labels
    # for i in range(time):
    #记录所有time次的预测概率，和预测标签
    b_time_predict_probabity = []
    k_time_predict_probabity = []
    s_time_predict_probabity = []
    r_time_predict_probabity = []
    b_time_predict_labels = []
    k_time_predict_labels = []
    s_time_predict_labels = []
    r_time_predict_labels = []
    nowpick = result  # 特征名为数字
    print(nowpick)
    y_train = ravel(y_train)
    print(y_train,X_train[nowpick])
    # **Naïve Bayes (朴素贝叶斯)**
    from sklearn.naive_bayes import GaussianNB
    bayesmodel = GaussianNB()
    bayesmodel.fit(X_train[nowpick], y_train)
    bpredicted_labels = bayesmodel.predict(X_test[nowpick])
    bprobabilities = bayesmodel.predict_proba(X_test[nowpick])
    # **k-Nearest Neighbors (k-最近邻算法)**:
    from sklearn.neighbors import KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train[nowpick], y_train)
    kpredicted_labels = knn_classifier.predict(X_test[nowpick])
    kprobabilities = knn_classifier.predict_proba(X_test[nowpick])
    # **Support Vector Machines (支持向量机)**:
    from sklearn.svm import SVC
    svm_classifier = SVC(probability=True)
    svm_classifier.fit(X_train[nowpick], y_train)
    spredicted_labels = svm_classifier.predict(X_test[nowpick])
    sprobabilities = svm_classifier.predict_proba(X_test[nowpick])
    # random Forest (随机森林)**:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train[nowpick], y_train)
    rpredicted_labels = rf_classifier.predict(X_test[nowpick])
    rprobabilities = rf_classifier.predict_proba(X_test[nowpick])
    #
    b_time_predict_probabity.append(bprobabilities)#记录n种分类器单次概率
    k_time_predict_probabity.append(kprobabilities)
    s_time_predict_probabity.append(sprobabilities)
    r_time_predict_probabity.append(rprobabilities)
    b_time_predict_labels.append(bpredicted_labels)
    k_time_predict_labels.append(kpredicted_labels)
    s_time_predict_labels.append(spredicted_labels)
    r_time_predict_labels.append(rpredicted_labels)
    predict_pro   = {'b': b_time_predict_probabity, 'k': k_time_predict_probabity, 's': s_time_predict_probabity, 'r': r_time_predict_probabity,}
    predict_lable = {'b': b_time_predict_labels, 'k': k_time_predict_labels, 's': s_time_predict_labels, 'r': r_time_predict_labels,}
    # print(predict_pro,predict_lable)
    # predict_pro   = {'b': b_time_predict_probabity, 'k': k_time_predict_probabity, 's': s_time_predict_probabity, 'r': r_time_predict_probabity,}
    # predict_lable = {'b': b_time_predict_labels, 'k': k_time_predict_labels, 's': s_time_predict_labels, 'r': r_time_predict_labels,}
    file_name1 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_pro.mat"
    file_name2 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)
base_url = r"./dataset/"
count = pd.DataFrame(index=range(len(datasets)), columns=range(1))
count.columns = ['eambsp']
count.index = datasets
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
    # print(data_processed,X.shape[1])
    # kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    # for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
        # if fold > 4:
        # print("\n当前处理到第", fold, "折")
    # result_dict = {}
    ifdep.call_out = 0
    for nowy in Class:
        print("nowy:", nowy)
        # data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
        # print(data_train.shape, data_train.columns)
        result = EAMB_sp(str(data_processed.shape[1]-1), data_processed, 0.15, nowy)
    methodname = 'eambsp'
    count[methodname][dataname] = ifdep.call_out
    print(count)
    count.to_excel('tables/' + 'eambspCIcount.xlsx', index=True)
        # result_dict[nowy] = set(result)
        # print("result:",result,)
    # 保存到txt文件

    # print(dataname)
    # print(result_dict)



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
#
# # 创建LabelEncoder实例
# result_dict = {}
# for i in df_encoded.columns:
#     print(i)
#     unique_y = df_encoded[i].unique()
#     print(unique_y)
#     for ay_val in unique_y:
#         result = EAMB_sp(i,df_encoded,30,ay_val)
#         dictresult = dict()
#         result_dict[i] = set(result)
# print(result_dict)

















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







