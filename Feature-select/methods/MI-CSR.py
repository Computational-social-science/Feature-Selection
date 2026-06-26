import warnings
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import scipy.io as scio
# from skfeature.utility.entropy_estimators import midd
from sklearn.feature_selection import mutual_info_classif
# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from skfeature.utility.entropy_estimators import *
from sklearn.preprocessing import LabelEncoder

from main_X import *

def middspecial(x, y, c):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """
    return -entropyed(list(zip(x, y)),c)+entropydpro(list(zip(x, y)),c)+entropyed(y,c)
def middspecialpro(x, y, c):
    """
    Discrete mutual information estimator given a list of samples which can be any hashable object
    """

    t = list(zip(*x.values, y))
    # print(-entropydpro(t,c))
    return -entropydpro(t,c)+entropyed(y,c)
def entropyed(sx,sy,base=2):
    return entropyfromprobs(histspecial(sx,sy), base=base)
def entropydpro(sx,sy,base=2):
    sumnum = 0
    d = dict()
    ds = dict()
    for s in sx:
        s = s[:-1]
        d[s] = d.get(s, 0) + 1
    for s in sx:
        # 让最后一个是需要指定的，就可以无限循环了
        if s[-1] == sy:
            ds[s] = ds.get(s, 0) + 1
    for p in ds:
        # print(p,d.get(p,1),ds.get(p[:-1], 2))
        pda = ds.get(p, 0)/d.get(p[:-1], 0)
        sumnum += d.get(p[:-1],0)/len(sx) * pda * log(pda)
    return -sumnum/log(base)
def histspecial(sx,sy):
    # Histogram from list of samples
    d = dict()
    for s in sx:
        if isinstance(s, tuple):
            nowy = s[1]
        else:
            nowy = s
        if (nowy == sy):
            d[s] = d.get(s, 0) + 1
    # print(d.values())
    return map(lambda z: float(z)/len(sx), d.values())
#
#
import math
from collections import Counter

# def middspecialpro(x, y, c):
#     """
#     Discrete mutual information estimator given a list of samples which can be any hashable object
#     """
#     t = list(zip(x, y))
#     return -entropyed(t, c) + entropydpro(t, c) + entropyed(y, c)
#
# def entropyed(sx, sy, base=2):
#     return entropyfromprobs(histspecial(sx, sy), base=base)
#
# def entropydpro(sx, sy, base=2):
#     sumnum = 0
#     d = Counter(s[:-1] for s in sx)
#     ds = Counter(s for s in sx if s[-1] == sy)
#     for p in ds:
#         if d[p[:-1]] > 0:  # Avoid log(0)
#             sumnum += ds[p] / len(sx) * log(d[p[:-1]] / len(sx))
#     return -sumnum/log(base)




def Core(data,C,d):
    MI = middspecialpro(data[C].T,label,d)
    result = []
    for c in C:
        x=0
        newC = [element for element in C if element != c]
        # print(data[newC].T)
        # print(data[newC])
        x = middspecialpro(data[newC].T,label,d)
        if x < MI:
            result.append(c)
    return result

def SigMI(data,R,c,d):
    newR = list(R)
    t2 = middspecialpro(data[R].T, label, d)
    newR.append(c)
    t1 = middspecialpro(data[newR].T,label,d)
    return t1-t2

# data = pd.DataFrame({
#     0: [0,0,0,1,0,1,1,1,1,],
#     1: [1,1,1,0,0,0,0,0,1,],
#     2: [3,3,3,1,1,2,1,1,2,],
#     3: [0,0,2,0,0,2,0,0,2,],
#     4: [0,0,2,1,1,0,2,2,1,],
#     'd': [1,1,1,2,2,2,3,3,3]
# })
# data = pd.DataFrame({
#     0: [1, 0, 0, 0, 0, 0, 0, ],
#     1: [0,1,0,1,0,2,1,],
#     2: [0,1,1,1,1,1,0,],
#     'd': [1,1,1,2,2,2,2]
# })
# X = data.iloc[:,:-1]
# y = data['d']
kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
datanames = [  'Yale', ]#'WarpAR10P', 'GLIOMA', 'Arcene','Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet', 'ORL', 'WarpPIE10P', 'Su',
            # 'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111', 'Yeoh' 'Musk1', 'Sorlie',
for dataname in datanames:
    print("====================================================================================================")
    print(dataname)
    # dataname = 'Wdbc'
    base_url = r"./dataset/"
    data_url = dataname + '.mat'
    # dataname = "movement_libras"
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    # print(data)
    # print(X0,y0)
    # 将y标签控制在0-n
    # print(X0,y0)
    def changetosinge(x):
        return float(x)
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    y = pd.DataFrame(y_encoded)
    # 数据离散化
    X = pd.DataFrame()
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(X0[33])
    # data = {33:y_encoded}
    # newf = pd.DataFrame(data)
    # X0[33]=newf[33]
    # print(X0,y)
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
        print("当前处理到第", fold, "折\n")
        bacc, kacc, sacc, racc = [], [], [], []
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
        result = []  # 特征提取结果数组
        C = list(set(X_train.columns))
        # print(C)
        label = list(y_train[0])
        # label = list(y)
        D = list(set(label))
        # print(len(t),len(t1))
        # print(D,C,X,y)
        result = []
        for i in D:
            R = []
            R = Core(X,C,i)
            # print(i,R)
            MI = middspecialpro(X[C].T,label,i)
            # print(MI)
            # # print(X[R].T,label)
            # print(555,X[:,5])
            if  not R:
                # print(C)
                mi = np.zeros(len(C))
                for q in C:
                    # print(q)
                    f = X[q]
                    # print(f)
                    mi[q] = midd(f, label)
                idx = np.argmax(mi)
                R.append(idx)
            # print(X,11111111111111111111111111)
            while middspecialpro(X[R].T,label,i) < MI:
                C_R = [element for element in C if element not in R]
                # print("C-R",C_R)
                # print(R)
                sig = np.zeros(len(C_R))
                for j in range(len(C_R)):
                    sig[j] = SigMI(X,R,C_R[j],i)
                idx = np.argmax(sig)
                R.append(C_R[idx])
            newmi = middspecialpro(X[R].T,label,i)
            oR = list(R)
            nR = list(R)
            for r in oR:
                nowR = [element for element in nR if element != r]
                if middspecialpro(X[nowR].T,label,i) == newmi :
                    R.remove(r)
            print(i,":    ",R)
            result.append(R)
        file_name = r".\feature_select_result\MI-CSR/" + dataname + str(fold) + "_result.txt"
        np.savetxt(file_name, result, fmt='%s')  # 使用savetxt()函数保存数组到文件
        #
    #
    #
#
# import pandas as pd
# import numpy as np
#
# # Given dataset
# data = pd.DataFrame({
#     1: [0, 0, 0, 1, 0, 1, 1, 1, 1],
#     2: [1, 1, 1, 0, 0, 0, 0, 0, 1],
#     3: [3, 3, 3, 1, 1, 2, 1, 1, 2],
#     4: [0, 0, 2, 0, 0, 2, 0, 0, 2],
#     5: [0, 0, 2, 1, 1, 0, 2, 2, 1],
#     'd': [1, 1, 1, 2, 2, 2, 3, 3, 3]
# })
#
# # Probability of each class Dj
# p_Dj = data['d'].value_counts(normalize=True).sort_index()
# p_Dj
#
#
# # Calculate conditional probabilities p(A_i | D_j)
# conditional_probs = {}
# for col in data.columns[:-1]:
#     conditional_probs[col] = data.groupby('d')[col].value_counts(normalize=True).unstack().fillna(0)
#
# conditional_probs
#
# # Calculate p(A_i | D_j) * log2(p(A_i | D_j))
# entropy_components = {}
# for col in conditional_probs:
#     entropy_components[col] = conditional_probs[col] * np.log2(conditional_probs[col])
#     entropy_components[col] = entropy_components[col].replace(-np.inf, 0)  # Replace -inf with 0 for log2(0)
#
# entropy_components
#
# # Calculate p(A_i | D_j) * log2(p(A_i | D_j))
# entropy_components = {}
# for col in conditional_probs:
#     entropy_components[col] = conditional_probs[col] * np.log2(conditional_probs[col])
#     entropy_components[col] = entropy_components[col].replace(-np.inf, 0)  # Replace -inf with 0 for log2(0)
#
# entropy_components
#
# # Sum weighted by p(D_j)
# weighted_entropy = sum(p_Dj[d] * entropy_components[col].sum(axis=1)[d] for d in p_Dj.index for col in entropy_components)
# weighted_entropy


