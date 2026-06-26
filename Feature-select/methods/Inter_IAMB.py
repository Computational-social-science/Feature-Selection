import random
import numpy as np
import pandas as pd
from scipy.special import gammaincc
from scipy.stats import chi2
from math import log
import  scipy.io as scio
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

N_CASES_PER_DF = 5
alpha = 0.05
k_tradeoff = 1.0  # 用于 Inter-IAMB 中 k_greedy 函数
def k_greedy(dep, n_vars):
    # Step 1: Identify candidates where dep[i] >= 1.0
    cand = [i for i in range(n_vars) if dep[i] >= 1.0]
    n_cands = len(cand)

    # Step 2: Adjust number of candidates to pick based on k_tradeoff
    n_cands = int(n_cands * k_tradeoff)
    if n_cands < 1:
        n_cands = 1

    # Step 3: Select the maximum element with a greedy approach
    max_index = cand[0]
    n_cands_left = len(cand)

    for i in range(n_cands):
        j = random.randint(0, n_cands_left - 1)

        if i == 0 or dep[cand[j]] >= dep[max_index]:
            if i != 0 and dep[cand[j]] == dep[max_index]:
                if max_index > cand[j]:
                    max_index = cand[j]
            else:
                max_index = cand[j]

        # Remove the selected candidate from the list
        cand.pop(j)
        n_cands_left -= 1

    return max_index
def gammq(a, x):
    """Equivalent to gammq in C++ using chi-squared survival function."""
    return chi2.sf(x, 2 * a)
# def compute_dep(var, target, cond, data, n_states, n_cases):
#     n_cond_states = 1
#     cond = [c for c in cond if c != -1]
#     for c in cond:
#         n_cond_states *= n_states[c]
#     # if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
#     #     return 0.0
#
#     ss_cond = np.zeros(n_cond_states, dtype=int)
#     ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
#     ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
#     ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)
#
#     for i in range(n_cases):
#         cond_state = 0
#         for c in cond:
#             cond_state = cond_state * n_states[c] + data[i, c]
#         ss_cond[cond_state] += 1
#         ss_target[cond_state][data[i, target]] += 1
#         ss_var[cond_state][data[i, var]] += 1
#         ss[cond_state][data[i, target]][data[i, var]] += 1
#
#     statistic = 0.0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             for j in range(n_states[target]):
#                 if ss_target[i][j] > 0:
#                     for k in range(n_states[var]):
#                         if ss[i][j][k] > 0:
#                             expected = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
#                             statistic += ss[i][j][k] * (log(ss[i][j][k]) - log(expected))
#     statistic *= 2.0
#
#     df = 0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             df_target = np.sum(ss_target[i] > 0)
#             df_var = np.sum(ss_var[i] > 0)
#             df += (df_target - 1) * (df_var - 1)
#     if df <= 0:
#         df = 1
#
#     dep = gammaincc(0.5 * df, 0.5 * statistic)
#     if dep <= alpha:
#         dep = 2.0 - dep
#         if dep == 2.0:
#             dep += 2.0 +statistic / df
#         return dep
#     else:
#         dep = -1.0 - dep
#         if dep == -2.0:
#             dep -= -2.0 -statistic / df
#         return dep

def compute_dep(var, target, cond, data, n_states, n_cases):
    # 清理cond列表
    cond = [c for c in cond if c != -1]
    n_cond_states = 1
    for c in cond:
        n_cond_states *= n_states[c]

    n_cond = len(cond)

    if n_cond == 0:
        cond_states = np.zeros(n_cases, dtype=int)
        n_cond_states = 1
    else:
        multipliers = np.ones(n_cond, dtype=int)
        for idx in range(n_cond - 1):
            multipliers[idx] = np.prod([n_states[c] for c in cond[idx + 1:]])
        cond_states = (data[:, cond] * multipliers).sum(axis=1)
        n_cond_states = np.prod([n_states[c] for c in cond])

    # 初始化统计量
    ss_cond = np.bincount(cond_states, minlength=n_cond_states)

    # 初始化 ss_target, ss_var, ss
    ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
    ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
    ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)

    # 用numpy的高级索引批量累加
    np.add.at(ss_target, (cond_states, data[:, target]), 1)
    np.add.at(ss_var, (cond_states, data[:, var]), 1)
    np.add.at(ss, (cond_states, data[:, target], data[:, var]), 1)

    statistic = 0.0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            ss_target_i = ss_target[i]
            ss_var_i = ss_var[i]
            ss_i = ss[i]
            for j in range(n_states[target]):
                if ss_target_i[j] > 0:
                    for k in range(n_states[var]):
                        if ss_i[j][k] > 0:
                            expected = (ss_target_i[j] * ss_var_i[k]) / ss_cond[i]
                            statistic += ss_i[j][k] * (log(ss_i[j][k]) - log(expected))
    statistic *= 2.0
    # print(var, target, cond, statistic)
    # 计算自由度
    df = 0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            df_target = np.count_nonzero(ss_target[i])
            df_var = np.count_nonzero(ss_var[i])
            df += (df_target - 1) * (df_var - 1)
    if df <= 0:
        df = 1

    # p值计算
    dep = gammaincc(0.5 * df, 0.5 * statistic)

    # 返回的规则，保持和你原来完全一致
    if dep <= alpha:
        dep = 2.0 - dep
        if dep == 2.0:
            dep += 2.0 + statistic / df
        return dep
    else:
        dep = -1.0 - dep
        if dep == -2.0:
            dep -= -2.0 - statistic / df
        return dep#返回的 dep 值越接近 2.0，表示结果越可能是独立的。

def estimate_max_cond_size(n_cases, n_states):
    min_n_states = min([s for s in n_states if s > 1])
    try:
        max_max_cond_size = int(
            log(n_cases / (N_CASES_PER_DF * (min_n_states - 1) ** 2)) / log(min_n_states)
        )
    except:
        max_max_cond_size = 0
    return max(max_max_cond_size, 3)
def changetosinge(x):
    return float(x)
def discretize_data(df):
    discretized_df = df.copy()

    for col in df.columns:
        unique_values = df[col].nunique()  # 获取每列数据的唯一值个数
        if unique_values >= 5:
            num = 6
        else:
            num = unique_values +1
        bins = np.linspace(df[col].min(), df[col].max(), num)  # 生成5个区间，6个切分点
        labels = list(range(num - 1))
        discretized_df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

    return discretized_df


def Inter_IAMB(df):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    target = n_vars - 1
    n_states = np.full(n_vars, 10)
    n_states[-1] = len(Class)
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)

    mb = np.full(n_vars, 0, dtype=int)
    removals = np.zeros(n_vars, dtype=int)
    cond = [-1] * max_max_cond_size
    dep = np.zeros(n_vars)
    mb[target] = -1
    n_conds = 0

    while True:
        stop = True
        if n_conds < max_max_cond_size:
            for i in range(n_vars):
                if mb[i] == 0:
                    dep[i] = compute_dep(i, target, cond, data, n_states, n_cases)
                    if dep[i] >= 1.0:
                        stop = False
        if stop:
            break
        aux = k_greedy(dep,n_vars)
        if aux == -1:
            break
        mb[aux] = 1
        cond[n_conds] = aux
        n_conds += 1

        # Shrinking immediately
        for i in range(n_vars):
            if mb[i] == 1 and removals[i] < 10:
                for j in range(n_conds):
                    if cond[j] == i:
                        for k in range(j, n_conds - 1):
                            cond[k] = cond[k + 1]
                        cond[n_conds - 1] = -1
                        break
                print(cond)
                dep[i] = compute_dep(i, target, cond, data, n_states, n_cases)
                if dep[i] <= -1.0:
                    n_conds -= 1
                    mb[i] = 0
                    removals[i] += 1
                else:
                    cond[n_conds - 1] = i

    return [i for i in range(n_vars) if mb[i] == 1]
# datasets = ['aaaaa']
# datasets = ['madelon','dexter','SRBCT','Colon','spambase','splice','dna','Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
datasets = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
            'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene','madelon','dexter','SRBCT','Colon','spambase','splice','dna',]#
# datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#

base_url = r"./dataset/"

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
    X = pd.DataFrame()
    #离散化并确定每个特征的取值
    n_states = []
    X = discretize_data(X0)
    # 打印每一列的唯一值
    for col in X.columns:
        n_states.append(X[col].max() + 1)
    n_states.append(y.nunique()[0])
    new_columns = [str(i) for i in range(X.shape[1] + 1)]  # 将列名转换为字符串
    X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))  # 重命名 X 的列名
    y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))  # 重命名 y 的列名
    data_processed = pd.concat([X, y], axis=1)

    # data = pd.read_csv('aaaaa.csv', encoding='utf_8_sig', index_col="𝑈")
    # C = data.iloc[:, -1]  # 类标签C type：series
    # Class = set(C)
    # label_encoder = LabelEncoder()
    # data_e = data.iloc[:, :].apply(lambda col: label_encoder.fit_transform(col))  # 对每列独
    # data_processed = pd.DataFrame(data_e)
    # # print(X)


    # kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    # for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
    # print("\n当前处理到第", fold, "折")
    # data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
    result =Inter_IAMB(data_processed)
    print("result:",result,)
    np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        # print(np_result)
        # 保存到txt文件
        # file_name = r"feature_select_result/" + 'New/Inter-IAMB' + '/0' + dataname  + "_result-"+str(fold)+".txt"
        # np.savetxt(file_name, result, fmt='%d')  # 使用savetxt()函数保存数组到文件