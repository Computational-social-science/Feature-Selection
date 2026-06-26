import time

import numpy as np
import pandas as pd
from scipy.special import gammaincc
from scipy.stats import chi2
from math import log
import  scipy.io as scio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import random
N_CASES_PER_DF = 5
alpha = 0.05
def k_greedy(dep, n_vars, k_tradeoff=0.5):
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


import numpy as np
from scipy.special import gammaincc
from math import log


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
        return dep


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
def LRH(df):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    target = n_vars - 1
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)

    mb = np.full(n_vars, 0, dtype=int)
    mb[target] = -1
    cond = [-1] * max_max_cond_size
    dep = np.zeros(n_vars)
    M1 = np.full(n_vars, -1, dtype=int)
    M2 = np.full(n_vars, -1, dtype=int)
    n_conds = 0

    # --- Growth Phase ---
    stop = False
    while not stop:
        print(111)
        stop = True
        stop3 = True
        K = 3

        if n_conds < max_max_cond_size:
            M1[:] = -1
            M2[:] = -1
            dep[:] = 0.0

            for i in range(n_vars):
                if mb[i] == 0:
                    dep[i] = compute_dep(i, target, cond, data, n_states, n_cases)
                    if dep[i] >= 1.0:
                        stop = False
                        M1[i] = 1

        if not stop:
            for i in range(n_vars):
                if M1[i] == 1:
                    belongtoM2 = True
                    for j in range(n_vars):
                        if M1[j] == 1 and j != i:
                            # cond[n_conds] = j
                            if compute_dep(i, j, cond, data, n_states, n_cases) >= 1.0:
                                cond[n_conds] = j
                                if compute_dep(i, target, cond, data, n_states, n_cases) <= -1.0:
                                    belongtoM2 = False
                                    break
                            cond[n_conds] = -1
                    if belongtoM2:
                        M2[i] = 1

            # Add top-K from M2
            stop2 = False
            while K > 0 and not stop2:
                print(222)
                stop2 = True
                maxdep = 0.0
                maxdep_index = -1
                for i in range(n_vars):
                    if M2[i] == 1 and dep[i] > maxdep:
                        maxdep = dep[i]
                        maxdep_index = i
                        stop2 = False

                if not stop2:
                    stop3 = False
                    if n_conds < max_max_cond_size:
                        M2[maxdep_index] = -1
                        mb[maxdep_index] = 1
                        cond[n_conds] = maxdep_index
                        n_conds += 1
                    K -= 1

            # If M2 为空，添加 M1 中依赖最大的点
            if stop3:
                maxdep = 0.0
                maxdep_index = -1
                for i in range(n_vars):
                    if M1[i] == 1 and dep[i] > maxdep:
                        maxdep = dep[i]
                        maxdep_index = i
                if maxdep_index != -1 and n_conds < max_max_cond_size:
                    mb[maxdep_index] = 1
                    cond[n_conds] = maxdep_index
                    n_conds += 1

    # --- Shrinking Phase ---
    dep[:] = 0.0
    for i in range(n_vars):
        if mb[i] == 1:
            for j in range(n_conds):
                if cond[j] == i:
                    for k in range(j, n_conds - 1):
                        cond[k] = cond[k + 1]
                    cond[n_conds - 1] = -1
                    break

            dep[i] = compute_dep(i, target, cond, data, n_states, n_cases)
            if dep[i] <= -1.0:
                n_conds -= 1
                mb[i] = 0
            else:
                cond[n_conds - 1] = i

    mb[mb == -1] = 0
    mb[target] = -1
    return [i for i in range(n_vars) if mb[i] == 1]
#
# ,'madelon','dexter','SRBCT','Colon','spambase','splice','dna', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171','CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',
datasets = ['Movement_libras']
# datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#

base_url = r"./dataset/"
import numpy as np

def compare_compute_dep(old_compute_dep, new_compute_dep, data, n_states, n_cases, num_tests=100, tol=1e-6):
    np.random.seed(42)  # 保证每次测试一样

    n_features = data.shape[1]
    matches = 0
    diffs = []

    for _ in range(num_tests):
        var = np.random.randint(0, n_features)
        target = np.random.randint(0, n_features)
        while target == var:
            target = np.random.randint(0, n_features)

        # 随机挑选一些条件变量（不包括var和target）
        cond_candidates = [i for i in range(n_features) if i != var and i != target]
        n_cond = np.random.randint(0, min(len(cond_candidates), 5))  # 最多5个条件
        cond = list(np.random.choice(cond_candidates, n_cond, replace=False))

        old_result = old_compute_dep(var, target, cond, data, n_states, n_cases)
        new_result = new_compute_dep(var, target, cond, data, n_states, n_cases)

        if np.isclose(old_result, new_result, atol=tol):
            matches += 1
        else:
            diffs.append({
                'var': var,
                'target': target,
                'cond': cond,
                'old_result': old_result,
                'new_result': new_result
            })

    print(f"✅ 测试总数：{num_tests}")
    print(f"🎯 完全一致的数量：{matches}")
    print(f"⚡ 不一致的数量：{len(diffs)}")
    if diffs:
        print("\n🚨 以下是不同的case示例：")
        for diff in diffs[:5]:  # 最多展示5个
            print(diff)

    return diffs

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
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
        print("\n当前处理到第", fold, "折")
        # if fold in [0,1,2]:
            # continue
        data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
        # n_cases, n_vars = data_train.shape
        # data = data_train.to_numpy().astype(int)
        # if isinstance(data_train, pd.DataFrame):
            # data_train = data_train.values
        # diffs = compare_compute_dep(compute_dep_old, compute_dep, data_train, n_states, n_cases)
        # start_time_old = time.time()  # 记录开始时间
        result = LRH(data_train)
        # end_time_old = time.time()  # 记录结束时间
        # elapsed_time_old = end_time_old - start_time_old  # 计算时间差
        # print("result:",elapsed_time_old,)
        np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        # 保存到txt文件
        file_name = r"FeatureSelect/" + 'LRH' + '/' + dataname  + "_result"+str(fold)+".txt"
        np.savetxt(file_name, result, fmt='%d')  # 使用savetxt()函数保存数组到文件


