import numpy as np
import pandas as pd
from scipy.stats import chi2
from math import log
import  scipy.io as scio
from sklearn.preprocessing import LabelEncoder

N_CASES_PER_DF = 5
alpha = 0.05

def gammq(a, x):
    """Equivalent to gammq in C++ using chi-squared survival function."""
    return chi2.sf(x, 2 * a)

def compute_dep(var, target, cond, data, n_states, n_cases):
    n_cond_states = 1
    cond = [c for c in cond if c != -1]
    for c in cond:
        n_cond_states *= n_states[c]
    if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
        return 0.0

    ss_cond = np.zeros(n_cond_states, dtype=int)
    ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
    ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
    ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)

    for i in range(n_cases):
        cond_state = 0
        for c in cond:
            cond_state = cond_state * n_states[c] + data[i, c]
        ss_cond[cond_state] += 1
        ss_target[cond_state][data[i, target]] += 1
        # print(ss_var.shape,cond_state,data[i, var],i,var)
        ss_var[cond_state][data[i, var]] += 1
        ss[cond_state][data[i, target]][data[i, var]] += 1

    statistic = 0.0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            for j in range(n_states[target]):
                if ss_target[i][j] > 0:
                    for k in range(n_states[var]):
                        if ss[i][j][k] > 0:
                            expected = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
                            statistic += ss[i][j][k] * (log(ss[i][j][k]) - log(expected))
    statistic *= 2.0

    df = 0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            df_target = np.sum(ss_target[i] > 0)
            df_var = np.sum(ss_var[i] > 0)
            df += (df_target - 1) * (df_var - 1)
    if df <= 0:
        df = 1

    dep = gammq(0.5 * df, 0.5 * statistic)
    if dep <= alpha:
        dep = 2.0 - dep
        if dep == 2.0:
            dep += statistic / df
        return dep
    else:
        dep = -1.0 - dep
        if dep == -2.0:
            dep -= statistic / df
        return dep
def estimate_max_cond_size(n_cases, n_states):
    min_n_states = min([s for s in n_states if s > 1])
    try:
        max_max_cond_size = int(
            log(n_cases / (N_CASES_PER_DF * (min_n_states - 1) ** 2)) / log(min_n_states)
        )
    except:
        max_max_cond_size = 0
    return max(max_max_cond_size, 1)

def Fast_IAMB(df):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    target = n_vars - 1  # 最后一列为目标
    data_processed = data

    # 动态估计 max_max_cond_size
    # n_states = [len(np.unique(data[:, i])) for i in range(n_vars)]
    n_states = np.full(n_vars, 5)
    n_states[-1] = len(Class)
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)

    mb = np.full(n_vars, 0, dtype=int)
    removals = np.zeros(n_vars, dtype=int)
    cond = [-1] * max_max_cond_size
    S = [-1] * (n_vars - 1)
    dep = np.zeros(n_vars)
    mb[target] = -1

    n_conds = 0
    n_Ss = 0

    for i in range(n_vars):
        if mb[i] == 0:
            dep[i] = compute_dep(i, target, cond, data_processed, n_states, n_cases)
            if dep[i] >= 1.0:
                S[n_Ss] = i
                n_Ss += 1

    stop = False
    while not stop:
        if n_Ss == 0:
            break
        # 按 dep 降序排列 S
        S = [s for s in S if s != -1]
        S = sorted(S, key=lambda x: -dep[x]) + [-1] * (n_vars - 1 - len(S))

        insufficient_data = False
        for i in range(n_vars - 1):
            if S[i] == -1:
                break
            B_size = np.prod([n_states[j] for j in range(n_vars) if mb[j] == 1]) if np.any(mb == 1) else 1.0
            if n_cases / (B_size * n_states[target] * n_states[S[i]]) > 5:
                mb[S[i]] = 1
                cond[n_conds] = S[i]
                n_conds += 1
            else:
                insufficient_data = True
                break

        removal = False
        dep[:] = 0.0
        for i in range(n_vars):
            if mb[i] == 1:
                if i in cond:
                    cond = [c for c in cond if c != i] + [-1]
                    n_conds -= 1
                dep[i] = compute_dep(i, target, cond, data_processed, n_states, n_cases)
                if dep[i] <= -1.0:
                    mb[i] = 0
                    removals[i] += 1
                    removal = True
                else:
                    cond[n_conds] = i
                    n_conds += 1

        if not removal and insufficient_data:
            stop = True
        else:
            S = [-1] * (n_vars - 1)
            n_Ss = 0
            for i in range(n_vars):
                if mb[i] == 0 and i != target:
                    dep[i] = compute_dep(i, target, cond, data_processed, n_states, n_cases)
                    if dep[i] >= 1.0 and removals[i] < 10:
                        S[n_Ss] = i
                        n_Ss += 1

    mb[mb == -1] = 0
    mb[target] = -1
    return [i for i in range(n_vars) if mb[i] == 1]
datasets = ['Colon','SRBCT','dexter','madelon','spambase','splice','dna' ]#
base_url = r"./dataset/testdatasets/"
def changetosinge(x):
    return float(x)

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
    #离散化
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    # 将列名命名为字符串
    new_columns = [str(i) for i in range(X.shape[1] + 1)]  # 将列名转换为字符串
    X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))  # 重命名 X 的列名
    y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))  # 重命名 y 的列名
    data_processed = pd.concat([X, y], axis=1)

    result_dict = {}
    result = Fast_IAMB(data_processed)
    print(dataname)
    print(result)
    result_array = np.array(result)
    file_name = r"feature_select_result/" + 'New/Fast-IAMB' + '/' + dataname  + "_result.txt"
    np.savetxt(file_name, result_array, fmt='%d')  # 使用savetxt()函数保存数组到文件


