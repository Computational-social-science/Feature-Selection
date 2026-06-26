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
def compute_dep(var, target, cond, data, n_states, n_cases):
    n_cond_states = 1
    cond = [c for c in cond if c != -1]
    for c in cond:
        n_cond_states *= n_states[c]
    # if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
    #     return 0.0

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

    dep = gammaincc(0.5 * df, 0.5 * statistic)
    if dep <= alpha:
        dep = 2.0 - dep
        if dep == 2.0:
            dep += 2.0 +statistic / df
        return dep
    else:
        dep = -1.0 - dep
        if dep == -2.0:
            dep -= -2.0 -statistic / df
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
def FBED(df):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    target = n_vars - 1
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)

    mb = np.full(n_vars, 0, dtype=int)
    cond = [-1] * max_max_cond_size
    R = np.zeros(n_vars, dtype=int)
    dep = np.zeros(n_vars)
    mb[target] = -1
    n_conds = 0

    K = 1
    K_cur = 0

    # --- Growth Phase ---
    stop = False
    while K_cur <= K and not stop:
        stop = True

        if n_conds < max_max_cond_size:
            # 初始化候选集合 R
            n_Rs = 0
            for i in range(n_vars):
                if mb[i] == 0:
                    R[i] = 1
                    n_Rs += 1

            tag = True
            while n_Rs > 0 and tag:
                tag = False#record a single round of growth phase, whether there is node and T dependent under the condition of cond
                if n_conds < max_max_cond_size:
                    for i in range(n_vars):
                        if R[i] == 1:
                            dep[i] = compute_dep(i, target, cond, data, n_states, n_cases)
                            if dep[i] >= 1.0:
                                tag = True
                                stop = False

                    if tag:
                        aux = k_greedy(dep,n_vars)
                        mb[aux] = 1
                        cond[n_conds] = aux
                        n_conds += 1
                        for i in range(n_vars):
                            if i == aux or dep[i] <= -1.0:
                                R[i] = 0
                                n_Rs -= 1
        K_cur += 1

    # --- Shrinking Phase ---
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


# datasets = ['madelon','dexter','SRBCT','Colon','spambase','splice','dna','Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
# datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
datasets = ['Movement_libras']
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
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
        print("\n当前处理到第", fold, "折")
        data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
        result = FBED(data_train)
        print("result:",result,)
        np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        # 保存到txt文件
        file_name = r"FeatureSelect/" + 'FBED/' + dataname + "_result" + str(fold) + ".txt"
        np.savetxt(file_name, result, fmt='%d')  # 使用savetxt()函数保存数组到文件


