import math

import numpy as np
import pandas as pd
from itertools import combinations

import scipy.io as scio
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


# 使用G²（卡方）检验评估条件依赖
# 返回 log(G²) 分数（若p值显著）或 -1.0（独立）
def compute_g2(X, Y, cond_vars, data, alpha=0.05):
    if len(cond_vars) == 0:
        table = pd.crosstab(data.iloc[:, X], data.iloc[:, Y])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return -1.0
        g2, p, dof, exp = chi2_contingency(table)
        return np.log(g2 + 1e-6) if p < alpha else -1.0
    else:
        grouped = data.groupby([data.columns[i] for i in cond_vars])
        scores = []
        for _, subset in grouped:
            if len(subset) < 5:
                continue
            try:
                table = pd.crosstab(subset.iloc[:, X], subset.iloc[:, Y])
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue
                g2, p, dof, exp = chi2_contingency(table)
                scores.append(np.log(g2 + 1e-6) if p < alpha else -1.0)
            except:
                continue
        return max(scores) if scores else -1.0

def compute_dep(var, target, cond):
    return compute_g2(var, target, cond, global_data)
def check_spouse_removal(s_y, last_added, Z, max_cond_size, k_cond, sep, sep_tag):
    n_codes = len(Z)
    for cond_size in range(min(k_cond, n_codes)+1):
        if cond_size + 1 > max_cond_size:
            continue
        idx = list(range(cond_size))
        while True:
            cond = [-1] * max_cond_size
            cond[0] = last_added
            for i in range(cond_size):
                cond[i+1] = Z[idx[i]]
            if compute_dep(s_y, last_added, [c for c in cond if c != -1]) <= -1.0:
                if not sep_tag[s_y]:
                    sep[s_y][:] = cond[:]
                    sep_tag[s_y] = 1
                return True
            if next_cond_index(n_codes, cond_size, idx):
                break
    return False
def min_dep(var, target, other, pc, sep2):
    candidates = [i for i in range(len(pc)) if pc[i] == 1 and i != var and i != target and i != other]
    best_dep = -1.0
    best_subset = []
    for r in range(min(len(candidates), max_max_cond_size) + 1):
        for subset in combinations(candidates, r):
            dep = compute_g2(var, target, list(subset), global_data)
            if dep > best_dep:
                best_dep = dep
                best_subset = list(subset)
            if dep <= -1.0:
                break
        if best_dep <= -1.0:
            break
    for i in range(max_max_cond_size):
        sep2[i] = best_subset[i] if i < len(best_subset) else -1
    return best_dep

def next_cond_index(n, k, idx):
    for i in reversed(range(k)):
        if idx[i] != i + n - k:
            break
    else:
        return 1
    idx[i] += 1
    for j in range(i + 1, k):
        idx[j] = idx[j - 1] + 1
    return 0

def report_mb(target, mb):
    print(f"Markov Blanket for target {target}: {[i for i, v in enumerate(mb) if v == 1]}")
def BAMB(target,k_condition=2):
    pc = np.full(n_vars, 0)
    mb = np.zeros(n_vars, dtype=int)
    dep = np.zeros(n_vars, dtype=float)
    sep = [[-1]*max_max_cond_size for _ in range(n_vars)]
    sep2 = [-1]*max_max_cond_size
    cond = [-1]*max_max_cond_size
    sep_tag = [0]*n_vars

    sp_dep_rank = [[] for _ in range(n_vars)]
    spouse = [[] for _ in range(n_vars)]
    pc_dep_rank = []

    for i in range(n_vars):
        if n_states[i] <= 1 or n_states[target] <= 1:
            pc[i] = -1

    rank = []
    for i in range(n_vars):
        if pc[i] == 0:
            dep[i] = min_dep(i, target, -1, pc, sep2)
            if dep[i] <= -1.0:
                pc[i] = -1
                sep_tag[i] = 1
                sep[i] = sep2[:]
            elif dep[i] >= 1.0:
                rank.append((i, dep[i]))

    rank = [i for i, _ in sorted(rank, key=lambda x: -x[1])]

    for last_added in rank:
        pc[last_added] = 1
        pc[last_added] = 0
        if min_dep(last_added, target, -1, pc, sep2) <= -1.0:
            sep[last_added] = sep2[:]
            pc[last_added] = -1
            continue
        else:
            pc[last_added] = 1
            pc_dep_rank.append(last_added)

        for j in range(n_vars):
            if pc[j] == 1 and j != last_added:
                pc[j] = 0
                dep[j] = min_dep(j, target, last_added, pc, sep2)
                pc[j] = 1
                if dep[j] <= -1.0:
                    sep[j] = sep2[:]
                    pc[j] = -1
                    if j in pc_dep_rank:
                        pc_dep_rank.remove(j)

        dep.fill(0.0)
        for j in range(n_vars):
            if j != target and pc[j] != 1:
                cond = sep[j][:]
                n_conds = sum(1 for x in cond if x != -1)
                if last_added not in cond and n_conds < max_max_cond_size:
                    cond[n_conds] = last_added
                    dep[j] = compute_dep(j, target, cond)
                    if dep[j] <= -1.0 and not sep_tag[j]:
                        sep[j] = cond[:]
                        sep_tag[j] = 1

        pre_sps = sorted([(i, d) for i, d in enumerate(dep) if d > 0], key=lambda x: -x[1])
        sp_dep_rank[last_added] = [i for i, _ in pre_sps]

        while sp_dep_rank[last_added]:
            y = sp_dep_rank[last_added].pop(0)
            spouse[last_added].append(y)

            for k in reversed(range(len(spouse[last_added]))):
                s_y = spouse[last_added][k]
                Z = [sp for idx, sp in enumerate(spouse[last_added]) if idx != k]
                if check_spouse_removal(s_y, last_added, Z, max_max_cond_size, k_condition, sep, sep_tag):
                    spouse[last_added].pop(k)

    for i in range(n_vars):
        mb[i] = 0
    for i in pc_dep_rank:
        mb[i] = 1
        for s in spouse[i]:
            mb[s] = 1

    report_mb(target, mb)
    return mb

# 示例数据设置（最后一列为标签）



datanames = ['Colon','SRBCT','madelon','spambase','splice','dna','dexter' ]#
max_max_cond_size = 100  #设置最大条件子集数量
N_CASES_PER_DF = 20  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy
# algorithm = "IAMB"
def changetosinge(x):
    return float(x)
for dataname in datanames:
    print(dataname)
    data_file_path = "./dataset/testdatasets/" + dataname + ".mat"
    data = scio.loadmat(data_file_path)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    if dataname == 'Dermatology':
        Special = X0.iloc[:, -1]
        X0 = X0.iloc[:, :-1]  # 读取训练数据
        a = np.array([item[0] for item in Special])
        label_encoder = LabelEncoder()
        a33 = label_encoder.fit_transform(a)
        X0[33] = a33
    # 将y标签控制在0-n
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    # 数据离散化
    X = pd.DataFrame()
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    y = pd.DataFrame(y_encoded)
    # t = len(Class)
    y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
    # print(y)

    data_processed = pd.concat([X, y], axis=1)
    n_vars = data_processed.shape[1]
    n_cases = data_processed.shape[0]
    n_states = np.full(n_vars, 5)
    n_states[-1] = len(Class)
    global_data = data_processed.copy()  # 设置全局共享数据
    # print(f"algorithm: {algorithm}, alpha: {alpha}, n_cases: {n_cases}, n_vars: {n_vars}")
    # 处理目标变量，初始化MB
    # targets = list(range(n_vars))[-1]
    targets = list(range(n_vars))[-1]
    # print(targets, '11')
    # start_time = time.time()  # 记录时间
    result = []
    # kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    classnum = len(Class)  # 类别数，分为几类
    samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
    # print(targets)
    # result = Inter_IAMB([targets])
    # print(12321,result)
    # for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
    #     print(X_train, X_test)
        # csmdccmrtest(len(result), result, X_test, y_test, X_train, y_train)
        # result = []  # 特征提取结果数组
        # for ck in Class:  # 为每一个类别进行特征选择      #X为纯数组，y为series
            # csmdccmr(X_train.values, np.ravel(y_train), ck, n_selected_features=time)/
        # result = IAMB(Class)
        # print(result)
        # np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        # 保存到txt文件
        # file_name = r"feature_select_result/" + 'INMB' + '/' + dataname + str(fold) + "_result.txt"
        # np.savetxt(file_name, np_result, fmt='%s', delimiter="\t")
        # np.savetxt(file_name, uni, delimiter =’, ’ fmt = ‘ %s’)
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件//
        # result = np.loadtxt(file_name)  # 加载特征提取结果
        # result = np.array(result)

    # accmax = []  # 准确度记录数组，为了画折线图
#     # 十折交叉验证
    result = BAMB(X.shape[1])
    print("result:", result, )

    # print(result)
    np_result = np.array(result, dtype=object)
    # 保存到txt文件
    file_name = r"feature_select_result/" + 'IAMB' + '/' + dataname + "_result-bamb.txt"
    np.savetxt(file_name, np_result, fmt='%s', delimiter="\t")
# 例：运行 EEMB/BAMB
# EEMB(target=n_vars-1, n_vars=n_vars, n_states=n_states, max_max_cond_size=3, k_cond=2)
# BAMB(target=n_vars-1, n_vars=n_vars, n_states=n_states, max_max_cond_size=3, k_condition=2)
