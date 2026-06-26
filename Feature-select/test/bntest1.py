import json
import time

import numpy as np
import pandas as pd
from scipy.special import gammaincc
from scipy.stats import chi2
# from sklearn.model_selection import KFold
import  scipy.io as scio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import random
from scipy.stats import chi2_contingency
# from skfeature.utility.entropy_estimators import cmidd

from csmdccmr import middspecial, redundancy, entropydpro, entropyed
import pandas as pd
# from skfeature.utility.entropy_estimators import cmidd
from csmdccmr import middspecial, redundancy

# from eamb import EAMB
# from IAMB import IAMB
# from Fast_IAMB import Fast_IAMB
# from Inter_IAMB import Inter_IAMB
# from FBED import FBED
# from LRH import LRH
N_CASES_PER_DF = 5
alpha = 0.05
citest = 0
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
from eamb import ESMB,SRMB

def compute_dep(var, target, cond, data, n_states, n_cases):
    # compute_dep.call_count += 1
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
#EAMB Algorithm
from scipy.stats import chi2_contingency
from skfeature.utility.entropy_estimators import cmidd

from csmdccmr import middspecial, redundancy, entropydpro, entropyed
import pandas as pd
from skfeature.utility.entropy_estimators import cmidd
from csmdccmr import middspecial, redundancy

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

def ifdep(dataframe, C, Y, X, alpha=0.05):
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
    # ifdep.call_count +=1
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
            # w = 1 - p_ay
            if s.empty:
                # info += w * middspecial(x, y, ay_val)
                info +=  middspecial(x, y, ay_val)
            else:
                # info += w * cmiddspecial_fixed(x, y, s, 'Y', ay_val)
                info += cmiddspecial_fixed(x, y, s, 'Y', ay_val)
    return info

def findm(data,Cfeature,target,CMB,find):#输入为候选特征们、目标特征和当前MB子集
    nowf = None
    if find == 'max':
        maxinfo = 0
        for f in Cfeature:
            info = compinfo(data[f],data[target],s=data[CMB])
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
            info = compinfo(data[f], data[target], s=data[nowfind])
            if info <= mininfo:
                mininfo = info
                nowf = f
    return nowf

def searchk(features,CQ,target,k):
    column_values = {col: compinfo(features[col],features[target]) for col in CQ}
    # 选出前k个最大的数据，并将列名降序排序
    num = int(len(CQ) * k * 0.01)
    num = max(1, num)  # 至少保留一列
    Q = sorted(column_values.keys(), key=lambda x: column_values[x], reverse=True)[:num]
    return Q

def ESMB( target,features,):
    CMB = []
    data = features
    featuresname = data.columns.tolist()
    # print(featuresname,target)

    CF = featuresname.copy()
    CF.remove(target)
    oldCMB = ['fss']
    first = True
    while oldCMB != CMB and not CF==[]:
        # print(1,CF)
        oldCMB = CMB.copy()
        best = findm(features,CF,target,CMB,'max')#输入为候选特征们、目标特征和当前MB子集
        # print("CMB:",CMB,"\nCF:",CF)
        if not ifdep(features,best,target,CMB):
            CMB.append(best)
            CF.remove(best)
        if oldCMB != CMB or first :
            for f in CF.copy():  # 创建副本用于遍历
                if ifdep(features, f, target, CMB):
                    CF.remove(f)  # 修改原列表
            worst = findm(features,CMB,target,CMB,'min')
            nowmb = CMB.copy().remove(worst)
            if ifdep(features,worst,target,nowmb):
                CMB.remove(worst)
    CF = list(set(featuresname)-set(CMB))
    CF.remove(target)
    first = True
    while oldCMB != CMB and not CF==[]:
        # print(2,CF)
        oldCMB = CMB.copy()
        best = findm(features,CF,target,CMB,'max')#输入为候选特征们、目标特征和当前MB子集
        # print("CMB:",CMB,"\nCF:",CF)
        if not ifdep(features,best,target,CMB):
            CMB.append(best)
            CF.remove(best)
        if oldCMB != CMB or first :
            for f in CF.copy():  # 创建副本用于遍历
                if ifdep(features, f, target, CMB):
                    CF.remove(f)  # 修改原列表
            worst = findm(features,CMB,target,CMB,'min')
            nowmb = CMB.copy().remove(worst)
            if ifdep(features,worst,target,nowmb):
                CMB.remove(worst)
    return CMB

def SRMB(target,features, CMB,k):
    clist = list(set(features.columns.tolist())-set(CMB))
    Queue = searchk(features,clist,target,k)
    MB = CMB
    for f in Queue:
        CMB = ESMB(f,features)
        if target in CMB:
            MB.append(f)
    return MB

def EAMB(target,features,k):
    CMB = ESMB(target,features)
    MB = SRMB(target,features, CMB,k)
    return MB

import numpy as np

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
def LRH(df,target):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    # target = n_vars - 1
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)
    target = df.columns.get_loc(target)
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
        # print(111)
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
                # print(222)
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
def EAMB(target,features,k):
    CMB = ESMB(target,features)
    MB = SRMB(target,features, CMB,k)
    return MB
def IAMB(df,target):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    # target = n_vars - 1
    # n_states = data.unique()
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)
    target = df.columns.get_loc(target)
    mb = np.full(n_vars, 0, dtype=int)
    cond = [-1] * max_max_cond_size
    dep = np.zeros(n_vars)
    mb[target] = -1
    n_conds = 0

    # Growth phase
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
        mb[aux] = 1
        cond[n_conds] = aux
        n_conds += 1

    # Shrinkage phase
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

    return [i for i in range(n_vars) if mb[i] == 1]
def Inter_IAMB(df,target):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    # target = n_vars - 1
    # n_states = np.full(n_vars, 10)
    # n_states[-1] = len(Class)
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)
    target = df.columns.get_loc(target)
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
def Fast_IAMB(df,target):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    # target = n_vars - 1  # 最后一列为目标
    data_processed = data

    # 动态估计 max_max_cond_size
    # n_states = [len(np.unique(data[:, i])) for i in range(n_vars)]
    # n_states = np.full(n_vars, 6)
    # n_states[-1] = len(Class)
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)
    target = df.columns.get_loc(target)
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
    #记录不独立特征
    stop = False
    while not stop:
        if n_Ss == 0:
            break
        # 按 dep 降序排列 S
        S = [s for s in S if s != -1]
        S = sorted(S, key=lambda x: -dep[x]) + [-1] * (n_vars - 1 - len(S))

        insufficient_data = False
        for i in range(n_vars - 1):
            if S[i] == -1:#所有特征都已经筛选过了
                break
            # B_size = np.prod([n_states[j] for j in range(n_vars) if mb[j] == 1]) if np.any(mb == 1) else 1.0
            # if n_cases / (B_size * n_states[target] * n_states[S[i]]) > 5:
            if n_conds < max_max_cond_size:
                mb[S[i]] = 1
                cond[n_conds] = S[i]
                n_conds += 1
            else:
                insufficient_data = True
            # break

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
def FBED(df,target):
    data = df.to_numpy().astype(int)
    n_cases, n_vars = data.shape
    # target = n_vars - 1
    max_max_cond_size = estimate_max_cond_size(n_cases, n_states)
    target = df.columns.get_loc(target)
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
def FSMB(data: pd.DataFrame,target, alpha=0.01) -> Set[int]:
    n_cases, n_vars = data.shape
    # target = n_vars - 1
    variables = set(range(n_vars)) - {target}
    target = data.columns.get_loc(target)

    # --- Phase 1: PC discovery ---
    CPCT = set()
    dep_scores = {x: compinfo(data[x], data[target]) for x in variables}
    sorted_vars = sorted(dep_scores, key=lambda x: -dep_scores[x])
    # print(111111111111111,)
    for X in sorted_vars[:10]:
        print(X)
        CPCT.add(X)
        for r in range(len(CPCT)):
            for Z in itertools.combinations(CPCT - {X}, r):
                if ifdep(data, X,target, list(Z)):
                    CPCT.remove(X)
                    break
            else:
                continue
            break
    # print(333)
    PCT = CPCT.copy()
    CSPT = dict()
    # --- Phase 2: SP discovery & refinement ---
    for Y in list(PCT):
        CSPT[Y] = set()
        candidates = variables - PCT
        for X in candidates:
            if not ifdep(data,X, target,[Y]):
                CSPT[Y].add(X)

        # First screening
        sorted_CSP = sorted(CSPT[Y], key=lambda x: compinfo(data[x], data[Y]))
        for X in list(sorted_CSP):
            for r in range(len(CSPT[Y]) + 1):
                for Z in itertools.combinations(CSPT[Y] | {target} - {X}, r):
                    if ifdep(data,X, Y, list(Z)):
                        CSPT[Y].remove(X)
                        break
                else:
                    continue
                break
        print(list(CSPT[Y]))
        # Second screening
        for X in list(CSPT[Y]):
            for r in range(len(PCT | CSPT[Y]) + 1):
                for Z in itertools.combinations((PCT | CSPT[Y]) - {X, Y}, r):
                    if ifdep(data,X, target, list(Z) + [Y]):
                        CSPT[Y].remove(X)
                        break
                else:
                    continue
                break
        # print(444)
        # Remove false PC
        for F in list(PCT):
            for r in range(len(PCT)):
                for Z in itertools.combinations(PCT - {F}, r):
                    if ifdep(data,F, target, list(Z) + list(CSPT[Y])):
                        PCT.remove(F)
                        CSPT[F] = set()
                        break
                else:
                    continue
                break

    SPT = set().union(*CSPT.values())
    MB = PCT.union(SPT)
    return MB
class GSMB:
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        :param data: Pandas DataFrame，每列为一个变量，每行为一个样本
        :param alpha: 显著性阈值（默认0.05）
        """
        self.data = data
        self.time = 0
        self.alpha = alpha
        self.variables = data.columns.tolist()
        # print(self.variables,data.columns.tolist())

    def is_conditionally_independent(self, X: str, Y: str, S: list) -> bool:
        """
        卡方条件独立性检验
        :param X: 变量1
        :param Y: 变量2
        :param S: 条件变量列表
        :return: True若独立，False若依赖
        """
        self.time += 1
        if not S:
            # 无条件集时直接检验X和Y的独立性
            contingency = pd.crosstab(self.data[X], self.data[Y])
            _, p, _, _ = chi2_contingency(contingency)
            return p >= self.alpha
        else:
            # 按条件集S分组后合并卡方统计量（Fisher's方法）
            p_values = []
            for _, group in self.data.groupby(S,observed=False):
                contingency = pd.crosstab(group[X], group[Y])
                if contingency.size > 0 and contingency.values.sum() > 0:
                    _, p, _, _ = chi2_contingency(contingency)
                    p_values.append(p)

            if not p_values:
                return True  # 无有效分组，默认独立

            # 合并p值（Fisher's方法）
            combined_stat = -2 * np.sum(np.log(p_values))
            df = 2 * len(p_values)
            combined_p = 1 - chi2.cdf(combined_stat, df)  # 使用chi2.cdf而非chi2_contingency.pdf
            return combined_p >= self.alpha

    def compute_markov_blanket(self, target: str) -> list:
        """
        计算目标变量的马尔可夫毯
        :param target: 目标变量名
        :return: 马尔可夫毯变量列表
        """
        MB = []
        remaining_vars = [v for v in self.variables if v != target]

        # === 增长阶段 ===
        while True:
            # 找到与target在给定MB下依赖的变量
            dependent_vars = []
            for Y in remaining_vars:
                if not self.is_conditionally_independent(target, Y, MB):
                    dependent_vars.append(Y)

            if not dependent_vars:
                break

            # 随机选择一个依赖变量（原始GSMB策略）
            Y = np.random.choice(dependent_vars)
            MB.append(Y)
            remaining_vars.remove(Y)

        # === 收缩阶段 ===
        for Y in MB.copy():
            cond_set = [x for x in MB if x != Y]
            if self.is_conditionally_independent(target, Y, cond_set):
                MB.remove(Y)

        return MB

    def fit(self) -> dict:
        """
        计算所有变量的马尔可夫毯
        :return: 字典 {变量: MB列表}
        """
        return {var: self.compute_markov_blanket(var) for var in self.variables},self.time
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
#'Isolet','EAMB',EAMB,
methodsname = ['FSMB',]#,,,,'FBED','LRH''IAMB','Inter_IAMB','Fast_IAMB','GSMB','FBED','LRH','IAMB',
methods = [FSMB,]#,,,,FBEDLRHIAMB,,Inter_IAMB,Fast_IAMB,GSMB,FBED,LRH,IAMB
# ,'madelon','dexter','SRBCT','Colon','spambase','splice','dna', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171','CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',
# datasets = [ 'madelon','dexter','SRBCT','Colon','spambase','splice','dna', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171','CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]
# datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
# datasets = [ 'Arcene','Authorship','CLL_SUB_111','Colon','Dermatology','dexter','dna', 'Factors','GLIOMA','Isolet','madelon','Movement_libras','Musk1','ORL','Orlraws10P','Pixel',  'Prostate_GE', 'Sorlie', 'Su','SRBCT','spambase','splice','Synthetic_control', 'TOX_171', 'Waveform','Wdbc',  'WarpPIE10P','WarpAR10P', 'Yeoh', 'Yale', ]#
datasets = ["insurance","pigs","child","hepar2","alarm",]#
base_url = r"./dataset/bnnet/"
import numpy as np


count = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
count.columns = methodsname
count.index = datasets
for dataname in datasets:
    for i in ["1000","5000"]:
        print(dataname)
        data_url = dataname + i+'.csv'
        url = base_url + data_url  # 数据路径
        data = pd.read_csv(url)  # 读取数据文件

        # X0 = pd.DataFrame(data['X'])  # 读取训练数据
        # y0 = pd.DataFrame(data['Y'])  # 读取标签
        # print(dataname)
        # print("====================================================================================================")
        # #=======================Dermatology==================
        # if dataname == 'Dermatology':
        #     Special = X0.iloc[:,-1]
        #     # print(Special.values)
        #     X0 = X0.iloc[:,:-1]  # 读取训练数据
        #     a = np.array([item[0] for item in Special])
        #     label_encoder = LabelEncoder()
        #     a33 = label_encoder.fit_transform(a)
        #     X0[33] = a33
        # #=======================Dercomatoly==================
        # # 将y标签控制在0-n
        # # 应用函数到 DataFrame 的每个元素
        # X0 = X0.applymap(changetosinge)
        # y0 = y0.applymap(changetosinge)
        encoders = {}

        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])  # +1 使编码从1开始
                encoders[col] = le  # 保存编码器以备后续使用
        # print("over deal")

        # y = pd.DataFrame(y_encoded)
        # X = pd.DataFrame()
        #离散化并确定每个特征的取值
        n_states = []
        # X = discretize_data(X0)
        # 打印每一列的唯一值
        for col in data.columns:
            n_states.append(data[col].max() + 1)
        # print(data)
        # n_states.append(y.nunique()[0])
        # new_columns = [str(i) for i in range(X.shape[1] + 1)]  # 将列名转换为字符串
        # X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))  # 重命名 X 的列名
        # y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))  # 重命名 y 的列名
        # data_processed = pd.concat([X, y], axis=1)
        for method in methods:
            # compute_dep.call_count = 0
            if method == GSMB :
                gsmb = GSMB(data, alpha=0.05)
                result,time = gsmb.fit()
                # result = method(data_processed)# data_processed, 0.1
                methodname = method.__name__
                # count[methodname][dataname] = int(time)
                print(result)
                path = "./bnresult/"
                with open(path+methodname+"_"+dataname+i+'=.json', 'w') as f:
                    json.dump(result, f, indent=2)
            elif method == EAMB:
                result = {}
                for var in data.columns:
                    Class = set(data[var])
                    result[var] = method(var,data, 0.1)
                # # print(result)
                # # 假设df是您的DataFrame
                # column_names = data.columns.tolist()
                # result_dict = {}
                # for key, indices in result.items():
                #     # 将每个数字索引映射为对应的列名
                #     mapped_names = [column_names[idx] for idx in indices]
                #     result_dict[key] = mapped_names
                print(result)
                # 然后使用上面的方法进行映射
                path = "./bnresult/"
                methodname = method.__name__
                with open(path + methodname + "_" + dataname + i + '.json', 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                result = {}
                for var in data.columns:
                    Class = set(data[var])
                    result[var] = method(data,var)
                # print(result)
                # 假设df是您的DataFrame
                column_names = data.columns.tolist()
                result_dict = {}
                for key, indices in result.items():
                    # 将每个数字索引映射为对应的列名
                    mapped_names = [column_names[idx] for idx in indices]
                    result_dict[key] = mapped_names
                print(result_dict)
                # 然后使用上面的方法进行映射
                path ="./bnresult/"
                methodname = method.__name__
                with open(path+methodname+"_"+dataname+i+'.json', 'w') as f:
                    json.dump(result, f, indent=2)
            # count.to_excel('tables/' + 'gsmbCIcount.xlsx', index=True)
    # kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    # for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
    #     print("\n当前处理到第", fold, "折")
    #     if fold == 0:
    #         continue
    #         data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
    #         print(data_train)
    #         for method in methods:
    #             print(method.__name__)
    #             result = method(str(X.shape[1]), data_train, 0.1)
    #             # print(ifdep.call_count,111)
    #             methodname = method.__name__
    #             # print(methodname)
    #             count[methodname][dataname] = int(ifdep.call_count)
    #             print(count)
    #             count.to_excel('tables/' + 'CIcounteamb4.xlsx', index=True)
    #             print(int(compute_dep.call_count),12345)
    #         np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
    #         # 保存到txt文件
    #         file_name = r"FeatureSelect/" + 'LRH' + '/' + dataname  + "_result"+str(fold)+".txt"
    #         np.savetxt(file_name, result, fmt='%d')  # 使用savetxt()函数保存数组到文件

#{'BirthAsphyxia': ['Disease'], 'HypDistrib': ['Disease', 'DuctFlow'], 'HypoxiaInO2': ['DuctFlow', 'LowerBodyO2', 'CardiacMixing'], 'CO2': ['ChestXray', 'CO2Report', 'LungParench'], 'ChestXray': ['Grunting', 'XrayReport', 'LungFlow'], 'Grunting': ['GruntingReport', 'LungParench'], 'LVHreport': ['LVH'], 'LowerBodyO2': ['HypoxiaInO2', 'Disease'], 'RUQO2': ['HypoxiaInO2'], 'CO2Report': ['CO2'], 'XrayReport': ['ChestXray'], 'Disease': ['ChestXray', 'LungFlow', 'DuctFlow', 'CardiacMixing'], 'GruntingReport': ['Grunting'], 'Age': ['Sick', 'LungFlow', 'Disease'], 'LVH': ['LVHreport', 'CardiacMixing', 'Disease'], 'DuctFlow': ['Disease'], 'CardiacMixing': ['Disease', 'HypoxiaInO2'], 'LungParench': ['Disease', 'Grunting', 'ChestXray'], 'LungFlow': ['Disease', 'XrayReport'], 'Sick': ['Disease', 'Age']}

#{'HISTORY': ['LVFAILURE'], 'CVP': ['LVEDVOLUME'], 'PCWP': ['LVEDVOLUME', 'SAO2', 'CO'], 'HYPOVOLEMIA': ['LVEDVOLUME', 'PCWP'], 'LVEDVOLUME': ['CVP', 'HYPOVOLEMIA', 'PCWP'], 'LVFAILURE': ['STROKEVOLUME', 'LVEDVOLUME'], 'STROKEVOLUME': ['VENTTUBE', 'CVP', 'CO'], 'ERRLOWOUTPUT': ['HRBP', 'HREKG'], 'HRBP': ['HR', 'ERRLOWOUTPUT'], 'HREKG': ['ERRCAUTER', 'CATECHOL', 'HR'], 'ERRCAUTER': ['HRSAT', 'HREKG', 'HRBP'], 'HRSAT': ['HR', 'ERRCAUTER'], 'INSUFFANESTH': ['PAP'], 'ANAPHYLAXIS': ['TPR'], 'TPR': ['BP', 'CO'], 'EXPCO2': ['VENTLUNG', 'ARTCO2'], 'KINKEDTUBE': ['VENTLUNG', 'VENTTUBE'], 'MINVOL': ['VENTLUNG', 'VENTALV'], 'FIO2': ['SAO2', 'ARTCO2'], 'PVSAT': ['SAO2', 'ARTCO2'], 'SAO2': ['SHUNT', 'PVSAT'], 'PAP': ['PRESS', 'PULMEMBOLUS', 'INSUFFANESTH'], 'PULMEMBOLUS': ['SHUNT', 'PAP', 'HREKG'], 'SHUNT': ['VENTLUNG', 'VENTALV'], 'INTUBATION': ['VENTALV', 'EXPCO2'], 'PRESS': ['VENTALV', 'VENTTUBE'], 'DISCONNECT': ['VENTTUBE', 'VENTMACH'], 'MINVOLSET': ['VENTMACH', 'LVEDVOLUME'], 'VENTMACH': ['MINVOLSET', 'VENTTUBE'], 'VENTTUBE': ['ARTCO2', 'PRESS', 'EXPCO2', 'MINVOLSET'], 'VENTLUNG': ['VENTTUBE', 'PVSAT', 'EXPCO2'], 'VENTALV': ['VENTLUNG', 'ARTCO2', 'INTUBATION'], 'ARTCO2': ['EXPCO2', 'MINVOL', 'VENTALV'], 'CATECHOL': ['HRBP', 'BP', 'ARTCO2'], 'HR': ['HRBP', 'HREKG'], 'CO': ['BP', 'STROKEVOLUME', 'HR', 'TPR'], 'BP': ['TPR', 'CO']}

#{'alcoholism': ['THepatitis', 'Steatosis'], 'vh_amn': ['hbsag_anti', 'ChHepatitis'], 'hepatotoxic': ['bilirubin'], 'THepatitis': ['alcoholism', 'ast'], 'hospital': ['injections', 'surgery'], 'surgery': ['cholesterol', 'transfusion', 'injections'], 'gallstones': ['fat', 'choledocholithotomy'], 'choledocholithotomy': ['gallstones'], 'injections': ['ChHepatitis', 'hospital', 'surgery'], 'transfusion': ['surgery', 'hospital', 'Cirrhosis'], 'ChHepatitis': ['fibrosis', 'hbsag'], 'sex': ['PBC'], 'age': ['PBC', 'Hyperbilirubinemia'], 'PBC': ['skin', 'ESR', 'ama'], 'fibrosis': ['ChHepatitis'], 'diabetes': [], 'obesity': ['Cirrhosis'], 'Steatosis': ['Cirrhosis', 'triglycerides'], 'Cirrhosis': ['fibrosis', 'Steatosis'], 'Hyperbilirubinemia': ['age'], 'triglycerides': ['hepatotoxic', 'THepatitis', 'Steatosis'], 'RHepatitis': ['cholesterol', 'amylase'], 'fatigue': ['alcohol', 'transfusion'], 'bilirubin': ['PBC', 'skin'], 'itching': ['bilirubin', 'triglycerides'], 'upper_pain': ['amylase'], 'fat': ['hbc_anti', 'albumin'], 'pain_ruq': ['Hyperbilirubinemia'], 'pressure_ruq': ['urea'], 'phosphatase': ['alcohol', 'ChHepatitis'], 'skin': ['bilirubin'], 'ama': ['PBC'], 'le_cells': ['PBC'], 'joints': ['pain', 'ESR', 'PBC'], 'pain': ['ESR', 'joints'], 'proteins': ['phosphatase'], 'edema': ['Cirrhosis', 'gallstones'], 'platelet': ['PBC', 'Cirrhosis', 'bleeding'], 'inr': ['bleeding', 'flatulence', 'hbc_anti', 'Steatosis'], 'bleeding': ['platelet', 'inr'], 'flatulence': ['inr'], 'alcohol': ['fatigue'], 'encephalopathy': ['consciousness'], 'urea': ['encephalopathy', 'cholesterol'], 'ascites': ['proteins'], 'hepatomegaly': ['Hyperbilirubinemia', 'hepatalgia'], 'hepatalgia': ['hepatomegaly', 'fat', 'itching'], 'density': ['encephalopathy'], 'ESR': ['PBC', 'ChHepatitis'], 'alt': ['THepatitis', 'pain'], 'ast': ['ChHepatitis', 'alt'], 'amylase': ['bilirubin'], 'ggtp': ['PBC'], 'cholesterol': ['le_cells', 'ama', 'bilirubin'], 'hbsag': ['ChHepatitis', 'vh_amn', 'flatulence'], 'hbsag_anti': ['vh_amn'], 'anorexia': ['inr'], 'nausea': ['PBC'], 'spleen': ['ESR', 'Cirrhosis'], 'consciousness': ['encephalopathy', 'hospital'], 'spiders': ['Cirrhosis', 'ESR', 'platelet'], 'jaundice': ['bilirubin'], 'albumin': [], 'edge': ['Cirrhosis'], 'irregular_liver': ['Cirrhosis', 'alt'], 'hbc_anti': ['ChHepatitis', 'phosphatase'], 'hcv_anti': [], 'palms': ['Cirrhosis'], 'hbeag': ['platelet'], 'carcinoma': ['Cirrhosis']}

#{'GoodStudent': ['Age'], 'Age': ['DrivQuality', 'SeniorTrain', 'GoodStudent'], 'SocioEcon': ['CarValue', 'HomeBase', 'RiskAversion'], 'RiskAversion': ['DrivHist', 'ThisCarDam', 'DrivingSkill', 'AntiTheft'], 'VehicleYear': ['MakeModel', 'CarValue', 'Airbag'], 'ThisCarDam': ['ThisCarCost', 'Accident'], 'RuggedAuto': ['Cushioning', 'MakeModel'], 'Accident': ['MakeModel', 'ThisCarDam'], 'MakeModel': ['VehicleYear', 'CarValue', 'SocioEcon', 'RuggedAuto'], 'DrivQuality': ['ThisCarCost', 'DrivingSkill', 'RiskAversion'], 'Mileage': ['PropCost', 'CarValue', 'Airbag'], 'Antilock': ['VehicleYear', 'MakeModel'], 'DrivingSkill': ['DrivQuality', 'DrivHist', 'RiskAversion'], 'SeniorTrain': ['DrivHist', 'AntiTheft', 'Age'], 'ThisCarCost': ['PropCost', 'ThisCarDam'], 'Theft': ['ThisCarCost', 'DrivingSkill'], 'CarValue': ['MakeModel', 'Mileage'], 'HomeBase': ['SocioEcon', 'RiskAversion'], 'AntiTheft': ['SocioEcon', 'RiskAversion'], 'PropCost': ['ThisCarCost', 'OtherCarCost'], 'OtherCarCost': ['Accident', 'PropCost'], 'OtherCar': ['SocioEcon'], 'MedCost': ['RuggedAuto', 'ThisCarDam'], 'Cushioning': ['RuggedAuto', 'Airbag'], 'Airbag': ['VehicleYear', 'Cushioning'], 'ILiCost': ['Accident', 'Mileage'], 'DrivHist': ['DrivingSkill', 'RiskAversion']}

#