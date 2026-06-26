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
        print(1,CF)
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
        print(2,CF)
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
# from scipy.stats import chi2_contingency
# def g2():#g2独立性检验
#     # 执行 G² 检验
#     g2_statistic, p_value, dof, expected = chi2_contingency(data, lambda_="log-likelihood")
#
#     # 输出结果
#     print(f"G² Statistic: {g2_statistic}")
#     print(f"P-value: {p_value}")
#     print(f"Degrees of Freedom: {dof}")
#     print("Expected Frequencies:\n", expected)
#
#     # 判断是否拒绝原假设
#     alpha = 0.05  # 显著性水平
#     if p_value < alpha:
#         print("拒绝原假设：两个变量不独立。")
#     else:
#         pri nt("无法拒绝原假设：两个变量独立。")