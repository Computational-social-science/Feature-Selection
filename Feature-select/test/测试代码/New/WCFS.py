#W-CFS Algorithm
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



def compinfo(data,xn, yn, zn=None, sn=None):
    info = 0
    x = data[xn]
    y = data[yn]
    if zn is None and sn is None:#x为计算特征，y为目标特征
        # 计算I(x, y)
        unique_y = y.unique()
        for ay in unique_y:
            p_ay = y.value_counts(normalize=True)[ay]
            w = 1 - p_ay
            info += w * middspecial(x, y, ay)
    elif zn is not None:#xy都为计算特征，z为目标特征
        # 计算Iz(x, y)
        z = data[zn]
        unique_z = z.unique()
        for az in unique_z:
            p_az = z.value_counts(normalize=True)[az]
            w = 1 - p_az
            info += w * redundancy(x, y, z, az)
    else:
        # 计算I(x, y | s)
        s = data[sn]
        unique_x = x.unique()
        for ax_val in unique_x:
            p_ax = x.value_counts(normalize=True)[ax_val]
            w = 1 - p_ax
            info += w * cmiddspecial_fixed(x, y, s, 'X', ax_val)
    return info

def learnPC(features, target):
    CPC = []
    PC = []
    data = features
    featuresname = data.columns.tolist()
    for feature in featuresname:
        # 构造条件集s: 排除当前特征
        s = [f for f in featuresname if not f == feature and not f == target]
        # print(s)
        # targetlist = data[target]
        # featurelist = data[feature]
        # slist = data[s]
        # print(target, feature, s)
        if not ifdep(data,target, feature, s):
            PC.append(feature)
            # print("PC:",PC)
        else:
            # 检查空条件下是否相关
            if not ifdep(data,target, feature,[]):
                CPC.append(feature)
                # print("CPC:",CPC)
    # 处理CPC中的特征
    CPC_copy = CPC.copy()
    for x in CPC_copy:
        for y in [y for y in CPC if not y != x]:
            diff = compinfo(data,target, y) - compinfo(data,x, y, target)
            if diff < 0 or ifdep(data, target, y, [x]):
                if y in CPC:
                    CPC.remove(y)
    print("PC：",PC )
    print("CPC：",CPC)
    return PC + CPC

def LearnALL(features, target):
    SP = []
    PC = learnPC(features, target)
    # targetlist = features[target]
    for x in PC:
        PC_new = learnPC(features, x)
        for y in PC_new:
            print(PC_new)
            # xlist = features[x]
            # ylist = features[y]
            diff = compinfo(features,target, y) - compinfo(features,target, y, sn=x)
            if diff < 0:
                SP.append(y)
    print("!!!!!!!",SP ,'1111', PC)
    return SP + PC

import numpy as np
from scipy.stats import chi2_contingency
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
#         print("无法拒绝原假设：两个变量独立。")