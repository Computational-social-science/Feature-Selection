import numpy as np
import itertools
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from typing import Set
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from csmdccmr import middspecial, redundancy, entropydpro, entropyed
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
    ifdep.call_count += 1
    if not X:
        contingency_table = pd.crosstab(dataframe[C], dataframe[Y])
        _, p_value, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return p_value >= alpha

    # 获取条件变量 X 的所有唯一组合
    grouped = dataframe.groupby(X,observed=False)

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
def ifdep_sp(dataframe, C, Y, X, y_value, alpha=0.05):
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

def FSMB_sp(data: pd.DataFrame, alpha=0.01) -> Set[int]:
    n_cases, n_vars = data.shape
    target = n_vars - 1
    Class = set(data[target])
    variables = set(range(n_vars)) - {target}
    # print(variables)
    # --- Phase 1: PC discovery ---
    result = {}
    for nowy in Class:
        CPCT = set()
        dep_scores = {x: sp_compinfo(data[x], data[target],nowy) for x in variables}
        sorted_vars = sorted(dep_scores, key=lambda x: -dep_scores[x])
        # print(111111111111111,)
        print(sorted_vars)
        for X in sorted_vars[:10]:
            # print(X)
            CPCT.add(X)
            for r in range(len(CPCT)):
                for Z in itertools.combinations(CPCT - {X}, r):
                    # print(ifdep(data, X,target, list(Z)))
                    # print(ifdep(data, X, target, list(Z)))
                    # if ifdep(data, X, target, list(Z)):
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
                if not ifdep(data, X,target, [Y]):
                    CSPT[Y].add(X)

            # First screening
            # print(data[Y])
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
            # print(list(CSPT[Y]))
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
        result[nowy]=MB
    return result


datasets = [ 'Su', 'Prostate_GE','TOX_171', 'WarpPIE10P','WarpAR10P', 'Yeoh', ]

# datasets = []#'dexter',
# datasets = []
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
    # =======================Dermatology==================
    if dataname == 'Dermatology':
        Special = X0.iloc[:, -1]
        # print(Special.values)
        X0 = X0.iloc[:, :-1]  # 读取训练数据
        a = np.array([item[0] for item in Special])
        label_encoder = LabelEncoder()
        a33 = label_encoder.fit_transform(a)
        X0[33] = a33
    # =======================Dercomatoly==================
    # 将y标签控制在0-n
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    y = pd.DataFrame(y_encoded)
    X = pd.DataFrame()
    # 离散化并确定每个特征的取值
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
    ifdep.call_count = 0
    data_processed.columns = data_processed.columns.astype(int)
    result = FSMB_sp(data_processed, alpha=0.05)
    # result,time = gsmb.fit()
    # result = method(data_processed)# data_processed, 0.1
    methodname = 'eambsp'
    count[methodname][dataname] = ifdep.call_count
    print(count)
    count.to_excel('tables/' + 'eambspCIcount.xlsx', index=True)
