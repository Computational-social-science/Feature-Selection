import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.metrics import mutual_info_score
from functools import lru_cache
import itertools
from collections import Counter
from itertools import combinations
from sklearn.metrics import mutual_info_score
def changetosinge(x):
    return float(x)

def entropy_cond_ci(series, mask):
    """H(X | C=ci)"""
    s = series[mask]
    n = len(s)
    if n == 0:
        return 0.0
    counts = np.array(list(Counter(s).values()), dtype=float)
    p = counts / n
    return -np.sum(p * np.log2(p))

def I_ci(X, Y, mask):
    """I_{Ci}(X;Y) = sum p(x,y|ci) log( p(x,y|ci)/(p(x|ci)p(y|ci)) )"""
    Xc, Yc = X[mask], Y[mask]
    n = len(Xc)
    if n == 0:
        return 0.0
    joint_counts = Counter(zip(Xc, Yc))
    I = 0.0
    for (x, y), cnt in joint_counts.items():
        p_xy = cnt / n
        p_x = np.sum(Xc == x) / n
        p_y = np.sum(Yc == y) / n
        I += p_xy * np.log2(p_xy / (p_x * p_y))
    return I

def SU_ci(X, Y, mask):
    """类条件对称不确定性 SU_{Ci}(X;Y)"""
    HX = entropy_cond_ci(X, mask)
    HY = entropy_cond_ci(Y, mask)
    Ixy = I_ci(X, Y, mask)
    return 0.0 if HX + HY == 0 else 2 * Ixy / (HX + HY)

def SU_ci_single(X, mask):
    """
    SU(Ci, X) = 2 * I(Ci;X) / (H(Ci)+H(X))
    其中 I(Ci;X) = sum_x p(ci,x) log p(ci,x)/(p(ci)p(x))
    但因 Ci 固定，H(Ci)=0，所以可用 2*I / H(X|Ci)
    实际常用等价： 2*(H(X|Ci^c)-H(X|Ci))/ (H(X)+0) 视需求调整。
    这里实现为 2*I(Ci;X)/H(X|Ci) 更符合你的描述。
    """
    HX = entropy_cond_ci(X, mask)
    p_ci = mask.mean()
    # p(ci,x)
    n = len(mask)
    Xc = X[mask]
    n_ci = len(Xc)
    I = 0.0
    if n_ci > 0:
        counts_x_ci = Counter(Xc)
        for x, cnt in counts_x_ci.items():
            p_cx = cnt / n
            p_x = np.sum(X == x) / n
            I += p_cx * np.log2(p_cx / (p_ci * p_x))
    return 0.0 if HX == 0 else 2 * I / HX


def mi_class_value(Ci_value, class_col, x_col, data: pd.DataFrame):
    """
    I(Ci;X)  -- 其中 Ci 是class_col中某个具体的类别值
    """
    x_vals = data[x_col].unique()
    n = len(data)
    p_ci = (data[class_col] == Ci_value).sum() / n
    mi = 0.0
    for xv in x_vals:
        p_x = (data[x_col] == xv).sum() / n
        p_ci_x = ((data[class_col] == Ci_value) & (data[x_col] == xv)).sum() / n
        if p_ci_x > 0 and p_ci > 0 and p_x > 0:
            mi += p_ci_x * np.log(p_ci_x / (p_ci * p_x))
    return mi / np.log(2)  # 以log2为底

def cmi_class_value(Ci_value, class_col, x_col, y_col, data: pd.DataFrame):
    """
    I(Ci; X | Y) = Σ_y p(y) Σ_x p(Ci,x | y) log [ p(Ci,x | y) / ( p(Ci|y) * p(x|y) ) ]
    """
    n = len(data)
    cmi = 0.0
    for yv in data[y_col].unique():
        sub_y = data[data[y_col] == yv]
        p_y = len(sub_y) / n
        if p_y == 0:
            continue
        # p(Ci|y) 与 p(x|y) 都在条件 y 下计算
        p_ci_y = (sub_y[class_col] == Ci_value).sum() / len(sub_y)
        for xv in data[x_col].unique():
            p_x_y = (sub_y[x_col] == xv).sum() / len(sub_y)
            # 这里是 p(Ci,x | y)
            p_ci_x_y = ((sub_y[class_col] == Ci_value) & (sub_y[x_col] == xv)).sum() / len(sub_y)
            if p_ci_x_y > 0 and p_ci_y > 0 and p_x_y > 0:
                cmi += p_y * p_ci_x_y * np.log(
                    p_ci_x_y / (p_ci_y * p_x_y)
                )
    return cmi / np.log(2)  # 以 log2 为底

def entropy(x):
    vals, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return -np.sum(p * np.log2(p + 1e-12))

def symmetrical_uncertainty(x, y):
    h_x, h_y = entropy(x), entropy(y)
    h_xy = entropy(list(zip(x, y)))
    ig = h_x - (h_xy - h_y)
    return 0 if (h_x + h_y) == 0 else 2.0 * ig / (h_x + h_y)

def conditional_mutual_info(x, y, z):
    """
    I(x;y|z) = sum_z p(z)*I(x;y | Z=z)
    简单枚举条件z的所有取值计算。
    """
    z = np.asarray(z)
    cmi = 0.0
    for v in np.unique(z):
        idx = (z == v)
        if idx.sum() == 0:
            continue
        cmi += (idx.sum() / len(z)) * mutual_info_score(x[idx], y[idx])
    return cmi / np.log(2)  # 以log2为底

# ========= FCBF（可复用） =========
def fcbf(X: pd.DataFrame, y: pd.Series, delta=0.5):
    su_with_class = {}
    for f in X.columns:
        su = symmetrical_uncertainty(X[f].values, y.values)
        if su >= delta:
            su_with_class[f] = su
    sorted_feats = sorted(su_with_class.items(), key=lambda x: x[1], reverse=True)
    selected = []
    while sorted_feats:
        f_best, _ = sorted_feats[0]
        selected.append(f_best)
        new_list = []
        for f, su_fc in sorted_feats[1:]:
            su_ff = symmetrical_uncertainty(X[f_best].values, X[f].values)
            if su_ff < su_fc:
                new_list.append((f, su_fc))
        sorted_feats = new_list
    return selected

# ========= Algorithm 1: DiscoverPC =========
def DiscoverPC(data: pd.DataFrame, target_col: str, ci_value, alpha=0.3):
    """
    DiscoverPC with class-conditional SU modifications.
    data       : pandas DataFrame (离散化)
    target_col : 类标签列名
    ci_value   : 固定的类别取值 Ci
    alpha      : 阈值
    """
    C = data[target_col].values
    mask_ci = (C == ci_value)

    features = [col for col in data.columns if col != target_col]
    selected = []

    # ---------- Step 1 : SU(Ci, X) ----------
    su_scores = {}
    for feat in features:
        su_scores[feat] = SU_ci_single(data[feat].values, mask_ci)
    # 按得分排序
    ranked = sorted(su_scores.items(), key=lambda x: x[1], reverse=True)

    # ---------- Step 2 : Redundancy pruning ----------
    for feat, score in ranked:
        if score < alpha:      # relevance threshold
            continue
        keep = True
        for sel in selected:
            # SU_{Ci}(feat, sel)
            su_cixy = SU_ci(data[feat].values, data[sel].values, mask_ci)
            # 这里的“差值”即 SU(Ci,feat) - SU_{Ci}(feat,sel)
            if su_scores[feat] - su_cixy < alpha:
                keep = False
                break
        if keep:
            selected.append(feat)
            # if len(selected) >= 10:
            #     break

    return selected


# ========= Algorithm 2: LaCFS =========
def LaCFS(data: pd.DataFrame, class_col: str, delta=0.3):
    """
    Label-Aware Causal Feature Selection
    return: (PC, SP)  PC: parents/children of C, SP: spouses
    """
    result = {}
    C = class_col
    Cis = np.unique(data[C])
    for Ci_value in Cis:
        # print(Ci_value)
        PC = DiscoverPC(data, C,Ci_value,delta,)
        SP = []
        # 假设 Ci_value 是你关心的具体类别，比如 1
        for X in PC:
            PC_X = fcbf(data.drop(columns=[class_col]), data[X], 0.5)
            for Y in PC_X:
                # 使用新的定义
                I_Ci_Y = mi_class_value(Ci_value, class_col, Y, data)
                I_Ci_Y_X = cmi_class_value(Ci_value, class_col, Y, X, data)
                if I_Ci_Y - I_Ci_Y_X < 0:
                    SP.append(Y)
        SP = list(set(SP))
        # print('PC+SP', PC, SP)
        result[Ci_value] = set(PC+SP)
    return result

# ========= 使用示例 =========
if __name__ == "__main__":
    # from sklearn.datasets import load_breast_cancer
    # from sklearn.preprocessing import KBinsDiscretizer
    #
    # data = load_breast_cancer()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = pd.Series(data.target, name='target')
    #
    # # 连续数据离散化
    # kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # X_disc = pd.DataFrame(kb.fit_transform(X), columns=X.columns)
    # df = pd.concat([X_disc, y], axis=1)
    #
    # PC, SP = Lacfs(df, class_col='target', delta=0.05)
    # print("Parents/Children (PC):", PC)
    # print("Spouses (SP):", SP)
    datasets = ['Authorship', 'CLL_SUB_111', 'Factors', 'Movement_libras', 'Musk1',
                'Orlraws10P', 'Pixel', 'Prostate_GE', 'Su', 'SRBCT', 'splice', 'Synthetic_control', 'TOX_171',
                'Waveform', 'Wdbc', 'WarpPIE10P', 'WarpAR10P', 'Yeoh', 'Yale', 'Arcene', ]
    for dataname in datasets:  #
        #  #,'Wdbc','Synthetic_control',1a2
        base_url = r"./dataset/"
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
        # 数据离散化
        X = pd.DataFrame()
        # print(X0,y)
        #离散化
        for col in X0.columns:
            X[col] = pd.cut(X0[col], bins=5, labels=False)
        # 使用rename函数重新命名列名，将x列名控制在0-n
        # 将列名命名为字符串
        new_columns = [str(i) for i in range(X.shape[1] + 1)]  # 将列名转换为字符串
        X = X.rename(columns=dict(zip(X.columns, new_columns[:-1])))  # 重命名 X 的列名
        y = y.rename(columns=dict(zip(y.columns, [new_columns[-1]])))  # 重命名 y 的列名
        data_processed = pd.concat([X, y], axis=1)
        # print(data_processed,'11111111',str(new_columns[-1]))
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
        for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
            print("\n当前处理到第", fold, "折")
            result_dict = {}
            data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
            # print(data_train.shape, data_train.columns)
            result = LaCFS(data_train, str(new_columns[-1]), )
            # result_dict[nowy] = set(result)
            print("result:", result, )
            # result = EAMB_sp(str(data_train.shape[1]-1), data_train, 0.15, nowy)
            # result_dict[nowy] = set(result)
            # print("result:",result,)
            # 保存到txt文件
            import json
            result_dict_serializable = {
                key: [int(x) for x in value] for key, value in result.items()
            }
            result_dict_fixed = {
                int(key): value for key, value in result_dict_serializable.items()
            }
            print(result_dict_fixed)
            # 保存为 JSON 文件
            file_name = r"./FeatureSelect/LaCFS" +'/' + dataname + "_sp_result" + str(fold) + ".txt"
            with open(file_name, 'w') as f:
                json.dump(result_dict_fixed, f, indent=4)