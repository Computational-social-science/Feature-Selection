import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency, chi2
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
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


class GSMB:
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        :param data: Pandas DataFrame，每列为一个变量，每行为一个样本
        :param alpha: 显著性阈值（默认0.05）
        """
        self.data = data
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
        if not S:
            # 无条件集时直接检验X和Y的独立性
            contingency = pd.crosstab(self.data[X], self.data[Y])
            _, p, _, _ = chi2_contingency(contingency)
            return p >= self.alpha
        else:
            # 按条件集S分组后合并卡方统计量（Fisher's方法）
            p_values = []
            for _, group in self.data.groupby(S):
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
        return {var: self.compute_markov_blanket(var) for var in self.variables[-1:]}


# ===== 使用示例 =====
if __name__ == "__main__":
    # 示例数据（离散变量）
    datasets = [ 'Orlraws10P',   ]
    # datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
    #             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
    # datasets = ['Movement_libras']
    base_url = r"./dataset/"

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
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
        for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
            if fold >6:
                print("\n当前处理到第", fold, "折")
                data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
                gsmb = GSMB(data_train, alpha=0.05)
                result = gsmb.fit()
                print(result)
                fresult = np.array(list(map(int, list(result.values())[0])))
                # print()
                # print("马尔可夫毯计算结果:")
                # for var, mb in result.items():
                #     print(f"{var}: {mb}")
                # np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
                # 保存到txt文件
                file_name = r"FeatureSelect/" + 'GSMB/' + dataname + "_result" + str(fold) + ".txt"
                np.savetxt(file_name, fresult, fmt='%d')  # 使用savetxt()函数保存数组到文件

    # 运行GSMB
    # gsmb = GSMB(data, alpha=0.05)
    # result = gsmb.fit()

    # 打印结果
    # print("马尔可夫毯计算结果:")
    # for var, mb in result.items():
    #     print(f"{var}: {mb}")