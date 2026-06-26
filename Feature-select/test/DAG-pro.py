import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.metrics import f1_score
import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
# 1. 计算互信息矩阵
def calculate_mutual_info(X, Y):#计算互信息函数
    return mutual_info_score(X, Y)

def mutual_info_matrix(data):#计算整个数据集的互信息矩阵，两两之间计算互信息。因为互信息矩阵是对称的，所以对位相等
    n_vars = data.shape[1]
    mi_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = calculate_mutual_info(data.iloc[:, i], data.iloc[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # 对称矩阵
    return mi_matrix

# 2. 根据互信息构建无向图
def build_undirected_graph(mi_matrix, threshold=0.1):
    n_vars = mi_matrix.shape[0]
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(i)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if mi_matrix[i, j] > threshold:#大于阈值即设定一条边
                G.add_edge(i, j)
    return G

# 3. 定向边
def orient_edges(G):#为边定向
    directed_G = nx.DiGraph()
    for (u, v) in G.edges():
        if u < v:
            directed_G.add_edge(u, v)
        else:
            directed_G.add_edge(v, u)
    return directed_G

# 4. 检查并绘制DAG
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

# 示例数据
# import pandas as pd
# data = pd.DataFrame({
#     'X1': [1, 2, 3, 4, 5],
#     'X2': [5, 6, 7, 8, 9],
#     'X3': [2, 1, 4, 3, 5],
#     'X4': [9, 8, 7, 6, 5]
# })
datanames = ['Dermatology']#
base_url = r"./dataset/"
for dataname in datanames:
    data_url = dataname + '.mat'
    # dataname = "movement_libras"
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    Special = X0.iloc[:,-1]
    # print(Special.values)
    X0 = X0.iloc[:,:-1]  # 读取训练数据
    a = np.array([item[0] for item in Special])
    label_encoder = LabelEncoder()
    a33 = label_encoder.fit_transform(a)
    X0[33] = a33
    print("====================================================================================================")
    # 将y标签控制在0-n
    def changetosinge(x):
        return float(x)
    # 应用函数到 DataFrame 的每个元素
    X0 = X0.applymap(changetosinge)
    y0 = y0.applymap(changetosinge)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y0)
    Class = set(y_encoded)
    y = pd.DataFrame(y_encoded)
    # 数据离散化
    X = pd.DataFrame()

    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    print(X, y)

# 执行过程
mi_matrix = mutual_info_matrix(X)
# 查看互信息分布
import matplotlib.pyplot as plt
plt.hist(mi_matrix.flatten(), bins=50)
plt.show()

# 使用互信息的75百分位作为阈值
threshold = np.percentile(mi_matrix, 90)
print(threshold)
G = build_undirected_graph(mi_matrix, threshold=threshold)
directed_G = orient_edges(G)

# 检查DAG并绘图
if nx.is_directed_acyclic_graph(directed_G):
    print("The graph is a DAG.")
    draw_graph(directed_G)
else:
    print("The graph contains cycles.")
