import numpy as np
import scipy.io as scio
import pandas as pd
from scipy.special import gammaincc
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import math

# pd.set_option('display.max_columns', None)  # 显示所有列

def changetosinge(x):
    return float(x)
# Example usage:
# Constants (adjust these values as needed)常量
# max_max_cond_size = 10  #设置最大条件子集数量
# N_CASES_PER_DF = 10  # Example value, adjust as needed
# alpha = 0.05  # Significance level, example value
# k_tradeoff = 0.5  # Example value for k-greedy
# # mb_write_in_file_path = 'mb_results.txt'  # File path to write results'Isolet',
datanames = ['madelon','spambase','splice','dna' ,'dexter','Colon','SRBCT',]#
# datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
max_max_cond_size = 100  #设置最大条件子集数量
N_CASES_PER_DF = 20  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy
algorithm = "IAMB"

# # 对每一列进行重新归一化
# for column in data_processed.columns:
#     data_processed[column] = label_encoder.fit_transform(data_processed[column])
# new_columns = list(range(data_processed.shape[1]))
# # 使用 rename 函数重命名列
# # net_file_path = #网结构文件
# alpha = 0.05
# algorithm = "IAMB"
# #变量数量和样本数量
# n_vars = data_processed.shape[1]
# n_cases = data_processed.shape[0]
# n_states = data_processed.nunique()
# print(f"algorithm: {algorithm}, alpha: {alpha}, n_cases: {n_cases}, n_vars: {n_vars}")
# # 处理目标变量，初始化MB
# targets = list(range(n_vars))[-1]
# # print(data_processed)
# # print(n_states)
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects import r
# 激活 R 和 Pandas 的自动转换
# pandas2ri.activate()

# 加载 R 的 base 包
# base = ro.packages.importr('base')

# 加载 .rda 文件
# base.load('./alarm.rda')
# print(r['load']('./alarm.rda'))
# import rpy2.robjects as ro
# import pandas as pd
# import rpy2.robjects.pandas2ri as p2r
# from pgmpy.models import BayesianModel
# from pgmpy.sampling import BayesianModelSampling
#
# # 读取 R 的 .rda 文件
# ro.r['load']('./alarm.rda')
#
# # 获取加载的 R 对象
# bn_r = ro.r['bn']
#
# # 显示 R 对象的内容
# print(bn_r)
#
# # 假设你的 R 对象是贝叶斯网络的结构，可以通过以下方式转换为 pgmpy 模型
# # 需要根据实际的 R 对象内容来定义这个转换函数
# def convert_r_bn_to_pgmpy(r_bn):
#     # 从 R 中提取网络结构
#     edges = ro.r['bn'](r_bn)[0]  # 假设 R 对象中包含边信息
#     edges = edges[ro.r['which'](ro.r['is'](edges, 'data.frame'))]  # 确保是数据框
#
#     # 从 R 中提取 CPT
#     cpts = ro.r['bn'](r_bn)[1]  # 假设 R 对象中包含 CPT 信息
#
#     # 创建 pgmpy 贝叶斯网络模型
#     model = BayesianModel()
#
#     # 添加边
#     for edge in edges.itertuples(index=False):
#         model.add_edge(edge[0], edge[1])
#
#     # 添加 CPT
#     for node in cpts.keys():
#         cpt = cpts[node]
#         parents = model.get_parents(node)
#         cpt = p2r.ri2py(cpt)  # 将 R 数据框转换为 Pandas 数据框
#         model.add_cpds(ParameterEstimator(cpt, parents))
#
#     return model
#
# # 使用转换函数将 R 贝叶斯网络转换为 pgmpy 模型
# model = convert_r_bn_to_pgmpy(bn_r)
#
# # 检查模型的有效性
# assert model.check_model()
#
# # 从模型中生成数据样本
# sampler = BayesianModelSampling(model)
# data = sampler.forward_sample(size=5000)
#
# # 打印前几行样本数据
# print(data.head())
#
# # 如果需要将数据保存为 CSV 文件
# data.to_csv('sample_data.csv', index=False)



# r_data_frame = r['bn']
# print(r_data_frame)
# with (ro.default_converter + pandas2ri.converter).context():
#   pd_from_r_df = ro.conversion.get_conversion().rpy2py(r_data_frame)
# 将R中的数据框转换为Python中的DataFrame
# data = pd_from_r_df
# 检查 R 环境中加载的对象
# r_objects = base._env.keys()
# print(r_objects)  # 打印加载的对象名
#
# # 假设你的 .rda 文件中包含一个数据框对象名为 'dataframe_name'
# r_data_frame = ro.globalenv['dataframe_name']
#
# # 将 R 数据框转换为 Pandas 数据框
# import pandas as pd
# df = pandas2ri.rpy2py_dataframe(r_data_frame)
#

# data =scio.loadmat('./Alarm1_s5000_v1.txt')

# file_path ='./Alarm1_s5000_v1.txt'
# data_processed = pd.read_csv(file_path, delimiter='  ', header=None)  # 使用制表符作为分隔符
# print(data_processed)

# def compute_dep(var, target, cond, n_conds, data_processed, n_states, n_cases):#计算独立性水平
#     average_n_G2 = 0  # This variable should be declared in the actual implementation context
#     n_cond_states = 1 #所有条件变量可能取值的组合数
#     for i in range(n_conds):
#         if cond[i] != -1: #条件变量的集合（作为条件独立性的条件），存储的是与目标变量和var相关联的变量索引。
#             n_cond_states *= n_states[cond[i]]
#
#     if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
#         return 0.0
#
#     # Initialize arrays
#     ss_cond = np.zeros(n_cond_states, dtype=int)
#     ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
#     ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
#     ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)
#
#     for i in range(n_cases):
#         cond_state = 0
#         for j in range(n_conds):
#             if cond[j] != -1:
#                 cond_state = cond_state * n_states[cond[j]] + data_processed[cond[j]][i]
#
#         ss_cond[cond_state] += 1
#         ss_target[cond_state][data_processed[target][i]] += 1
#         ss_var[cond_state][data_processed[var][i]] += 1
#         ss[cond_state][data_processed[target][i]][data_processed[var][i]] += 1
#
#     # G^2 statistic based on observed and expected frequencies
#     statistic = 0.0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             for j in range(n_states[target]):
#                 if ss_target[i][j] > 0:
#                     for k in range(n_states[var]):
#                         if ss[i][j][k] > 0:
#                             aux = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
#                             statistic += ss[i][j][k] * (math.log(ss[i][j][k]) - math.log(aux))
#     statistic *= 2.0
#
#     # Calculate degrees of freedom
#     df = 0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             df_target = np.sum(ss_target[i] > 0)
#             df_var = np.sum(ss_var[i] > 0)
#             df += (df_target - 1) * (df_var - 1)
#
#     if statistic < 0.0:
#         statistic = 0.0
#
#     if df <= 0:
#         df = 1
#
#     # Compute p-value
#     dep = gammq(0.5 * df, 0.5 * statistic)
#
#     if dep < 0.0:
#         dep = 0.0
#     elif dep > 1.0:
#         dep = 1.0
#
#     if dep <= alpha:
#         dep = 2.0 - dep
#         if dep == 2.0:
#             dep = 2.0 + statistic / df
#         return dep
#     else:
#         dep = -1.0 - dep
#         if dep == -2.0:
#             dep = -2.0 - statistic / df
#         return dep


# 假设的 alpha 值，用于判断依赖显著性
alpha = 0.05


def compute_dep(var, target, cond):
    n_cond_states = 1
    for i in range(max_max_cond_size):
        if cond[i] != -1:
            n_cond_states *= n_states[cond[i]]

    # 检查是否满足最低样本要求
    if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
        return 0.0

    # 初始化频数表
    ss_cond = np.zeros(n_cond_states, dtype=int)
    ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
    ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
    ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)

    # 填充频数表
    for i in range(n_cases):
        cond_state = 0
        for j in range(max_max_cond_size):
            if cond[j] != -1:
                cond_state = cond_state * n_states[cond[j]] + data_processed[cond[j]][i]

        ss_cond[cond_state] += 1
        # print(cond_state,data_processed[target][i],target,i)
        ss_target[cond_state][data_processed[target][i]] += 1
        ss_var[cond_state][data_processed[var][i]] += 1
        ss[cond_state][data_processed[target][i]][data_processed[var][i]] += 1

    # 计算 G² 统计量
    statistic = 0.0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            for j in range(n_states[target]):
                if ss_target[i][j] > 0:
                    for k in range(n_states[var]):
                        if ss[i][j][k] > 0:
                            expected = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
                            statistic += ss[i][j][k] * (np.log(ss[i][j][k]) - np.log(expected))

    statistic *= 2.0

    # 计算自由度
    df = 0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            df_target = np.sum(ss_target[i] > 0)
            df_var = np.sum(ss_var[i] > 0)
            df += (df_target - 1) * (df_var - 1)

    # 防止统计量为负值
    if statistic < 0.0:
        statistic = 0.0

    if df <= 0:
        df = 1

    # 计算依赖度（通过伽马不完全函数 gammaincc）
    dep = gammaincc(0.5 * df, 0.5 * statistic)

    # 将依赖度限制在 [0, 1] 之间
    dep = max(0.0, min(1.0, dep))

    # 检查依赖是否显著
    if dep <= alpha:
        dep = 2.0 - dep
        if dep == 2.0:
            dep = 2.0 + statistic / df
    else:
        dep = -1.0 - dep
        if dep == -2.0:
            dep = -2.0 - statistic / df

    return dep

#cond像是初步确定的MB子集
# def compute_mi(var, target, cond, data_processed, n_states, n_cases):
#     # 初始化条件状态的数量，计算一共有多少种状态需要计算
#     n_cond_states = 1
#     for i in range(max_max_cond_size):
#         if cond[i] != -1:
#             n_cond_states *= n_states[cond[i]]
#
#     # 初始化统计数组
#     ss_cond = np.zeros(n_cond_states, dtype=int)  # 条件状态计数
#     ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)  # 目标变量在各条件状态下的计数
#     ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)  # var在各条件状态下的计数
#     ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)  # 联合计数
#
#     # 遍历所有样本，填充统计表
#     for i in range(n_cases):
#         cond_state = 0
#         for j in range(max_max_cond_size):
#             if cond[j] != -1:
#                 cond_state = cond_state * n_states[cond[j]] + data_processed[i][cond[j]]
#
#         ss_cond[cond_state] += 1
#         ss_target[cond_state][data_processed[i][target]] += 1
#         ss_var[cond_state][data_processed[i][var]] += 1
#         ss[cond_state][data_processed[i][target]][data_processed[i][var]] += 1
#
#     # 计算互信息
#     mutual_information = 0.0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:  # 只有条件状态出现时，才有必要计算
#             for j in range(n_states[target]):
#                 if ss_target[i][j] > 0:  # 只有目标状态出现时，才有必要计算
#                     for k in range(n_states[var]):
#                         if ss[i][j][k] > 0:  # 只有联合计数非零时，才计算互信息
#                             # 计算联合概率 P(target = j, var = k | cond_state = i)
#                             p_joint = ss[i][j][k] / ss_cond[i]
#
#                             # 计算条件概率 P(target = j | cond_state = i) 和 P(var = k | cond_state = i)
#                             p_target_given_cond = ss_target[i][j] / ss_cond[i]
#                             p_var_given_cond = ss_var[i][k] / ss_cond[i]
#
#                             # 计算互信息的增量
#                             mutual_information += p_joint * math.log(p_joint / (p_target_given_cond * p_var_given_cond))
#
#     return mutual_information

def k_greedy(dep):
    n_cands = np.sum(dep >= 1.0)
    cand = [i for i in range(len(dep)) if dep[i] >= 1.0]

    n_cands_left = len(cand)
    n_cands = int(len(cand) * k_tradeoff)
    if n_cands < 1:
        n_cands = 1

    max_idx = cand[0]
    for i in range(n_cands):
        j = np.random.randint(n_cands_left)
        if i == 0 or dep[cand[j]] >= dep[max_idx]:
            if i != 0 and dep[cand[j]] == dep[max_idx] and max_idx > cand[j]:
                max_idx = cand[j]
            else:
                max_idx = cand[j]
        cand.pop(j)
        n_cands_left -= 1

    return max_idx

def Inter_IAMB(target):
    result = []
    # for target in targets:
    mb = [0 if state > 1 else -1 for i, state in enumerate(n_states)]
    removals = [0 for _ in range(n_vars)]
    mb[target] = -1
    n_conds = 0
    cond = np.full(max_max_cond_size, -1, dtype=int)
    dep = np.zeros(n_vars, dtype=float)
    # 增长阶段
    while True:
         stop = True
         if n_conds < max_max_cond_size:
             dep[:] = 0.0
             for i in range(n_vars):
                 if mb[i] == 0:
                     dep[i] = compute_dep(i, target, cond)
                     if dep[i] >= 1.0:
                         stop = False
         if not stop:
             aux = k_greedy(dep)
             mb[aux] = 1
             cond[n_conds] = aux
             n_conds += 1
             # 开始剪枝
             dep[:] = 0.0
             for i in range(n_vars):
                 if mb[i] == 1 and removals[i] < 10:
                     if i in cond:
                         index = np.where(cond == i)[0][0]
                         cond[index:-1] = cond[index + 1:]
                         cond[-1] = -1  # 最后一位置为-1
                     dep[i] = compute_dep(i, target, cond)
                     if dep[i] <= -1.0:
                         n_conds -= 1
                         mb[i] = 0
                         removals[i] += 1  # 避免死循环
                     else:
                         # pass
                         if cond[-1] == -1:
                             cond[-1] = i
         else:
             break
    # 最终调整
    # mb = [0 if x == -1 else x for x in mb]
    # print(mb)
    # result.append(mb)
    mb[mb == -1] = 0
    for index, x in enumerate(mb):
         if x == 1:
             # print(index)
             # print(mb[mb == 1].)
             result.append(index)
    relation_matrix = np.array(result)
    # np.set_printoptions(threshold=np.inf)
    # print(dataname, relation_matrix)
    return relation_matrix
    # 保存结果
    # relation_matrix = np.array(result)
    # np.savetxt('result.txt', relation_matrix)
def IAMB(target):
    # print(targets)
    result = []
    # for target in targets:
    mb = [0 if state > 1 else -1 for i, state in enumerate(n_states)]
    mb[target] = -1
    n_conds = 0
    cond = np.full(max_max_cond_size, -1, dtype=int)
    dep = np.zeros(n_vars, dtype=float)
    #增长阶段
    while True:
        stop = False
        if n_conds < max_max_cond_size:
            dep[:] = 0.0
            for i in range(n_vars):
                if mb[i] == 0:
                    dep[i] = compute_dep(i, target, cond)#计算独立性
                    if dep[i] >= 1.0:
                        stop = True
        if stop:
            aux = k_greedy(dep)
            mb[aux] = 1
            cond[n_conds] = aux
            n_conds += 1
        else:
            break
    # 缩减阶段
    for i in range(n_vars):
        if mb[i] == 1:
            if i in cond:#检查每一个重新计算独立性
                index = np.where(cond == i)[0][0]
                cond[index:-1] = cond[index + 1:]
                cond[-1] = -1  # Set the last position to -1
            dep[i] = compute_dep(i, target, cond)
            if dep[i] <= -1.0:
                n_conds -= 1
                mb[i] = 0
            else:
                pass
                # Restore i to cond
                # if cond[-1] == -1:
                #     cond[-1] = i
    # Final adjustment
    mb[mb == -1] = 0
    for index,x in enumerate(mb):
        if x == 1:
            # print(index)
    # print(mb[mb == 1].)
            result.append(index)
    relation_matrix = np.array(result)
    return relation_matrix
    # np.set_printoptions(threshold=np.inf)
    # print(dataname,relation_matrix)
    # np.savetxt('result.txt', relation_matrix)
    # # 创建空图
    # import matplotlib.pyplot as plt
    # import networkx as nx
    # G = nx.Graph()
    #
    # # 获取矩阵的节点数
    # num_nodes = relation_matrix.shape[0]
    #
    # # 添加节点
    # G.add_nodes_from(range(num_nodes))
    #
    # # 添加边，根据矩阵中的值添加连线
    # for i in range(num_nodes):
    #     for j in range(i + 1, num_nodes):
    #         if relation_matrix[i, j] == 1:
    #             G.add_edge(i, j)
    #
    # # 画图
    # plt.figure(figsize=(8, 8))
    # pos = nx.spring_layout(G,k=10, iterations=50)  # 使用spring布局方式
    #
    # # 绘制节点和边
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=15, font_color='black',
    #         edge_color='gray',width=5 )
    #
    # # 显示图形
    # plt.title('关系图')
    # plt.show()




#
#
#
#
#     # print(result)
# thresh = 0.05
# def IAMB(targets):
#     result = []
#     for target in targets:
#         for c in range(t-1):
#             mb = [0 if state > 1 else -1 for i, state in enumerate(n_states)]
#             mb[target] = -1
#
#             n_conds = 0
#             cond = np.full(max_max_cond_size, -1, dtype=int)
#             dep = np.zeros(n_vars, dtype=float)
#
#             #增长阶段
#             while True:
#                 stop = False
#                 if n_conds < max_max_cond_size:
#                     dep[:] = 0.0#初始化所有dep为0
#
#                     for i in range(n_vars): #对所有变量，若未被选中，则计算其dep，若dep大于，记录
#                         if mb[i] == 0:
#                             # dep[i] = compute_dep([i, target, [cond], c)  # 计算独立性
#                             # print(i,target,cond)
#                             nowcond = [x for x in cond if x != -1]
#                             # dep[i] = compute_info([i], target, [cond], c)#计算独立性
#                             dep[i] = cmiddspecial(data_processed[i], data_processed[nowcond], data_processed[target], c)
#                             # print(dep)
#                             if dep[i] >= thresh:
#                                 stop = True
#
#                 if stop:#记录dep最大，将其归于mb子集
#                     aux = k_greedy(dep)
#                     mb[aux] = 1
#                     cond[n_conds] = aux
#                     n_conds += 1
#                 else:
#                     break
#             # print(mb)
#             # 缩减阶段
#             for i in range(n_vars):
#                 if mb[i] == 1:#对于所有已经选定的mb元素
#                     if i in cond:#检查每一个重新计算独立性
#                         index = np.where(cond == i)[0][0]
#                         cond[index:-1] = cond[index + 1:]#index后面的所有向左移动一格，覆盖index所在位置
#                         cond[-1] = -1  # Set the last position to -1
#                     # dep[i] = compute_info([i], target, [cond], c)#重新计算（将i拿出来和其他进行比较）
#                         nowcond = [x for x in cond if x != -1]
#                         # dep[i] = compute_info([i], target, [cond], c)#计算独立性
#                         dep[i] = cmiddspecial(data_processed[i], data_processed[nowcond], data_processed[target], c)
#                         # print(dep)
#                     if dep[i] <= thresh:#若新dep小于阈值，则去除该元素
#                         n_conds -= 1
#                         mb[i] = 0
#                     else:
#                         # Restore i to cond
#                         if cond[-1] == -1:
#                             cond[-1] = i
#
#             # Final adjustment
#             # print(mb)
#             mbn = []
#             mb[mb == -1] = 0
#             for index,x in enumerate(mb):
#                 if x == 1:
#                     mbn.append(index)
#             # print(mb[mb == 1].)
#             result.append(mbn)
#     # print(11111111111,result)
#     return result
#         # file_name = r"feature_select_result/" + 'INMB' + '/' + dataname + str(fold) + "_result.txt"
#         # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
#         # print(data_processed[target].nunique())
#         # relation_matrix = np.array(result)
#         # print(relation_matrix)
#         # np.set_printoptions(threshold=np.inf)
#         # np.savetxt('result.txt', relation_matrix)
#     # # 创建空图
#     # import matplotlib.pyplot as plt
#     # import networkx as nx
#     # G = nx.Graph()
#     #
#     # # 获取矩阵的节点数
#     # num_nodes = relation_matrix.shape[0]
#     #
#     # # 添加节点
#     # G.add_nodes_from(range(num_nodes))
#     #
#     # # 添加边，根据矩阵中的值添加连线
#     # for i in range(num_nodes):
#     #     for j in range(i + 1, num_nodes):
#     #         if relation_matrix[i, j] == 1:
#     #             G.add_edge(i, j)
#     #
#     # # 画图
#     # plt.figure(figsize=(8, 8))
#     # pos = nx.spring_layout(G,k=10, iterations=50)  # 使用spring布局方式
#     #
#     # # 绘制节点和边
#     # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=15, font_color='black',
#     #         edge_color='gray',width=5 )
#     #
#     # # 显示图形
#     # plt.title('关系图')
#     # plt.show()
#
#
#
#
#
#     # print(result)
#
#
#     #参数设置
#     # parser = argparse.ArgumentParser(description="Run structure learning algorithms")
#     # parser.add_argument('data_file_path', type=str, help='Path to the data file')
#     # parser.add_argument('net_file_path', type=str, help='Path to the network file')
#     # parser.add_argument('alpha', type=float, help='Alpha value')
#     # parser.add_argument('algorithm', type=str, help='Algorithm to use')
#     # parser.add_argument('targets_str', type=str, help='Target nodes (comma-separated) or "all"')
#     # parser.add_argument('k_tradeoff', type=float, help='K tradeoff value')
#     # parser.add_argument('k_conditon', type=int, help='Maximum test set for conditional independence tests')
#     # args = parser.parse_args()
#     #数据基础信息
#     # data_file_path = args.data_file_path
#     # net_file_path = args.net_file_path
#     # alpha = args.alpha
#     # algorithm = args.algorithm
#     # k_tradeoff = args.k_tradeoff
#     # k_conditon = args.k_conditon
#     # 文件路径
#
#     # 初始化矩阵和缓存
#     # families = [list() for _ in range(n_vars)]  #家庭
#     # neighs = [list() for _ in range(n_vars)]  #邻居
#     # adjMat = np.zeros((n_vars, n_vars), dtype=bool)  #邻接矩阵
#     # pc_cache = [[] for _ in range(n_vars)]  #候选父子
#     # pc_iscomputed = [False] * n_vars  #候选父子是否已经计算
#
#     # 选择算法并运行
#
#     # 输出性能指标
#     # print(f"Performance time: {perform_time}")
#
#     # # 将结果写入文件
#     # with open("./indicator.out", "a") as f:
#     #     f.write(f"Precision: {average_Precision}\n")
#     #     f.write(f"Recall: {average_Recall}\n")
#     #     f.write(f"F1: {average_F1}\n")
#     #     f.write(f"Distance: {average_Distance}\n")
#     #     f.write(f"Time: {perform_time}\n")

#


def csmdccmrtest(time, result, X_test, y_test, X_train, y_train):
    # print()
    # for i in range(time):#对不同数量的特征选择结果进行测试
    nowpick = result  # 当前i个特征的特征选择结果

    # 初始化平均值和结果概率矩阵
    bresultmax = np.zeros((samplenum, classnum))
    kresultmax = np.zeros((samplenum, classnum))
    sresultmax = np.zeros((samplenum, classnum))
    rresultmax = np.zeros((samplenum, classnum))
    #记录所有time次的预测概率，和预测标签
    b_time_predict_probabity = []
    k_time_predict_probabity = []
    s_time_predict_probabity = []
    r_time_predict_probabity = []
    b_time_predict_labels = []
    k_time_predict_labels = []
    s_time_predict_labels = []
    r_time_predict_labels = []
    #记录每个分类器单次的可能性结果，共Class个分类器
    bsingleclasspro = []
    ksingleclasspro = []
    ssingleclasspro = []
    rsingleclasspro = []
    # 对于每个建立一个对应分类器
    # for item in nowpick:
    item = nowpick
    # print(item)
    # print(X_train[item])
    # **Naïve Bayes (朴素贝叶斯)**
    from sklearn.naive_bayes import GaussianNB
    bayesmodel = GaussianNB()
    bayesmodel.fit(X_train[item], y_train)
    bprobabilities = bayesmodel.predict_proba(X_test[item])
    if not bprobabilities.shape == bresultmax.shape:
        bprobabilities = np.vstack([bprobabilities, np.zeros_like(bprobabilities[-1])])
    bresultmax += bprobabilities
    #单次预测类概率
    # **k-Nearest Neighbors (k-最近邻算法)**:
    from sklearn.neighbors import KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train[item], y_train)
    kprobabilities = knn_classifier.predict_proba(X_test[item])
    if not kprobabilities.shape == kresultmax.shape:
        kprobabilities = np.vstack([kprobabilities, np.zeros_like(kprobabilities[-1])])
    kresultmax += kprobabilities
    # **Support Vector Machines (支持向量机)**:
    from sklearn.svm import SVC
    svm_classifier = SVC(probability=True)
    svm_classifier.fit(X_train[item], y_train)
    sprobabilities = svm_classifier.predict_proba(X_test[item])
    if not sprobabilities.shape == sresultmax.shape:
        sprobabilities = np.vstack([sprobabilities, np.zeros_like(sprobabilities[-1])])
    sresultmax += sprobabilities
    # random Forest (随机森林)**:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train[item], y_train)
    rprobabilities = rf_classifier.predict_proba(X_test[item])
    if not rprobabilities.shape == rresultmax.shape:
        rprobabilities = np.vstack([rprobabilities, np.zeros_like(rprobabilities[-1])])
    rresultmax += rprobabilities
    #每个分类器在当前情况下的分类概率列表（一次测试的所有分类器结果）
    bsingleclasspro.append(bprobabilities)#记录n种分类器单次概率
    ksingleclasspro.append(kprobabilities)
    ssingleclasspro.append(sprobabilities)
    rsingleclasspro.append(rprobabilities)
    #根据单次测试分类概率得到分类标签，共计算50次
    bsinglepredicted_labels = (np.argmax(bresultmax, axis=1))
    ksinglepredicted_labels = (np.argmax(kresultmax, axis=1))
    ssinglepredicted_labels = (np.argmax(sresultmax, axis=1))
    rsinglepredicted_labels = (np.argmax(rresultmax, axis=1))
    #记录time次分类结果，50次，其中每次都有class个预测矩阵
    b_time_predict_probabity.append(bsingleclasspro)
    k_time_predict_probabity.append(ksingleclasspro)
    s_time_predict_probabity.append(ssingleclasspro)
    r_time_predict_probabity.append(rsingleclasspro)
    # 记录time次分类结果，50次，其中每次都是单次的预测标签
    b_time_predict_labels.append(bsinglepredicted_labels)
    k_time_predict_labels.append(ksinglepredicted_labels)
    s_time_predict_labels.append(ssinglepredicted_labels)
    r_time_predict_labels.append(rsinglepredicted_labels)
    predict_pro   = {'b': b_time_predict_probabity, 'k': k_time_predict_probabity, 's': s_time_predict_probabity, 'r': r_time_predict_probabity,}
    predict_lable = {'b': b_time_predict_labels, 'k': k_time_predict_labels, 's': s_time_predict_labels, 'r': r_time_predict_labels,}
    file_name1 = r"result/" + dataname + str(fold) +  "_predict_pro.mat"
    file_name2 = r"result/" + dataname + str(fold) +  "_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)
#指定数据类型
def changetosinge(x):
    return float(x)
#函数主体：数据处理，十折验证
for dataname in datanames:
    print(dataname)
    data_file_path = "../dataset/testdatasets/" + dataname + ".mat"
    data = scio.loadmat(data_file_path)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    print(X0,y0)
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
    print(data_processed)
    n_vars = data_processed.shape[1]
    n_cases = data_processed.shape[0]
    n_states = np.full(n_vars, 5)
    n_states[-1] = len(Class)
    # print(f"algorithm: {algorithm}, alpha: {alpha}, n_cases: {n_cases}, n_vars: {n_vars}")
    # 处理目标变量，初始化MB
    # targets = list(range(n_vars))[-1]
    targets = list(range(n_vars))[-1]
    # print(targets, '11')
    # start_time = time.time()  # 记录时间
    result = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
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
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
        print("当前处理到第", fold, "折\n")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
        # result = []  # 特征提取结果数组
        # for ck in Class:  # 为每一个类别进行特征选择      #X为纯数组，y为series
            # csmdccmr(X_train.values, np.ravel(y_train), ck, n_selected_features=time)/
        # print(X.shape[1])
        result1 = IAMB(X.shape[1])
        result2 = Inter_IAMB(X.shape[1])
        print("\nresult1:",result1,"\nresult2:",result2)
        # print(result)
        # np_result1 = np.array(result1, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        # np_result2 = np.array(result2, dtype=object)
        # 保存到txt文件
        # file_name1 = r"feature_select_result/" + 'IAMB' + '/' + dataname + str(fold) + "_result-0.txt"
        # file_name2 = r"feature_select_result/" + 'IAMB' + '/' + dataname + str(fold) + "_result-i.txt"
        # np.savetxt(file_name1, np_result1, fmt='%s', delimiter="\t")
        # np.savetxt(file_name2, np_result2, fmt='%s', delimiter="\t")
        # np.savetxt(file_name, uni, delimiter =’, ’ fmt = ‘ %s’)
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件//
        # result = np.loadtxt(file_name)  # 加载特征提取结果
        # result = np.array(result)

        # 对每一个数量的特征选择做测试
        # csmdccmrtest(time, result, X_test, y_test, X_train, y_train)//
    # if algorithm == "IAMB":
    #     for target in [[targets]]:
    #         result.append(IAMB(target))

    # 继续添加其他算法选择逻辑...

    # end_time = time.time()
    # perform_time = end_time - start_time
    # print(result)

# methods = ['Inter-IAMB']
# datasets = datanames
# bf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# bf1table.columns = methods
# bf1table.index = datasets
# sf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# sf1table.columns = methods
# sf1table.index = datasets
# kf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# kf1table.columns = methods
# kf1table.index = datasets
# rf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# rf1table.columns = methods
# rf1table.index = datasets
#
# bauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# bauctable.columns = methods
# bauctable.index = datasets
# sauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# sauctable.columns = methods
# sauctable.index = datasets
# kauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# kauctable.columns = methods
# kauctable.index = datasets
# rauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# rauctable.columns = methods
# rauctable.index = datasets
#
# # methods = ['csmdccmr',]#'cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs',
# #设置最大化展示
# # pd.set_option('display.max_rows', None)
# # pd.set_option('display.max_columns', None)
# kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
# for method in methods:#对每个方法进行处理
#     print(method,"=============================================================================")
#     #存储不同方法所有数据集23个的结果
#     for dataname in datasets:#对于每个数据集进行处理
#         iposition = datasets.index(dataname)#定方法的位置
#         #读取数据及其真实值
#         data_url = dataname + '.mat'
#         base_url = r"./dataset/"
#         url = base_url + data_url  # 数据路径
#         data = scio.loadmat(url)  # 读取数据文件
#         print(dataname)
#         #读取文件中的真实标签
#         y0 = pd.DataFrame(data['Y'])  # 读取标签
#         # 应用函数到 DataFrame 的每个元素
#         y0 = y0.applymap(changetosinge)
#         #将标签控制在0-n
#         label_encoder = LabelEncoder()
#         y_encoded = label_encoder.fit_transform(y0)
#         Class = set(y_encoded)
#         y = pd.DataFrame(y_encoded)
#         classes = list(Class)
#
#         #当前数据集的十折数据所有样本结果存储
#         # 十折交叉验证
#         allfold_b_f1 = []
#         allfold_k_f1 = []
#         allfold_s_f1 = []
#         allfold_r_f1 = []
#         allfold_b_auc = []
#         allfold_k_auc = []
#         allfold_s_auc = []
#         allfold_r_auc = []
#
#         for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
#             print("当前是第",fold)
#             fold_b_f1 = []
#             fold_k_f1 = []
#             fold_s_f1 = []
#             fold_r_f1 = []
#             fold_b_auc = []
#             fold_k_auc = []
#             fold_s_auc = []
#             fold_r_auc = []
#             b_all_auc = []
#             k_all_auc = []
#             s_all_auc = []
#             r_all_auc = []
#             y_test = y.iloc[test_index]  # 提取当前折测试集条目
#             #读取预测结果
#             file_name1 = r"result/" + dataname + str(fold) + "_predict_pro.mat"
#             file_name2 = r"result/" + dataname + str(fold) + "_predict_lable.mat"
#             predict_pro = scio.loadmat(file_name1)  # 读取数据文件
#             predict_lable = scio.loadmat(file_name2)  # 读取数据文件
#             lenth = len(y_test)#当前折的测试集数量
#
#             #对于每一折的数据都读取对应数量
#             bpredict_probabity = np.array(predict_pro['b'])[:, :, :lenth, :]#time个结果，每个结果有class个分类器，每个分类器有n个样本和class种可能
#             kpredict_probabity = np.array(predict_pro['k'])[:, :, :lenth, :]
#             spredict_probabity = np.array(predict_pro['s'])[:, :, :lenth, :]
#             rpredict_probabity = np.array(predict_pro['r'])[:, :, :lenth, :]
#             bpredict_labels = pd.DataFrame(predict_lable['b']).iloc[:,:lenth]#time个结果，每个结果n个样本，
#             kpredict_labels = pd.DataFrame(predict_lable['k']).iloc[:,:lenth]
#             spredict_labels = pd.DataFrame(predict_lable['s']).iloc[:,:lenth]
#             rpredict_labels = pd.DataFrame(predict_lable['r']).iloc[:,:lenth]
#             #转置矩阵，使行为样本列为特征数量
#             bpredict_labels = bpredict_labels.T
#             kpredict_labels = kpredict_labels.T
#             spredict_labels = spredict_labels.T
#             rpredict_labels = rpredict_labels.T
#             feature_num = bpredict_labels.shape[1]
#             for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
#                 # print(y_true_binarized,bprobabily_all[i])
#                 # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
#                 time_bf1 = f1_score(bpredict_labels[i], y_test, average='micro')
#                 time_kf1 = f1_score(kpredict_labels[i], y_test, average='micro')
#                 time_sf1 = f1_score(spredict_labels[i], y_test, average='micro')
#                 time_rf1 = f1_score(rpredict_labels[i], y_test, average='micro')
#                 # 种类个数特殊处理，为了多分类的auc进行处理
#                 if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
#                     y_true_binarized = pd.get_dummies(y_test[0])
#                     y_true_binarized_array = y_true_binarized.values
#                 else:  # 多类别可以直接调用函数
#                     y_true_binarized = label_binarize(y_test, classes=classes)
#                 # for j in range(0, len(classes)):  # 不同分类器结果进行分别处理
#                 b_class_auc = roc_auc_score(y_true_binarized, bpredict_probabity[i][0], average='macro', multi_class='ovr')
#                 k_class_auc = roc_auc_score(y_true_binarized, kpredict_probabity[i][0], average='macro', multi_class='ovr')
#                 s_class_auc = roc_auc_score(y_true_binarized, spredict_probabity[i][0], average='macro', multi_class='ovr')
#                 r_class_auc = roc_auc_score(y_true_binarized, rpredict_probabity[i][0], average='macro', multi_class='ovr')
#                 b_all_auc.append(b_class_auc)#存储不同分类器的auc结果
#                 k_all_auc.append(k_class_auc)
#                 s_all_auc.append(s_class_auc)
#                 r_all_auc.append(r_class_auc)
#                 time_bauc = np.mean(b_all_auc)
#                 time_kauc = np.mean(k_all_auc)
#                 time_sauc = np.mean(s_all_auc)
#                 time_rauc = np.mean(r_all_auc)
#                 fold_b_f1.append(time_bf1)
#                 fold_k_f1.append(time_kf1)
#                 fold_s_f1.append(time_sf1)
#                 fold_r_f1.append(time_rf1)
#                 fold_b_auc.append(time_bauc)
#                 fold_k_auc.append(time_kauc)
#                 fold_s_auc.append(time_sauc)
#                 fold_r_auc.append(time_rauc)
#             allfold_b_f1.append(fold_b_f1)
#             allfold_k_f1.append(fold_k_f1)
#             allfold_s_f1.append(fold_s_f1)
#             allfold_r_f1.append(fold_r_f1)
#             allfold_b_auc.append(fold_b_auc)
#             allfold_k_auc.append(fold_k_auc)
#             allfold_s_auc.append(fold_s_auc)
#             allfold_r_auc.append(fold_r_auc)
#         #当前数据集的结果为所有特征数量结果的平均数
#         #处理完后存入表格记录
#         bf1table[method][iposition] =  np.mean(allfold_b_f1 )
#         sf1table[method][iposition] =  np.mean(allfold_k_f1 )
#         kf1table[method][iposition] =  np.mean(allfold_s_f1 )
#         rf1table[method][iposition] =  np.mean(allfold_r_f1 )
#         bauctable[method][iposition] = np.mean(allfold_b_auc)
#         sauctable[method][iposition] = np.mean(allfold_k_auc)
#         kauctable[method][iposition] = np.mean(allfold_s_auc)
#         rauctable[method][iposition] = np.mean(allfold_r_auc)
#
# tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
# # 计算每列的平均值
# for i, table in enumerate(tables):
#     average_row = table.mean(axis=0)
#     table.loc['average'] = average_row
#
# #对每个表格求平均值并按照大小进行升序排序
# bf1table = bf1table.sort_values(by='average', axis=1)
# sf1table = sf1table.sort_values(by='average', axis=1)
# kf1table = kf1table.sort_values(by='average', axis=1)
# rf1table = rf1table.sort_values(by='average', axis=1)
# bauctable = bauctable.sort_values(by='average', axis=1)
# sauctable = sauctable.sort_values(by='average', axis=1)
# kauctable = kauctable.sort_values(by='average', axis=1)
# rauctable = rauctable.sort_values(by='average', axis=1)
# print(bf1table,kf1table,sf1table,rf1table)
# # print(bauctable,sauctable,kauctable,rauctable)
# #存储
# T = 3
# # bf1table.to_excel( 'tables/'+T+'/ten-bf1table.xlsx', index=True)
# # sf1table.to_excel( 'tables/'+T+'/ten-sf1table.xlsx', index=True)
# # kf1table.to_excel( 'tables/'+T+'/ten-kf1table.xlsx', index=True)
# # rf1table.to_excel( 'tables/'+T+'/ten-rf1table.xlsx', index=True)
# # bauctable.to_excel('tables/'+T+'/ten-bauctable.xlsx', index=True)
# # sauctable.to_excel('tables/'+T+'/ten-sauctable.xlsx', index=True)
# # kauctable.to_excel('tables/'+T+'/ten-kauctable.xlsx', index=True)
# # rauctable.to_excel('tables/'+T+'/ten-rauctable.xlsx', index=True)
# #绘制箱线图
# import matplotlib.pyplot as plt
# # import numpy as np
# # bf1table = pd.read_excel( 'tables/'+T+'/ten-bf1table.xlsx', index_col=0)
# # sf1table = pd.read_excel( 'tables/'+T+'/ten-sf1table.xlsx', index_col=0)
# # kf1table = pd.read_excel( 'tables/'+T+'/ten-kf1table.xlsx', index_col=0)
# # rf1table = pd.read_excel( 'tables/'+T+'/ten-rf1table.xlsx', index_col=0)
# # bauctable = pd.read_excel('tables/'+T+'/ten-bauctable.xlsx', index_col=0)
# # sauctable = pd.read_excel('tables/'+T+'/ten-sauctable.xlsx', index_col=0)
# # kauctable = pd.read_excel('tables/'+T+'/ten-kauctable.xlsx', index_col=0)
# # rauctable = pd.read_excel('tables/'+T+'/ten-rauctable.xlsx', index_col=0)
# # tables = [bf1table,sf1table,kf1table,rf1table,bauctable,sauctable,kauctable,rauctable]
#
#
#
