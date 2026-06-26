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
# datanames = [
            # 'Prostate_GE', 'Musk1','Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel',  ]#
datanames = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
            'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
max_max_cond_size = 100  #设置最大条件子集数量
N_CASES_PER_DF = 20  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy
algorithm = "IAMB"


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

def GSMB(target):
    result = []
    mb = [0 if n_states[i] > 1 and n_states[target] > 1 else -1 for i in range(n_vars)]
    mb[target] = -1  # 目标变量不在自身的MB中

    n_conds = 0
    cond = [-1] * max_max_cond_size

    # Growth Phase
    stop = False
    while not stop:
        stop = True
        if n_conds < max_max_cond_size:
            for i in range(n_vars):
                if mb[i] == 0:
                    dep = compute_dep(i, target, cond)
                    if dep >= 1.0:
                        stop = False
                        mb[i] = 1
                        cond[n_conds] = i
                        n_conds += 1
                        break
    print(mb)
    # Shrink Phase
    for i in range(n_vars):
        if mb[i] == 1:
            # 移除 cond 中的 i
            for j in range(n_conds):
                if cond[j] == i:
                    for k in range(j, n_conds - 1):
                        cond[k] = cond[k + 1]
                    cond[n_conds - 1] = -1
                    break

            dep = compute_dep(i, target, cond)

            if dep <= -1.0:
                n_conds -= 1
                mb[i] = 0
            else:
                cond[n_conds - 1] = i

    # 将标记为 -1 的变量设回 0
    print(mb)
    for var in range(n_vars):
        if mb[var] == -1:
            mb[var] = 0



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

    data_file_path = "./dataset/" + dataname + ".mat"
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
        print(X.shape[1])
        # result1 = IAMB(X.shape[1])
        result2 = GSMB(X.shape[1])
        print("\nresult2:",result2)
        # print(result)
        # np_result1 = np.array(result1, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        np_result2 = np.array(result2, dtype=object)
        # 保存到txt文件
        # file_name1 = r"feature_select_result/" + 'IAMB' + '/' + dataname + str(fold) + "_result-0.txt"
        file_name2 = r"feature_select_result/" + 'GSMB' + '/' + dataname + str(fold) + "_result.txt"
        # np.savetxt(file_name1, np_result1, fmt='%s', delimiter="\t")
        np.savetxt(file_name2, np_result2, fmt='%s', delimiter="\t")
        # np.savetxt(file_name, uni, delimiter =’, ’ fmt = ‘ %s’)
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件//
        # result = np.loadtxt(file_name)  # 加载特征提取结果
        result = np.array(result)

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
