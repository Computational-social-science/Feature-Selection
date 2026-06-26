import scipy.io as scio
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import math

from csmdccmr import cmiddspecial,middspecial

#
pd.set_option('display.max_columns', None)  # 显示所有列

def changetosinge(x):
    return float(x)

# datanames = [ 'Waveform', 'Synthetic_control','Authorship','Factors','Musk1','Su','Sorlie', 'Movement_libras','Yeoh','TOX_171', 'Orlraws10P', 'CLL_SUB_111', 'Yale', 'WarpAR10P', 'GLIOMA',   'ORL', 'WarpPIE10P','Pixel',]#,],,
datanames = ['Dermatology', 'Waveform','Wdbc','Movement_libras', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
            'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene']#,


N_CASES_PER_DF = 20  # Example value, adjust as needed
alpha = 0.05  # Significance level, example value
k_tradeoff = 0.5  # Example value for k-greedy
# algorithm = "IAMB"

import numpy as np
from scipy.special import gammaincc

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



def k_greedy(dep):
    n_cands = np.sum(dep >= thresh)
    cand = [i for i in range(len(dep)) if dep[i] >= thresh]
    # print(12345,cand)
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

def Inter_IAMB(targets):
    result = []
    for target in targets:
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
                            if cond[-1] == -1:
                                cond[-1] = i
            else:
                break

        # 最终调整
        mb = [0 if x == -1 else x for x in mb]
        print(mb)
        result.append(mb)

    # 保存结果
    relation_matrix = np.array(result)
    np.savetxt('result.txt', relation_matrix)

thresh = 0
def changed_IAMB(target):
    result = []
    for c in range(t):
        mb = [0 if state > 1 else -1 for i, state in enumerate(n_states)]
        mb[target] = -1
        n_conds = 0
        cond = np.full(max_max_cond_size, -1, dtype=int)
        dep = np.zeros(n_vars, dtype=float)
        #增长阶段
        while True:
            stop = False
            if n_conds < max_max_cond_size:
                dep[:] = 0.0#初始化所有dep为0
                for i in range(n_vars): #对所有变量，若未被选中，则计算其dep，若dep大于，记录
                    if mb[i] == 0:
                        # dep[i] = compute_dep([i, target, [cond], c)  # 计算独立性
                        # print(i,target,cond)
                        nowcond = [x for x in cond if x != -1]
                        if nowcond == []:
                            x = middspecial(data_processed[i], data_processed[target], c)
                        # dep[i] = compute_info([i], target, [cond], c)#计算独立性
                        else:
                            x = cmiddspecial(data_processed[i], data_processed[nowcond], data_processed[target], c)
                        dep[i] = x
                        if dep[i] > 0:
                            stop = True
                # print("a",dep)
                # print(nowcond)
                # print(data_processed[nowcond])
                # print(nowcond,n_conds)
            # print(dep)
            if stop:#记录dep最大，将其归于mb子集
                aux = k_greedy(dep)
                mb[aux] = 1
                cond[n_conds] = aux
                n_conds += 1
                # print(n_conds)
            else:
                break
        # print(mb)
        # 缩减阶段
        for i in range(n_vars):
            if mb[i] == 1:#对于所有已经选定的mb元素
                if i in cond:#检查每一个重新计算独立性,先把index从cond中取出
                    index = np.where(cond == i)[0][0]
                    cond[index:-1] = cond[index + 1:]#index后面的所有向左移动一格，覆盖index所在位置
                    cond[-1] = -1  # Set the last position to -1
                    nowcond = [x for x in cond if x != -1]#当前的条件子集序列
                    # dep[i] = compute_info([i], target, [cond], c)#计算独立性
                    dep[i] = cmiddspecial(data_processed[i],data_processed[nowcond],  data_processed[target], c)

                # print(dep[i])
                if dep[i] <= 0:#若新dep小于阈值，则去除该元素
                    n_conds -= 1
                    mb[i] = 0
                else:
                    # Restore i to cond
                    if cond[-1] == -1:
                        cond[-1] = i
        # Final adjustment
        # print(mb)
        mbn = []
        mb[mb == -1] = 0
        for index,x in enumerate(mb):
            if x == 1:
                mbn.append(index)
        # print(mb[mb == 1].)
        result.append(mbn)
    print(result)
    return result
        # file_name = r"feature_select_result/" + 'INMB' + '/' + dataname + str(fold) + "_result.txt"
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
        # print(data_processed[target].nunique())
        # relation_matrix = np.array(result)
        # print(relation_matrix)
        # np.set_printoptions(threshold=np.inf)
        # np.savetxt('result.txt', relation_matrix)
def csmdccmrtest(time, result, X_test, y_test, X_train, y_train):
    # for i in range(time):#对不同数量的特征选择结果进行测试
    nowpick = result # 当前i个特征的特征选择结果
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
    for item in nowpick:
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
        # bsingleclasspro.append(bprobabilities)#记录n种分类器单次概率
        # ksingleclasspro.append(kprobabilities)
        # ssingleclasspro.append(sprobabilities)
        # rsingleclasspro.append(rprobabilities)
        #根据单次测试分类概率得到分类标签，共计算50次
        # bsinglepredicted_labels = (np.argmax(bresultmax, axis=1))
        # ksinglepredicted_labels = (np.argmax(kresultmax, axis=1))
        # ssinglepredicted_labels = (np.argmax(sresultmax, axis=1))
        # rsinglepredicted_labels = (np.argmax(rresultmax, axis=1))
        #记录time次分类结果，50次，其中每次都有class个预测矩阵
        b_time_predict_probabity.append(bprobabilities)
        k_time_predict_probabity.append(kprobabilities)
        s_time_predict_probabity.append(sprobabilities)
        r_time_predict_probabity.append(rprobabilities)
        # 记录time次分类结果，50次，其中每次都是单次的预测标签
        b_time_predict_labels.append(np.argmax(bresultmax, axis=1))
        k_time_predict_labels.append(np.argmax(kresultmax, axis=1))
        s_time_predict_labels.append(np.argmax(sresultmax, axis=1))
        r_time_predict_labels.append(np.argmax(rresultmax, axis=1))
    predict_pro   = {'b': b_time_predict_probabity, 'k': k_time_predict_probabity, 's': s_time_predict_probabity, 'r': r_time_predict_probabity,}
    predict_lable = {'b': b_time_predict_labels, 'k': k_time_predict_labels, 's': s_time_predict_labels, 'r': r_time_predict_labels,}
    file_name1 = r"result/new+" + dataname + str(fold) +  "_predict_pro.mat"
    file_name2 = r"result/new+" + dataname + str(fold) +  "_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)

for dataname in datanames:
    print(dataname)
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
    t = len(Class)
    y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
    # print(y)

    data_processed = pd.concat([X, y], axis=1)
    n_vars = data_processed.shape[1]
    n_cases = data_processed.shape[0]
    n_states = np.full(n_vars, 5)
    n_states[-1] = t
    max_max_cond_size = n_vars  # 设置最大条件子集数量
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
    accmax = []  # 准确度记录数组，为了画折线图
    # 十折交叉验证
    result = changed_IAMB(n_vars - 1)
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
    #     print("当前处理到第", fold, "折\n")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
    #     # result = []  # 特征提取结果数组
    #     # for ck in Class:  # 为每一个类别进行特征选择      #X为纯数组，y为series
    #         # csmdccmr(X_train.values, np.ravel(y_train), ck, n_selected_features=time)/
    #     result = changed_IAMB(n_vars-1)
    #     # print(result)
    #     np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
        csmdccmrtest(len(result), result, X_test, y_test, X_train, y_train)
    #     # 保存到txt文件
    #     file_name = r"feature_select_result/" + 'INMB' + '/' + dataname + str(fold) + "_result.txt"
    #     np.savetxt(file_name, np_result, fmt='%s', delimiter="\t")

        # np.savetxt(file_name, uni, delimiter =’, ’ fmt = ‘ %s’)
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件//
        # result = np.loadtxt(file_name)  # 加载特征提取结果
        # result = np.array(result)




methods = ['Info-']
#==================================数据预测完毕，进行预测结果的处理=======================================================================
#处理得到结果计算出的F1和AUC
T = '1'#用于区别储存
datasets = datanames
# methods = ['IAMB']#'cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs',
#定义八个表格用于存储最后的不同方法和不同数据集的F1和AUC结果
bf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
bf1table.columns = methods
bf1table.index = datasets
sf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
sf1table.columns = methods
sf1table.index = datasets
kf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
kf1table.columns = methods
kf1table.index = datasets
rf1table = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
rf1table.columns = methods
rf1table.index = datasets

bacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
bacctable.columns = methods
bacctable.index = datasets
sacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
sacctable.columns = methods
sacctable.index = datasets
kacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
kacctable.columns = methods
kacctable.index = datasets
racctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
racctable.columns = methods
racctable.index = datasets


bauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
bauctable.columns = methods
bauctable.index = datasets
sauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
sauctable.columns = methods
sauctable.index = datasets
kauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
kauctable.columns = methods
kauctable.index = datasets
rauctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
rauctable.columns = methods
rauctable.index = datasets

def calculate_accuracy(predicted_labels, true_labels):
    # 确保预测结果和真实标签的长度相同
    if len(predicted_labels) != len(true_labels):
        raise ValueError("预测结果和真实标签的长度不一致。")
    predicted_labels = [int(x) for x in predicted_labels]
    true_labels = [int(x) for x in true_labels]
    # print(predicted_labels,'\n',true_labels)
    # 计算预测正确的数量
    correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)

    # 计算准确度
    accuracy = correct_count / len(true_labels)

    return accuracy

#设置最大化展示
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
for method in methods:#对每个方法进行处理
    print(method,"=============================================================================")
    #存储不同方法所有数据集23个的结果
    for dataname in datasets:#对于每个数据集进行处理
        iposition = datasets.index(dataname)#定方法的位置
        #读取数据及其真实值
        data_url = dataname + '.mat'
        base_url = r"./dataset/"
        url = base_url + data_url  # 数据路径
        data = scio.loadmat(url)  # 读取数据文件
        print(dataname)
        #读取文件中的真实标签
        y0 = pd.DataFrame(data['Y'])  # 读取标签
        # 应用函数到 DataFrame 的每个元素
        y0 = y0.applymap(changetosinge)
        #将标签控制在0-n
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y0)
        Class = set(y_encoded)
        y = pd.DataFrame(y_encoded)
        classes = list(Class)

        #当前数据集的十折数据所有样本结果存储
        # 十折交叉验证
        allfold_b_f1 = []
        allfold_k_f1 = []
        allfold_s_f1 = []
        allfold_r_f1 = []
        allfold_b_auc = []
        allfold_k_auc = []
        allfold_s_auc = []
        allfold_r_auc = []
        bpredict_labels_all = pd.DataFrame()
        kpredict_labels_all = pd.DataFrame()
        spredict_labels_all = pd.DataFrame()
        rpredict_labels_all = pd.DataFrame()
        bprobabily_all = []
        kprobabily_all = []
        sprobabily_all = []
        rprobabily_all = []
        bpredict_probabity_list = []
        kpredict_probabity_list = []
        spredict_probabity_list = []
        rpredict_probabity_list = []
        bpredict_labels_list = []
        kpredict_labels_list = []
        spredict_labels_list = []
        rpredict_labels_list = []
        y_all = pd.DataFrame()  # 真实值的所有样本结果
        for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
            print("当前是第",fold)
            fold_b_f1 = []
            fold_k_f1 = []
            fold_s_f1 = []
            fold_r_f1 = []
            fold_b_acc = []
            fold_k_acc = []
            fold_s_acc = []
            fold_r_acc = []
            fold_b_auc = []
            fold_k_auc = []
            fold_s_auc = []
            fold_r_auc = []
            b_all_auc = []
            k_all_auc = []
            s_all_auc = []
            r_all_auc = []
            y_test = y.iloc[test_index]  # 提取当前折测试集条目
            #读取预测结果
            file_name1 = r"result/new+" + dataname + str(fold) + "_predict_pro.mat"
            file_name2 = r"result/new+" + dataname + str(fold) + "_predict_lable.mat"
            predict_pro = scio.loadmat(file_name1)  # 读取数据文件
            predict_lable = scio.loadmat(file_name2)  # 读取数据文件
            lenth = len(y_test)#当前折的测试集数量
            #对于每一折的数据都读取对应数量
            bpredict_probabity = np.array(predict_pro['b'])[:, :lenth, :]#time个结果，每个结果有class个分类器，每个分类器有n个样本和class种可能
            kpredict_probabity = np.array(predict_pro['k'])[:, :lenth, :]
            spredict_probabity = np.array(predict_pro['s'])[:, :lenth, :]
            rpredict_probabity = np.array(predict_pro['r'])[:, :lenth, :]
            spredict_labels = pd.DataFrame(predict_lable['s']).iloc[:, :lenth]
            bpredict_labels = pd.DataFrame(predict_lable['b']).iloc[:, :lenth]#time个结果，每个结果n个样本，
            kpredict_labels = pd.DataFrame(predict_lable['k']).iloc[:, :lenth]
            rpredict_labels = pd.DataFrame(predict_lable['r']).iloc[:, :lenth]
            # print(bpredict_probabity)
            bpredict_probabity_list.append(bpredict_probabity)
            kpredict_probabity_list.append(kpredict_probabity)
            spredict_probabity_list.append(spredict_probabity)
            rpredict_probabity_list.append(rpredict_probabity)
            bpredict_labels_list.append(bpredict_labels)
            kpredict_labels_list.append(kpredict_labels)
            spredict_labels_list.append(spredict_labels)
            rpredict_labels_list.append(rpredict_labels)

            bpredict_labels_all = pd.concat([bpredict_labels_all, bpredict_labels], axis=1)
            kpredict_labels_all = pd.concat([kpredict_labels_all, kpredict_labels], axis=1)
            spredict_labels_all = pd.concat([spredict_labels_all, spredict_labels], axis=1)
            rpredict_labels_all = pd.concat([rpredict_labels_all, rpredict_labels], axis=1)
            y_all = pd.concat([y_all, y_test], axis=0)
        # print(bpredict_labels_all,y_all)
        bprobabily_all = np.concatenate(bpredict_probabity_list, axis=1)
        kprobabily_all = np.concatenate(kpredict_probabity_list, axis=1)
        sprobabily_all = np.concatenate(spredict_probabity_list, axis=1)
        rprobabily_all = np.concatenate(rpredict_probabity_list, axis=1)
        # print(bprobabily_all.shape,bpredict_labels_all.shape)
        # 转置矩阵，使行为样本列为特征数量
        sample_num = bpredict_labels_all.shape[1]
        feature_num = bpredict_labels_all.shape[0]
        new_column_names = range(sample_num)
        bpredict_labels_all.columns = new_column_names[:sample_num]
        bpredict_labels_all = bpredict_labels_all.T
        kpredict_labels_all.columns = new_column_names[:sample_num]
        kpredict_labels_all = kpredict_labels_all.T
        spredict_labels_all.columns = new_column_names[:sample_num]
        spredict_labels_all = spredict_labels_all.T
        rpredict_labels_all.columns = new_column_names[:sample_num]
        rpredict_labels_all = rpredict_labels_all.T
        # print(rpredict_labels_all,bprobabily_all,y_all)
        # for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
            # print(y_true_binarized,bprobabily_all[i])
            # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
        # print(bpredict_labels_all)
        time_bf1 = f1_score(bpredict_labels_all[0], y_all, average='micro')
        time_kf1 = f1_score(kpredict_labels_all[0], y_all, average='micro')
        time_sf1 = f1_score(spredict_labels_all[0], y_all, average='micro')
        time_rf1 = f1_score(rpredict_labels_all[0], y_all, average='micro')
        # print(time_bf1, time_kf1, time_sf1, time_rf1)
        acc_b = calculate_accuracy(bpredict_labels_all[0], y_all[0], )
        acc_k = calculate_accuracy(kpredict_labels_all[0], y_all[0], )
        acc_s = calculate_accuracy(spredict_labels_all[0], y_all[0], )
        acc_r = calculate_accuracy(rpredict_labels_all[0], y_all[0], )
        # print(acc_b, acc_k, acc_s, acc_r)
        # 种类个数特殊处理，为了多分类的auc进行处理
        if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
            y_true_binarized = pd.get_dummies(y_all[0])
            y_true_binarized_array = y_true_binarized.values
        else:  # 多类别可以直接调用函数
            y_true_binarized = label_binarize(y_all, classes=classes)
        # for j in range(0, len(classes)):  # 不同分类器结果进行分别处理
        # print(y_true_binarized,bprobabily_all)
        b_all_auc = roc_auc_score(y_true_binarized, bprobabily_all[0], average='macro', multi_class='ovr')
        k_all_auc = roc_auc_score(y_true_binarized, kprobabily_all[0], average='macro', multi_class='ovr')
        s_all_auc = roc_auc_score(y_true_binarized, sprobabily_all[0], average='macro', multi_class='ovr')
        r_all_auc = roc_auc_score(y_true_binarized, rprobabily_all[0], average='macro', multi_class='ovr')
        # print(b_all_auc, k_all_auc, s_all_auc, r_all_auc)
        # time_bauc = np.mean(b_all_auc)
        # time_kauc = np.mean(k_all_auc)
        # time_sauc = np.mean(s_all_auc)
        # time_rauc = np.mean(r_all_auc)
        fold_b_f1.append(time_bf1)
        fold_k_f1.append(time_kf1)
        fold_s_f1.append(time_sf1)
        fold_r_f1.append(time_rf1)
        fold_b_acc.append(acc_b)
        fold_k_acc.append(acc_k)
        fold_s_acc.append(acc_s)
        fold_r_acc.append(acc_r)
        fold_b_auc.append(b_all_auc)
        fold_k_auc.append(k_all_auc)
        fold_s_auc.append(s_all_auc)
        fold_r_auc.append(r_all_auc)
        # allfold_b_f1.append(fold_b_f1)
        # allfold_k_f1.append(fold_k_f1)
        # allfold_s_f1.append(fold_s_f1)
        # allfold_r_f1.append(fold_r_f1)
        # allfold_b_auc.append(fold_b_auc)
        # allfold_k_auc.append(fold_k_auc)
        # allfold_s_auc.append(fold_s_auc)
        # allfold_r_auc.append(fold_r_auc)
        #当前数据集的结果为所有特征数量结果的平均数
        #处理完后存入表格记录
        bacctable[method][iposition] = np.mean(fold_b_acc)
        sacctable[method][iposition] = np.mean(fold_s_acc)
        kacctable[method][iposition] = np.mean(fold_k_acc)
        racctable[method][iposition] = np.mean(fold_r_acc)
        bf1table[method][iposition]  = np.mean(fold_b_f1)
        sf1table[method][iposition]  = np.mean(fold_s_f1)
        kf1table[method][iposition]  = np.mean(fold_k_f1)
        rf1table[method][iposition]  = np.mean(fold_r_f1)
        bauctable[method][iposition] = np.mean(fold_b_auc)
        sauctable[method][iposition] = np.mean(fold_s_auc)
        kauctable[method][iposition] = np.mean(fold_k_auc)
        rauctable[method][iposition] = np.mean(fold_r_auc)

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
print(bacctable,'\n',kacctable,'\n',sacctable,'\n',racctable,'\n')
print(bf1table,'\n',kf1table,'\n',sf1table,'\n',rf1table,'\n')
print(bauctable,'\n',sauctable,'\n',kauctable,'\n',rauctable,'\n')
#存储
bf1table.to_excel( 'tables/'+'T'+'/info-ten-bf1table.xlsx', index=True)
sf1table.to_excel( 'tables/'+'T'+'/info-ten-sf1table.xlsx', index=True)
kf1table.to_excel( 'tables/'+'T'+'/info-ten-kf1table.xlsx', index=True)
rf1table.to_excel( 'tables/'+'T'+'/info-ten-rf1table.xlsx', index=True)
bacctable.to_excel('tables/'+'T'+'/info-ten-bacctable.xlsx', index=True)
sacctable.to_excel('tables/'+'T'+'/info-ten-sacctable.xlsx', index=True)
kacctable.to_excel('tables/'+'T'+'/info-ten-kacctable.xlsx', index=True)
racctable.to_excel('tables/'+'T'+'/info-ten-racctable.xlsx', index=True)
bauctable.to_excel('tables/'+'T'+'/info-ten-bauctable.xlsx', index=True)
sauctable.to_excel('tables/'+'T'+'/info-ten-sauctable.xlsx', index=True)
kauctable.to_excel('tables/'+'T'+'/info-ten-kauctable.xlsx', index=True)
rauctable.to_excel('tables/'+'T'+'/info-ten-rauctable.xlsx', index=True)

# import pandas as pd
#
# # 定义文件路径和表格名称
# tables_path = 'tables/T/'
# file_names = [
#     'info-ten-bf1table.xlsx', 'info-ten-sf1table.xlsx', 'info-ten-kf1table.xlsx', 'info-ten-rf1table.xlsx',
#     'info-ten-bacctable.xlsx', 'info-ten-sacctable.xlsx', 'info-ten-kacctable.xlsx', 'info-ten-racctable.xlsx',
#     'info-ten-bauctable.xlsx', 'info-ten-sauctable.xlsx', 'info-ten-kauctable.xlsx', 'info-ten-rauctable.xlsx'
# ]
#
# # 初始化一个空的 DataFrame 来存储合并后的数据
# combined_df = pd.DataFrame()
#
# # 循环读取每个表格，并将其中的一列数据添加到合并表中
# for file_name in file_names:
#     file_path = tables_path + file_name
#     df = pd.read_excel(file_path, index_col=0)  # 假设数据第一列为索引
#     column_name = file_name.split('-')[2].split('.')[0]  # 使用文件名部分作为列名
#     combined_df[column_name] = df.iloc[:, 0]  # 提取表格中的第一列数据
#
# # 保存合并后的表格
# output_file = tables_path + 'info-ten-combined.xlsx'
# combined_df.to_excel(output_file, index=True)
#
# print(f"数据已成功合并并保存到 {output_file}")
