import numpy as np
import pandas as pd
from scipy.special import gammaincc
from scipy.stats import chi2
from math import log
import  scipy.io as scio
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import KFold
import warnings

# 忽略 FutureWarning 警告
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DataConversionWarning)

N_CASES_PER_DF = 5
alpha = 0.05
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
def compute_dep(var, target, cond, data, n_states, n_cases):
    n_cond_states = 1
    cond = [c for c in cond if c != -1]
    for c in cond:
        n_cond_states *= n_states[c]
    # if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
    #     return 0.0

    ss_cond = np.zeros(n_cond_states, dtype=int)
    ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
    ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
    ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)

    for i in range(n_cases):
        cond_state = 0
        for c in cond:
            cond_state = cond_state * n_states[c] + data[i, c]
        ss_cond[cond_state] += 1
        ss_target[cond_state][data[i, target]] += 1
        ss_var[cond_state][data[i, var]] += 1
        ss[cond_state][data[i, target]][data[i, var]] += 1

    statistic = 0.0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            for j in range(n_states[target]):
                if ss_target[i][j] > 0:
                    for k in range(n_states[var]):
                        if ss[i][j][k] > 0:
                            expected = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
                            statistic += ss[i][j][k] * (log(ss[i][j][k]) - log(expected))
    statistic *= 2.0

    df = 0
    for i in range(n_cond_states):
        if ss_cond[i] > 0:
            df_target = np.sum(ss_target[i] > 0)
            df_var = np.sum(ss_var[i] > 0)
            df += (df_target - 1) * (df_var - 1)
    if df <= 0:
        df = 1

    dep = gammaincc(0.5 * df, 0.5 * statistic)
    if dep <= alpha:
        dep = 2.0 - dep
        if dep == 2.0:
            dep += 2.0 +statistic / df
        return dep
    else:
        dep = -1.0 - dep
        if dep == -2.0:
            dep -= -2.0 -statistic / df
        return dep
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
def resulttest(result, data_train, data_test):
    if result.shape == ():  # 标量（单个数据）
        nowpick = np.array([result])  # 将标量转换为包含一个元素的数组
    else:  # 数组（多个数据）
        nowpick = result.astype(int)
    print(type(nowpick),nowpick)
    X_test = data_test.iloc[:, nowpick].to_numpy(dtype=float)

    # print(data_train.iloc[:, nowpick],data_train.iloc[-1])
    # **Naïve Bayes (朴素贝叶斯)**
    from sklearn.naive_bayes import GaussianNB
    bayesmodel = GaussianNB()
    bayesmodel.fit(data_train.iloc[:, nowpick], data_train.iloc[:,-1])
    bpredicted_labels = bayesmodel.predict(X_test)
    bprobabilities = bayesmodel.predict_proba(X_test)

    # **k-Nearest Neighbors (k-最近邻算法)**:
    from sklearn.neighbors import KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(data_train.iloc[:, nowpick], data_train.iloc[:,-1])
    kpredicted_labels = knn_classifier.predict(X_test)
    kprobabilities = knn_classifier.predict_proba(X_test)

    # **Support Vector Machines (支持向量机)**:
    from sklearn.svm import SVC
    svm_classifier = SVC(probability=True)
    svm_classifier.fit(data_train.iloc[:, nowpick], data_train.iloc[:,-1])
    spredicted_labels = svm_classifier.predict(X_test)
    sprobabilities = svm_classifier.predict_proba(X_test)

    # random Forest (随机森林)**:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(data_train.iloc[:, nowpick], data_train.iloc[:,-1])
    rpredicted_labels = rf_classifier.predict(X_test)
    rprobabilities = rf_classifier.predict_proba(X_test)

    bpredict_probabity = bprobabilities
    kpredict_probabity = kprobabilities
    spredict_probabity = sprobabilities
    rpredict_probabity = rprobabilities
    bpredict_labels = bpredicted_labels
    kpredict_labels = kpredicted_labels
    spredict_labels = spredicted_labels
    rpredict_labels = rpredicted_labels

    bf1 = f1_score(data_test.iloc[:,-1], bpredict_labels, average='micro')
    kf1 = f1_score(data_test.iloc[:,-1], kpredict_labels, average='micro')
    sf1 = f1_score(data_test.iloc[:,-1], spredict_labels, average='micro')
    rf1 = f1_score(data_test.iloc[:,-1], rpredict_labels, average='micro')# 种类个数
    bacc = accuracy_score(data_test.iloc[:,-1], bpredict_labels)
    kacc = accuracy_score(data_test.iloc[:,-1], kpredict_labels)
    sacc = accuracy_score(data_test.iloc[:,-1], spredict_labels)
    racc = accuracy_score(data_test.iloc[:,-1], rpredict_labels)# 种类个数
    # classnum = data_test.iloc[:,-1].nunique()
    # classes = data_test.iloc[:,-1].unique()
    # if (classnum == 2):  # 若只分为两个类，需要特殊处理
    #     y_binarized = pd.get_dummies(data_test.iloc[:,-1])
    #     y_true_binarized = y_binarized.values
    # else:  # 多类别可以直接调用函数
    #     y_true_binarized = label_binarize(data_test.iloc[:,-1], classes=classes)

    # bauc = roc_auc_score(y_true_binarized, bpredict_probabity, average='macro', multi_class='ovr')
    # kauc = roc_auc_score(y_true_binarized, kpredict_probabity, average='macro', multi_class='ovr')
    # sauc = roc_auc_score(y_true_binarized, spredict_probabity, average='macro', multi_class='ovr')
    # rauc = roc_auc_score(y_true_binarized, rpredict_probabity, average='macro', multi_class='ovr')

    f1   = {'b': bf1, 'k': kf1, 's': sf1, 'r': rf1,}
    acc = {'b': bacc, 'k': kacc, 's': sacc, 'r': racc,}
    pred ={ 'b': bpredict_probabity, 'k': kpredict_probabity, 's': spredict_probabity, 'r': rpredict_probabity,}
    # auc = {'b': bauc, 'k': kauc, 's': sauc, 'r': rauc,}
    # print(predict_pro,predict_lable)
    # file_name1 = r"result/" + "eamb" + '/' + dataname + str(fold) +"_predict_pro.mat"
    # file_name2 = r"result/" + "eamb" + '/' + dataname + str(fold) +"_predict_lable.mat"
    # scio.savemat(file_name1, predict_pro)
    # scio.savemat(file_name2, predict_lable)
    return f1,acc,pred


from sklearn.metrics import roc_auc_score
import numpy as np


def safe_auc_score(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    unique_labels = np.unique(y_true)

    # 如果是二分类
    if len(unique_labels) == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            return roc_auc_score(y_true, y_prob[:, 1])  # 正类概率
        elif y_prob.ndim == 1:
            return roc_auc_score(y_true, y_prob)
        else:
            raise ValueError(f"二分类时 y_prob 的维度不符合要求，shape={y_prob.shape}")

    # 如果是多分类
    elif len(unique_labels) > 2:
        if y_prob.ndim == 2 and y_prob.shape[1] >= len(unique_labels):
            return roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
        else:
            raise ValueError(f"多分类时 y_prob 的 shape 应为 (n_samples, n_classes)，当前为 {y_prob.shape}")

    else:
        raise ValueError("y_true 中没有足够的类别标签")


# methods = []'GSMB',
# # datasets = ['madelon','SRBCT','Colon','spambase','splice','dna','dexter',]#'Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
# #             # 'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',
# datasets = ['Dermatology',]# 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             # 'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
methods = ['FSMB',]#,'madelon','SRBCT','Colon','spambase','splice','dna','dexter',
# datasets = ['madelon','SRBCT','Colon','spambase','splice','dna',]#'Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             # 'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',
# datasets = ['Dermatology', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene','madelon','SRBCT','Colon','spambase','splice','dna','dexter']#
datasets = [ 'Arcene','Authorship','CLL_SUB_111','Colon','Dermatology','dna', 'Factors','GLIOMA','madelon','Movement_libras','Musk1','Orlraws10P','Pixel',  'Prostate_GE', 'Sorlie', 'Su','SRBCT','spambase','splice','Synthetic_control', 'TOX_171', 'Waveform','Wdbc',  'WarpPIE10P','WarpAR10P', 'Yeoh',  ]#
base_url = r"./dataset/"

# 定义八个表格用于存储最后的不同方法和不同数据集的F1和AUC结果
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

bstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
bstdtable.columns = methods
bstdtable.index = datasets
sstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
sstdtable.columns = methods
sstdtable.index = datasets
kstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
kstdtable.columns = methods
kstdtable.index = datasets
rstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
rstdtable.columns = methods
rstdtable.index = datasets

for dataname in datasets:
    # jposition = datasets.index(dataname)
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
    X = pd.DataFrame()
    #离散化并确定每个特征的取值
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
    for method in methods:
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
        allf1b,allf1k,allf1s,allf1r  = [],[],[],[]
        allaccb,allacck,allaccs,allaccr = [],[],[],[]
        y_true,bpred,kpred,spred,rpred= [],[],[],[],[]
        for fold, (train_index, test_index) in enumerate(kfold.split(data_processed)):  # fold赋值之后就是0-9
            print("\n当前处理到第", fold, "折")
            data_train, data_test = data_processed.iloc[train_index], data_processed.iloc[test_index]
            # np_result = np.array(result, dtype=object)  # 使用 dtype=object 以允许不规则的长度
            # 保存到txt文件
            file_name = r"FeatureSelect/" + method+'/' + dataname  + "_result"+str(fold)+".txt"
            result = np.loadtxt(file_name)#加载特征提取结果
            print("result:",result)
            result = np.array(result)
            if result.size == 0:
                continue
            else:
                f1,acc,pred= resulttest(result, data_train, data_test)
                allf1b.append(f1['b'])
                allf1k.append(f1['k'])
                allf1s.append(f1['s'])
                allf1r.append(f1['r'])
                allaccb.append(acc['b'])
                allacck.append(acc['k'])
                allaccs.append(acc['s'])
                allaccr.append(acc['r'])
                y_true.extend(data_test.iloc[:,-1])
                bpred.extend(pred['b'])
                kpred.extend(pred['k'])
                spred.extend(pred['s'])
                rpred.extend(pred['r'])
            # allaucb.append(auc['b'])
            # allauck.append(auc['k'])
            # allaucs.append(auc['s'])
            # allaucr.append(auc['r'])
        bf1table[method][dataname] = np.mean(allf1b)
        kf1table[method][dataname] = np.mean(allf1k)
        sf1table[method][dataname] = np.mean(allf1s)
        rf1table[method][dataname] = np.mean(allf1r)
        bacctable[method][dataname] = np.mean(allaccb)
        kacctable[method][dataname] = np.mean(allacck)
        sacctable[method][dataname] = np.mean(allaccs)
        racctable[method][dataname] = np.mean(allaccr)
        bstdtable[method][dataname] = np.std(allaccb,ddof=1)
        kstdtable[method][dataname] = np.std(allacck,ddof=1)
        sstdtable[method][dataname] = np.std(allaccs,ddof=1)
        rstdtable[method][dataname] = np.std(allaccr,ddof=1)
        bauctable[method][dataname] = safe_auc_score(y_true,bpred)
        kauctable[method][dataname] = safe_auc_score(y_true,kpred)
        sauctable[method][dataname] = safe_auc_score(y_true,spred)
        rauctable[method][dataname] = safe_auc_score(y_true,rpred)
        # bauctable[method][dataname] = np.mean(allaucb)
        # kauctable[method][dataname] = np.mean(allauck)
        # sauctable[method][dataname] = np.mean(allaucs)
        # rauctable[method][dataname] = np.mean(allaucr)
# print(bauctable,kauctable,sauctable,rauctable)
print(bf1table,kf1table,sf1table,rf1table)
bf1table.to_excel( 'tables/new/' + 'fsmb_bf1table.xlsx', index=True)
kf1table.to_excel( 'tables/new/' + 'fsmb_kf1table.xlsx', index=True)
sf1table.to_excel( 'tables/new/' + 'fsmb_sf1table.xlsx', index=True)
rf1table.to_excel( 'tables/new/' + 'fsmb_rf1table.xlsx', index=True)
bauctable.to_excel('tables/new/' + 'fsmb_bauctable.xlsx', index=True)
kauctable.to_excel('tables/new/' + 'fsmb_kauctable.xlsx', index=True)
sauctable.to_excel('tables/new/' + 'fsmb_sauctable.xlsx', index=True)
rauctable.to_excel('tables/new/' + 'fsmb_rauctable.xlsx', index=True)
bstdtable.to_excel('tables/new/' + 'fsmb_bastdctable.xlsx', index=True)
kstdtable.to_excel('tables/new/' + 'fsmb_kastdctable.xlsx', index=True)
sstdtable.to_excel('tables/new/' + 'fsmb_sastdctable.xlsx', index=True)
rstdtable.to_excel('tables/new/' + 'fsmb_rastdctable.xlsx', index=True)
bacctable.to_excel('tables/new/' + 'fsmb_bacctable.xlsx', index=True)
kacctable.to_excel('tables/new/' + 'fsmb_kacctable.xlsx', index=True)
sacctable.to_excel('tables/new/' + 'fsmb_sacctable.xlsx', index=True)
racctable.to_excel('tables/new/' + 'fsmb_racctable.xlsx', index=True)

# 计算标准差

