import math
import warnings
from collections import Counter

from sklearn.metrics import f1_score
import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from ClassSpecific5_CSMI import csmi
from OtherWay import *
import pandas as pd
# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#'Dermatology','Waveform',
datanames = ['Yale','Authorship', 'Factors', 'Pixel',  'Sorlie', 'Su', 'Yeoh']#'FeatMIAS',
base_url = r"./dataset/"
def compute_N(X):
    m, n = X.shape
    N = np.zeros(m, dtype=int)

    for i in range(m):
        candidates = []
        for j in range(n):
            if X[i, j] == j:
                candidates.append(j)

        if len(candidates) == 1:
            N[i] = candidates[0]
        elif len(candidates) >= 2:
            counts = Counter(candidates)
            max_count = max(counts.values())
            max_candidates = [val for val, count in counts.items() if count == max_count]
            N[i] = random.choice(max_candidates)
        else:
            counts = Counter(X[i])
            max_count = max(counts.values())
            max_values = [val for val, count in counts.items() if count == max_count]
            N[i] = random.choice(max_values)

    return N
def calculate_accuracy(predicted_labels, true_labels):
    """
    计算分类器的准确度。
    参数:
    predicted_labels: 预测的标签列表。
    true_labels: 真实的标签列表。
    返回值:
    accuracy: 分类器的准确度。
    """
    # 确保预测结果和真实标签的长度相同
    if len(predicted_labels) != len(true_labels):
        raise ValueError("预测结果和真实标签的长度不一致。")
    predicted_labels = [int(x) for x in predicted_labels]
    true_labels = [int(x) for x in true_labels]
    # 计算预测正确的数量
    correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)

    # 计算准确度
    accuracy = correct_count / len(true_labels)

    return accuracy
def csmdccmrtest(time, result, X_test, y_test, X_train, y_train):
    # all_bpredict_probabity = []#全部的预测结果
    # all_kpredict_probabity = []
    # all_spredict_probabity = []
    # all_rpredict_probabity = []
    # all_bpredict_labels = []
    # all_kpredict_labels = []
    # all_spredict_labels = []
    # all_rpredict_labels = []
    nowpick = result  # 特征名为数字
    # 初始化平均值和结果概率矩阵
    bresultmax = np.zeros((len(X_test), classnum))
    kresultmax = np.zeros((len(X_test), classnum))
    sresultmax = np.zeros((len(X_test), classnum))
    rresultmax = np.zeros((len(X_test), classnum))
    bpredict_probabity = []
    kpredict_probabity = []
    spredict_probabity = []
    rpredict_probabity = []
    bpredict_labels = []
    kpredict_labels = []
    spredict_labels = []
    rpredict_labels = []
    b10auc = []
    k10auc = []
    s10auc = []
    r10auc = []
    # 对于每个建立一个对应分类器
    for index, item in enumerate(nowpick):
        # print(len(nowpick))
        # **Naïve Bayes (朴素贝叶斯)**
        from sklearn.naive_bayes import GaussianNB
        bayesmodel = GaussianNB()
        bayesmodel.fit(X_train[item], y_train)
        bpredicted_labels = bayesmodel.predict(X_test[item])
        bprobabilities = bayesmodel.predict_proba(X_test[item])
        bresultmax[:, index] = bpredicted_labels

        # **k-Nearest Neighbors (k-最近邻算法)**:
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train[item], y_train)
        kpredicted_labels = knn_classifier.predict(X_test[item])
        kprobabilities = knn_classifier.predict_proba(X_test[item])
        kresultmax[:, index] = kpredicted_labels

        # **Support Vector Machines (支持向量机)**:
        from sklearn.svm import SVC
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(X_train[item], y_train)
        sprobabilities = svm_classifier.predict_proba(X_test[item])
        spredicted_labels = svm_classifier.predict(X_test[item])
        sresultmax[:, index] = spredicted_labels

        # random Forest (随机森林)**:
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train[item], y_train)
        rprobabilities = rf_classifier.predict_proba(X_test[item])
        rpredicted_labels = rf_classifier.predict(X_test[item])
        rresultmax[:, index] = rpredicted_labels
        b10auc.append(bprobabilities)
        k10auc.append(kprobabilities)
        s10auc.append(sprobabilities)
        r10auc.append(rprobabilities)
    #根据分类概率得到分类标签
    bpredicted_labels = compute_N(bresultmax)
    kpredicted_labels = compute_N(kresultmax)
    spredicted_labels = compute_N(sresultmax)
    rpredicted_labels = compute_N(rresultmax)
    #分类标签和分类概率
    bpredict_probabity.append(b10auc)#记录time次分类结果
    kpredict_probabity.append(k10auc)
    spredict_probabity.append(s10auc)
    rpredict_probabity.append(r10auc)
    bpredict_labels.append(bpredicted_labels)
    kpredict_labels.append(kpredicted_labels)
    spredict_labels.append(spredicted_labels)
    rpredict_labels.append(rpredicted_labels)
    # all_bpredict_probabity.append(bpredict_probabity)  # 全部的预测结果
    # all_kpredict_probabity.append(kpredict_probabity)
    # all_spredict_probabity.append(spredict_probabity)
    # all_rpredict_probabity.append(rpredict_probabity)
    # all_bpredict_labels.append(bpredict_labels)
    # all_kpredict_labels.append(kpredict_labels)
    # all_spredict_labels.append(spredict_labels)
    # all_rpredict_labels.append(rpredict_labels)
    predict_pro   = {'b': bpredict_probabity, 'k': kpredict_probabity, 's': spredict_probabity, 'r': rpredict_probabity,}
    predict_lable = {'b': bpredict_labels, 'k': kpredict_labels, 's': spredict_labels, 'r': rpredict_labels,}
    # print(predict_pro,predict_lable)
    file_name1 = r"result/"+ method +'/'+ dataname + str(fold) +"_predict_pro.mat"
    file_name2 = r"result/"+ method +'/'+ dataname + str(fold) +"_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)
    # print(len(bpredict_probabity),len(bpredict_labels))
    return bnacc, knacc, rnacc, snacc, bnf1, knf1, rnf1, snf1, bnauc, knauc, rnauc, snauc
def othertest(time, result, X_test, y_test, X_train, y_train):
    for i in range(time):
        nowpick = result[:i + 1]  # 特征名为数字
        # **Naïve Bayes (朴素贝叶斯)**
        from sklearn.naive_bayes import GaussianNB
        bayesmodel = GaussianNB()
        bayesmodel.fit(X_train[nowpick], y_train)
        bpredicted_labels = bayesmodel.predict(X_test[nowpick])

        # **k-Nearest Neighbors (k-最近邻算法)**:
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train[nowpick], y_train)
        kpredicted_labels = knn_classifier.predict(X_test[nowpick])

        # **Support Vector Machines (支持向量机)**:
        from sklearn.svm import SVC
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(X_train[nowpick], y_train)
        spredicted_labels = svm_classifier.predict(X_test[nowpick])

        # random Forest (随机森林)**:
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train[nowpick], y_train)
        rpredicted_labels = rf_classifier.predict(X_test[nowpick])

        lenth = len(y_test[0])
        baccuracy = calculate_accuracy(bpredicted_labels[:lenth], y_test[0])
        kaccuracy = calculate_accuracy(kpredicted_labels[:lenth], y_test[0])
        saccuracy = calculate_accuracy(spredicted_labels[:lenth], y_test[0])
        raccuracy = calculate_accuracy(rpredicted_labels[:lenth], y_test[0])
        bacc.append(baccuracy)
        sacc.append(saccuracy)
        kacc.append(kaccuracy)
        racc.append(raccuracy)
    bnacc.append(bacc)
    rnacc.append(racc)
    knacc.append(kacc)
    snacc.append(sacc)
    accresult = {'b': bnacc, 'k': knacc, 's': snacc, 'r': rnacc, }
    file_name = r"accuracy/" + dataname + '_' + method + "_all_accuracy.mat"
    scio.savemat(file_name, accresult)
    return bnacc, knacc, rnacc, snacc
def calculate(time):
    for i in range(time):
        print('选择特征为 ', i + 1, ' 时数据集测试结果：')
        baveacc, saveacc, raveacc, kaveacc = 0, 0, 0, 0
        for j in range(10):
            baveacc += bnacc[j][i]
            raveacc += rnacc[j][i]
            saveacc += snacc[j][i]
            kaveacc += knacc[j][i]
        print('贝叶斯    十折交叉验证平均准确度为：', baveacc / 10)
        print('K邻近     十折交叉验证平均准确度为：', kaveacc / 10)
        print('支持向量机 十折交叉验证平均准确度为：', saveacc / 10)
        print('随机森林   十折交叉验证平均准确度为：', raveacc / 10)
        # print(bnacc,knacc,snacc,rnacc)
        accmax.append((baveacc + saveacc + kaveacc + raveacc) / 40)
    print(accmax)
    file_name = r"accuracy/" + dataname + '_' + method + "_accuracy.txt"
    np.savetxt(file_name, accmax)  # 使用savetxt()函数保存数组到文件

for dataname in datanames:
    data_url = dataname + '.mat'
    print(dataname)
    # dataname = "movement_libras"
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    # Special = X0.iloc[:,-1]
    # # print(Special.values)
    # X0 = X0.iloc[:,:-1]  # 读取训练数据
    # a = np.array([item[0] for item in Special])
    # label_encoder = LabelEncoder()
    # a33 = label_encoder.fit_transform(a)
    # X0[33] = a33
    # print(a,"!!!!!!!!!!!!!!!!!!!!!!!")
    # print(data)
    # print(X0,y0)
    print("====================================================================================================")
    # 将y标签控制在0-n
    # print(X0,y0)
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
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(X0[33])
    # data = {33:y_encoded}
    # newf = pd.DataFrame(data)
    # X0[33]=newf[33]

    print(X0,y)
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    print(X, y)
    # ====================================数据导入=====================================
    methods1 = [csmi]
    unique_values = set(y)
    # print(unique_values)
    # print(data0)
    allresult = []
    for n in methods1:
        print(n.__name__)
        method = n.__name__
        # 创建一个十折交叉验证对象
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子

        classnum = len(Class)  # 类别数，分为几类
        samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
        accmax = []  # 准确度记录数组，为了画折线图
        Fall = X.shape[1]  # 特征数
        # 选取的特征数量
        if Fall < 50:
            time = Fall
        else:
            time = 50
        # 对每一个9折，进行特征选择
        baverage, kaverage, saverage, raverage = 0, 0, 0, 0
        bnacc, knacc, snacc, rnacc = [], [], [], []
        bnf1, knf1, rnf1, snf1 = [], [], [], []
        bnauc, knauc, rnauc, snauc = [], [], [], []
        # 十折交叉验证
        for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
            print("当前处理到第", fold, "折\n")
            bacc, kacc, sacc, racc = [], [], [], []
            b0f1, k0f1, s0f1, r0f1 = [], [], [], []
            bauc, kauc, rauc, sauc = [], [], [], []
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
            result = []  # 特征提取结果数组
            # -----------------csmi
            # for ck in Class:# 为每一个类别进行特征选择      #X为纯数组，y为series
            #     # print(u)
            #     newy = y_train.applymap(lambda val: 0 if val == ck else 1)
            #     othercount = (newy[0] == 1).sum()
            #     # for j in range(data.num_instances):
            #     #     if data.get_instance(j).get_value(data.class_index) != u:
            #     #         data.get_instance(j).set_value(data.class_index, 0.0)
            #     #         othercount += 1
            #     #     else:
            #     #         data.get_instance(j).set_value(data.class_index, 1.0)
            #     count = len(y_train) - othercount
            #     # Oversample by repeating instances in each class
            #     if count < othercount:
            #         # Calculate the number of instances to repeat
            #         repeat_count = othercount - count
            #         # Find instances in class u
            #         # 从原始样本中随机选择需要增加的样本
            #         minority_indices = y_train[y_train[0] == 0].index
            #         sample_indices = np.random.choice(minority_indices, count, replace=True)
            #         # 添加这些样本到原始数据中
            #         X_train = pd.concat([X_train, X_train.loc[sample_indices]], ignore_index=True)
            #         y_train = pd.concat([y_train, y_train.loc[sample_indices]], ignore_index=True)
            #     cpick = csmdccmr(X_train.values, np.ravel(y_train), 0, n_selected_features=time)
            #     result.append(cpick)
            # print(result)
            # -----------------cmdccmr
            # result,a,b = n(X_train.values, np.ravel(y_train), n_selected_features=time)#只提取一次特征
            # result = n(X_train.values, np.ravel(y_train), n_selected_features=time)  # 只提取一次特征
            # -----------------otherway
            file_name = r"feature_select_result/" + method + '/' + dataname+ str(fold) + "_result.txt"
            # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
            result = np.loadtxt(file_name)#加载特征提取结果
            result = np.array(result)
            # print(result)
            # 对每一个数量的特征选择做测试
            bnacc, knacc, rnacc, snacc, bnf1, knf1, rnf1, snf1, bnauc, knauc, rnauc, snauc= csmdccmrtest(time, result, X_test, y_test, X_train, y_train)
            # bnacc, knacc, rnacc, snacc, bnf1, knf1, rnf1, snf1,bnauc, knauc, rnauc, snauc = othertest(time, result, X_test, y_test, X_train, y_train)
        # calculate(time)

        # =======================================画图==========================================
        # import matplotlib.pyplot as plt
        #
        # # 数据读取
        # y_values = np.loadtxt("accuracy/" + dataname + '_' + method + "_accuracy.txt")
        # x_values = np.arange(1, time + 1)
        # # 创建折线图
        # plt.figure(figsize=(8, 4))
        # # plt.ylim(0,1)
        # # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        # plt.plot(x_values, y_values, color='black')  # 黑色
        # # 添加标题和标签
        # plt.title(dataname + ' ' + method)
        # plt.xlabel('Number of selected features')
        # plt.ylabel('Classification accuracy')
        # # 显示图形
        # plt.show()
