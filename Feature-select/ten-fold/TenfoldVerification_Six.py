import ast
import math
import warnings

import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import random
from collections import Counter
from OtherWay import *

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






# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#'Dermatology',
datanames = [ 'Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene', 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet', 'ORL', 'WarpPIE10P', 'Su',
            'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111', 'Yeoh']#
base_url = r"./dataset/"

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
    # for i in range(time):
    nowpick = result # 特征名为数字
    # 初始化平均值和结果概率矩阵
    bresultmax = np.zeros((len(X_test), classnum))
    kresultmax = np.zeros((len(X_test), classnum))
    sresultmax = np.zeros((len(X_test), classnum))
    rresultmax = np.zeros((len(X_test), classnum))
    # 对于每个建立一个对应分类器
    for index,item in enumerate(nowpick):
        # print(index)
        # **Naïve Bayes (朴素贝叶斯)**
        from sklearn.naive_bayes import GaussianNB
        bayesmodel = GaussianNB()
        bayesmodel.fit(X_train[item], y_train)
        bpredicted_labels = bayesmodel.predict(X_test[item])
        bresultmax[:,index] = bpredicted_labels

        # **k-Nearest Neighbors (k-最近邻算法)**:
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train[item], y_train)
        kpredicted_labels = knn_classifier.predict(X_test[item])
        kresultmax[:,index] = kpredicted_labels

        # **Support Vector Machines (支持向量机)**:
        from sklearn.svm import SVC
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(X_train[item], y_train)
        spredicted_labels = svm_classifier.predict(X_test[item])
        sresultmax[:,index] = spredicted_labels

        # random Forest (随机森林)**:
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train[item], y_train)
        rpredicted_labels = rf_classifier.predict(X_test[item])
        rresultmax[:,index] = rpredicted_labels

    bpredicted_labels = compute_N(bresultmax)
    kpredicted_labels = compute_N(kresultmax)
    spredicted_labels = compute_N(sresultmax)
    rpredicted_labels = compute_N(rresultmax)
    lenth = len(y_test[0])
    baccuracy = calculate_accuracy(bpredicted_labels[:lenth], y_test[0])
    kaccuracy = calculate_accuracy(kpredicted_labels[:lenth], y_test[0])
    saccuracy = calculate_accuracy(spredicted_labels[:lenth], y_test[0])
    raccuracy = calculate_accuracy(rpredicted_labels[:lenth], y_test[0])
    # bacc.append(baccuracy)
    # kacc.append(kaccuracy)
    # sacc.append(saccuracy)
    # racc.append(raccuracy)
    # bnacc.append(bacc)
    # rnacc.append(racc)
    # knacc.append(kacc)
    # snacc.append(sacc)
    bnacc.append(baccuracy)
    snacc.append(saccuracy)
    knacc.append(kaccuracy)
    rnacc.append(raccuracy)
    accresult = {'b': bnacc, 'k': knacc, 's': snacc, 'r': rnacc, }
    file_name = r"accuracy/2/" + dataname + '_' + method + "_all_accuracy.mat"
    scio.savemat(file_name, accresult)
    return bnacc, knacc, rnacc, snacc
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
    # for i in range(time):
    #     print('选择特征为 ', i + 1, ' 时数据集测试结果：')
    #     baveacc, saveacc, raveacc, kaveacc = 0, 0, 0, 0
    #     for j in range(10):
    #         baveacc += bnacc[j][i]
    #         raveacc += rnacc[j][i]
    #         saveacc += snacc[j][i]
    #         kaveacc += knacc[j][i]
    #     print('贝叶斯    十折交叉验证平均准确度为：', baveacc / 10)
    #     print('K邻近     十折交叉验证平均准确度为：', kaveacc / 10)
    #     print('支持向量机 十折交叉验证平均准确度为：', saveacc / 10)
    #     print('随机森林   十折交叉验证平均准确度为：', raveacc / 10)
    #     # print(bnacc,knacc,snacc,rnacc)
    #     accmax.append((baveacc + saveacc + kaveacc + raveacc) / 40)
    # print(accmax)
    # file_name = r"accuracy/" + dataname + '_' + method + "_accuracy.txt"
    # np.savetxt(file_name, accmax)  # 使用savetxt()函数保存数组到文件
    # print('选择特征为 ', time + 1, ' 时数据集测试结果：')
    baveacc, saveacc, raveacc, kaveacc = 0, 0, 0, 0
    for j in range(10):
        baveacc += bnacc[j]
        raveacc += rnacc[j]
        saveacc += snacc[j]
        kaveacc += knacc[j]
    print('贝叶斯    十折交叉验证平均准确度为：', baveacc / 10)
    print('K邻近     十折交叉验证平均准确度为：', kaveacc / 10)
    print('支持向量机 十折交叉验证平均准确度为：', saveacc / 10)
    print('随机森林   十折交叉验证平均准确度为：', raveacc / 10)
    # print(bnacc,knacc,snacc,rnacc)
    accmax.append((baveacc + saveacc + kaveacc + raveacc) / 40)
    print(accmax)
    file_name = r"accuracy/2/" + dataname + '_' + method + "_accuracy.txt"
    np.savetxt(file_name, accmax)  # 使用savetxt()函数保存数组到文件

for dataname in datanames:
    data_url = dataname + '.mat'
    # dataname = "movement_libras"
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
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
    methods = ['CFS-BF','CON-BF','CON-GEN','CFS-GEN',]
    for method in methods:
        print(method)
        # 创建一个十折交叉验证对象
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子

        classnum = len(Class)  # 类别数，分为几类
        samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
        accmax = []  # 准确度记录数组，为了画折线图
        Fall = X.shape[1]  # 特征数
        # 选取的特征数量
        if Fall < 50:
            time = Fall/2
        else:
            time = 50
        # 对每一个9折，进行特征选择
        baverage, kaverage, saverage, raverage = 0, 0, 0, 0
        bnacc, knacc, snacc, rnacc = [], [], [], []

        # 十折交叉验证
        for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
            print("当前处理到第", fold, "折\n")
            bacc, kacc, sacc, racc = [], [], [], []
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
            result = []  # 特征提取结果数组
            # -----------------直接读取特征提取结果
            file_name = r"feature_select_result/" + method + '/' + method + '_' + dataname + str(fold) + ".txt"
            # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
            with open(file_name, 'r') as file:
                lines = file.readlines()
            result = [ast.literal_eval(line.strip()) for line in lines]
            # 转换为二维的 NumPy 数组
            # result = np.array(result, dtype=int)  # 假设数据为整数类型，可以根据实际数据类型调整
            # data = np.loadtxt(file_name, skiprows=)
            # result = pd.read_csv(file_name,header=None)  # 读取文件
            # result = result.iloc[:-1]
            # result = result.astype(int)
            # print(result)
            # 对每一个数量的特征选择做测试
            bnacc, knacc, rnacc, snacc = csmdccmrtest(len(result), result, X_test, y_test, X_train, y_train)
        calculate(len(result))

        # =======================================画图==========================================
        # import matplotlib.pyplot as plt
        #
        # # 数据读取
        # y_values = np.loadtxt("accuracy/" + dataname + '_' + method + "_accuracy.txt")
        # x_values = np.arange(1, len(result) + 1)
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
