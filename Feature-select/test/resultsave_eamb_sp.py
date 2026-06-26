import math
import warnings

import numpy as np
import scipy.io as scio
# from skfeature.function.information_theoretical_based import CIFE
# from skfeature.function.information_theoretical_based import MRMR
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
import json
# from imblearn.over_sampling import SMOTE


# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
# 'ORL','Yale',
# datanames = [  'Authorship', 'Factors','Pixel', 'Sorlie','Su', 'Isolet','Dermatology','Movement_libras','Musk1','Synthetic_control','Waveform','Wdbc',,'CLL_SUB_111','GLIOMA','Yeoh','Orlraws10P', 'Prostate_GE', 'TOX_171','WarpAR10P','WarpPIE10P', ]#,]
# datanames = ['Arcene']#,]'Factors','Pixel','Sorlie', 'Isolet','ORL', 'Yale','GLIOMA','WarpAR10P'
datanames = ['Colon','SRBCT','madelon','spambase','splice','dna', ]#'dexter'
base_url = r"./dataset/"
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
    file_name = r"accuracy/1/" + dataname + '_' + method + "_accuracy.txt"
    np.savetxt(file_name, accmax)  # 使用savetxt()函数保存数组到文件
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
def csmdccmrtest(num,result, X_test, y_test, X_train, y_train):
    # print(num)
    result = list(result.values())  # 特征名为数字
    nowpick = [[item for item in sublist if item != num] for sublist in result]
    # print(nowpick)
    # 初始化平均值和结果概率矩阵
    bresultmax = np.zeros((samplenum, classnum))
    kresultmax = np.zeros((samplenum, classnum))
    sresultmax = np.zeros((samplenum, classnum))
    rresultmax = np.zeros((samplenum, classnum))
    # 对于每个建立一个对应分类器
    for item in nowpick:
        # print(item)
        # **Naïve Bayes (朴素贝叶斯)**
        from sklearn.naive_bayes import GaussianNB
        bayesmodel = GaussianNB()
        bayesmodel.fit(X_train[item], y_train)
        bprobabilities = bayesmodel.predict_proba(X_test[item])
        if not bprobabilities.shape == bresultmax.shape:
            bprobabilities = np.vstack([bprobabilities, np.zeros_like(bprobabilities[-1])])
        bresultmax += bprobabilities
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
    lenth = len(y_test[0])
    bpredicted_labels = np.argmax(bresultmax, axis=1)[:lenth]
    kpredicted_labels = np.argmax(kresultmax, axis=1)[:lenth]
    spredicted_labels = np.argmax(sresultmax, axis=1)[:lenth]
    rpredicted_labels = np.argmax(rresultmax, axis=1)[:lenth]

    # baccuracy = calculate_accuracy(bpredicted_labels[:lenth], y_test[0])
    # kaccuracy = calculate_accuracy(kpredicted_labels[:lenth], y_test[0])
    # saccuracy = calculate_accuracy(spredicted_labels[:lenth], y_test[0])
    # raccuracy = calculate_accuracy(rpredicted_labels[:lenth], y_test[0])
    # bacc.append(baccuracy)
    # kacc.append(kaccuracy)
    # sacc.append(saccuracy)
    # racc.append(raccuracy)
    # bnacc.append(bacc)
    # rnacc.append(racc)
    # knacc.append(kacc)
    # snacc.append(sacc)
    # print(len(y_test),len(bpredicted_labels))
    predict_lable = {'b': bpredicted_labels, 'k': kpredicted_labels, 's': spredicted_labels, 'r': rpredicted_labels, }
    file_name1 = r"result/" + "eamb" + '/sp/' + dataname + str(fold) +"_predict_lable.mat"
    # file_name2 = r"result/" + "eamb" + '/' + dataname + str(fold) +"_predict_lable.mat"
    # scio.savemat(file_name2, predict_pro)
    scio.savemat(file_name1, predict_lable)
    return  predict_lable

f1table = pd.DataFrame(index=range(len(datanames)), columns= range(4))
f1table.columns = ["b","k","s","r"]
f1table.index = datanames

acctable = pd.DataFrame(index=range(len(datanames)), columns = range(4))
acctable.columns = ["b","k","s","r"]
acctable.index = datanames

predict_pro, predict_lable = {},{}
for dataname in datanames:
    iposition = datanames.index(dataname)
    data_url = dataname + '.mat'
    # dataname = "movement_libras"
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
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
    # print(data)
    # print(X0,y0)
    print("====================================================================================================")
    print(dataname)
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
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X0, y)
    # print(X0,y)
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    # y = y_resampled

    # print(len(X_resampled), len(X_resampled))
    # ====================================数据导入=====================================
    # methods1 = [csmdccmr]
    # methods2 = [MRMR.mrmr, CIFE.cife]
    # methods3 = [cfr ,mri , dcsf, mrmd, iwfs, ucrfs,jmim]#,
    # for n in methods3:
    #     print(n.__name__)
    #     method = n.__name__
        # 创建一个十折交叉验证对象
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    classnum = len(Class)  # 类别数，分为几类
    classes = list(Class)
    samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
    Fall = X.shape[1]  # 特征数
    # 选取的特征数量
    # if Fall < 50:
    #     time = Fall
    # else:
    #     time = 50
    # 对每一个9折，进行特征选择
    baverage, kaverage, saverage, raverage = 0, 0, 0, 0
    bnacc, knacc, snacc, rnacc = [], [], [], []
    # 十折交叉验证
    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
        fold_b_f1 = []
        fold_k_f1 = []
        fold_s_f1 = []
        fold_r_f1 = []
        fold_b_acc = []
        fold_k_acc = []
        fold_s_acc = []
        fold_r_acc = []
        print("当前处理到第", fold, "折\n")
        bacc, kacc, sacc, racc = [], [], [], []
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
        # result = []  # 特征提取结果数组
        # -----------------cmdccmr
        # for ck in Class:# 为每一个类别进行特征选择      #X为纯数组，y为series
        #     cpick = csmdccmr(X_train.values, np.ravel(y_train), ck, n_selected_features=time)
        #     result.append(cpick)
        # -----------------cmdccmr
        # result,a,b = n(X_train.values, np.ravel(y_train), n_selected_features=time)#只提取一次特征
        # result = n(X_train.values, np.ravel(y_train), n_selected_features=time)  # 只提取一次特征
        # -----------------otherway
        file_name = r"feature_select_result/" + 'EAMB' + '/' + dataname + "_sp_result.txt"
        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
        #{'0': [34, 27, 21, 9, 18, 20, 8, 19, 15, 22], '1': [27, 34, 2, 21, 25, 4, 20, 8, 19, 13, 15], '2': [34, 5, 7, 20, 28, 13, 15, 32], '3': [27, 34, 25, 21, 9, 15, 4, 20, 8, 19, 3], '4': [34, 21, 4, 1, 20, 14, 8, 9, 13, 15], '5': [34, 21, 4, 20, 33, 10, 8, 30, 6, 3]}
        # 打开文件并读取内容
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
        # 将 JSON 字符串解析为 Python 字典
        result = json.loads(content)
        # print(X.shape[1] + 1)

        # 对每一个数量的特征选择做测试
        predict_lable = csmdccmrtest(X.shape[1],result, X_test, y_test, X_train, y_train)
        # predict_pro, predict_lable = othertest(result, X_test, y_test, X_train, y_train)
        bpredict_labels = predict_lable['b']
        kpredict_labels = predict_lable['k']
        spredict_labels = predict_lable['s']
        rpredict_labels = predict_lable['r']

        # bpredict_probabity = predict_pro['b']
        # kpredict_probabity = predict_pro['k']
        # spredict_probabity = predict_pro['s']
        # rpredict_probabity = predict_pro['r']
        bacc = accuracy_score(y_test,bpredict_labels,)
        kacc = accuracy_score(y_test,kpredict_labels,)
        sacc = accuracy_score(y_test,spredict_labels,)
        racc = accuracy_score(y_test,rpredict_labels,)
        bf1 = f1_score( y_test,bpredict_labels,average='micro')
        kf1 = f1_score( y_test,kpredict_labels,average='micro')
        sf1 = f1_score( y_test,spredict_labels,average='micro')
        rf1 = f1_score( y_test,rpredict_labels,average='micro')
        # 种类个数
        # if (classnum == 2):  # 若只分为两个类，需要特殊处理
        #     y_true_binarized = pd.get_dummies(y_test[0])
        #     y_true_binarized_array = y_true_binarized.values
        # else:  # 多类别可以直接调用函数
        #     y_true_binarized = label_binarize(y_test, classes=classes)

        # bauc = roc_auc_score(y_true_binarized, bpredict_probabity, average='macro', multi_class='ovr')
        # kauc = roc_auc_score(y_true_binarized, kpredict_probabity, average='macro', multi_class='ovr')
        # sauc = roc_auc_score(y_true_binarized, spredict_probabity, average='macro', multi_class='ovr')
        # rauc = roc_auc_score(y_true_binarized, rpredict_probabity, average='macro', multi_class='ovr')
        fold_b_f1.append(bf1)
        fold_k_f1.append(kf1)
        fold_s_f1.append(sf1)
        fold_r_f1.append(rf1)
        fold_b_acc.append(bacc)
        fold_k_acc.append(kacc)
        fold_s_acc.append(sacc)
        fold_r_acc.append(racc)

    f1table["b"][iposition] = np.mean(fold_b_f1)
    f1table["k"][iposition] = np.mean(fold_k_f1)
    f1table["s"][iposition] = np.mean(fold_s_f1)
    f1table["r"][iposition] = np.mean(fold_r_f1)
    acctable["b"][iposition] = np.mean(fold_b_acc)
    acctable["k"][iposition] = np.mean(fold_k_acc)
    acctable["s"][iposition] = np.mean(fold_s_acc)
    acctable["r"][iposition] = np.mean(fold_r_acc)
print(f1table,acctable)
# print(f1table, auctable)
f1table.to_excel('tables/' + '/auctable_eamb_spsm-new.xlsx', index=True)
