import math
import warnings

import numpy as np
import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd

# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#'Yeoh',
# datanames = ['Waveform','Wdbc', 'Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet','ORL', 'WarpPIE10P', 'Su',
#             'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111','Yeoh','Musk1', 'Sorlie', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',]#
# datanames = ['Dermatology','Musk1','Synthetic_control','Waveform','Wdbc','Arcene','CLL_SUB_111','GLIOMA', 'Isolet', 'ORL','Orlraws10P', 'Prostate_GE', 'TOX_171','WarpAR10P', 'WarpPIE10P',  'Yale',  'Authorship', 'Factors','Pixel', 'Sorlie','Su',  'Yeoh']#,]
datanames = ['Colon','SRBCT','madelon','spambase','splice','dna','dexter' ]#
base_url = "./dataset/testdatasets/"

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
    for i in range(time):
        nowpick = result[:, :i + 1]  # 特征名为数字
        # 初始化平均值和结果概率矩阵
        bresultmax = np.zeros((samplenum, classnum))
        kresultmax = np.zeros((samplenum, classnum))
        sresultmax = np.zeros((samplenum, classnum))
        rresultmax = np.zeros((samplenum, classnum))
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

        bpredicted_labels = (np.argmax(bresultmax, axis=1))
        kpredicted_labels = (np.argmax(kresultmax, axis=1))
        spredicted_labels = (np.argmax(sresultmax, axis=1))
        rpredicted_labels = (np.argmax(rresultmax, axis=1))
        lenth = len(y_test[0])
        baccuracy = calculate_accuracy(bpredicted_labels[:lenth], y_test[0])
        kaccuracy = calculate_accuracy(kpredicted_labels[:lenth], y_test[0])
        saccuracy = calculate_accuracy(spredicted_labels[:lenth], y_test[0])
        raccuracy = calculate_accuracy(rpredicted_labels[:lenth], y_test[0])
        bacc.append(baccuracy)
        kacc.append(kaccuracy)
        sacc.append(saccuracy)
        racc.append(raccuracy)
    bnacc.append(bacc)
    rnacc.append(racc)
    knacc.append(kacc)
    snacc.append(sacc)
    accresult = {'b': bnacc, 'k': knacc, 's': snacc, 'r': rnacc, }
    file_name = r"accuracy/1/" + dataname + '_' + method + "_all_accuracy.mat"
    scio.savemat(file_name, accresult)
    return bnacc, knacc, rnacc, snacc
def othertest(result, X_test, y_test, X_train, y_train):
    print(result,)
    bpredict_probabity = []
    kpredict_probabity = []
    spredict_probabity = []
    rpredict_probabity = []
    bpredict_labels = []
    kpredict_labels = []
    spredict_labels = []
    rpredict_labels = []
    # for i in range(time):if isinstance(s, (int, float)):  # 判断 s 是否为数字
    #     r = [s]
    # else:
    #     r = s
    # nowpick = result # 特征名为数字
    import numpy as np

    if result.shape == ():  # 标量（单个数据）
        nowpick = np.array([result])  # 将标量转换为包含一个元素的数组
    else:  # 数组（多个数据）
        nowpick = result
    # print(nowpick,y_train)
    # **Naïve Bayes (朴素贝叶斯)**
    from sklearn.naive_bayes import GaussianNB
    bayesmodel = GaussianNB()
    bayesmodel.fit(X_train[nowpick], y_train)
    bpredicted_labels = bayesmodel.predict(X_test[nowpick])
    bprobabilities = bayesmodel.predict_proba(X_test[nowpick])
    # **k-Nearest Neighbors (k-最近邻算法)**:
    from sklearn.neighbors import KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train[nowpick], y_train)
    kpredicted_labels = knn_classifier.predict(X_test[nowpick])
    kprobabilities = knn_classifier.predict_proba(X_test[nowpick])
    # **Support Vector Machines (支持向量机)**:
    from sklearn.svm import SVC
    svm_classifier = SVC(probability=True)
    svm_classifier.fit(X_train[nowpick], y_train)
    spredicted_labels = svm_classifier.predict(X_test[nowpick])
    sprobabilities = svm_classifier.predict_proba(X_test[nowpick])
    # random Forest (随机森林)**:
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train[nowpick], y_train)
    rpredicted_labels = rf_classifier.predict(X_test[nowpick])
    rprobabilities = rf_classifier.predict_proba(X_test[nowpick])


    bpredict_probabity = bprobabilities
    kpredict_probabity = kprobabilities
    spredict_probabity = sprobabilities
    rpredict_probabity = rprobabilities
    bpredict_labels = bpredicted_labels
    kpredict_labels = kpredicted_labels
    spredict_labels = spredicted_labels
    rpredict_labels = rpredicted_labels

    predict_pro   = {'b': bpredict_probabity, 'k': kpredict_probabity, 's': spredict_probabity, 'r': rpredict_probabity,}
    predict_lable = {'b': bpredict_labels, 'k': kpredict_labels, 's': spredict_labels, 'r': rpredict_labels,}
    # print(predict_pro,predict_lable)
    file_name1 = r"result/" + "eamb" + '/' + dataname + str(fold) +"_predict_pro.mat"
    file_name2 = r"result/" + "eamb" + '/' + dataname + str(fold) +"_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)
    return predict_pro, predict_lable

f1table = pd.DataFrame(index=range(len(datanames)), columns= range(4))
f1table.columns = ["b","k","s","r"]
f1table.index = datanames

acctable = pd.DataFrame(index=range(len(datanames)), columns = range(4))
acctable.columns = ["b","k","s","r"]
acctable.index = datanames
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

    # print(X0,y)
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    print(X, y)
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
        file_name = r"feature_select_result/" + 'IAMB' + '/' + dataname + str(fold) + "_result-0.txt"

        # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
        result = np.loadtxt(file_name)#加载特征提取结果
        result = np.array(result)
        if X.shape[1] in result:
            result = np.array([x for x in result if x != X.shape[1]])
        if result.size == 0:
            continue
        # 对每一个数量的特征选择做测试
        # bnacc,knacc,rnacc,snacc = csmdccmrtest(time, result, X_test, y_test, X_train, y_train)
        predict_pro, predict_lable = othertest(result, X_test, y_test, X_train, y_train)
        bpredict_labels = predict_lable['b']
        kpredict_labels = predict_lable['k']
        spredict_labels = predict_lable['s']
        rpredict_labels = predict_lable['r']

        bpredict_probabity = predict_pro['b']
        kpredict_probabity = predict_pro['k']
        spredict_probabity = predict_pro['s']
        rpredict_probabity = predict_pro['r']

        bacc = accuracy_score(y_test, bpredict_labels, )
        kacc = accuracy_score(y_test, kpredict_labels, )
        sacc = accuracy_score(y_test, spredict_labels, )
        racc = accuracy_score(y_test, rpredict_labels, )
        bf1 = f1_score(y_test, bpredict_labels, average='micro')
        kf1 = f1_score(y_test, kpredict_labels, average='micro')
        sf1 = f1_score(y_test, spredict_labels, average='micro')
        rf1 = f1_score(y_test, rpredict_labels, average='micro')# 种类个数
        if (classnum == 2):  # 若只分为两个类，需要特殊处理
            y_true_binarized = pd.get_dummies(y_test[0])
            y_true_binarized_array = y_true_binarized.values
        else:  # 多类别可以直接调用函数
            y_true_binarized = label_binarize(y_test, classes=classes)

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
# print(f1table)
print(f1table, acctable)
f1table.to_excel('tables/' + '/auctable_iamb-new.xlsx', index=True)
# auctable.to_excel('tables/' + '/f1table_eamb.xlsx', index=True)
# for dataname in datanames:
#     iposition = datanames.index(dataname)
#     # 读取数据及其真实值
#     kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
#     data_url = dataname + '.mat'
#     # print(dataname)
#     url = base_url + data_url  # 数据路径
#     data = scio.loadmat(url)  # 读取数据文件
#     # 读取文件中的真实标签
#     y0 = pd.DataFrame(data['Y'])  # 读取标签
#     # 应用函数到 DataFrame 的每个元素
#     y0 = y0.applymap(changetosinge)
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y0)
#     Class = set(y_encoded)
#     y = pd.DataFrame(y_encoded)
#     classes = list(Class)
#     # 当前数据集的十折数据所有样本结果存储
#     bpredict_labels_all = pd.DataFrame()
#     kpredict_labels_all = pd.DataFrame()
#     spredict_labels_all = pd.DataFrame()
#     rpredict_labels_all = pd.DataFrame()
#     pre_pro_all_b = []
#     pre_pro_all_k = []
#     pre_pro_all_s = []
#     pre_pro_all_r = []
#     # y_all = pd.DataFrame()#真实值的所有样本结果
#     # 十折交叉验证

#     for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
#         print("当前是第", fold)
#         all_f1_b = []
#         all_f1_k = []
#         all_f1_s = []
#         all_f1_r = []
#         y_test = y.iloc[test_index]  # 提取测试集条目
#         # 读取预测结果
#         # file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
#         # file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
#         # predict_pro = scio.loadmat(file_name1)  # 读取数据文件
#         # predict_lable = scio.loadmat(file_name2)  # 读取数据文件
#         # predict_pro, predict_lable
#         lenth = len(y_test)
#         pre_prob_sum_b = np.zeros((lenth, len(classes)))
#         pre_prob_sum_k = np.zeros((lenth, len(classes)))
#         pre_prob_sum_s = np.zeros((lenth, len(classes)))
#         pre_prob_sum_r = np.zeros((lenth, len(classes)))
#         # print(lenth)
#         # 对于每一折的数据都读取对应数量
#         # if method == 'csmdccmr':
#         # bpredict_labels_squeezed = np.squeeze(predict_lable['b'], axis=1)
#         # kpredict_labels_squeezed = np.squeeze(predict_lable['k'], axis=1)
#         # spredict_labels_squeezed = np.squeeze(predict_lable['s'], axis=1)
#         # rpredict_labels_squeezed = np.squeeze(predict_lable['r'], axis=1)
#         # bpredict_probabity_squeezed = np.squeeze(predict_pro['b'], axis=1)
#         # kpredict_probabity_squeezed = np.squeeze(predict_pro['k'], axis=1)
#         # spredict_probabity_squeezed = np.squeeze(predict_pro['s'], axis=1)
#         # rpredict_probabity_squeezed = np.squeeze(predict_pro['r'], axis=1)
#         # bpredict_probabity = np.array(bpredict_probabity_squeezed)[:, :, :lenth, :]
#         # kpredict_probabity = np.array(kpredict_probabity_squeezed)[:, :, :lenth, :]
#         # spredict_probabity = np.array(spredict_probabity_squeezed)[:, :, :lenth, :]
#         # rpredict_probabity = np.array(rpredict_probabity_squeezed)[:, :, :lenth, :]
#         # else:
#         bpredict_labels_squeezed = pd.DataFrame(predict_lable['b'])
#         kpredict_labels_squeezed = pd.DataFrame(predict_lable['k'])
#         spredict_labels_squeezed = pd.DataFrame(predict_lable['s'])
#         rpredict_labels_squeezed = pd.DataFrame(predict_lable['r'])
#         bpredict_probabity_squeezed = np.array(predict_pro['b'])
#         kpredict_probabity_squeezed = np.array(predict_pro['k'])
#         spredict_probabity_squeezed = np.array(predict_pro['s'])
#         rpredict_probabity_squeezed = np.array(predict_pro['r'])
#
#         bpredict_probabity = np.array(bpredict_probabity_squeezed)[ :lenth, :]
#         kpredict_probabity = np.array(kpredict_probabity_squeezed)[ :lenth, :]
#         spredict_probabity = np.array(spredict_probabity_squeezed)[ :lenth, :]
#         rpredict_probabity = np.array(rpredict_probabity_squeezed)[ :lenth, :]
#         # print(bpredict_probabity_squeezed.shape, kpredict_probabity_squeezed.shape,lenth)
#
#         # print(bpredict_probabity.shape,kpredict_probabity.shape)
#         bpredict_labels = pd.DataFrame(bpredict_labels_squeezed).iloc[:lenth]
#         kpredict_labels = pd.DataFrame(kpredict_labels_squeezed).iloc[:lenth]
#         spredict_labels = pd.DataFrame(spredict_labels_squeezed).iloc[:lenth]
#         rpredict_labels = pd.DataFrame(rpredict_labels_squeezed).iloc[:lenth]
#
#         bpredict_labels = bpredict_labels
#         kpredict_labels = kpredict_labels
#         spredict_labels = spredict_labels
#         rpredict_labels = rpredict_labels
#         feature_num = bpredict_labels.shape[1]
#         # print(bpredict_probabity.shape)
#         # print(feature_num)
#         # for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
#             # print(y_true_binarized,bprobabily_all[i])
#             # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
#         bf1 = f1_score(bpredict_labels, y_test, average='micro')
#         kf1 = f1_score(kpredict_labels, y_test, average='micro')
#         sf1 = f1_score(spredict_labels, y_test, average='micro')
#         rf1 = f1_score(rpredict_labels, y_test, average='micro')
#         # 种类个数
#         if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
#             y_true_binarized = pd.get_dummies(y_test[0])
#             y_true_binarized_array = y_true_binarized.values
#         else:  # 多类别可以直接调用函数
#             y_true_binarized = label_binarize(y_test, classes=classes)
        # print(y_true_binarized)
        # if method in ['csmdccmr', 'csmdccmr2']:
        #         for j in range(0, len(classes)):  # 不同种类分类器的预测概率求平均
        #             pre_prob_sum_b += bpredict_probabity[i][j]
        #             pre_prob_sum_k += kpredict_probabity[i][j]
        #             pre_prob_sum_s += spredict_probabity[i][j]
        #             pre_prob_sum_r += rpredict_probabity[i][j]
        #         pre_prob_sum_b = pre_prob_sum_b / len(classes)
        #         pre_prob_sum_k = pre_prob_sum_k / len(classes)
        #         pre_prob_sum_s = pre_prob_sum_s / len(classes)
        #         pre_prob_sum_r = pre_prob_sum_r / len(classes)
        #         bauc = roc_auc_score(y_true_binarized, pre_prob_sum_b, average='macro', multi_class='ovr')
        #         kauc = roc_auc_score(y_true_binarized, pre_prob_sum_k, average='macro', multi_class='ovr')
        #         sauc = roc_auc_score(y_true_binarized, pre_prob_sum_s, average='macro', multi_class='ovr')
        #         rauc = roc_auc_score(y_true_binarized, pre_prob_sum_r, average='macro', multi_class='ovr')
        #     else:
        #     print(y_true_binarized)
        # bauc = roc_auc_score(y_true_binarized, bpredict_probabity, average='macro',multi_class='ovr')
        # kauc = roc_auc_score(y_true_binarized, kpredict_probabity, average='macro',multi_class='ovr')
        # sauc = roc_auc_score(y_true_binarized, spredict_probabity, average='macro',multi_class='ovr')
        # rauc = roc_auc_score(y_true_binarized, rpredict_probabity, average='macro',multi_class='ovr')
        # # print(bauc)
        # # all_auc_b.append(bauc)
        # # all_auc_k.append(kauc)
        # # all_auc_s.append(sauc)
        # # all_auc_r.append(rauc)
        #     # 计算后将不同特征数量结果储存
        # # all_f1_b.append()
        # # all_f1_k.append()
        # # all_f1_s.append()
        # # all_f1_r.append()

        # 这里的lables行为特征数量，列为样本;这里的概率shape为（特征数，样本数，类别数）
        # 将每一折的数据都综合到all中，使所有样本都放在一起
        # print(bpredict_probabity.shape)
        # if method in ['csmdccmr','csmdccmr2']:
        #     pre_pro_all_b.append(bpredict_probabity[:,:,:lenth,:])
        #     pre_pro_all_k.append(kpredict_probabity[:,:,:lenth,:])
        #     pre_pro_all_s.append(spredict_probabity[:,:,:lenth,:])
        #     pre_pro_all_r.append(rpredict_probabity[:,:,:lenth,:])
        # else:
        #     pre_pro_all_b.append(bpredict_probabity[:, :lenth, :])
        #     pre_pro_all_k.append(kpredict_probabity[:, :lenth, :])
        #     pre_pro_all_s.append(spredict_probabity[:, :lenth, :])
        #     pre_pro_all_r.append(rpredict_probabity[:, :lenth, :])
        # print(bpredict_probabity.shape)
        # bpredict_labels_all = pd.concat([bpredict_labels_all, bpredict_labels], axis=1)
        # kpredict_labels_all = pd.concat([kpredict_labels_all, kpredict_labels], axis=1)
        # spredict_labels_all = pd.concat([spredict_labels_all, spredict_labels], axis=1)
        # rpredict_labels_all = pd.concat([rpredict_labels_all, rpredict_labels], axis=1)
        # y_all = pd.concat([y_all, y_test], axis=0)
    # 对于概率数据进行特殊处理，将矩阵连接在一起
    # print(pre_pro_all_r)
    # print(bpredict_labels_all.shape, len(pre_pro_all_r),len(pre_pro_all_r[0]),len(pre_pro_all_r[0][0]),len(pre_pro_all_r[0][0][0]),len(pre_pro_all_r[0][0][0][0]))
    # if method in ['csmdccmr','csmdccmr2']:
    #      bprobabily_all = np.concatenate(pre_pro_all_b, axis=2)
    #      sprobabily_all = np.concatenate(pre_pro_all_k, axis=2)
    #      kprobabily_all = np.concatenate(pre_pro_all_s, axis=2)
    #      rprobabily_all = np.concatenate(pre_pro_all_r, axis=2)
    #  else:
    #      bprobabily_all = np.concatenate(pre_pro_all_b, axis=1)
    #      sprobabily_all = np.concatenate(pre_pro_all_k, axis=1)
    #      kprobabily_all = np.concatenate(pre_pro_all_s, axis=1)
    #      rprobabily_all = np.concatenate(pre_pro_all_r, axis=1)
    # print(bprobabily_all.shape)
    # 将数据样本重新命名
    # sample_num = bpredict_labels_all.shape[1]
    # new_column_names = range(sample_num)
    #
    # bpredict_labels_all.columns = new_column_names[:sample_num]
    # bpredict_labels_all = bpredict_labels_all.T
    #
    # kpredict_labels_all.columns = new_column_names[:sample_num]
    # kpredict_labels_all = kpredict_labels_all.T
    #
    # spredict_labels_all.columns = new_column_names[:sample_num]
    # spredict_labels_all = spredict_labels_all.T
    #
    # rpredict_labels_all.columns = new_column_names[:sample_num]
    # rpredict_labels_all = rpredict_labels_all.T

    # 经过处理之后label的shape是（总样本数，特征数） 概率的shape是（特征数，总样本数，类别数）
    # 用于记录所有特征数情况下的f1和auc值，应当有特征数量大小

    # 当前数据集的结果为所有特征数量结果的平均数
    # print(fold_b_f1,fold_r_auc)
    # alldataset_f1_b.append(np.mean(fold_b_f1))
    # alldataset_f1_k.append(np.mean(fold_k_f1))
    # alldataset_f1_s.append(np.mean(fold_s_f1))
    # alldataset_f1_r.append(np.mean(fold_r_f1))
    # alldataset_auc_b.append(np.mean(fold_b_auc))
    # alldataset_auc_k.append(np.mean(fold_k_auc))
    # alldataset_auc_s.append(np.mean(fold_s_auc))
    # alldataset_auc_r.append(np.mean(fold_r_auc))
    # 处理完后存入表格记录

    # 对每个表格求平均值并按照大小进行升序排序
# tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
# for i, table in enumerate(tables):
#     # 计算每列的平均值
#     average_row = table.mean(axis=0)
#     table.loc['average'] = average_row
# bf1table = bf1table.sort_values(by='average', axis=1)
# sf1table = sf1table.sort_values(by='average', axis=1)
# kf1table = kf1table.sort_values(by='average', axis=1)
# rf1table = rf1table.sort_values(by='average', axis=1)
# bauctable = bauctable.sort_values(by='average', axis=1)
# sauctable = sauctable.sort_values(by='average', axis=1)
# kauctable = kauctable.sort_values(by='average', axis=1)
# rauctable = rauctable.sort_values(by='average', axis=1)
# print(bf1table, kf1table, sf1table, rf1table)
# print(bauctable, sauctable, kauctable, rauctable)
# bf1table.to_excel('tables/' + T + '/ten-bf1table.xlsx', index=True)
# sf1table.to_excel('tables/' + T + '/ten-sf1table.xlsx', index=True)
# kf1table.to_excel('tables/' + T + '/ten-kf1table.xlsx', index=True)
# rf1table.to_excel('tables/' + T + '/ten-rf1table.xlsx', index=True)

# kauctable.to_excel('tables/' + T + '/ten-kauctable.xlsx', index=True)
# rauctable.to_excel('tables/' + T + '/ten-rauctable.xlsx', index=True)







    # 定义八个表格用于存储最后的不同方法和不同数据集的F1和AUC结果
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
    # # pd.set_option('display.max_rows', None)
    # # pd.set_option('display.max_columns', None)
    # for method in methods:
    #     print(method, "=============================================================================")
    #     # 存储不同方法所有数据集23个的结果
    #
    # # 绘制箱线图
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # bf1table = pd.read_excel('tables/' + T + '/ten-bf1table.xlsx', index_col=0)
    # sf1table = pd.read_excel('tables/' + T + '/ten-sf1table.xlsx', index_col=0)
    # kf1table = pd.read_excel('tables/' + T + '/ten-kf1table.xlsx', index_col=0)
    # rf1table = pd.read_excel('tables/' + T + '/ten-rf1table.xlsx', index_col=0)
    # bauctable = pd.read_excel('tables/' + T + '/ten-bauctable.xlsx', index_col=0)
    # sauctable = pd.read_excel('tables/' + T + '/ten-sauctable.xlsx', index_col=0)
    # kauctable = pd.read_excel('tables/' + T + '/ten-kauctable.xlsx', index_col=0)
    # rauctable = pd.read_excel('tables/' + T + '/ten-rauctable.xlsx', index_col=0)
    # tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
    # # for table in tables:
    # #     # 计算每列的平均值
    # #     average_row = table.mean(axis=0)
    # #     table.loc['average'] = average_row
    # #     df_sorted = table.sort_values(by='average', axis=1)
    # #     print(table)
    # #
    # # print(bf1table)
    # # 创建一些示例数据
    # f1datas = [bf1table, sf1table, kf1table, rf1table]
    # aucdatas = [bauctable, sauctable, kauctable, rauctable]
    # names = ['Naive Bayes', '3NN', 'SVM', 'RandomForest']
    # # 创建箱线图
    # i = -1
    # for data in f1datas:
    #     i += 1
    #     plt.figure(figsize=(8, 6))
    #     plt.boxplot(data, labels=data.columns)
    #     # plt.ylim(0.3, 1)
    #     # 添加标题和标签
    #     plt.title(names[i])
    #     plt.ylabel('F1')
    #     plt.grid(False)
    #     # 显示图形
    #     plt.grid(True)
    #     plt.savefig("picture/" + names[i] + T + "_ten_F1.png", dpi=300, bbox_inches='tight')
    #     plt.show()
    # j = -1
    # for data in aucdatas:
    #     j += 1
    #     plt.figure(figsize=(8, 6))
    #     plt.boxplot(data, labels=data.columns)
    #     # plt.ylim(0.3, 1)
    #     # 添加标题和标签
    #     plt.title(names[j])
    #     plt.ylabel('AUC')
    #     plt.grid(False)
    #     # 显示图形
    #     plt.grid(True)
    #     plt.savefig("picture/" + names[j] + T + "_ten_AUC.png", dpi=300, bbox_inches='tight')
    #     plt.show()

    # =======================================画图==========================================
    # import matplotlib.pyplot as plt
    #
    # # 数据读取
    # y_values = np.loadtxt("accuracy/1/" + dataname + '_' + method + "_accuracy.txt")
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