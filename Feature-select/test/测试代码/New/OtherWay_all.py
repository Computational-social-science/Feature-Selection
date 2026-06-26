import math
import warnings

import scipy.io as scio
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
from OtherWay import *

# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#
datasets = ['Dermatology',]
# 'Waveform','Wdbc', 'Synthetic_control','Authorship', 'Musk1', 'Factors', 'Pixel', 'Isolet', 'Su','Sorlie', 'Movement_libras','Yeoh']#,'Prostate_GE', 'TOX_171', 'Orlraws10P', 'CLL_SUB_111', 'Yale', 'WarpAR10P', 'GLIOMA', 'Arcene',  'ORL', 'WarpPIE10P',

base_url = r"./dataset/"
def changetosinge(x):
    return float(x)

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
def othertest(time, result, X_test, y_test, X_train, y_train):
    for i in range(time):
        #记录所有time次的预测概率，和预测标签
        b_time_predict_probabity = []
        k_time_predict_probabity = []
        s_time_predict_probabity = []
        r_time_predict_probabity = []
        b_time_predict_labels = []
        k_time_predict_labels = []
        s_time_predict_labels = []
        r_time_predict_labels = []
        nowpick = result[:i + 1]  # 特征名为数字
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
        #每个分类器在当前情况下的分类概率列表（一次测试的所有分类器结果）
        b_time_predict_probabity.append(bprobabilities)#记录n种分类器单次概率
        k_time_predict_probabity.append(kprobabilities)
        s_time_predict_probabity.append(sprobabilities)
        r_time_predict_probabity.append(rprobabilities)
        b_time_predict_labels.append(bpredicted_labels)
        k_time_predict_labels.append(kpredicted_labels)
        s_time_predict_labels.append(spredicted_labels)
        r_time_predict_labels.append(rpredicted_labels)
    predict_pro   = {'b': b_time_predict_probabity, 'k': k_time_predict_probabity, 's': s_time_predict_probabity, 'r': r_time_predict_probabity,}
    predict_lable = {'b': b_time_predict_labels, 'k': k_time_predict_labels, 's': s_time_predict_labels, 'r': r_time_predict_labels,}
    file_name1 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_pro.mat"
    file_name2 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)

for dataname in datasets:
    data_url = dataname + '.mat'
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    print("====================================================================================================")
    #=======================Dercomatoly==================
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
    # 数据离散化
    X = pd.DataFrame()
    print(X0,y)
    #离散化
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    print(X, y)
    # ====================================数据导入=====================================
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    classnum = len(Class)  # 类别数，分为几类
    samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整

    methods2 = [MRMR.mrmr, CIFE.cife]
    methods3 = [cfr ]#,,mri , dcsf, mrmd, iwfs, ucrfs,jmim
    for n in methods3:
        print(n.__name__)
        method = n.__name__
        # 创建一个十折交叉验证对象
        Fall = X.shape[1]  # 特征数
        # 选取的特征数量
        if Fall < 50:
            time = Fall
        else:
            time = 50
        # 十折交叉验证
        for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
            print("当前处理到第", fold, "折\n")
            bacc, kacc, sacc, racc = [], [], [], []
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
            result = []  # 特征提取结果数组
            # result,a,b = n(X_train.values, np.ravel(y_train), n_selected_features=time)#只提取一次特征
            result = n(X_train.values, np.ravel(y_train), n_selected_features=time)  # 只提取一次特征
            file_name = r"feature_select_result/" + method + '/' + dataname + str(fold) + "_result.txt"
            np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
            result = np.loadtxt(file_name)#加载特征提取结果
            result = np.array(result)
            # 对每一个数量的特征选择做测试
            othertest(time, result, X_test, y_test, X_train, y_train)
        # calculate(time)

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
#==================================数据预测完毕，进行预测结果的处理
#处理得到结果计算出的F1和AUC
T = '1'#用于区别储存
methods = ['jmim']#'cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs',
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
            file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
            file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
            predict_pro = scio.loadmat(file_name1)  # 读取数据文件
            predict_lable = scio.loadmat(file_name2)  # 读取数据文件
            lenth = len(y_test)#当前折的测试集数量
            #对于每一折的数据都读取对应数量
            bpredict_probabity = np.array(predict_pro['b'])[:, :lenth, :]#time个结果，每个结果有class个分类器，每个分类器有n个样本和class种可能
            kpredict_probabity = np.array(predict_pro['k'])[:, :lenth, :]
            spredict_probabity = np.array(predict_pro['s'])[:, :lenth, :]
            rpredict_probabity = np.array(predict_pro['r'])[:, :lenth, :]
            bpredict_labels = pd.DataFrame(predict_lable['b']).iloc[:, :lenth]#time个结果，每个结果n个样本，
            kpredict_labels = pd.DataFrame(predict_lable['k']).iloc[:, :lenth]
            spredict_labels = pd.DataFrame(predict_lable['s']).iloc[:, :lenth]
            rpredict_labels = pd.DataFrame(predict_lable['r']).iloc[:, :lenth]
            # print(bpredict_probabity,bpredict_labels)
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
        bprobabily_all = np.concatenate(bpredict_probabity_list, axis=2)
        kprobabily_all = np.concatenate(kpredict_probabity_list, axis=2)
        sprobabily_all = np.concatenate(spredict_probabity_list, axis=2)
        rprobabily_all = np.concatenate(rpredict_probabity_list, axis=2)
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
        # print()
        for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
            # print(y_true_binarized,bprobabily_all[i])
            # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
            time_bf1 = f1_score(bpredict_labels_all[i], y_all, average='micro')
            time_kf1 = f1_score(kpredict_labels_all[i], y_all, average='micro')
            time_sf1 = f1_score(spredict_labels_all[i], y_all, average='micro')
            time_rf1 = f1_score(rpredict_labels_all[i], y_all, average='micro')
            # 种类个数特殊处理，为了多分类的auc进行处理
            if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
                y_true_binarized = pd.get_dummies(y_all[0])
                y_true_binarized_array = y_true_binarized.values
            else:  # 多类别可以直接调用函数
                y_true_binarized = label_binarize(y_all, classes=classes)
            for j in range(0, len(classes)):  # 不同分类器结果进行分别处理
                b_class_auc = roc_auc_score(y_true_binarized, bprobabily_all[i][j], average='macro', multi_class='ovr')
                k_class_auc = roc_auc_score(y_true_binarized, kprobabily_all[i][j], average='macro', multi_class='ovr')
                s_class_auc = roc_auc_score(y_true_binarized, sprobabily_all[i][j], average='macro', multi_class='ovr')
                r_class_auc = roc_auc_score(y_true_binarized, rprobabily_all[i][j], average='macro', multi_class='ovr')
                b_all_auc.append(b_class_auc)#存储不同分类器的auc结果
                k_all_auc.append(k_class_auc)
                s_all_auc.append(s_class_auc)
                r_all_auc.append(r_class_auc)
            time_bauc = np.mean(b_all_auc)
            time_kauc = np.mean(k_all_auc)
            time_sauc = np.mean(s_all_auc)
            time_rauc = np.mean(r_all_auc)
            fold_b_f1.append(time_bf1)
            fold_k_f1.append(time_kf1)
            fold_s_f1.append(time_sf1)
            fold_r_f1.append(time_rf1)
            fold_b_auc.append(time_bauc)
            fold_k_auc.append(time_kauc)
            fold_s_auc.append(time_sauc)
            fold_r_auc.append(time_rauc)
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
        bf1table[method][iposition]  = np.mean(fold_b_f1)
        sf1table[method][iposition]  = np.mean(fold_k_f1)
        kf1table[method][iposition]  = np.mean(fold_s_f1)
        rf1table[method][iposition]  = np.mean(fold_r_f1)
        bauctable[method][iposition] = np.mean(fold_b_auc)
        sauctable[method][iposition] = np.mean(fold_k_auc)
        kauctable[method][iposition] = np.mean(fold_s_auc)
        rauctable[method][iposition] = np.mean(fold_r_auc)

tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
# 计算每列的平均值
for i, table in enumerate(tables):
    average_row = table.mean(axis=0)
    table.loc['average'] = average_row

#对每个表格求平均值并按照大小进行升序排序
bf1table = bf1table.sort_values(by='average', axis=1)
sf1table = sf1table.sort_values(by='average', axis=1)
kf1table = kf1table.sort_values(by='average', axis=1)
rf1table = rf1table.sort_values(by='average', axis=1)
bauctable = bauctable.sort_values(by='average', axis=1)
sauctable = sauctable.sort_values(by='average', axis=1)
kauctable = kauctable.sort_values(by='average', axis=1)
rauctable = rauctable.sort_values(by='average', axis=1)
print(bf1table,'\n',kf1table,'\n',sf1table,'\n',rf1table,'\n')
print(bauctable,'\n',sauctable,'\n',kauctable,'\n',rauctable,'\n')
#存储
bf1table.to_excel( 'tables/'+T+'/ten-bf1table.xlsx', index=True)
sf1table.to_excel( 'tables/'+T+'/ten-sf1table.xlsx', index=True)
kf1table.to_excel( 'tables/'+T+'/ten-kf1table.xlsx', index=True)
rf1table.to_excel( 'tables/'+T+'/ten-rf1table.xlsx', index=True)
bauctable.to_excel('tables/'+T+'/ten-bauctable.xlsx', index=True)
sauctable.to_excel('tables/'+T+'/ten-sauctable.xlsx', index=True)
kauctable.to_excel('tables/'+T+'/ten-kauctable.xlsx', index=True)
rauctable.to_excel('tables/'+T+'/ten-rauctable.xlsx', index=True)
#绘制箱线图
import matplotlib.pyplot as plt
import numpy as np
bf1table = pd.read_excel( 'tables/'+T+'/ten-bf1table.xlsx', index_col=0)
sf1table = pd.read_excel( 'tables/'+T+'/ten-sf1table.xlsx', index_col=0)
kf1table = pd.read_excel( 'tables/'+T+'/ten-kf1table.xlsx', index_col=0)
rf1table = pd.read_excel( 'tables/'+T+'/ten-rf1table.xlsx', index_col=0)
bauctable = pd.read_excel('tables/'+T+'/ten-bauctable.xlsx', index_col=0)
sauctable = pd.read_excel('tables/'+T+'/ten-sauctable.xlsx', index_col=0)
kauctable = pd.read_excel('tables/'+T+'/ten-kauctable.xlsx', index_col=0)
rauctable = pd.read_excel('tables/'+T+'/ten-rauctable.xlsx', index_col=0)
tables = [bf1table,sf1table,kf1table,rf1table,bauctable,sauctable,kauctable,rauctable]
# for table in tables:
#     # 计算每列的平均值
#     average_row = table.mean(axis=0)
#     table.loc['average'] = average_row
#     df_sorted = table.sort_values(by='average', axis=1)
#     print(table)
#
# print(bf1table)
# 创建一些示例数据
f1datas = [bf1table,sf1table,kf1table,rf1table]
aucdatas = [bauctable,sauctable,kauctable,rauctable]
names = ['Naive Bayes','3NN','SVM','RandomForest']
# 创建箱线图
i=-1
for data in f1datas:
    i+=1
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=data.columns)
    # plt.ylim(0.3, 1)
    # 添加标题和标签
    plt.title(names[i])
    plt.ylabel('F1')
    plt.grid(False)
    # 显示图形
    plt.grid(True)
    plt.savefig("picture/" + names[i] + T +"_ten_F1.png", dpi=300, bbox_inches='tight')
    plt.show()
j=-1
for data in aucdatas:
    j+=1
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=data.columns)
    # plt.ylim(0.3, 1)
    # 添加标题和标签
    plt.title(names[j])
    plt.ylabel('AUC')
    plt.grid(False)
    # 显示图形
    plt.grid(True)
    plt.savefig("picture/" + names[j] + T +"_ten_AUC.png", dpi=300, bbox_inches='tight')
    plt.show()
