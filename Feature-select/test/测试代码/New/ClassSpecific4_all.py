import copy
import random

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from weka.core.converters import Loader,Saver
import weka.core.jvm as jvm
from weka.filters import AttributeSelection,ASEvaluation,ASSearch
import scipy.io as scio
import numpy as np
import ast
import math
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from collections import Counter
# from OtherWay import *

# 读取mat文件'Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','Prostate_GE','TOX_171','GLIOMA','Arcene','Waveform','Wdbc','Synthetic_control','Authorship','Factors','Pixel','Isolet','ORL','WarpPIE10P','Su',
datasets = ['Dermatology',]
base_url = r"./dataset/"

# 启动jvm
# jvm.start(packages=True)
# for dataname in datasets:
#     for i in range(0,10):
#         filename = base_url +'arff/' + dataname + str(i) + '.arff'
#         #读取加载arff文件
#         print(filename)
#         loader = Loader(classname="weka.core.converters.ArffLoader")
#         data0 = loader.load_file(filename)
#         data0.class_is_last()#设置标签
#         last_column_values = []
#         # 去重并打印所有取值（取所有可能标签取值）
#         for k in range(data0.num_instances):
#             last_column_values.append(data0.get_instance(k).get_value(data0.class_index))
#         unique_values = set(last_column_values)
#         allresult = []
#         #对每个标签进行二分类
#         for u in unique_values:
#             othercount = 0 #计数
#             data = copy.deepcopy(data0)
#             for j in range(data.num_instances):
#                 if data.get_instance(j).get_value(data.class_index) != u:
#                     data.get_instance(j).set_value(data.class_index, 0.0)
#                     othercount+=1
#                 else:
#                     data.get_instance(j).set_value(data.class_index, 1.0)
#             count = data.num_instances - othercount
#             #样本数量平衡
#             if count < othercount:
#                 repeat_count = othercount - count
#                 instances_to_repeat = [inst for inst in data if inst.get_value(data0.class_index) == 1.0]
#                 for _ in range(repeat_count):
#                     instance_to_add = random.choice(instances_to_repeat)
#                     data.add_instance(instance_to_add)
#             # 转换数值类标签为分类类型标签
#             if data.class_attribute.is_numeric:
#                 from weka.filters import Filter
#                 num_to_nom = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", str(data.class_index + 1)])
#                 num_to_nom.inputformat(data)
#                 data = num_to_nom.filter(data)
#             # # 获取已安装的包列表，安装包
#             # import weka.core.packages as packages
#             # packages.install_package("attributeSelectionSearchMethods")
#             # installed_packages = packages.installed_packages()
#             # print(installed_packages)
#             #=================================搜索器和评估器的选择
#             #使用filter进行特征选取#四种方法两两组合
#             # search=ASSearch(classname='weka.attributeSelection.GeneticSearch')
#             search=ASSearch(classname='weka.attributeSelection.BestFirst',options=["-N", "50"])
#             evaluation = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
#             # evaluation = ASEvaluation(classname="weka.attributeSelection.ConsistencySubsetEval")
#             #方法名记录和特征选择开始
#             method = 'CFS-BF'
#             attsel=AttributeSelection()
#             attsel.evaluator=evaluation
#             attsel.search=search
#             #===========================================================训练数据提取特征
#             attsel.inputformat(data)
#             select=attsel.filter(data)
#             #结果保存
#             result = pd.DataFrame(select.attribute_names())
#             allresult.append(list(result[0]))
#         file_name = "./feature_select_result/"+method + '/'+method +'_' + dataname + str(i) + '.txt'
#         np.savetxt(file_name, allresult, fmt='%s')  # 使用savetxt()函数保存数组到文件
# jvm.stop()

#投票机制的标签计算
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
def changetosinge(x):
    return float(x)
# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# ====================================数据导入=====================================
#
# datanames = ['Wdbc','Synthetic_control','Authorship','Factors','Pixel','Isolet','ORL','WarpPIE10P','Su','TOX_171','Orlraws10P','CLL_SUB_111','Yeoh']'Dermatology','CLL_SUB_111','Orlraws10P',
# 'Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene','Waveform','Sorlie', 'ORL','WarpPIE10P','Musk1', 'Sorlie', 'Movement_libras','Waveform','Prostate_GE','Wdbc', 'Su','Synthetic_control','Authorship', 'Factors', 'Pixel', 'Isolet', 'Yale',  'WarpAR10P', 'GLIOMA', 'Arcene',
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
    nowpick = result  # 当前i个特征的特征选择结果
    # 初始化平均值和结果概率矩阵
    bresultmax = np.zeros((len(X_test), classnum))
    kresultmax = np.zeros((len(X_test), classnum))
    sresultmax = np.zeros((len(X_test), classnum))
    rresultmax = np.zeros((len(X_test), classnum))
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
    for index,item in enumerate(nowpick):
        # **Naïve Bayes (朴素贝叶斯)**
        from sklearn.naive_bayes import GaussianNB
        bayesmodel = GaussianNB()
        bayesmodel.fit(X_train[item], y_train)
        bpredicted_labels = bayesmodel.predict(X_test[item])
        bprobabilities = bayesmodel.predict_proba(X_test[item])
        bresultmax[:,index] = bpredicted_labels
        # **k-Nearest Neighbors (k-最近邻算法)**:
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(X_train[item], y_train)
        kpredicted_labels = knn_classifier.predict(X_test[item])
        kprobabilities = knn_classifier.predict_proba(X_test[item])
        kresultmax[:,index] = kpredicted_labels
        # **Support Vector Machines (支持向量机)**:
        from sklearn.svm import SVC
        svm_classifier = SVC(probability=True)
        svm_classifier.fit(X_train[item], y_train)
        sprobabilities = svm_classifier.predict_proba(X_test[item])
        spredicted_labels = svm_classifier.predict(X_test[item])
        sresultmax[:,index] = spredicted_labels
        # random Forest (随机森林)**:
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train[item], y_train)
        rprobabilities = rf_classifier.predict_proba(X_test[item])
        rpredicted_labels = rf_classifier.predict(X_test[item])
        rresultmax[:,index] = rpredicted_labels
        #每个分类器在当前情况下的分类概率列表（一次测试的所有分类器结果）
        bsingleclasspro.append(bprobabilities)#记录n种分类器单次概率
        ksingleclasspro.append(kprobabilities)
        ssingleclasspro.append(sprobabilities)
        rsingleclasspro.append(rprobabilities)
    bsinglepredicted_labels = compute_N(bresultmax)
    ksinglepredicted_labels = compute_N(kresultmax)
    ssinglepredicted_labels = compute_N(sresultmax)
    rsinglepredicted_labels = compute_N(rresultmax)
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
    file_name1 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_pro.mat"
    file_name2 = r"result/" + method +'/'+ dataname + str(fold) +  "_predict_lable.mat"
    scio.savemat(file_name1, predict_pro)
    scio.savemat(file_name2, predict_lable)

for dataname in datasets:
    data_url = dataname + '.mat'
    print(dataname)
    url = base_url + data_url  # 数据路径
    data = scio.loadmat(url)  # 读取数据文件
    X0 = pd.DataFrame(data['X'])  # 读取训练数据
    y0 = pd.DataFrame(data['Y'])  # 读取标签
    #================='Dermatology'特殊处理===================
    Special = X0.iloc[:, -1]
    X0 = X0.iloc[:, :-1]  # 读取训练数据
    a = np.array([item[0] for item in Special])
    label_encoder = LabelEncoder()
    a33 = label_encoder.fit_transform(a)
    X0[33] = a33
    # ================='Dermatology'特殊处理===================
    print(X0,y0)
    print("====================================================================================================")
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
    for col in X0.columns:
        X[col] = pd.cut(X0[col], bins=5, labels=False)
    # 使用rename函数重新命名列名，将x列名控制在0-n
    new_columns = list(range(X.shape[1] + 1))
    X = X.rename(columns=dict(zip(X.columns, new_columns)))
    # ====================================数据导入=====================================
    methods = ['CFS-BF',]#'CON-GEN','CFS-GEN','CON-GEN',
    classnum = len(Class)  # 类别数，分为几类
    samplenum = math.ceil(X.shape[0] / 10)  # 样本数向上取整
    kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
    for method in methods:
        print(method)
        # 创建一个十折交叉验证对象
        Fall = X.shape[1]  # 特征数
        # 选取的特征数量
        if Fall < 50:
            time = Fall/2
        else:
            time = 50
        # 对每一个9折，进行特征选择
        # 十折交叉验证
        for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):  # fold赋值之后就是0-9
            print("当前处理到第", fold, "折\n")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # 提取训练集条目
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # 提取测试集条目
            result = []  # 特征提取结果数组
            # -----------------直接读取特征提取结果
            file_name = r"feature_select_result/" + method + '/' + method + '_' + dataname + str(fold) + ".txt"
            # np.savetxt(file_name, result)  # 使用savetxt()函数保存数组到文件
            with open(file_name, 'r') as file:
                lines = file.readlines()
            result = [ast.literal_eval(line.strip()) for line in lines]
            # 对每一个数量的特征选择做测试
            csmdccmrtest(len(result), result, X_test, y_test, X_train, y_train)

#==================================数据预测完毕，进行预测结果的处理
#处理得到结果计算出的F1和AUC
T = '2'#用于区别储存
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

methods = ['CFS-BF',]#'cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs',
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
            bpredict_probabity = np.array(predict_pro['b'])[:, :, :lenth, :]#time个结果，每个结果有class个分类器，每个分类器有n个样本和class种可能
            kpredict_probabity = np.array(predict_pro['k'])[:, :, :lenth, :]
            spredict_probabity = np.array(predict_pro['s'])[:, :, :lenth, :]
            rpredict_probabity = np.array(predict_pro['r'])[:, :, :lenth, :]
            bpredict_labels = pd.DataFrame(predict_lable['b']).iloc[:, :lenth]#time个结果，每个结果n个样本，
            kpredict_labels = pd.DataFrame(predict_lable['k']).iloc[:, :lenth]
            spredict_labels = pd.DataFrame(predict_lable['s']).iloc[:, :lenth]
            rpredict_labels = pd.DataFrame(predict_lable['r']).iloc[:, :lenth]
            #转置矩阵，使行为样本列为特征数量
            bpredict_labels = bpredict_labels.T
            kpredict_labels = kpredict_labels.T
            spredict_labels = spredict_labels.T
            rpredict_labels = rpredict_labels.T
            feature_num = bpredict_labels.shape[1]
            for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
                # print(y_true_binarized,bprobabily_all[i])
                # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
                time_bf1 = f1_score(bpredict_labels[i], y_test, average='micro')
                time_kf1 = f1_score(kpredict_labels[i], y_test, average='micro')
                time_sf1 = f1_score(spredict_labels[i], y_test, average='micro')
                time_rf1 = f1_score(rpredict_labels[i], y_test, average='micro')
                # 种类个数特殊处理，为了多分类的auc进行处理
                if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
                    y_true_binarized = pd.get_dummies(y_test[0])
                    y_true_binarized_array = y_true_binarized.values
                else:  # 多类别可以直接调用函数
                    y_true_binarized = label_binarize(y_test, classes=classes)
                for j in range(0, len(classes)):  # 不同分类器结果进行分别处理
                    b_class_auc = roc_auc_score(y_true_binarized, bpredict_probabity[i][j], average='macro', multi_class='ovr')
                    k_class_auc = roc_auc_score(y_true_binarized, kpredict_probabity[i][j], average='macro', multi_class='ovr')
                    s_class_auc = roc_auc_score(y_true_binarized, spredict_probabity[i][j], average='macro', multi_class='ovr')
                    r_class_auc = roc_auc_score(y_true_binarized, rpredict_probabity[i][j], average='macro', multi_class='ovr')
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
            allfold_b_f1.append(fold_b_f1)
            allfold_k_f1.append(fold_k_f1)
            allfold_s_f1.append(fold_s_f1)
            allfold_r_f1.append(fold_r_f1)
            allfold_b_auc.append(fold_b_auc)
            allfold_k_auc.append(fold_k_auc)
            allfold_s_auc.append(fold_s_auc)
            allfold_r_auc.append(fold_r_auc)
        #当前数据集的结果为所有特征数量结果的平均数
        #处理完后存入表格记录
        bf1table[method][iposition] =  np.mean(allfold_b_f1 )
        sf1table[method][iposition] =  np.mean(allfold_k_f1 )
        kf1table[method][iposition] =  np.mean(allfold_s_f1 )
        rf1table[method][iposition] =  np.mean(allfold_r_f1 )
        bauctable[method][iposition] = np.mean(allfold_b_auc)
        sauctable[method][iposition] = np.mean(allfold_k_auc)
        kauctable[method][iposition] = np.mean(allfold_s_auc)
        rauctable[method][iposition] = np.mean(allfold_r_auc)

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
print(bf1table,kf1table,sf1table,rf1table)
print(bauctable,sauctable,kauctable,rauctable)
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

