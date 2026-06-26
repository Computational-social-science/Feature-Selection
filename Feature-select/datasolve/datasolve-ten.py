#处理得到结果计算出的F1和AUC
import warnings

import numpy as np
import scipy.io as scio
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize

warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def changetosinge(x):
    return float(x)
#读取数据
datasets = ['Sorlie','Movement_libras','Yale','Yeoh','WarpAR10P','GLIOMA', 'ORL', 'WarpPIE10P','Dermatology', 'Orlraws10P','CLL_SUB_111','Waveform','Wdbc', 'Su','Authorship', 'Pixel', 'Isolet','Arcene', 'Prostate_GE','TOX_171','Factors', 'Musk1','Synthetic_control',
         ]#
base_url = r"./dataset/"
# methods = ['cife','mrmr','jmim', 'mri', 'cfr', 'dcsf', 'mrmd', 'iwfs', 'ucrfs','csmdccmr',]#
methods = ['CON-BF', 'CON-GEN', 'CFS-GEN','csmdccmr2','CFS-BF','CSMI' ]
T = '2'
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

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
for method in methods:
    print(method,"=============================================================================")
    #存储不同方法所有数据集23个的结果
    alldataset_f1_b = []
    alldataset_f1_k = []
    alldataset_f1_s = []
    alldataset_f1_r = []
    alldataset_auc_b = []
    alldataset_auc_k = []
    alldataset_auc_s = []
    alldataset_auc_r = []
    for dataname in datasets:
        iposition = datasets.index(dataname)
        #读取数据及其真实值
        kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
        data_url = dataname + '.mat'
        print(dataname)
        url = base_url + data_url  # 数据路径
        data = scio.loadmat(url)  # 读取数据文件
        #读取文件中的真实标签
        y0 = pd.DataFrame(data['Y'])  # 读取标签
        # 应用函数到 DataFrame 的每个元素
        y0 = y0.applymap(changetosinge)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y0)
        Class = set(y_encoded)
        y = pd.DataFrame(y_encoded)
        classes = list(Class)
        #当前数据集的十折数据所有样本结果存储
        bpredict_labels_all = pd.DataFrame()
        kpredict_labels_all = pd.DataFrame()
        spredict_labels_all = pd.DataFrame()
        rpredict_labels_all = pd.DataFrame()
        pre_pro_all_b = []
        pre_pro_all_k = []
        pre_pro_all_s = []
        pre_pro_all_r = []
        # y_all = pd.DataFrame()#真实值的所有样本结果
        # 十折交叉验证
        fold_b_f1 = []
        fold_k_f1 = []
        fold_s_f1 = []
        fold_r_f1 = []
        fold_b_auc = []
        fold_k_auc = []
        fold_s_auc = []
        fold_r_auc = []
        for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
            print("当前是第",fold)
            all_f1_b = []
            all_f1_k = []
            all_f1_s = []
            all_f1_r = []
            y_test = y.iloc[test_index]  # 提取测试集条目
            #读取预测结果
            file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
            file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
            predict_pro = scio.loadmat(file_name1)  # 读取数据文件
            predict_lable = scio.loadmat(file_name2)  # 读取数据文件
            lenth = len(y_test)
            pre_prob_sum_b = np.zeros((lenth,len(classes)))
            pre_prob_sum_k = np.zeros((lenth,len(classes)))
            pre_prob_sum_s = np.zeros((lenth,len(classes)))
            pre_prob_sum_r = np.zeros((lenth,len(classes)))
            # print(lenth)
            #对于每一折的数据都读取对应数量
            if method == 'csmdccmr':
                bpredict_labels_squeezed = np.squeeze(predict_lable['b'], axis=1)
                kpredict_labels_squeezed = np.squeeze(predict_lable['k'], axis=1)
                spredict_labels_squeezed = np.squeeze(predict_lable['s'], axis=1)
                rpredict_labels_squeezed = np.squeeze(predict_lable['r'], axis=1)
                bpredict_probabity_squeezed = np.squeeze(predict_pro['b'], axis=1)
                kpredict_probabity_squeezed = np.squeeze(predict_pro['k'], axis=1)
                spredict_probabity_squeezed = np.squeeze(predict_pro['s'], axis=1)
                rpredict_probabity_squeezed = np.squeeze(predict_pro['r'], axis=1)
                bpredict_probabity = np.array(bpredict_probabity_squeezed)[:, :, :lenth, :]
                kpredict_probabity = np.array(kpredict_probabity_squeezed)[:, :, :lenth, :]
                spredict_probabity = np.array(spredict_probabity_squeezed)[:, :, :lenth, :]
                rpredict_probabity = np.array(rpredict_probabity_squeezed)[:, :, :lenth, :]
            else:
                bpredict_labels_squeezed = pd.DataFrame(predict_lable['b'])
                kpredict_labels_squeezed = pd.DataFrame(predict_lable['k'])
                spredict_labels_squeezed = pd.DataFrame(predict_lable['s'])
                rpredict_labels_squeezed = pd.DataFrame(predict_lable['r'])
                bpredict_probabity_squeezed = np.array(predict_pro['b'])
                kpredict_probabity_squeezed = np.array(predict_pro['k'])
                spredict_probabity_squeezed = np.array(predict_pro['s'])
                rpredict_probabity_squeezed = np.array(predict_pro['r'])
                bpredict_probabity = np.array(bpredict_probabity_squeezed)[:,  :lenth, :]
                kpredict_probabity = np.array(kpredict_probabity_squeezed)[:,  :lenth, :]
                spredict_probabity = np.array(spredict_probabity_squeezed)[:,  :lenth, :]
                rpredict_probabity = np.array(rpredict_probabity_squeezed)[:,  :lenth, :]
            # print(bpredict_probabity_squeezed.shape, kpredict_probabity_squeezed.shape,lenth)

            # print(bpredict_probabity.shape,kpredict_probabity.shape)
            bpredict_labels = pd.DataFrame(bpredict_labels_squeezed).iloc[:,:lenth]
            kpredict_labels = pd.DataFrame(kpredict_labels_squeezed).iloc[:,:lenth]
            spredict_labels = pd.DataFrame(spredict_labels_squeezed).iloc[:,:lenth]
            rpredict_labels = pd.DataFrame(rpredict_labels_squeezed).iloc[:,:lenth]
            bpredict_labels = bpredict_labels.T
            kpredict_labels = kpredict_labels.T
            spredict_labels = spredict_labels.T
            rpredict_labels = rpredict_labels.T
            feature_num = bpredict_labels.shape[1]
            # print(bpredict_probabity.shape)
            # print(feature_num)
            for i in range(0, feature_num):  # 对不同特征数量的结果进行处理
                # print(y_true_binarized,bprobabily_all[i])
                # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
                bf1 = f1_score(bpredict_labels[i], y_test, average='micro')
                kf1 = f1_score(kpredict_labels[i], y_test, average='micro')
                sf1 = f1_score(spredict_labels[i], y_test, average='micro')
                rf1 = f1_score(rpredict_labels[i], y_test, average='micro')
                # 种类个数
                if (len(classes) == 2):  # 若只分为两个类，需要特殊处理
                    y_true_binarized = pd.get_dummies(y_test[0])
                    y_true_binarized_array = y_true_binarized.values
                else:  # 多类别可以直接调用函数
                    y_true_binarized = label_binarize(y_test, classes=classes)
                # print(y_true_binarized)
                if method in ['csmdccmr', 'csmdccmr2']:
                    for j in range(0, len(classes)):#不同种类分类器的预测概率求平均
                        pre_prob_sum_b += bpredict_probabity[i][j]
                        pre_prob_sum_k += kpredict_probabity[i][j]
                        pre_prob_sum_s += spredict_probabity[i][j]
                        pre_prob_sum_r += rpredict_probabity[i][j]
                    pre_prob_sum_b = pre_prob_sum_b/len(classes)
                    pre_prob_sum_k = pre_prob_sum_k/len(classes)
                    pre_prob_sum_s = pre_prob_sum_s/len(classes)
                    pre_prob_sum_r = pre_prob_sum_r/len(classes)
                    bauc = roc_auc_score(y_true_binarized, pre_prob_sum_b, average='macro', multi_class='ovr')
                    kauc = roc_auc_score(y_true_binarized, pre_prob_sum_k, average='macro', multi_class='ovr')
                    sauc = roc_auc_score(y_true_binarized, pre_prob_sum_s, average='macro', multi_class='ovr')
                    rauc = roc_auc_score(y_true_binarized, pre_prob_sum_r, average='macro', multi_class='ovr')
                else:
                    print(y_true_binarized)
                    bauc = roc_auc_score(y_true_binarized, bpredict_probabity[i], average='macro', multi_class='ovr')
                    kauc = roc_auc_score(y_true_binarized, kpredict_probabity[i], average='macro', multi_class='ovr')
                    sauc = roc_auc_score(y_true_binarized, spredict_probabity[i], average='macro', multi_class='ovr')
                    rauc = roc_auc_score(y_true_binarized, rpredict_probabity[i], average='macro', multi_class='ovr')
                    # print(bauc)
                    # all_auc_b.append(bauc)
                    # all_auc_k.append(kauc)
                    # all_auc_s.append(sauc)
                    # all_auc_r.append(rauc)
                # 计算后将不同特征数量结果储存
                all_f1_b.append(bf1)
                all_f1_k.append(kf1)
                all_f1_s.append(sf1)
                all_f1_r.append(rf1)
            fold_b_f1.append(all_f1_b)
            fold_k_f1.append(all_f1_k)
            fold_s_f1.append(all_f1_s)
            fold_r_f1.append(all_f1_r)
            fold_b_auc.append(bauc)
            fold_k_auc.append(kauc)
            fold_s_auc.append(sauc)
            fold_r_auc.append(rauc)
            #这里的lables行为特征数量，列为样本;这里的概率shape为（特征数，样本数，类别数）
            #将每一折的数据都综合到all中，使所有样本都放在一起
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
        #对于概率数据进行特殊处理，将矩阵连接在一起
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
        #将数据样本重新命名
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

        #经过处理之后label的shape是（总样本数，特征数） 概率的shape是（特征数，总样本数，类别数）
        #用于记录所有特征数情况下的f1和auc值，应当有特征数量大小

        #当前数据集的结果为所有特征数量结果的平均数
        # print(fold_b_f1,fold_r_auc)
        alldataset_f1_b.append(np.mean( fold_b_f1 ))
        alldataset_f1_k.append(np.mean( fold_k_f1 ))
        alldataset_f1_s.append(np.mean( fold_s_f1 ))
        alldataset_f1_r.append(np.mean( fold_r_f1 ))
        alldataset_auc_b.append(np.mean(fold_b_auc))
        alldataset_auc_k.append(np.mean(fold_k_auc))
        alldataset_auc_s.append(np.mean(fold_s_auc))
        alldataset_auc_r.append(np.mean(fold_r_auc))
        #处理完后存入表格记录
        bf1table[method][iposition] =  np.mean(fold_b_f1 )
        sf1table[method][iposition] =  np.mean(fold_k_f1 )
        kf1table[method][iposition] =  np.mean(fold_s_f1 )
        rf1table[method][iposition] =  np.mean(fold_r_f1 )
        bauctable[method][iposition] = np.mean(fold_b_auc)
        sauctable[method][iposition] = np.mean(fold_k_auc)
        kauctable[method][iposition] = np.mean(fold_s_auc)
        rauctable[method][iposition] = np.mean(fold_r_auc)
#对每个表格求平均值并按照大小进行升序排序
tables = [bf1table, sf1table, kf1table, rf1table, bauctable, sauctable, kauctable, rauctable]
for i, table in enumerate(tables):
    # 计算每列的平均值
    average_row = table.mean(axis=0)
    table.loc['average'] = average_row
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
