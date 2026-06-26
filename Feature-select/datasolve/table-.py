import pandas as pd
import numpy as np
import scipy.io as scio
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

T = '1'
# methods = ['POS-CSR','MI-CSR','CFS-GEN', 'CFS-BF', 'CON-BF', 'CON-GEN','csmdccmr2',]
methods = ['mRMR','CIFE','JMIM', 'IWFS', 'MRI', 'CFR', 'DCSF', 'MRMD', 'UCRFS','csmdccmr',]
datasets = ['Dermatology','Movement_libras', 'Musk1', 'Synthetic_control','Waveform','Wdbc','Arcene',
               'CLL_SUB_111', 'GLIOMA','Isolet', 'ORL', 'Orlraws10P','Prostate_GE','TOX_171',
              'WarpAR10P', 'WarpPIE10P', 'Yale','Authorship', 'Factors', 'Pixel',  'Sorlie', 'Su', 'Yeoh']#'FeatMIAS',
base_url = r"./dataset/"
#
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
def changetosinge(x):
    return float(x)
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
for method in methods:
    print(method,"=============================================================================")
    #存储不同方法所有数据集23个的结果
    alldataset_acc_b = []
    alldataset_acc_k = []
    alldataset_acc_s = []
    alldataset_acc_r = []
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
        #当前数据集的十折数据所有样本结果存储
        bpredict_labels_all = pd.DataFrame()
        kpredict_labels_all = pd.DataFrame()
        spredict_labels_all = pd.DataFrame()
        rpredict_labels_all = pd.DataFrame()
        pre_pro_all_b = []
        pre_pro_all_k = []
        pre_pro_all_s = []
        pre_pro_all_r = []
        y_all = pd.DataFrame()#真实值的所有样本结果
        # 十折交叉验证
        for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
            y_test = y.iloc[test_index]  # 提取测试集条目
            file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
            file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
            predict_pro = scio.loadmat(file_name1)  # 读取数据文件
            predict_lable = scio.loadmat(file_name2)  # 读取数据文件
            lenth = len(y_test)
            # print(lenth)
            #对于每一折的数据都读取对应数量
            if method == 'csmdccmr':
                bpredict_labels_squeezed = np.squeeze(predict_lable['b'], axis=1)
                kpredict_labels_squeezed = np.squeeze(predict_lable['k'], axis=1)
                spredict_labels_squeezed = np.squeeze(predict_lable['s'], axis=1)
                rpredict_labels_squeezed = np.squeeze(predict_lable['r'], axis=1)
            else:
                bpredict_labels_squeezed = pd.DataFrame(predict_lable['b'])
                kpredict_labels_squeezed = pd.DataFrame(predict_lable['k'])
                spredict_labels_squeezed = pd.DataFrame(predict_lable['s'])
                rpredict_labels_squeezed = pd.DataFrame(predict_lable['r'])

            bpredict_labels = pd.DataFrame(bpredict_labels_squeezed).iloc[:,:lenth]
            kpredict_labels = pd.DataFrame(kpredict_labels_squeezed).iloc[:,:lenth]
            spredict_labels = pd.DataFrame(spredict_labels_squeezed).iloc[:,:lenth]
            rpredict_labels = pd.DataFrame(rpredict_labels_squeezed).iloc[:,:lenth]

            #这里的lables行为特征数量，列为样本;这里的概率shape为（特征数，样本数，类别数）
            #将每一折的数据都综合到all中，使所有样本都放在一起
            bpredict_labels_all = pd.concat([bpredict_labels_all, bpredict_labels], axis=1)
            kpredict_labels_all = pd.concat([kpredict_labels_all, kpredict_labels], axis=1)
            spredict_labels_all = pd.concat([spredict_labels_all, spredict_labels], axis=1)
            rpredict_labels_all = pd.concat([rpredict_labels_all, rpredict_labels], axis=1)
            y_all = pd.concat([y_all, y_test], axis=0)
        #对于概率数据进行特殊处理，将矩阵连接在一起
        # print(bpredict_labels_all.shape, len(pre_pro_all_r),len(pre_pro_all_r[0]),len(pre_pro_all_r[0][0]),len(pre_pro_all_r[0][0][0]),len(pre_pro_all_r[0][0][0][0]))
        # print(bprobabily_all.shape)
        #将数据样本重新命名
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
        #种类个数
        classes = list(set(y_all[0]))
        #经过处理之后label的shape是（总样本数，特征数） 概率的shape是（特征数，总样本数，类别数）
        #用于记录所有特征数情况下的f1和auc值，应当有特征数量大小
        all_acc_b = []
        all_acc_k = []
        all_acc_s = []
        all_acc_r = []
        # print(bpredict_labels_all.shape)

        for i in range(0,feature_num):#对不同特征数量的结果进行处理
            # print(y_true_binarized,bprobabily_all[i])
            # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
            bacc = calculate_accuracy(bpredict_labels_all[i], y_all[0],)
            kacc = calculate_accuracy(kpredict_labels_all[i], y_all[0],)
            sacc = calculate_accuracy(spredict_labels_all[i], y_all[0],)
            racc = calculate_accuracy(rpredict_labels_all[i], y_all[0],)
            #计算后将不同特征数量结果储存
            all_acc_b.append(bacc)
            all_acc_k.append(kacc)
            all_acc_s.append(sacc)
            all_acc_r.append(racc)
        #当前数据集的结果为所有特征数量结果的平均数
        alldataset_acc_b.append(np.mean(all_acc_b))
        alldataset_acc_k.append(np.mean(all_acc_k))
        alldataset_acc_s.append(np.mean(all_acc_s))
        alldataset_acc_r.append(np.mean(all_acc_r))
        #处理完后存入表格记录
        bmean = np.mean(all_acc_b)
        smean = np.mean(all_acc_k)
        kmean = np.mean(all_acc_s)
        rmean = np.mean(all_acc_r)
        bstd = np.std(all_acc_b)
        sstd = np.std(all_acc_k)
        kstd = np.std(all_acc_s)
        rstd = np.std(all_acc_r)
        bacctable[method][iposition] = bmean
        sacctable[method][iposition] = smean
        kacctable[method][iposition] = kmean
        racctable[method][iposition] = rmean
        bstdtable[method][iposition] = bstd
        sstdtable[method][iposition] = sstd
        kstdtable[method][iposition] = kstd
        rstdtable[method][iposition] = rstd
#对每个表格求平均值并按照大小进行升序排序
tables = [bacctable, sacctable, kacctable, racctable]
for i, table in enumerate(tables):
    # 计算每列的平均值
    average_row = table.mean(axis=0)
    table.loc['average'] = average_row
# bacctable = bacctable.sort_values(by='average', axis=1)
# sacctable = sacctable.sort_values(by='average', axis=1)
# kacctable = kacctable.sort_values(by='average', axis=1)
# racctable = racctable.sort_values(by='average', axis=1)
bacctable.to_excel( 'tables/1/bacctable.xlsx', index=True)
sacctable.to_excel( 'tables/1/sacctable.xlsx', index=True)
kacctable.to_excel( 'tables/1/kacctable.xlsx', index=True)
racctable.to_excel( 'tables/1/racctable.xlsx', index=True)
bstdtable.to_excel( 'tables/1/bstdtable.xlsx', index=True)
sstdtable.to_excel( 'tables/1/sstdtable.xlsx', index=True)
kstdtable.to_excel( 'tables/1/kstdtable.xlsx', index=True)
rstdtable.to_excel( 'tables/1/rstdtable.xlsx', index=True)
print(bacctable,kacctable,sacctable,racctable)

# bacctable = pd.read_excel( 'tables/'+T+'/bacctable.xlsx',index_col=0)
# sacctable = pd.read_excel( 'tables/'+T+'/sacctable.xlsx',index_col=0)
# kacctable = pd.read_excel( 'tables/'+T+'/kacctable.xlsx',index_col=0)
# racctable = pd.read_excel( 'tables/'+T+'/racctable.xlsx',index_col=0)
#
# bstdtable = pd.read_excel( 'tables/'+T+'/bstdtable.xlsx',index_col=0)
# sstdtable = pd.read_excel( 'tables/'+T+'/sstdtable.xlsx',index_col=0)
# kstdtable = pd.read_excel( 'tables/'+T+'/kstdtable.xlsx',index_col=0)
# rstdtable = pd.read_excel( 'tables/'+T+'/rstdtable.xlsx',index_col=0)

balltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
balltable.columns = methods
balltable.index = datasets
salltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
salltable.columns = methods
salltable.index = datasets
kalltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
kalltable.columns = methods
kalltable.index = datasets
ralltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
ralltable.columns = methods
ralltable.index = datasets








