import Orange
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from Orange.evaluation import compute_CD, graph_ranks
# # # T = '2'
# # methods = ['EAMB','IAMB','Inter-IAMB','LRH','Fast-IAMB','GSMB','FBED']
# # # methods = ['mRMR','CIFE','JMIM', 'IWFS', 'MRI', 'CFR', 'DCSF', 'MRMD','UCRFS', 'csmdccmr',]#
# # datasets = ['Movement_libras','Dermatology', 'Musk1', 'Synthetic_control','Waveform','Wdbc','Arcene',#'FeatMIAS',
# #                'CLL_SUB_111', 'GLIOMA','Isolet', 'Orlraws10P','Prostate_GE','TOX_171','ORL',
# #               'WarpAR10P', 'WarpPIE10P', 'Yale','Authorship', 'Factors', 'Pixel',  'Sorlie', 'Su', 'Yeoh']
# # base_url = r"./dataset/"
# #
# # # def calculate_accuracy(predicted_labels, true_labels):
# # #     """
# # #     计算分类器的准确度。
# # #     参数:
# # #     predicted_labels: 预测的标签列表。
# # #     true_labels: 真实的标签列表。
# # #     返回值:
# # #     accuracy: 分类器的准确度。
# # #     """
# # #     # 确保预测结果和真实标签的长度相同
# # #     if len(predicted_labels) != len(true_labels):
# # #         raise ValueError("预测结果和真实标签的长度不一致。")
# # #     predicted_labels = [int(x) for x in predicted_labels]
# # #     true_labels = [int(x) for x in true_labels]
# # #     # 计算预测正确的数量
# # #     correct_count = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
# # #
# # #     # 计算准确度
# # #     accuracy = correct_count / len(true_labels)
# # #
# # #     return accuracy
# # # def changetosinge(x):
# # #     return float(x)
# # # bacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # bacctable.columns = methods
# # # bacctable.index = datasets
# # # sacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # sacctable.columns = methods
# # # sacctable.index = datasets
# # # kacctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # kacctable.columns = methods
# # # kacctable.index = datasets
# # # racctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # racctable.columns = methods
# # # racctable.index = datasets
# # # bstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # bstdtable.columns = methods
# # # bstdtable.index = datasets
# # # sstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # sstdtable.columns = methods
# # # sstdtable.index = datasets
# # # kstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # kstdtable.columns = methods
# # # kstdtable.index = datasets
# # # rstdtable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # rstdtable.columns = methods
# # # rstdtable.index = datasets
# # # for method in methods:
# # #     print(method,"=============================================================================")
# # #     #存储不同方法所有数据集23个的结果
# # #     alldataset_acc_b = []
# # #     alldataset_acc_k = []
# # #     alldataset_acc_s = []
# # #     alldataset_acc_r = []
# # #     for dataname in datasets:
# # #         iposition = datasets.index(dataname)
# # #         #读取数据及其真实值
# # #         kfold = KFold(n_splits=10, shuffle=True, random_state=19)  # 十折交叉验证，固定种子
# # #         data_url = dataname + '.mat'
# # #         print(dataname)
# # #         url = base_url + data_url  # 数据路径
# # #         data = scio.loadmat(url)  # 读取数据文件
# # #         #读取文件中的真实标签
# # #         y0 = pd.DataFrame(data['Y'])  # 读取标签
# # #         # 应用函数到 DataFrame 的每个元素
# # #         y0 = y0.applymap(changetosinge)
# # #         label_encoder = LabelEncoder()
# # #         y_encoded = label_encoder.fit_transform(y0)
# # #         Class = set(y_encoded)
# # #         y = pd.DataFrame(y_encoded)
# # #         #当前数据集的十折数据所有样本结果存储
# # #         bpredict_labels_all = pd.DataFrame()
# # #         kpredict_labels_all = pd.DataFrame()
# # #         spredict_labels_all = pd.DataFrame()
# # #         rpredict_labels_all = pd.DataFrame()
# # #         pre_pro_all_b = []
# # #         pre_pro_all_k = []
# # #         pre_pro_all_s = []
# # #         pre_pro_all_r = []
# # #         y_all = pd.DataFrame()#真实值的所有样本结果
# # #         # 十折交叉验证
# # #         all_acc_b = []
# # #         all_acc_k = []
# # #         all_acc_s = []
# # #         all_acc_r = []
# # #         for fold, (train_index, test_index) in enumerate(kfold.split(y)):  # fold赋值之后就是0-9
# # #             y_test = y.iloc[test_index]  # 提取测试集条目
# # #             file_name1 = r"result/" + method + '/' + dataname + str(fold) + "_predict_pro.mat"
# # #             file_name2 = r"result/" + method + '/' + dataname + str(fold) + "_predict_lable.mat"
# # #             predict_pro = scio.loadmat(file_name1)  # 读取数据文件
# # #             predict_lable = scio.loadmat(file_name2)  # 读取数据文件
# # #             lenth = len(y_test)
# # #             # print(predict_lable)
# # #             # print(lenth)
# # #             #对于每一折的数据都读取对应数量
# # #             if method in ['csmdccmr'] :
# # #                 bpredict_labels_squeezed = np.squeeze(predict_lable['b'], axis=1)
# # #                 kpredict_labels_squeezed = np.squeeze(predict_lable['k'], axis=1)
# # #                 spredict_labels_squeezed = np.squeeze(predict_lable['s'], axis=1)
# # #                 rpredict_labels_squeezed = np.squeeze(predict_lable['r'], axis=1)
# # #             else:
# # #                 bpredict_labels_squeezed = pd.DataFrame(predict_lable['b'])
# # #                 kpredict_labels_squeezed = pd.DataFrame(predict_lable['k'])
# # #                 spredict_labels_squeezed = pd.DataFrame(predict_lable['s'])
# # #                 rpredict_labels_squeezed = pd.DataFrame(predict_lable['r'])
# # #
# # #             bpredict_labels = pd.DataFrame(bpredict_labels_squeezed).iloc[:,:lenth].T
# # #             kpredict_labels = pd.DataFrame(kpredict_labels_squeezed).iloc[:,:lenth].T
# # #             spredict_labels = pd.DataFrame(spredict_labels_squeezed).iloc[:,:lenth].T
# # #             rpredict_labels = pd.DataFrame(rpredict_labels_squeezed).iloc[:,:lenth].T
# # #             # print(bpredict_labels)
# # #             bacc = calculate_accuracy(bpredict_labels[0], y_test[0],)
# # #             kacc = calculate_accuracy(kpredict_labels[0], y_test[0],)
# # #             sacc = calculate_accuracy(spredict_labels[0], y_test[0],)
# # #             racc = calculate_accuracy(rpredict_labels[0], y_test[0],)
# # #             #计算后将不同特征数量结果储存
# # #             all_acc_b.append(bacc)
# # #             all_acc_k.append(kacc)
# # #             all_acc_s.append(sacc)
# # #             all_acc_r.append(racc)
# # #             #这里的lables行为特征数量，列为样本;这里的概率shape为（特征数，样本数，类别数）
# # #             #将每一折的数据都综合到all中，使所有样本都放在一起
# # #             # bpredict_labels_all = pd.concat([bpredict_labels_all, bpredict_labels], axis=1)
# # #             # kpredict_labels_all = pd.concat([kpredict_labels_all, kpredict_labels], axis=1)
# # #             # spredict_labels_all = pd.concat([spredict_labels_all, spredict_labels], axis=1)
# # #             # rpredict_labels_all = pd.concat([rpredict_labels_all, rpredict_labels], axis=1)
# # #             # y_all = pd.concat([y_all, y_test], axis=0)
# # #         #对于概率数据进行特殊处理，将矩阵连接在一起
# # #         # print(bpredict_labels_all.shape, len(pre_pro_all_r),len(pre_pro_all_r[0]),len(pre_pro_all_r[0][0]),len(pre_pro_all_r[0][0][0]),len(pre_pro_all_r[0][0][0][0]))
# # #         # print(bprobabily_all.shape)
# # #         #将数据样本重新命名
# # #         # sample_num = bpredict_labels_all.shape[1]
# # #         feature_num = bpredict_labels_all.shape[0]
# # #         # new_column_names = range(sample_num)
# # #         #
# # #         # bpredict_labels_all.columns = new_column_names[:sample_num]
# # #         # bpredict_labels_all = bpredict_labels_all.T
# # #         #
# # #         # kpredict_labels_all.columns = new_column_names[:sample_num]
# # #         # kpredict_labels_all = kpredict_labels_all.T
# # #         #
# # #         # spredict_labels_all.columns = new_column_names[:sample_num]
# # #         # spredict_labels_all = spredict_labels_all.T
# # #         #
# # #         # rpredict_labels_all.columns = new_column_names[:sample_num]
# # #         # rpredict_labels_all = rpredict_labels_all.T
# # #         #种类个数
# # #         # classes = list(set(y_all[0]))
# # #         #经过处理之后label的shape是（总样本数，特征数） 概率的shape是（特征数，总样本数，类别数）
# # #         #用于记录所有特征数情况下的f1和auc值，应当有特征数量大小
# # #
# # #         # print(bpredict_labels_all.shape)
# # #
# # #         # for i in range(0,feature_num):#对不同特征数量的结果进行处理
# # #         #     # print(y_true_binarized,bprobabily_all[i])
# # #         #     # print(len(bprobabily_all[i]),len(bprobabily_all[0][0]), len(bprobabily_all[i][0][0]))
# # #         #
# # #         #     print(all_acc_b)
# # #         #当前数据集的结果为所有特征数量结果的平均数
# # #         # alldataset_acc_b.append(np.mean(all_acc_b))
# # #         # alldataset_acc_k.append(np.mean(all_acc_k))
# # #         # alldataset_acc_s.append(np.mean(all_acc_s))
# # #         # alldataset_acc_r.append(np.mean(all_acc_r))
# # #         #处理完后存入表格记录
# # #         bmean = np.mean(all_acc_b)
# # #         smean = np.mean(all_acc_k)
# # #         kmean = np.mean(all_acc_s)
# # #         rmean = np.mean(all_acc_r)
# # #         bstd = np.std(all_acc_b)
# # #         sstd = np.std(all_acc_k)
# # #         kstd = np.std(all_acc_s)
# # #         rstd = np.std(all_acc_r)
# # #         # print(alldataset_acc_b)
# # #         # print(bstd,rstd,kstd,sstd)
# # #         bacctable[method][iposition] = bmean
# # #         sacctable[method][iposition] = smean
# # #         kacctable[method][iposition] = kmean
# # #         racctable[method][iposition] = rmean
# # #         bstdtable[method][iposition] = bstd
# # #         sstdtable[method][iposition] = sstd
# # #         kstdtable[method][iposition] = kstd
# # #         rstdtable[method][iposition] = rstd
# # # #对每个表格求平均值并按照大小进行升序排序
# # # tables = [bacctable, sacctable, kacctable, racctable]
# # # for i, table in enumerate(tables):
# # #     # 计算每列的平均值
# # #     average_row = table.mean(axis=0)
# # #     table.loc['average'] = average_row
# # # # bacctable = bacctable.sort_values(by='average', axis=1)
# # # # sacctable = sacctable.sort_values(by='average', axis=1)
# # # # kacctable = kacctable.sort_values(by='average', axis=1)
# # # # racctable = racctable.sort_values(by='average', axis=1)
# # # bacctable.to_excel( 'tables/'+T+'/bacctable.xlsx', index=True)
# # # sacctable.to_excel( 'tables/'+T+'/sacctable.xlsx', index=True)
# # # kacctable.to_excel( 'tables/'+T+'/kacctable.xlsx', index=True)
# # # racctable.to_excel( 'tables/'+T+'/racctable.xlsx', index=True)
# # # bstdtable.to_excel( 'tables/'+T+'/bstdtable.xlsx', index=True)
# # # sstdtable.to_excel( 'tables/'+T+'/sstdtable.xlsx', index=True)
# # # kstdtable.to_excel( 'tables/'+T+'/kstdtable.xlsx', index=True)
# # # rstdtable.to_excel( 'tables/'+T+'/rstdtable.xlsx', index=True)
# # # print(bacctable,kacctable,sacctable,racctable)
# # #
# # # bacctable = pd.read_excel( 'tables/'+T+'/bacctable.xlsx',index_col=0)
# # # sacctable = pd.read_excel( 'tables/'+T+'/sacctable.xlsx',index_col=0)
# # # kacctable = pd.read_excel( 'tables/'+T+'/kacctable.xlsx',index_col=0)
# # # racctable = pd.read_excel( 'tables/'+T+'/racctable.xlsx',index_col=0)
# # #
# # # bstdtable = pd.read_excel( 'tables/'+T+'/bstdtable.xlsx',index_col=0)
# # # sstdtable = pd.read_excel( 'tables/'+T+'/sstdtable.xlsx',index_col=0)
# # # kstdtable = pd.read_excel( 'tables/'+T+'/kstdtable.xlsx',index_col=0)
# # # rstdtable = pd.read_excel( 'tables/'+T+'/rstdtable.xlsx',index_col=0)
# # #
# # # balltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # balltable.columns = methods
# # # balltable.index = datasets
# # # salltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # salltable.columns = methods
# # # salltable.index = datasets
# # # kalltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # kalltable.columns = methods
# # # kalltable.index = datasets
# # # ralltable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # ralltable.columns = methods
# # # ralltable.index = datasets
# # # # print(list(bacctable.columns), list(bstdtable.columns),list(bacctable.index), list(bstdtable.index))
# # # # assert list(bacctable.columns) == list(bstdtable.columns), "两个DataFrame的列名不一致"
# # # # assert list(bacctable.index) == list(bstdtable.index), "两个DataFrame的索引不一致"
# # #
# # # for row in bstdtable.index:
# # #     for col in bstdtable.columns:
# # #         balltable.at[row, col] = f"{bacctable.at[row, col]*100:.2f}±{bstdtable.at[row, col]:.2f}"
# # #         salltable.at[row, col] = f"{sacctable.at[row, col]*100:.2f}±{sstdtable.at[row, col]:.2f}"
# # #         kalltable.at[row, col] = f"{kacctable.at[row, col]*100:.2f}±{kstdtable.at[row, col]:.2f}"
# # #         ralltable.at[row, col] = f"{racctable.at[row, col]*100:.2f}±{rstdtable.at[row, col]:.2f}"
# # # balltable.loc['average'] = bacctable.loc['average']
# # # salltable.loc['average'] = sacctable.loc['average']
# # # kalltable.loc['average'] = kacctable.loc['average']
# # # ralltable.loc['average'] = racctable.loc['average']
# # # balltable.to_excel( 'tables/'+T+'/balltable.xlsx', index=True)
# # # salltable.to_excel( 'tables/'+T+'/salltable.xlsx', index=True)
# # # kalltable.to_excel( 'tables/'+T+'/kalltable.xlsx', index=True)
# # # ralltable.to_excel( 'tables/'+T+'/ralltable.xlsx', index=True)
# # #
# # # avetable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # avetable.columns = methods
# # # avetable.index = datasets
# # # # 四种方法平均
# # # acctable = pd.DataFrame(index=range(len(datasets)), columns=range(len(methods)))
# # # acctable.columns = methods
# # # acctable.index = datasets
# # # for row in bstdtable.index:
# # #     for col in bstdtable.columns:
# # #         avetable.at[row, col] = f"{(bacctable.at[row, col]+kacctable.at[row, col]+sacctable.at[row, col]+racctable.at[row, col])/4*100:.2f}±{(bstdtable.at[row, col]+kstdtable.at[row, col]+sstdtable.at[row, col]+rstdtable.at[row, col])/4:.2f}"
# # #         acctable.at[row, col] = (bacctable.at[row, col] + kacctable.at[row, col] + sacctable.at[row, col] + racctable.at[row, col]) / 4 * 100
# # # average_row = acctable.mean(axis=0)
# # # # print(acctable.mean(axis=0))
# # # # print(acctable)
# # # avetable.loc['average'] = average_row
# # # avetable.to_excel( 'tables/'+T+'/avetable.xlsx', index=True)
# # # avetable = avetable.rank(method='max',ascending=False, axis=1)
# # # avetable_rank = avetable.mean(axis=0)
# # # print(avetable_rank,111)
# # # names = ['Naive Bayes','3NN','SVM','RandomForest']
# # # ranktable = pd.DataFrame(index=range(len(names)), columns=range(len(methods)))
# # # ranktable.columns = methods
# # # ranktable.index = names
# # # branked_acctable = bacctable.rank(method='max',ascending=False, axis=1)
# # # sranked_acctable = sacctable.rank(method='max',ascending=False, axis=1)
# # # kranked_acctable = kacctable.rank(method='max',ascending=False, axis=1)
# # # rranked_acctable = racctable.rank(method='max',ascending=False, axis=1)
# # # ranktable.loc['Naive Bayes'] = branked_acctable.mean(axis=0)
# # # ranktable.loc['3NN'] = kranked_acctable.mean(axis=0)
# # # ranktable.loc['SVM'] = sranked_acctable.mean(axis=0)
# # # ranktable.loc['RandomForest'] = rranked_acctable.mean(axis=0)
# # # ranktable.to_excel( 'tables/'+T+'/ranktable.xlsx', index=True)
# # # print(ranktable)
# # rank = pd.read_excel( '../tables/1'+'/ranktable.xlsx',index_col=0)
# # avetable_rank = pd.DataFrame(rank)
# # datasets_num = 20
# # CD = Orange.evaluation.scoring.compute_CD(avetable_rank, datasets_num, alpha='0.05', test='nemenyi')
# # Orange.evaluation.scoring.graph_ranks(avetable_rank, avetable_rank.index, cd=CD,width=6, textspace=1, cdmethod=6,reverse=True)
# # plt.savefig("../picture/"  +"now.png", dpi=300, bbox_inches='tight')
# # plt.show()
# # pd.set_option('display.max_rows', None)
# # pd.set_option('display.max_columns', None)
# # def friedman_test(data):
# #     """
# #     执行 Friedman 检验并计算 F 统计量及其 p 值
# #
# #     参数:
# #     data (2D numpy array): 形状为 (N, M) 的二维数组，表示 N 个块，每个块有 M 个处理的排名。
# #
# #     返回:
# #     tuple: (χ²_F, F_statistic, p_value, df_num, df_den)
# #     """
# #     N, M = data.shape
# #
# #     # 计算每个处理的排名总和
# #     R_j = np.mean(data, axis=0)
# #
# #     # 计算 Friedman 统计量 χ²_F
# #     chi2_F = (12 * N / (M * (M + 1))) * (np.sum(R_j**2) - (M * (M + 1)**2) / 4)
# #
# #     # 计算 F 统计量
# #     F_statistic = (N - 1) * chi2_F / (N * (M - 1) - chi2_F)
# #
# #     # 计算 F 统计量的 p 值
# #     df_num = M - 1
# #     df_den = (M - 1) * (N - 1)
# #     p_value = stats.f.sf(F_statistic, df_num, df_den)
# #
# #     return chi2_F, F_statistic, p_value, df_num, df_den
# # # print(ranktable)
# # chi2_F, F_statistic, p_value, df_num, df_den = friedman_test(ranktable)
# #
# # print(f"Friedman Statistic (χ²_F): {chi2_F:.4f}")
# # print(f"F Statistic: {F_statistic:.4f}")
# # print(f"P-value: {p_value:.4f}")
# # print(f"Degrees of Freedom: Numerator = {df_num}, Denominator = {df_den}")
# #
# # # 检查 p 值是否小于显著性水平（例如 0.05）
# # alpha = 0.05
# # if p_value < alpha:
# #     print(f"The result is statistically significant at α = {alpha}.")
# # else:
# #     print(f"The result is not statistically significant at α = {alpha}.")
# import Orange
# import pandas as pd
# import matplotlib.pyplot as plt
# import scikit_posthocs as sp
# import numpy as np
#
# # 读取排名数据
# rank = pd.read_excel('./tables/ranktable.xlsx', index_col=0)
# avetable_rank = pd.DataFrame(rank)
# datasets_num = 20
# import Orange  # version: Orange3==3.32.0
# # import matplotlib
# # matplotlib.use('TkAgg')  # 不显示图则加上这两行
# import matplotlib.pyplot as plt
#
# names = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7']
# avranks = [5.9, 4.37, 4.25, 5.39, 2.19, 2.85, 3.04]
# datasets_num = 135
# CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', type='nemenyi')
# Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
# # CD = Orange.evaluation.scoring.compute_CD(avetable_rank, datasets_num, alpha='0.05', test='nemenyi')
# # Orange.evaluation.scoring.graph_ranks(avetable_rank, avetable_rank.index, cd=CD,width=6, textspace=1, cdmethod=6,reverse=True)
# plt.show()
# # 计算 Nemenyi 检验和临界差（CD）
# # 注意：scikit-posthocs 需要输入原始性能指标（而非排名），但如果你只有排名，也可以直接使用
# # 这里假设 avetable_rank 是每个算法在多个数据集上的排名（行是数据集，列是算法）
#
# # 如果你有原始性能值（如准确率），可以这样计算排名：
# # ranks = avetable_rank.rank(axis=1, ascending=False)  # 如果数值越大越好
# # ranks_mean = ranks.mean()
#
# # 如果你已经计算好了平均排名（avetable_rank 是平均排名向量）
# ranks_mean = avetable_rank.iloc[0]  # 假设第一行是平均排名
#
# # 计算 Nemenyi 检验的临界差（CD）
# # 首先需要 Friedman 检验的卡方统计量（但 scikit-posthocs 没有直接计算 CD 的函数，我们可以手动计算）
#
# # 手动计算 CD（根据 Nemenyi 检验公式）
# # k = len(ranks_mean)  # 算法数量
# # N = datasets_num  # 数据集数量
# # q_alpha = sp.posthoc_nemenyi_friedman.get_distribution(k, alpha=0.05)  # 获取临界值
# # CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
# #
# # print(f"Critical Difference (CD): {CD}")
# #
# # # 绘制 CD 图
# # plt.figure(figsize=(10, 6))
# # sp.sign_plot(ranks_mean, CD, reverse=True)  # 注意：scikit-posthocs 的 sign_plot 可能不直接支持 CD 图，但可以自己画
# #
# #
# # # 手动绘制 CD 图（类似 Orange 的 graph_ranks）
# # def plot_cd_diagram(ranks, cd, labels=None, reverse=False):
# #     if labels is None:
# #         labels = ranks.index
# #     if reverse:
# #         ranks = ranks.max() - ranks  # 反转排名（如果需要）
# #
# #     # 按排名排序
# #     sorted_idx = np.argsort(ranks.values)
# #     sorted_ranks = ranks.values[sorted_idx]
# #     sorted_labels = [labels[i] for i in sorted_idx]
# #
# #     # 绘制排名轴
# #     plt.figure(figsize=(10, 6))
# #     plt.hlines(y=sorted_labels, xmin=sorted_ranks - cd / 2, xmax=sorted_ranks + cd / 2, color='black', linewidth=2)
# #     plt.scatter(sorted_ranks, sorted_labels, color='red', s=100)
# #     plt.xlabel('Average Rank')
# #     plt.title(f'CD Diagram (CD = {cd:.3f})')
# #     plt.grid(axis='x', linestyle='--', alpha=0.7)
# #     plt.tight_layout()
# #
# #
# # plot_cd_diagram(ranks_mean, CD, labels=ranks_mean.index, reverse=True)
# # plt.savefig("../picture/now.png", dpi=300, bbox_inches='tight')
# # plt.show()
rank = pd.read_excel( 'ranktable.xlsx',index_col=0)
# print(rank)
avetable_rank = pd.DataFrame(rank)
# print(avetable_rank)
# mean_values = avetable_rank.drop(index="Average").mean()
datasets_num = 26
# print(avetable_rank.index)
CD = compute_CD(avetable_rank.mean(), datasets_num, alpha='0.05', test='nemenyi',)
print(CD)
graph_ranks(avetable_rank.mean(), avetable_rank.columns, cd=CD,width=4, textspace=1, cdmethod=7,reverse=True)
x_start = 8  # 线条起始位置（可以根据方法数量调整）
x_end = x_start + CD  # 长度为 CD
y = 2.05
# plt.hlines(y, x_start, x_end, colors='red', linewidth=3)
plt.text((x_start+x_end)/2, y+0.02, f'CD = {CD:.2f}', color='red',
         ha='center', va='bottom', fontweight='bold')
plt.savefig("./picture/"  +"1111CDpicture.png", dpi=800, bbox_inches='tight')
plt.show()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# import Orange  # version: Orange3==3.32.0
# # import matplotlib
# # matplotlib.use('TkAgg')  # 不显示图则加上这两行
# import matplotlib.pyplot as plt
# #
# names = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7']
# avranks = [5.9, 4.37, 4.25, 5.39, 2.19, 2.85, 3.04]
# datasets_num = 135
# CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05')
# Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
# plt.show()

