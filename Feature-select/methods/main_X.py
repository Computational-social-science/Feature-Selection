import math
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

#计算互信息
# def mutualInfo(X,Y):
#     if(len(X)!=len(Y)):
#         raise ValueError;
#     # 使用字典统计每一个x元素出现的次数
#     d_x = dict()  # x的字典
#     for x in X:
#         if x in d_x:
#             d_x[x] += 1
#         else:
#             d_x[x] = 1
#     # 计算每个元素出现的概率
#     p_x = dict()#字典表示x的出现概率
#     for x in d_x.keys():
#         p_x[x] = d_x[x] / X.size
#
#     # 使用字典统计每一个y元素出现的次数
#     d_y = dict()  # y的字典
#     for y in Y:
#         if y in d_y:
#             d_y[y] += 1
#         else:
#             d_y[y] = 1
#     # 计算每个元素出现的概率
#     p_y = dict()
#     for y in d_y.keys():
#         p_y[y] = d_y[y] / Y.size
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_xy = dict()  # x的字典
#     for i in range(X.size):
#         if (X[i], Y[i]) in d_xy:
#             d_xy[X[i], Y[i]] += 1
#         else:
#             d_xy[X[i], Y[i]] = 1
#     # 计算每个元素出现的概率
#     p_xy = dict()
#     for xy in d_xy.keys():
#         p_xy[xy] = d_xy[xy] / X.size
#     # print(d_x, d_y, d_xy)
#     # print(p_x, p_y, p_xy)
#
#     # 初始化互信息值为0
#     mi = 0
#     for xy in p_xy.keys():#对每一对xy
#         mi += p_xy[xy] * np.log(p_xy[xy] / (p_x[xy[0]] * p_y[xy[1]]))/np.log(2)
#         # print(p_xy[xy],p_x[xy[0]], p_y[xy[1]])
#     # print(mi)
#
#     return mi
#计算特定类互信息
# def mutualInfoys(X,Y,ys):#ys指定特定类cj
#     if(len(X)!=len(Y)):
#         raise ValueError;
#     # 使用字典统计每一个x元素出现的次数
#     d_x = dict()  # x的字典
#     for x in X:
#         if x in d_x:
#             d_x[x] += 1
#         else:
#             d_x[x] = 1
#     # 计算每个元素出现的概率
#     p_x = dict()#字典表示x的出现概率
#     for x in d_x.keys():
#         p_x[x] = d_x[x] / X.size
#
#     # 使用字典统计每一个y元素出现的次数
#     d_y = dict()  # y的字典
#     for y in Y:
#         if y in d_y:
#             d_y[y] += 1
#         else:
#             d_y[y] = 1
#     # 计算每个元素出现的概率
#     p_y = dict()
#     for y in d_y.keys():
#         p_y[y] = d_y[y] / Y.size
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_xy = dict()  # x的字典
#     for i in range(X.size):
#         if (X[i], Y[i]) in d_xy:
#             d_xy[X[i], Y[i]] += 1
#         else:
#             d_xy[X[i], Y[i]] = 1
#     # 计算每个元素出现的概率
#     p_xy = dict()
#     for xy in d_xy.keys():
#         p_xy[xy] = d_xy[xy] / X.size
#     # print(d_x, d_y, d_xy)
#     # print(p_x, p_y, p_xy)
#
#     # 初始化互信息值为0
#     mi = 0
#     for xy in p_xy.keys():#对每一对xy
#         if(xy[1]==ys):
#             mi += p_xy[xy] * np.log(p_xy[xy] / (p_x[xy[0]] * p_y[xy[1]]))/np.log(2)
#     # print(mi)
#
#     return mi
#计算条件互信息
# def ConMutualInfo(X,Y,Z,sy):#X为特征列，Y为类列，Z为特征列，sy指定特定类
#     data = pd.DataFrame()
#     data.insert(data.shape[1], 'x', X)
#     data.insert(data.shape[1], 'y', Y)
#     data.insert(data.shape[1], 'z', Z)
#
#     # 重新设置输入数据data的列名为'x','y','z'
#     # 获取不重复的(x,y,z)
#     data_xyz = data.drop_duplicates(subset=None, keep='first', inplace=False)
#     # 初始化条件互信息mi
#     con = 0
#     red = 0
#     # 遍历每一个不同的(x,y,z)
#     for i in range(data_xyz.shape[0]):
#         x = data_xyz.iloc[i]['x']
#         y = data_xyz.iloc[i]['y']
#         z = data_xyz.iloc[i]['z']
#         # 统计(x,y,z)出现的次数count_xyz
#         # count_xyz = (data[data['x']==2].index & data[data['y']==1].index).shape[0]
#         count_xyz = data.query('x == @x & y == @y & z == @z').shape[0]
#         # 计算(x,y,z)出现的概率p_xyz
#         p_xyz = count_xyz / data.shape[0]
#
#         # 统计z出现的次数count_z
#         data_z = data[data['z'] == z]
#         count_z = data_z.shape[0]
#         p_z = count_z / data.shape[0]
#
#         # 计算条件概率：在当前的z下(x,y)出现的概率p_xy|z
#         p_xy_z = count_xyz / count_z
#
#         # 计算条件概率：在当前的z下x出现的概率p_x|z
#         count_xz = data.query("x == @x & z== @z").shape[0]
#         p_x_z = count_xz / count_z
#         p_xz = count_xz / data.shape[0]
#
#         # 计算条件概率：在当前的z下y出现的概率p_y|z
#         count_yz = data.query("y==@y & z==@z").shape[0]
#         p_y_z = count_yz / count_z
#
#         # 统计z出现的次数count_z
#         data_x = data[data['x'] == x]
#         count_x = data_x.shape[0]
#         p_x = count_x / data.shape[0]
#
#         # 计算条件概率：在当前的z下(x,y)出现的概率p_xy|z
#         p_zy_x = count_xyz / count_x
#
#         # 计算条件概率：在当前的z下x出现的概率p_x|z
#         count_zx = data.query("x == @x & z== @z").shape[0]
#         p_z_x = count_zx / count_x
#
#         # 计算条件概率：在当前的z下y出现的概率p_y|z
#         count_yx = data.query("y==@y & x==@x").shape[0]
#         p_y_x = count_yx / count_x
#         if(y == sy):
#             con += p_xyz * np.log(p_xy_z / (p_x_z * p_y_z))
#             con += p_xyz * np.log(p_zy_x / (p_z_x * p_y_x))
#         if (y == sy):
#             red += p_xyz * np.log(p_xz / (p_x * p_z))
#             # print(p_xyz,p_x_z,p_x , p_z)
#     # print(mi)
#     return (con,red)
#计算冗余
# def ConAndRed(X,Y,Z,ys):#X为特征列，Y为类列，Z为特征列，sy指定特定类
#     if(len(X)!=len(Y)):
#         raise ValueError;
#     lens = X.size
#     # 使用字典统计每一个x元素出现的次数
#     d_x = dict()  # x的字典
#     for x in X:
#         if x in d_x:
#             d_x[x] += 1
#         else:
#             d_x[x] = 1
#     # 计算每个元素出现的概率
#     p_x = dict()#字典表示x的出现概率
#     for x in d_x.keys():
#         p_x[x] = d_x[x] / lens
#
#     # 使用字典统计每一个y元素出现的次数
#     d_z = dict()  # y的字典
#     for z in Z:
#         if z in d_z:
#             d_z[z] += 1
#         else:
#             d_z[z] = 1
#     # 计算每个元素出现的概率
#     p_z = dict()
#     for z in d_z.keys():
#         p_z[z] = d_z[z] / lens
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_xz = dict()  # x的字典
#     for i in range(lens):
#         if (X[i],Z[i]) in d_xz:
#             d_xz[X[i], Z[i]] += 1
#         else:
#             d_xz[X[i], Z[i]] = 1
#     # 计算每个元素出现的概率
#     p_xz = dict()
#     for xz in d_xz.keys():
#         p_xz[xz] = d_xz[xz] / lens
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_xy = dict()  # x的字典
#     for i in range(lens):
#         if (X[i],Y[i]) in d_xy:
#             d_xy[X[i], Y[i]] += 1
#         else:
#             d_xy[X[i], Y[i]] = 1
#     # 计算每个元素出现的概率
#     p_xy = dict()
#     for xy in d_xy.keys():
#         p_xy[xy] = d_xy[xy] / lens
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_yz = dict()  # x的字典
#     for i in range(lens):
#         if (Y[i],Z[i]) in d_yz:
#             d_yz[Y[i], Z[i]] += 1
#         else:
#             d_yz[Y[i], Z[i]] = 1
#     # 计算每个元素出现的概率
#     p_yz = dict()
#     for yz in d_yz.keys():
#         p_yz[yz] = d_yz[yz] / lens
#
#     # 使用字典统计每一个(x,y)元素出现的次数，xy同时出现
#     d_xyz = dict()  # x的字典
#     for i in range(lens):
#         if (X[i], Y[i],Z[i]) in d_xyz:
#             d_xyz[X[i], Y[i], Z[i]] += 1
#         else:
#             d_xyz[X[i], Y[i], Z[i]] = 1
#     # 计算每个元素出现的概率
#     p_xyz = dict()
#     for xyz in d_xyz.keys():
#         p_xyz[xyz] = d_xyz[xyz] / lens
#     # print(d_x, d_y, d_xy)
#     # print(p_x, p_y, p_xy)
#
#     # 初始化互信息值为0
#     con = 0
#     red = 0
#     for xyz in p_xyz.keys():#对每一对xy
#         if(xyz[1]==ys):
#             xy = (xyz[0], xyz[1])
#             xz = (xyz[0], xyz[2])
#             yz = (xyz[1], xyz[2])
#             red += p_xyz[xyz] * np.log(p_xz[xz] / (p_x[xyz[0]] * p_z[xyz[2]]))/np.log(2)
#             # print(p_xyz[xyz] ,p_xz[xz] , p_x[xyz[0]], p_z[xyz[2]])
#             con += p_xyz[xyz] * np.log(p_z[xyz[2]] * p_xyz[xyz] / (p_xz[xz] * p_yz[yz]))/np.log(2)
#             con += p_xyz[xyz] * np.log(p_x[xyz[0]] * p_xyz[xyz] / (p_xz[xz] * p_xy[xy]))/np.log(2)
#     # print(mi)
#     return con,red
# -------------------------------计算第一部分REL-----------------------------------------------
#第一部分时互信息，使用sklearn中的函数计算# metrics.mutual_info_score()
# result = pd.DataFrame()
# allresult = pd.DataFrame()
# # 对于每个类别，构建相应的特征子集
# # f1_c1 = data[0:3][C == 'c1']
# # f1_c2 = data[3:6][C == 'c2']
# # f1_c3 = data[6:10][C == 'c3']
# for f in data.iloc[:,:8]:#对于每一个特征f：这个f是特征变量名
#     # 将f按照分类标签C进行归类————待优化
#     f_c1 = data[f][0:3]
#     f_c2 = data[f][3:6]
#     f_c3 = data[f][6:10]
#     # 计算每个类别的互信息量
#     mi_f    = mutual_info_score(data[f],C)#标签和样本的数量不对应，无法进行特定类的计算
#     mi_f_c1 = mutual_info_score(f_c1,C[0:3])
#     mi_f_c2 = mutual_info_score(f_c2,C[3:6])
#     mi_f_c3 = mutual_info_score(f_c3,C[6:10])
#     # 结果展示
#     # print("互信息量 mi("+f+", c1):", mi_f_c1)
#     # print("互信息量 mi("+f+", c2):", mi_f_c2)
#     # print("互信息量 mi("+f+", c3):", mi_f_c3)
#     # 结果保存
#     result.insert(result.shape[1],f,(mi_f,mi_f_c1,mi_f_c2,mi_f_c3))
# # print(result)

# 互信息计算公式 I(X;Y) = sigma(p_xy * ln(p_xy/(p_x * p_y)))
# 输入为一个dataframe，有两列数据，计算并返回的是这两列之间的互信息值
#需要优化

# # 自编程完成求取
# myresult = pd.DataFrame()
# for f in data.iloc[:,:8]:#对于每一个特征f：这个f是特征变量名
#     mymi = mutualInfo(data[f],C)
#     mymi_1 = mutualInfoys(data[f],C,'not_recom')
#     mymi_2 = mutualInfoys(data[f],C,'priority')
#     mymi_3 = mutualInfoys(data[f],C,'very_recom')
#     # 结果保存
#     myresult.insert(myresult.shape[1],f,(mymi,mymi_1,mymi_2,mymi_3))
# print(myresult)

# -----------------------计算第二部分CCDCC--------------------------------------------------------------------------------------------------------
#计算条件互信息
# conresult = pd.DataFrame()
# # print(data.iloc[:,:8])
# # print(pd.DataFrame(data.iloc[:,2:3]) == data.iloc[:,2:3])
# for f in data.iloc[:,:8]:#对于每一个特征f：这个f是特征变量名
#     # mymi = mutualInfo(data[f],C)
#     #第三个参数是fs
#     mycmi_1 = ConMutualInfo(data[f],C,data.iloc[:,2],'not_recom')
#     mycmi_2 = ConMutualInfo(data[f],C,data.iloc[:,2],'priority')
#     mycmi_3 = ConMutualInfo(data[f],C,data.iloc[:,2],'very_recom')
#     # 结果保存
#     conresult.insert(conresult.shape[1],f,(mycmi_1,mycmi_2,mycmi_3))
# print(conresult)
# ---------------------------------计算第三部分RED--------------------------------------------------------------------------
# redresult = pd.DataFrame()
# for f in data.iloc[:,:8]:#对于每一个特征f：这个f是特征变量名
#     # mymi = mutualInfo(data[f],C)
#     # 第三个参数是fs
#     mycmi_1 = RedmutualInfo(data[f],C,data.iloc[:,2],'not_recom')
#     mycmi_2 = RedmutualInfo(data[f],C,data.iloc[:,2],'priority')
#     mycmi_3 = RedmutualInfo(data[f],C,data.iloc[:,2],'very_recom')
#     # 结果保存
#     redresult.insert(redresult.shape[1],f,(mycmi_1,mycmi_2,mycmi_3))
# print(redresult)
# 打印选择后的特征
# print(X_selected)
#==========================================================================================================
# C = data.iloc[:,-1] #C是类集合：type：series
# Cname = set(C)#提取C中所有可能的取值
import time

from sklearn.preprocessing import LabelEncoder

from eamb import EAMB

# def Jfk(Fk,C,cname,Fs):#Fk是一列指定特征，j为类的序号：第j个类（类名），Fs是所有候选特征
#     #计算互信息
#     J_fk = mutualInfoys(Fk,C,cname)
#     #计算条件互信息和计算冗余度
#     aConMuInfo = 0
#     aRed = 0
#     # print(Fk)
#     for f in Fs:
#         # aConMuInfo += ConMutualInfo(Fk,C,Fs[f],cname)+ConMutualInfo(Fs[f],C,Fk,cname)
#         # aConMuInfo = ConMutualInfo(Fk,C,Fs[f],cname)
#         # aRed += ConMutualInfo(Fk,C,Fs[f],cname)[1]
#         # result1=ConMutualInfo(Fk, C, Fs[f], cname)
#         # aConMuInfo1 = result1[0]
#         # aRed1 = result1[1]
#         result= ConAndRed(Fk, C, Fs[f], cname)
#         aConMuInfo =result[0]
#         aRed = result[1]
#     aConMuInfo = aConMuInfo/Fs.shape[0]#shape[0]求行数
#     aRed = aRed/Fs.shape[0]
#     J_fk += aConMuInfo
#     J_fk -= aRed
#     # print(J_fk)
#     return J_fk

# 准备数据
#读取数据时，按照f和u进行表格行列分类
data = pd.read_csv('../aaaaa.csv',encoding='utf_8_sig',index_col="𝑈")
# print(data)
C = data.iloc[:,-1] #类标签C type：series
# print(C)
Class = set(C)
# print(data)
label_encoder = LabelEncoder()
data_e= data.iloc[:,:].apply(lambda col: label_encoder.fit_transform(col))  # 对每列独
# data_e = label_encoder.fit_transform(data)
# Class = set(y_encoded)
X = pd.DataFrame(data_e)
# 数据离散化
# X = pd.DataFrame()
# print(X0,y)
#离散化
# for col in X0.columns:
#     X[col] = pd.cut(X0[col], bins=5, labels=False)
# print(X)
# new_columns = list(range(X.shape[1] + 1))
new_columns = [str(i) for i in range(X.shape[1] + 1)]
X = X.rename(columns=dict(zip(X.columns, new_columns)))
# y = y.rename(columns=dict(zip(y.columns, [int(X.shape[1])])))
# data_processed = pd.concat([X, y], axis=1)
# print(X)
# print(X.shape[1]-1)
result_dict = {}
result = EAMB(str(X.shape[1]-1), X, 0.15)
dictresult = dict()
result_dict = set(result)
print(result_dict)
result_list = [int(x) for x in result_dict]
result_array = np.array(result_list)
print(result_array)
import time
# for fk in data:
    # print(y,ck,'-------------------------\n')
    # jfk = Jfk(data[fk], C, C[0], data)  # Jfk函数
    # print(fk)
    # mi1 = mutual_info_score(data[fk],C)
    # mi = ConAndRed(data[fk],C)
    # mi1 =ConAndRed(data[fk],C,C[0])
    # mi2 =ConAndRed(data[fk],C,C[4])
    # mi3 =ConAndRed(data[fk],C,data['𝑓3'],C[9])
    # print(mi3[0])
    # print(jfk)
    # start_time1 = time.time()
    # mutualInfo(data[fk],C)
    # end_time1 = time.time()  # 记录结束时间
    # execution_time1 = end_time1 - start_time1  # 计算执行时间
