#=======================================画图==========================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
#数据读取'dexter','Isolet',
datasets = [ 'Arcene','Authorship','CLL_SUB_111','Dermatology', 'Factors','Movement_libras','Musk1','Orlraws10P','Pixel', 'Prostate_GE', 'Su','SRBCT','splice','Synthetic_control', 'TOX_171', 'Waveform','Wdbc','WarpPIE10P','WarpAR10P', 'Yeoh', 'Yale', ]## datanames = ['Movement_libras','Musk1','Sorlie','Yale','WarpAR10P','GLIOMA','Arcene',]#'Dermatology',

tables = ['b','k','s','r']
names = ['Bayes','KNN','SVM','Random Forest']
index = 0
for table in tables:
    data = pd.read_excel('../tables/' +  '1esmbsp-'+table+'f1table.xlsx')
    df = pd.DataFrame(data)
    #
    # # 2️⃣ 设置折线图的颜色和标记
    markers = ['o','s','^','v','x','h','o','^','s']
    colors = ['gold','slateblue','lightseagreen','darkorange','deepskyblue','blue','darkgoldenrod','yellow','red']

    # 3️⃣ 绘图
    plt.figure(figsize=(15,6))
    for i, col in enumerate(df.columns[1:]):  # 跳过 Dataset 列
        plt.plot(datasets, df[col], marker=markers[i], color=colors[i], label=col, linewidth=2)

    # 4️⃣ 图表美化
    plt.title("Algorithm Performance Across Datasets ("+names[index]+')', fontsize=14)

    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45, ha='right')  # 旋转60度，右对齐
    plt.ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../picture/" + names[index]+ "_accuracy_line.png", dpi=300, bbox_inches='tight')
    # 5️⃣ 显示图表
    index += 1
    plt.show()

    # # melt 数据方便绘图
    # df_melt = df.melt(id_vars='Dataset', var_name='Method', value_name='Score')
    #
    # # 指定每个方法对应的颜色
    # methods = ['EAMB', 'IAMB', 'Inter-IAMB', 'LRH', 'Fast-IAMB', 'GSMB', 'FBED', 'EAMB-SP']
    # colors_list = ['gold', 'slateblue', 'lightseagreen', 'darkorange', 'deepskyblue', 'blue', 'darkgoldenrod', 'red']
    # colors = dict(zip(methods, colors_list))

    # plt.figure(figsize=(20, 6))
    # for method in methods:
    #     subset = df_melt[df_melt['Method'] == method]
    #     plt.plot(subset['Dataset'], subset['Score'], marker='o', label=method, color=colors[method])
    #
    # plt.xlabel('Dataset')
    # plt.xticks(rotation=60, ha='right')  # 旋转60度，右对齐
    # plt.ylabel('Score')
    # plt.title('Parallel Coordinates Style Plot with Specified Colors')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 2️⃣ 平行坐标绘制
    # plt.figure(figsize=(12, 6))
    # parallel_coordinates(df, 'Dataset', colormap=plt.get_cmap("tab10"), marker='o')
    # plt.xticks(rotation=45)
    # plt.ylabel("Performance")
    # plt.title("Parallel Coordinates Plot of Methods across Datasets")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()

# # dataname = 'Musk1'
# for dataname in datasets:
#     methods = ['EAMB','IAMB','Inter-IAMB','LRH','Fast-IAMB','GSMB','FBED','EAMBSP',]#
#     markers = ['o','s','^','v','x','h','o','^','v','.']
#     colors =['gold','slateblue','lightseagreen','darkorange','deepskyblue','blue','darkgoldenrod','red',]
#     x_datas =[]
#     y_datas =[]
#     for method in methods:
#         y_values = np.loadtxt("accuracy/1/" + dataname + '_' + method + "_accuracy.txt")
#         y_datas.append(y_values)
#     x_values = np.arange(1, len(y_values)+1)
#     # 创建折线图
#     plt.figure(figsize=(8, 4))
#     # plt.ylim(0,1 )#将y值固定再0-1
#     for i in range(0,8):
#         # print(i)
#         # plt.scatter(x_values, y_datas[i],marker=markers[i],color=colors[i],s = 15)
#         # if i in [7,8,9]:
#         #     plt.plot(x_values, y_datas[i],color=colors[i],marker=markers[i],markerfacecolor='none', markersize=4)#黑色
#         # else:
#         plt.plot(x_values, y_datas[i], color=colors[i], marker=markers[i], markersize=4)  # 黑色
#
#     # 添加标题和标签
#     plt.title(dataname)
#     plt.xlabel('Number of selected features')
#     plt.ylabel('Classification accuracy')
#     # 显示图形
#     plt.savefig("accuracy/" + dataname + "_accuracy.png", dpi=300, bbox_inches='tight')
#     plt.show()
