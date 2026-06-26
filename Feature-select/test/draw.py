# import pandas as pd
# import numpy as np
# import sns
# from matplotlib import pyplot as plt
#
# # 读取图片中提取的文本数据
# image_data1 = [
#     ["Datasets", "mRMR", "CIFE", "JMIM", "IWFS", "MRI", "CFR", "DCSF", "MRMD", "UCRFS", "CSMDCCMR"],
#     ["Dermatology", 82.17, 81.97, 83.29, 82.67, 83.13, 82.56, 81.72, 82.98, 84.44, 84.44],
#     ["FeatMIAS", 70.24, 63.79, 66.78, 66.79, 66.45, 66.28, 69.53, 70.77, 70.19, 69.17],
#     ["Movement_libras", 55.09, 47.62, 58.00, 51.93, 54.03, 56.97, 59.17, 60.15, 61.27, 54.03],
#     ["Musk1", 61.51, 52.81, 72.12, 65.11, 66.11, 66.03, 69.09, 69.17, 70.95, 63.89],
#     ["Synthetic_control", 86.10, 89.99, 82.24, 90.78, 89.79, 91.58, 91.37, 91.00, 93.94, 90.82],
#     ["Waveform", 77.74, 75.21, 77.33, 77.69, 77.72, 77.75, 77.78, 78.51, 78.78, 78.78],
#     ["Wdbc", 92.91, 90.07, 92.76, 93.35, 93.11, 93.08, 92.98, 93.10, 93.03, 93.40],
#     ["Arcene", 68.98, 66.10, 64.96, 66.53, 67.25, 66.88, 69.28, 71.86, 75.39, 71.66],
#     ["CLL_SUB_111", 62.38, 62.97, 65.90, 72.24, 71.12, 71.76, 71.12, 71.25, 71.66, 71.87],
#     ["GLIOMA", 66.16, 67.64, 65.40, 66.44, 66.03, 66.34, 66.04, 66.00, 65.85, 66.34],
#     ["Isolet", 53.23, 60.83, 50.20, 53.21, 64.25, 51.94, 51.94, 64.09, 71.63, 71.63],
#     ["ORL", 55.92, 61.43, 49.12, 53.84, 64.30, 58.44, 58.44, 62.94, 64.09, 65.01],
#     ["Orlraws10P", 75.62, 83.60, 74.60, 78.44, 82.48, 84.60, 84.20, 85.35, 80.91, 84.20],
#     ["Pixraw10P", 94.06, 89.98, 86.24, 93.24, 97.44, 93.24, 97.14, 97.36, 97.44, 97.14],
#     ["Prostate_GE", 90.38, 83.67, 87.28, 88.46, 86.79, 88.46, 88.46, 88.46, 88.46, 88.46],
#     ["TOX_171", 59.67, 61.11, 52.20, 57.26, 55.27, 51.94, 51.94, 53.27, 59.35, 59.35],
#     ["WarpAR10P", 61.15, 43.95, 41.68, 46.87, 57.25, 52.20, 52.20, 55.38, 64.78, 64.78],
#     ["WarpPIE10P", 69.21, 58.93, 66.30, 68.06, 69.66, 68.06, 68.06, 70.66, 64.78, 64.78],
#     ["Yale", 47.43, 40.47, 54.10, 53.22, 53.27, 53.27, 53.27, 53.27, 54.10, 53.27],
#     ["Authorship", 93.99, 94.42, 93.25, 95.18, 95.18, 95.18, 95.18, 95.18, 95.18, 95.18],
#     ["Factors", 85.53, 84.64, 84.60, 85.78, 84.60, 85.79, 84.60, 84.60, 84.60, 84.60],
#     ["Pixel", 76.93, 77.06, 74.60, 77.80, 77.80, 77.80, 77.80, 77.80, 77.80, 77.80],
#     ["Sorlie", 72.56, 67.89, 71.71, 71.71, 71.71, 71.71, 71.71, 71.71, 71.71, 71.71],
#     ["Su", 92.33, 91.72, 91.98, 90.27, 90.27, 90.27, 90.27, 90.27, 90.27, 90.27],
#     ["Yeoh", 88.96, 81.07, 87.95, 89.07, 89.07, 89.07, 89.07, 89.07, 89.07, 89.07],
#     # ["Average", 73.61, 70.67, 74.89, 70.32, 75.58, 74.89, 75.94, 74.89, 74.89, 74.89],
#     # ["W/T/L", "23/2/0", "24/1/0", "24/1/0", "24/1/0", "23/2/0", "19/4/2"]
# ]
#
# 80
# # Convert the image data to dataframes
# df1 = pd.DataFrame(image_data1[1:], columns=image_data1[0])
# df2 = pd.DataFrame(image_data2[1:], columns=image_data2[0])
# # print(df1,df2)
# df1['Features'] = df2['Features']
# # Sort df2 by the 'features' column
# df2_sorted = df2.sort_values(by="Features", ascending=True)  # Assuming 'features' is in the third column (index 2)
# # Sort df1 based on the order from df2_sorted by 'features' column
# # Map the order from df2_sorted to df1
# # df1_sorted = df1.set_index(0).reindex(df2_sorted[1]).reset_index(drop=True)
# # 将df1中数据按df2_sorted中的dataset顺序排列
# df1_sorted = df1.set_index("Datasets").reindex(df2_sorted.iloc[:, 1]).reset_index(drop=False)
# # 将df2中的features列加入到df1中
#
#
# # 提取准确率数据，确保转换为数值型
# df1_accuracy = df1_sorted.apply(lambda x: pd.Series(pd.to_numeric(x.drop('Features').values, errors='coerce')), axis=1)
#
# # 统计不同准确率水平的方法个数，排除 NaN 数据
# accuracy_bins = pd.cut(df1_accuracy.stack().dropna(), bins=5, labels=[f'{i*20}-{(i+1)*20}' for i in range(5)])
#
# # 分组：小规模、中规模、大规模数据集
# def classify_scale(features):
#     if features <= 100:
#         return 'Small'
#     elif 100 < features <= 1000:
#         return 'Medium'
#     else:
#         return 'Large'
#
# df2_sorted['Scale'] = df2_sorted['Features'].apply(classify_scale)
#
# # 创建一个新的 DataFrame 来存储统计结果
# accuracy_counts_df = df2_sorted.groupby(['Scale', accuracy_bins]).size().unstack(fill_value=0)
#
# # 绘制表格
# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# # 绘制小规模
# sns.heatmap(accuracy_counts_df['Small'], annot=True, cmap="YlGnBu", fmt="d", ax=axes[0])
# axes[0].set_title('Small Scale Accuracy Distribution')
# axes[0].set_xlabel('Accuracy Range')
# axes[0].set_ylabel('Methods Count')
#
# # 绘制中规模
# sns.heatmap(accuracy_counts_df['Medium'], annot=True, cmap="YlGnBu", fmt="d", ax=axes[1])
# axes[1].set_title('Medium Scale Accuracy Distribution')
#
# # 绘制大规模
# sns.heatmap(accuracy_counts_df['Large'], annot=True, cmap="YlGnBu", fmt="d", ax=axes[2])
# axes[2].set_title('Large Scale Accuracy Distribution')
#
# plt.tight_layout()
# plt.show()


import pandas as pd

# 数据集规模和数据定义
image_data2 = [
    ["ID", "Datasets", "Samples", "Features", "Classes", "Sources"],
    [1, "Dermatology", 366, 34, 6, "UCI"],
    [2, "FeatMIAS", 322, 280, 4, "UCI"],
    [3, "Movement_libras", 360, 90, 15, "UCI"],
    [4, "Musk1", 476, 166, 2, "UCI"],
    [5, "Synthetic_control", 600, 60, 6, "UCI"],
    [6, "Waveform", 5000, 40, 3, "UCI"],
    [7, "Wdbc", 569, 30, 2, "UCI"],
    [8, "Arcene", 200, 10000, 2, "ASU"],
    [9, "CLL_SUB_111", 111, 11340, 3, "ASU"],
    [10, "GLIOMA", 50, 4434, 4, "ASU"],
    [11, "Isolet", 1560, 617, 26, "ASU"],
    [12, "ORL", 400, 1024, 40, "ASU"],
    [13, "Orlraws10P", 100, 10304, 10, "ASU"],
    [14, "Pixraw10P", 100, 10000, 10, "ASU"],
    [15, "Prostate_GE", 102, 5966, 2, "ASU"],
    [16, "TOX_171", 171, 5748, 4, "ASU"],
    [17, "WarpAR10P", 130, 2400, 10, "ASU"],
    [18, "WarpPIE10P", 210, 2420, 10, "ASU"],
    [19, "Yale", 165, 1024, 15, "ASU"],
    [20, "Authorship", 841, 70, 4, "PMLB"],
    [21, "Factors", 2000, 216, 10, "PMLB"],
    [22, "Pixel", 2000, 240, 10, "PMLB"],
    [23, "Sorlie", 84, 456, 5, "DMA"],
    [24, "Su", 101, 5565, 4, "DMA"],
    [25, "Yeoh", 247, 12625, 6, "DMA"]
]

# 定义每个数据集的假设准确度范围与对应方法数量
dataset_accuracy = {
    "Dermatology": [0, 0, 10, 0, 0],
    "FeatMIAS": [0, 0, 6, 1, 0],
    "Movement_libras": [0, 0, 6, 0, 0],
    "Musk1": [0, 0, 5, 0, 0],
    "Synthetic_control": [0, 4, 6, 1, 0],
    "Waveform": [0, 0, 11, 3, 0],
    "Arcene": [1, 7, 7, 2, 0],
    "CLL_SUB_111": [2, 6, 3, 0, 0],
    "GLIOMA": [2, 3, 2, 0, 0],
    "Isolet": [0, 3, 2, 0, 0],
    "ORL": [0, 0, 3, 1, 0],
    "Orlraws10P": [0, 0, 0, 0, 6],
    "Pixraw10P": [0, 0, 0, 0, 10],
    "Prostate_GE": [0, 0, 1, 0, 3],
    "TOX_171": [0, 0, 3, 0, 3],
    "WarpAR10P": [0, 0, 4, 1, 0],
    "WarpPIE10P": [0, 0, 3, 0, 3],
    "Yale": [0, 0, 0, 0, 1],
    "Authorship": [0, 0, 3, 7, 0],
    "Factors": [0, 0, 4, 10, 0],
    "Pixel": [0, 4, 6, 5, 0],
    "Sorlie": [0, 0, 0, 0, 0],
    "Su": [0, 0, 0, 0, 0],
    "Yeoh": [0, 0, 0, 0, 6]
}

# 将 image_data2 转换为 DataFrame
data_info = pd.DataFrame(image_data2[1:], columns=image_data2[0])
data_info.set_index("Datasets", inplace=True)

# 定义规模大小分类
def classify_scale(features):
    if features <= 100:
        return "Small"
    elif features <= 1000:
        return "Medium"
    else:
        return "Large"

data_info["Scale"] = data_info["Features"].apply(classify_scale)

# 合并准确度信息和数据规模
accuracy_df = pd.DataFrame.from_dict(dataset_accuracy, orient="index",
                                     columns=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
result_df = data_info.join(accuracy_df)

# 按规模统计每个范围内的方法数量统计
scale_summary = result_df.groupby("Scale")["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"].sum()

print("Per Dataset:")
print(result_df)
print("\nSummary by Scale:")
print(scale_summary)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 假设 result_df 和 scale_summary 已经从前面代码生成
# 以下是一些可视化的示例代码


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 定义数据
data = {
    "Scale": ["Large", "Medium", "Small"],
    "50-60%": [5.0, 0.0, 0.0],
    "60-70%": [16.0, 7.0, 4.0],
    "70-80%": [26.0, 23.0, 36.0],
    "80-90%": [4.0, 16.0, 11.0],
    "90-100%": [32.0, 0.0, 0.0]
}

# 转换为DataFrame
summary_df = pd.DataFrame(data)
summary_df.set_index("Scale", inplace=True)
colors = ["#D84d3a", "#d68660", "#FFD687", "#8bb69b", "#327726"]
# 绘制堆叠柱状图
fig, ax = plt.subplots(figsize=(10, 6))

categories = summary_df.columns
x = np.arange(len(summary_df.index))

bottom = np.zeros(len(summary_df.index))

for category, color in zip(categories, colors):
    ax.bar(x, summary_df[category], label=category, bottom=bottom, color=color)
    bottom += summary_df[category]

ax.set_xlabel("Scale", fontsize=12)
ax.set_ylabel("Method Count", fontsize=12)
ax.set_title("Accuracy Distribution by Scale", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(summary_df.index, fontsize=10)
ax.legend(title="Accuracy Range")

plt.tight_layout()
plt.show()
