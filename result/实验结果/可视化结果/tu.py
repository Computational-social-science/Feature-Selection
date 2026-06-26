# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 1. 数据预处理
# df = pd.read_excel('classifier_average_table.xlsx', index_col=0)
# def parse_mean(val):
#     return float(val.split('±')[0]) if isinstance(val, str) and '±' in val else np.nan
# df_clean = df.applymap(parse_mean).drop('Average', errors='ignore')
#
# # 2. 绘图准备
# plt.figure(figsize=(12, 7))
# # 将算法按平均表现排序，使 CSMB 处于末尾压轴位置
# order = df_clean.mean().sort_values().index.tolist()
# df_melted = df_clean.reset_index().melt(id_vars='index', var_name='Algorithm', value_name='Accuracy')
#
# plt.figure(figsize=(12, 7))
#
# # 建议直接使用 Seaborn 内置色板，它会自动循环分配颜色给所有 Algorithm
# # 'Set2' 或 'muted' 是顶会论文中常用的柔和且对比清晰的配色
# sns.violinplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#                inner=None, color="skyblue", density_norm='width')
#
# # 保持箱线图和散点图的层次
# sns.boxplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#             width=0.15, boxprops={'zorder': 2}, whiskerprops={'zorder': 2}, showfliers=False)
# sns.stripplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#               size=4, color=".3", alpha=0.4, jitter=True, zorder=1)
#
# plt.title('Performance Distribution across 26 Datasets', fontsize=15, pad=20)
# plt.ylabel('Accuracy (%)', fontsize=12)
# plt.xlabel('Algorithm', fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
#
# plt.tight_layout()
# plt.savefig('violin_refined.png', dpi=800)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据预处理（与之前一致）
df = pd.read_excel('classifier_average_table.xlsx', index_col=0)
def parse_mean(val):
    return float(val.split('±')[0]) if isinstance(val, str) and '±' in val else np.nan
df_clean = df.applymap(parse_mean).drop('Average', errors='ignore')
order = df_clean.mean().sort_values().index.tolist()
df_melted = df_clean.reset_index().melt(id_vars='index', var_name='Algorithm', value_name='Accuracy')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置学术绘图风格
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(10, 6))

# 定义统一的学术色调（经典钢蓝色）
academic_color = "#8EBAE3"

# 1. 绘制小提琴图 - 统一色调，线条细致
sns.violinplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
               inner=None, color=academic_color, linewidth=1.2, alpha=0.7)

# 2. 嵌套箱线图 - 使用白色背景以增强对比，中位数设为红色
sns.boxplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
            width=0.12, color="white",
            boxprops={'zorder': 3, 'edgecolor': '#333333'},
            medianprops={'color': '#d62728', 'linewidth': 1.5, 'zorder': 4}, # 红色中位数
            showfliers=False)

# 3. 抖动散点 - 使用深灰色以体现数据的原始透明度
sns.stripplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
              size=3, color="#444444", alpha=0.3, jitter=True, zorder=2)

# 细节微调
plt.title('Performance Distribution across Benchmark Datasets', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Classification Accuracy (%)', fontsize=11)
plt.xlabel('Algorithm (Ordered by Mean Performance)', fontsize=16)
plt.xticks(rotation=30, ha='right')

# 移除冗余边框
sns.despine()

plt.tight_layout()
plt.savefig('violin_refined.svg',  format='svg', bbox_inches='tight',dpi=800)
# # 3. 绘制多层统计图
# # 绘制底层的分布密度 (小提琴图)
# sns.violinplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#                inner=None, color=".9", density_norm='width')
# # 绘制中间层的统计区间 (箱线图)
# sns.boxplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#             width=0.15, showfliers=False, zorder=2)
# # 绘制顶层的原始数据点 (抖动散点)
# sns.stripplot(data=df_melted, x='Algorithm', y='Accuracy', order=order,
#               size=4, color="orange", alpha=0.4, jitter=True, zorder=1)
#
# # 高亮 CSMB 颜色
# # csmb_idx = order.index('CSMB')
# # plt.gca().get_children()[csmb_idx].set_facecolor('#d62728')
#
# plt.title('Performance Distribution: CSMB vs Baselines', fontsize=15)
# plt.ylabel('Accuracy (%)', fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("violin_plot.png", dpi=800)
# # plt.savefig('violin_plot.pdf')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 数据归一化：每行数值除以该行最大值
df_norm = df_clean.div(df_clean.max(axis=1), axis=0)

# 2. 绘制热力图
plt.figure(figsize=(12, 10))
# 使用 YlGnBu 色板，数值越接近 1.0 颜色越深
sns.heatmap(df_norm, cmap='YlGnBu', annot=df_clean.round(1), fmt='.1f',
            annot_kws={"size": 8}, cbar_kws={'label': 'Relative Performance (1.0 = Best)'})

plt.title('Relative Performance Heatmap (Normalized by Dataset Max)', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Benchmark Datasets')
plt.xlabel('Feature Selection Algorithms')
plt.tight_layout()
plt.savefig("norm_heatmap.svg", format='svg', bbox_inches='tight',dpi=800)