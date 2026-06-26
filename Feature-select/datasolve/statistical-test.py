import numpy as np
from scipy import stats
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
def friedman_test(data):
    """
    执行 Friedman 检验并计算 F 统计量及其 p 值

    参数:
    data (2D numpy array): 形状为 (N, M) 的二维数组，表示 N 个块，每个块有 M 个处理的排名。

    返回:
    tuple: (χ²_F, F_statistic, p_value, df_num, df_den)
    """
    N, M = data.shape

    # 计算每个处理的排名总和
    R_j = np.sum(data, axis=0)

    # 计算 Friedman 统计量 χ²_F
    chi2_F = (12 * N / (M * (M + 1))) * (np.sum(R_j**2) - (M * (M + 1)**2) / 4)

    # 计算 F 统计量
    F_statistic = (N - 1) * chi2_F / (N * (M - 1) - chi2_F)

    # 计算 F 统计量的 p 值
    df_num = M - 1
    df_den = (M - 1) * (N - 1)
    p_value = stats.f.sf(F_statistic, df_num, df_den)

    return chi2_F, F_statistic, p_value, df_num, df_den

# 使用示例数据
T = '1'
rank = pd.read_excel( '../tables/'+T+'/ranktable.xlsx',index_col=0)
print(rank)

chi2_F, F_statistic, p_value, df_num, df_den = friedman_test(rank)

print(f"Friedman Statistic (χ²_F): {chi2_F:.4f}")
print(f"F Statistic: {F_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of Freedom: Numerator = {df_num}, Denominator = {df_den}")

# 检查 p 值是否小于显著性水平（例如 0.05）
alpha = 0.05
if p_value < alpha:
    print(f"The result is statistically significant at α = {alpha}.")
else:
    print(f"The result is not statistically significant at α = {alpha}.")
