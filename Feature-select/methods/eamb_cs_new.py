import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, fisher_exact

def ifdep_class_specific(dataframe, C, Y, X=None, y_value=None, alpha=0.05, min_group_size=10):
    """
    类特定条件独立性检验 (Class-Specific Conditional Independence Test)
    -----------------------------------------------------------
    检验 C ⟂ I(Y=y_value) | X 是否成立（离散型变量版本）

    参数：
    ----------
    dataframe : pandas.DataFrame
        数据集，包含 C, Y, X 等列（均为离散取值）。
    C : str
        目标特征列名。
    Y : str
        类别标签列名。
    X : list[str] or None
        条件集变量列表。如果为空，则直接检验 C 与 Y_bin 是否独立。
    y_value : 任意类型
        需要关注的类标签取值。
    alpha : float, default=0.05
        显著性水平。
    min_group_size : int, default=10
        每个条件分层的最小样本数，小于该阈值的分层将被跳过。

    返回：
    ----------
    (is_independent, p_value, note)
        is_independent : bool
            True 表示不能拒绝独立（即 C ⟂ Y_bin | X 成立或数据不足）
        p_value : float or None
            检验的总体 p 值，若无法计算则为 None
        note : str
            附加说明，包括分层信息、跳过数量等
    """
    if y_value is None:
        raise ValueError("请指定 y_value 参数。")

    df = dataframe.copy()

    # === Step 1: 二值化 Y ===
    df["_Y_bin"] = (df[Y] == y_value).astype(int)

    # === Step 2: 如果条件集 X 为空 ===
    if not X:
        cont = pd.crosstab(df[C], df["_Y_bin"])
        if cont.shape[0] < 2 or cont.shape[1] < 2:
            return True, None, "不足以构建 2x2 表（单列或单行）——无法检验"
        # 若为 2x2 且存在小格
        if cont.shape == (2, 2) and (cont.values < 5).any():
            _, p = fisher_exact(cont.values)
            return p >= alpha, p, "使用 Fisher 精确检验（2x2 小样本）"
        else:
            _, p, _, _ = chi2_contingency(cont, lambda_="log-likelihood")
            return p >= alpha, p, "使用 G² 检验（全样本）"

    # === Step 3: 有条件变量 X ===
    grouped = df.groupby(X)
    G2_total = 0.0
    df_total = 0
    valid_strata = 0
    skipped_strata = 0

    for name, group in grouped:
        if len(group) < min_group_size:
            skipped_strata += 1
            continue

        cont = pd.crosstab(group[C], group["_Y_bin"])
        if cont.shape[0] < 2 or cont.shape[1] < 2:
            skipped_strata += 1
            continue

        arr = cont.values.astype(float)
        n = arr.sum()
        row_sum = arr.sum(axis=1, keepdims=True)
        col_sum = arr.sum(axis=0, keepdims=True)
        expected = row_sum.dot(col_sum) / n

        # 忽略零格子的 log(0) 警告
        mask = arr > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(mask, arr * np.log(arr / expected), 0.0)

        G2_k = 2.0 * term.sum()
        df_k = (arr.shape[0] - 1) * (arr.shape[1] - 1)

        G2_total += G2_k
        df_total += df_k
        valid_strata += 1

    # === Step 4: 汇总统计量 ===
    if df_total == 0:
        return True, None, f"有效分层不足，valid={valid_strata}, skipped={skipped_strata}"

    p_value = chi2.sf(G2_total, df_total)
    decision = p_value >= alpha
    note = f"G²总统计量={G2_total:.3f}, 自由度={df_total}, 有效分层={valid_strata}, 跳过={skipped_strata}"
    return decision, p_value, note
