from __future__ import annotations
"""Vai trò: đánh giá chất lượng và tóm tắt kết quả phân cụm.

Hướng thực hiện sau:
- Chọn metric nội bộ phù hợp với khoảng cách và dữ liệu.
- So sánh nhiều k, lưu objective/dissimilarity và kích thước cụm.
- Nếu có nhãn thật, chỉ dùng metric ngoài sau khi fit như phân tích hậu nghiệm.
- Nêu định nghĩa, cách tính và giới hạn diễn giải của từng metric trong báo cáo.
"""
"""Đánh giá phân cụm: chọn k bằng silhouette và kiểm tra hậu nghiệm bằng target.
- Chọn k trong khoảng 2-8 theo silhouette (chỉ dùng đặc trưng/khoảng cách,
  KHÔNG dùng target).
- target chỉ dùng SAU khi đã phân cụm để đối chiếu (đánh giá ngoài).
"""


import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .data import NUMERIC_COLS, TARGET_COL
from .library_model import RANDOM_STATE, KMedoidsResult, fit_kmedoids

K_RANGE = range(2, 9)  # k = 2..8


def silhouette(data: np.ndarray, labels: np.ndarray, metric: str = "manhattan") -> float:
    """Silhouette; metric='precomputed' khi `data` là ma trận khoảng cách."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(data, labels, metric=metric))


def select_k(
    data: np.ndarray,
    metric: str = "manhattan",
    k_range=K_RANGE,
    random_state: int = RANDOM_STATE,
) -> tuple[int, pd.DataFrame, dict[int, KMedoidsResult]]:
    """Thử từng k, tính silhouette; trả (k tốt nhất, bảng tổng hợp, kết quả theo k)."""
    rows, results = [], {}
    for k in k_range:
        res = fit_kmedoids(data, k, metric=metric, random_state=random_state)
        results[k] = res
        sizes = np.bincount(res.labels, minlength=k)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette(data, res.labels, metric=metric),
                "inertia": res.inertia,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            }
        )
    table = pd.DataFrame(rows).set_index("k")
    best_k = int(table["silhouette"].idxmax())
    return best_k, table, results


def cluster_sizes(labels: np.ndarray) -> pd.Series:
    """Kích thước từng cụm."""
    return pd.Series(labels).value_counts().sort_index().rename("n")


def target_crosstab(labels: np.ndarray, target: pd.Series) -> pd.DataFrame:
    """Bảng chéo cụm x target (hậu nghiệm), kèm tỉ lệ có bệnh mỗi cụm."""
    ct = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(target.to_numpy(), name=TARGET_COL))
    ct["tỉ_lệ_bệnh"] = ct.get(1, 0) / ct.sum(axis=1)
    return ct


def posthoc_target_check(labels: np.ndarray, target: pd.Series) -> dict:
    """Chỉ số ngoài: ARI và độ thuần (purity) so với target.

    Giới hạn diễn giải: phân cụm không giám sát không nhằm tái tạo target, nên
    ARI/purity thấp không có nghĩa là cụm sai.
    """
    y = target.to_numpy()
    ct = pd.crosstab(labels, y)
    purity = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
    return {"ARI": float(adjusted_rand_score(y, labels)), "purity": purity}


def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Hồ sơ cụm trên giá trị gốc: kích thước, trung bình biến số, tỉ lệ nhị phân/target.

    `df` là bảng 13 cột + target đã khử trùng lặp, cùng thứ tự với `labels`.
    """
    d = df.copy()
    d["cluster"] = labels
    g = d.groupby("cluster")
    prof = g[NUMERIC_COLS].mean().round(2)
    for col in ["sex", "fbs", "exang"]:
        prof[f"{col}_rate"] = g[col].mean().round(3)
    if TARGET_COL in d.columns:
        prof["target_rate"] = g[TARGET_COL].mean().round(3)
    prof.insert(0, "n", g.size())
    return prof


def compare_labelings(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """ARI giữa hai kết quả (vd. Manhattan vs Gower, hoặc thư viện vs tự cài đặt)."""
    return float(adjusted_rand_score(labels_a, labels_b))
