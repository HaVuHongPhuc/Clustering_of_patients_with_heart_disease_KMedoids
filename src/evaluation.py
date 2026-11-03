"""Đánh giá phân cụm: chọn k bằng silhouette và kiểm tra hậu nghiệm bằng target.

- Chọn k trong khoảng 2-8 theo silhouette (chỉ dùng đặc trưng/khoảng cách,
  KHÔNG dùng target).
- target chỉ dùng SAU khi đã phân cụm để đối chiếu (đánh giá ngoài).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .data import NUMERIC_COLS, TARGET_COL, split_features_target
from .library_model import INIT, METHOD, RANDOM_STATE, KMedoidsResult, fit_kmedoids
from .preprocessing import gower_distance_matrix, preprocess_features

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


def run_and_save_metrics(df: pd.DataFrame, out_dir, extra_config: dict | None = None) -> dict:
    """Chạy k = 2..8 cho Manhattan (chính) và Gower (đối chứng), lưu vào out_dir.

    `df` = bảng 13 cột + target đã xử lý ca=4 và khử trùng lặp.
    Các file ghi ra (notebook 02 chỉ cần đọc, không tính lại):
    - k_selection.csv     : mỗi dòng một (distance, k): silhouette, inertia, kích thước
                            cụm nhỏ/lớn, n_iter, và ARI/purity so với target (hậu nghiệm)
    - labels.csv          : nhãn cụm từng dòng, cột <distance>_k<k>
    - medoids.csv         : chỉ số dòng medoid theo (distance, k, cluster)
    - cross_method_ari.csv: ARI giữa Manhattan và Gower theo từng k
    - run_config.json     : cấu hình chạy (seed, method, init, k, metric, phiên bản...)
    """
    import json
    import platform
    from pathlib import Path

    import sklearn
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    def _safe_version(pkg_name: str) -> str:
        try:
            return _pkg_version(pkg_name)
        except PackageNotFoundError:
            return "unknown"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = split_features_target(df)
    X_mat, _, _ = preprocess_features(X)
    D_gower = gower_distance_matrix(X)
    inputs = {"manhattan": (X_mat, "manhattan"), "gower": (D_gower, "precomputed")}

    rows, label_cols, medoid_rows, results_by = [], {}, [], {}
    for name, (data, metric) in inputs.items():
        best_k, table, results = select_k(data, metric=metric)
        results_by[name] = results
        for k, res in results.items():
            ext = posthoc_target_check(res.labels, y)
            row = table.loc[k].to_dict()
            row.update(
                distance=name, k=k, n_iter=res.n_iter, is_best_k=(k == best_k),
                ARI_target=ext["ARI"], purity_target=ext["purity"],
            )
            rows.append(row)
            label_cols[f"{name}_k{k}"] = res.labels
            for c, idx in enumerate(res.medoid_indices):
                medoid_rows.append({"distance": name, "k": k, "cluster": c, "medoid_index": int(idx)})

    cols = ["distance", "k", "silhouette", "inertia", "min_cluster_size",
            "max_cluster_size", "n_iter", "ARI_target", "purity_target", "is_best_k"]
    ksel = pd.DataFrame(rows)[cols]
    cross = pd.DataFrame(
        {"k": list(K_RANGE),
         "ARI_manhattan_vs_gower": [
             compare_labelings(results_by["manhattan"][k].labels, results_by["gower"][k].labels)
             for k in K_RANGE]}
    )

    paths = {
        "k_selection": out / "k_selection.csv",
        "labels": out / "labels.csv",
        "medoids": out / "medoids.csv",
        "cross_method_ari": out / "cross_method_ari.csv",
        "run_config": out / "run_config.json",
    }
    ksel.to_csv(paths["k_selection"], index=False)
    pd.DataFrame(label_cols).to_csv(paths["labels"], index=False)
    pd.DataFrame(medoid_rows).to_csv(paths["medoids"], index=False)
    cross.to_csv(paths["cross_method_ari"], index=False)

    config = {
        "n_rows": int(len(df)),
        "n_features_manhattan": int(X_mat.shape[1]),
        "k_range": list(K_RANGE),
        "random_state": RANDOM_STATE,
        "method": METHOD,
        "init": INIT,
        "main_distance": "manhattan (StandardScaler + OneHot + binary giữ nguyên)",
        "control_distance": "gower (13 cột gốc, KMedoids metric='precomputed')",
        "best_k": {n: int(ksel[(ksel.distance == n) & ksel.is_best_k].k.iloc[0]) for n in inputs},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "kmedoids": _safe_version("kmedoids")},
    }
    if extra_config:
        config.update(extra_config)
    paths["run_config"].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths