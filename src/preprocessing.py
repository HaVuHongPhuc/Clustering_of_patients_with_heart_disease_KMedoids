"""Chuẩn bị ma trận đặc trưng dùng chung cho hai mô hình K-Medoids.

Phương án chính (Manhattan):
  - 5 biến số (age, trestbps, chol, thalach, oldpeak): StandardScaler
  - 5 biến phân loại (cp, restecg, slope, ca, thal): OneHotEncoder
  - 3 biến nhị phân (sex, fbs, exang): giữ nguyên
Đối chứng (Gower): tính trên 13 cột gốc, ra ma trận khoảng cách để dùng với
KMedoids(metric='precomputed').
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY_COLS, CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: StandardScaler + OneHotEncoder + passthrough."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("bin", "passthrough", BINARY_COLS),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_features(X: pd.DataFrame) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    """Fit + transform 13 cột gốc -> ma trận số cho Manhattan.

    Trả về (ma trận X_mat, tên đặc trưng sau biến đổi, preprocessor đã fit).
    """
    missing = [c for c in FEATURE_COLS if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu cột đặc trưng: {missing}")
    pre = build_preprocessor()
    X_mat = pre.fit_transform(X[FEATURE_COLS]).astype(float)
    return X_mat, list(pre.get_feature_names_out()), pre


def manhattan_distance_matrix(X_mat: np.ndarray) -> np.ndarray:
    """Ma trận khoảng cách Manhattan (L1) n x n trên ma trận đã tiền xử lý."""
    return pairwise_distances(X_mat, metric="manhattan")


def gower_distance_matrix(X: pd.DataFrame) -> np.ndarray:
    """Khoảng cách Gower trên 13 cột gốc (chưa scale/one-hot).

    - Cột số (5): |xi - xj| / (max - min); cột có max == min thì đóng góp 0.
    - Cột phân loại (5) và nhị phân (3): 0 nếu khớp, 1 nếu không khớp.
    - Gower = trung bình cộng 13 thành phần, nằm trong [0, 1].
    """
    missing = [c for c in FEATURE_COLS if c not in X.columns]
    if missing:
        raise ValueError(f"Thiếu cột đặc trưng: {missing}")

    n = len(X)
    total = np.zeros((n, n), dtype=float)

    for col in NUMERIC_COLS:
        v = X[col].to_numpy(dtype=float)
        rng = v.max() - v.min()
        if rng > 0:
            total += np.abs(v[:, None] - v[None, :]) / rng

    for col in CATEGORICAL_COLS + BINARY_COLS:
        v = X[col].to_numpy()
        total += (v[:, None] != v[None, :]).astype(float)

    D = total / len(FEATURE_COLS)
    np.fill_diagonal(D, 0.0)
    return D


def save_processed(
    df: pd.DataFrame,
    X_mat: np.ndarray,
    feature_names: list[str],
    gower_matrix: np.ndarray,
    out_dir,
) -> dict:
    """Lưu dữ liệu tiền xử lý vào out_dir (thường là data/processed/).

    - heart_dedup.csv      : 13 cột gốc + target sau drop_duplicates (dùng cho hồ sơ cụm)
    - X_manhattan.csv      : ma trận đặc trưng đã Scale + One-Hot (đầu vào Manhattan)
    - target.csv           : nhãn target tách riêng (chỉ kiểm tra sau phân cụm)
    - gower_distance.npy   : ma trận khoảng cách Gower n x n (đối chứng)
    Thứ tự dòng của mọi file giống nhau.
    """
    from pathlib import Path

    from .data import TARGET_COL

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "dedup": out / "heart_dedup.csv",
        "X": out / "X_manhattan.csv",
        "target": out / "target.csv",
        "gower": out / "gower_distance.npy",
    }
    df.to_csv(paths["dedup"], index=False)
    pd.DataFrame(X_mat, columns=feature_names).to_csv(paths["X"], index=False)
    df[[TARGET_COL]].to_csv(paths["target"], index=False)
    np.save(paths["gower"], gower_matrix)
    return paths
