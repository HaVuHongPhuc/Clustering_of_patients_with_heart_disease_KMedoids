from __future__ import annotations
"""Chuẩn bị ma trận đặc trưng và ma trận khoảng cách (OOP).

Class chính:
- FeaturePreprocessor : Scale + OneHot + giữ binary; fit/transform ma trận 12 cột.
- DistanceCalculator   : Manhattan trên ma trận đã tiền xử lý, Gower trên 12 cột gốc.
- ProcessedDataSaver   : lưu data/processed/.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    BINARY_COLS,
    CLUSTER_CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    REFERENCE_COLS,
    TARGET_COL,
)


class FeaturePreprocessor:
    """Scale 5 biến số, OneHot 4 biến phân loại, giữ nguyên 3 binary.

    Sử dụng:
        prep = FeaturePreprocessor()
        X_mat, names, pre = prep.fit_transform(X)
    """

    def __init__(self):
        self._transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def build(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]), NUMERIC_COLS),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), CLUSTER_CATEGORICAL_COLS),
                ("bin", "passthrough", BINARY_COLS),
            ],
            verbose_feature_names_out=False,
        )

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str], ColumnTransformer]:
        missing = [c for c in FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Thiếu cột đặc trưng: {missing}")
        self._transformer = self.build()
        X_mat = self._transformer.fit_transform(X[FEATURE_COLS]).astype(float)
        self.feature_names_ = list(self._transformer.get_feature_names_out())
        return X_mat, self.feature_names_, self._transformer

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("Chưa fit; gọi fit_transform trước.")
        return self._transformer.transform(X[FEATURE_COLS]).astype(float)


class DistanceCalculator:
    """Tính khoảng cách Manhattan và Gower trên 12 đặc trưng."""

    @staticmethod
    def manhattan(X_mat: np.ndarray) -> np.ndarray:
        return pairwise_distances(X_mat, metric="manhattan")

    @staticmethod
    def gower(X: pd.DataFrame) -> np.ndarray:
        """Khoảng cách Gower trên 12 cột gốc (chưa scale/one-hot, không có cp)."""
        if X[FEATURE_COLS].isna().any().any():
            raise ValueError("Còn NaN trong X; hãy impute trước khi tính Gower")
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
        for col in CLUSTER_CATEGORICAL_COLS + BINARY_COLS:
            v = X[col].to_numpy()
            total += (v[:, None] != v[None, :]).astype(float)
        D = total / len(FEATURE_COLS)
        np.fill_diagonal(D, 0.0)
        return D


class ProcessedDataSaver:
    """Lưu dữ liệu đã tiền xử lý vào data/processed/."""

    @staticmethod
    def save(
        df: pd.DataFrame,
        X_mat: np.ndarray,
        feature_names: list[str],
        gower_matrix: np.ndarray,
        out_dir,
    ) -> dict:
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "dedup": out / "heart_dedup.csv",
            "X": out / "X_manhattan.csv",
            "target": out / "target.csv",
            "reference": out / "reference.csv",
            "gower": out / "gower_distance.npy",
        }
        df.to_csv(paths["dedup"], index=False)
        pd.DataFrame(X_mat, columns=feature_names).to_csv(paths["X"], index=False)
        df[[TARGET_COL]].to_csv(paths["target"], index=False)
        df[REFERENCE_COLS + [TARGET_COL]].to_csv(paths["reference"], index=False)
        np.save(paths["gower"], gower_matrix)
        return paths