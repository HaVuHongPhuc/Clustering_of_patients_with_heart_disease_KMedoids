from __future__ import annotations
"""Vai trò: đọc dữ liệu và xác thực cấu trúc bảng đầu vào (OOP).

Class chính:
- HeartDataLoader : đọc raw, harmonize encoding, làm sạch, impute, split.
- DataQualityAnalyzer : báo cáo chất lượng dữ liệu (missing, duplicate, abnormal).
"""
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_COLS = ["sex", "fbs", "exang"]
TARGET_COL = "target"
REFERENCE_COLS = ["cp"]
CLUSTER_CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c not in REFERENCE_COLS]
FEATURE_COLS = NUMERIC_COLS + CLUSTER_CATEGORICAL_COLS + BINARY_COLS
ALL_COLS = FEATURE_COLS + REFERENCE_COLS + [TARGET_COL]

EXPECTED_VALUES = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {1, 2, 3, 4},
    "restecg": {0, 1, 2},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    TARGET_COL: {0, 1},
}

_KAGGLE_SOURCES = {"kaggle_heart_1025", "heart_disease_1048", "fedesoriano_heart"}


class HeartDataLoader:
    """Đọc raw, harmonize encoding UCI, làm sạch, impute và tách features/target/reference.

    Sử dụng:
        loader = HeartDataLoader(path)
        df_raw = loader.raw
        df_clean = loader.clean(sources=None)      # trả DataFrame đã xử lý
        X, y = loader.split_features_target(df_clean)
        ref = loader.split_reference(df_clean)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._raw: pd.DataFrame | None = None
        self._clean: pd.DataFrame | None = None
        self.fill_categorical: dict = {}
        self.fill_numeric: dict = {}

    # ---------- đọc ----------
    @property
    def raw(self) -> pd.DataFrame:
        """Dữ liệu gốc (đã đọc, chưa làm sạch)."""
        if self._raw is None:
            self._raw = self._read_raw()
        return self._raw

    def _read_raw(self) -> pd.DataFrame:
        if not self.path.is_file():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu: {self.path}")
        df = pd.read_csv(self.path, na_values=["", " ", "?", "NA", "N/A"])
        df = self._harmonize_to_uci_encoding(df)
        if "source" in df.columns:
            df = df.drop(columns=["source"])
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột trong {self.path.name}: {missing}")
        for c in ALL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[TARGET_COL]).copy()
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df[ALL_COLS].copy()

    # ---------- harmonize ----------
    @staticmethod
    def _harmonize_to_uci_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """Đưa mọi source về encoding UCI 1-based (cp 1-4, slope 1-3, thal 3/6/7)."""
        out = df.copy()
        if "source" not in out.columns:
            return out
        mask = out["source"].isin(_KAGGLE_SOURCES)
        out.loc[mask, "cp"] = pd.to_numeric(out.loc[mask, "cp"], errors="coerce") + 1
        out.loc[mask, "slope"] = pd.to_numeric(out.loc[mask, "slope"], errors="coerce") + 1
        out.loc[mask, "thal"] = (
            pd.to_numeric(out.loc[mask, "thal"], errors="coerce")
            .map({0: 3, 1: 6, 2: 7, 3: 7})
        )
        return out

    # ---------- làm sạch ----------
    def clean(self, sources: list[str] | None = None) -> pd.DataFrame:
        """Pipeline làm sạch đầy đủ: filter source → replace abnormal → dedup → impute.

        Trả về DataFrame 14 cột (12 đặc trưng + cp + target) không còn NaN.
        Lưu fill_categorical / fill_numeric để tái sử dụng.
        """
        df = self.raw
        if sources is not None:
            if "source" not in self.raw.columns:
                raise ValueError("File không có cột 'source' để filter")
            unknown = set(sources) - set(self.raw["source"].unique())
            if unknown:
                raise ValueError(f"Source không tồn tại: {unknown}")
            df = self.raw[self.raw["source"].isin(sources)].copy()
            # re-harmonize đã làm trong _read_raw (không có source nên cần đọc lại)
            df = self._read_raw_filtered(sources)

        df = self._replace_abnormal_with_nan(df)
        df = df.drop_duplicates().reset_index(drop=True)
        df = self._impute_mode(df, cols=CATEGORICAL_COLS + BINARY_COLS)
        df = self._impute_median(df, cols=NUMERIC_COLS)

        remaining = df.isna().sum()
        remaining = remaining[remaining > 0].to_dict()
        if remaining:
            raise ValueError(f"Vẫn còn NaN sau impute: {remaining}")

        self._clean = df
        return df

    def _read_raw_filtered(self, sources: list[str]) -> pd.DataFrame:
        df = pd.read_csv(self.path, na_values=["", " ", "?", "NA", "N/A"])
        if "source" in df.columns:
            df = df[df["source"].isin(sources)].copy()
            df = self._harmonize_to_uci_encoding(df)
            df = df.drop(columns=["source"])
        for c in ALL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[TARGET_COL]).copy()
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df[ALL_COLS].copy()

    @staticmethod
    def _replace_abnormal_with_nan(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, valid in EXPECTED_VALUES.items():
            if col == TARGET_COL:
                continue
            out[col] = out[col].astype(float)
            invalid = out[col].notna() & ~out[col].isin(valid)
            out.loc[invalid, col] = np.nan
        return out

    def _impute_mode(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            if out[col].isna().any():
                mode_series = out[col].mode(dropna=True)
                fill_val = int(mode_series.iloc[0]) if len(mode_series) else 0
                out[col] = out[col].fillna(fill_val).astype(int)
                self.fill_categorical[col] = fill_val
        return out

    def _impute_median(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            if out[col].isna().any():
                med = float(out[col].median())
                if pd.isna(med):
                    med = 0.0
                out[col] = out[col].fillna(med)
                self.fill_numeric[col] = med
        return out

    # ---------- split ----------
    @staticmethod
    def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Trả (X 12 đặc trưng không có cp/target, target)."""
        return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()

    @staticmethod
    def split_reference(df: pd.DataFrame) -> pd.DataFrame:
        """Bảng tham chiếu gồm `cp` + `target`; không dùng để phân cụm."""
        return df[REFERENCE_COLS + [TARGET_COL]].copy()


class DataQualityAnalyzer:
    """Báo cáo chất lượng dữ liệu: missing, duplicate, giá trị bất thường."""

    @staticmethod
    def report(df: pd.DataFrame) -> dict:
        unexpected = {}
        for col, allowed in EXPECTED_VALUES.items():
            bad = df.loc[df[col].notna() & ~df[col].isin(allowed), col]
            if len(bad):
                unexpected[col] = bad.value_counts().sort_index().to_dict()
        return {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_missing": int(df.isna().sum().sum()),
            "missing_per_col": df.isna().sum()[df.isna().sum() > 0].to_dict(),
            "n_duplicates": int(df.duplicated().sum()),
            "unexpected_values": unexpected,
        }

    @staticmethod
    def missing_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
        return {
            "before": int(df_before.isna().sum().sum()),
            "after": int(df_after.isna().sum().sum()),
        }