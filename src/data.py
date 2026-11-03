from __future__ import annotations
"""Vai trò: đọc dữ liệu và xác thực cấu trúc bảng đầu vào.

Hướng thực hiện sau:
- Đọc đường dẫn từ cấu hình thay vì hard-code.
- Báo lỗi rõ ràng khi thiếu file/cột hoặc kiểu dữ liệu không hợp lệ.
- Giữ nguyên nhãn chẩn đoán ở biến riêng, không gộp vào ma trận phân cụm.
- Ghi lại số mẫu và thuộc tính trước/sau các bước xử lý.
"""
"""
bộ dữ liệu gồm 13 cột đặc trưng + cột `target`.
`target` luôn được giữ riêng, KHÔNG đưa vào ma trận phân cụm; chỉ dùng để
kiểm tra sau khi phân cụm.
"""
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_COLS = ["sex", "fbs", "exang"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
TARGET_COL = "target"

# Encoding chuẩn UCI 1-based (sau khi harmonize)
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

# Các source dùng encoding Kaggle 0-based
_KAGGLE_SOURCES = {"kaggle_heart_1025", "heart_disease_1048", "fedesoriano_heart"}


def harmonize_to_uci_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa mọi source về encoding UCI 1-based (cp 1-4, slope 1-3, thal 3/6/7).

    - uci_multi_*: giữ nguyên.
    - kaggle_heart_1025, heart_disease_1048, fedesoriano_heart: cộng 1 cho cp/slope,
      map thal {0:3, 1:6, 2:7, 3:7}.
    """
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


def load_heart_data(
    path: str | Path,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Đọc file raw, (tuỳ chọn) filter theo source, harmonize encoding về UCI 1-based.

    - sources=None: dùng tất cả (~2000+ dòng), tự harmonize encoding.
    - sources=[...]: chỉ lấy các source chỉ định.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")

    df = pd.read_csv(path, na_values=["", " ", "?", "NA", "N/A"])

    if sources is not None:
        if "source" not in df.columns:
            raise ValueError("File không có cột 'source' để filter")
        unknown = set(sources) - set(df["source"].unique())
        if unknown:
            raise ValueError(f"Source không tồn tại: {unknown}")
        df = df[df["source"].isin(sources)].copy()

    # Harmonize TRƯỚC khi bỏ cột source (cần biết source để map)
    df = harmonize_to_uci_encoding(df)

    if "source" in df.columns:
        df = df.drop(columns=["source"])

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong {path.name}: {missing}")

    for c in FEATURE_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df[FEATURE_COLS + [TARGET_COL]].copy()


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def replace_abnormal_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, valid in EXPECTED_VALUES.items():
        if col == TARGET_COL:
            continue
        out[col] = out[col].astype(float)
        invalid = out[col].notna() & ~out[col].isin(valid)
        out.loc[invalid, col] = np.nan
    return out


def impute_mode(df: pd.DataFrame, cols=CATEGORICAL_COLS) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    fills = {}
    for col in cols:
        if out[col].isna().any():
            mode_series = out[col].mode(dropna=True)
            if len(mode_series) == 0:
                # Cột toàn NaN — dùng 0 làm fallback (hoặc bỏ cột nếu muốn)
                fill_val = 0
            else:
                fill_val = int(mode_series.iloc[0])
            out[col] = out[col].fillna(fill_val).astype(int)
            fills[col] = fill_val
    return out, fills


def impute_median(df: pd.DataFrame, cols=NUMERIC_COLS) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    fills = {}
    for col in cols:
        if out[col].isna().any():
            med = float(out[col].median())
            if pd.isna(med):
                med = 0.0   # fallback nếu cột toàn NaN
            out[col] = out[col].fillna(med)
            fills[col] = med
    return out, fills


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


def data_quality_report(df: pd.DataFrame) -> dict:
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