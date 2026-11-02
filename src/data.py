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

# --- Phân nhóm 13 cột đặc trưng (theo ML Rà soát.docx) ---
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_COLS = ["sex", "fbs", "exang"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS  # 13 cột
TARGET_COL = "target"

# Giá trị hợp lệ của các biến phân loại/nhị phân (để cảnh báo bất thường)
EXPECTED_VALUES = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {0, 1, 2, 3},
    "restecg": {0, 1, 2},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3},  # mã 4 là giá trị bất thường trong bộ dữ liệu
    "thal": {0, 1, 2, 3},
    TARGET_COL: {0, 1},
}


def load_heart_data(path: str | Path) -> pd.DataFrame:
    """Đọc file CSV và kiểm tra đủ 13 cột đặc trưng + target.

    Báo lỗi rõ ràng khi thiếu file/cột hoặc cột số không phải kiểu số.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {path}. Đặt heart.csv vào data/raw/."
        )
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong {path.name}: {missing}")

    non_numeric = [
        c for c in FEATURE_COLS + [TARGET_COL] if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Các cột phải có kiểu số: {non_numeric}")

    return df[FEATURE_COLS + [TARGET_COL]].copy()


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """df.drop_duplicates() trước khi phân cụm; đặt lại chỉ số 0..n-1."""
    return df.drop_duplicates().reset_index(drop=True)

def replace_abnormal_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Thay giá trị bất thường ca == 4 bằng NaN (ca hợp lệ chỉ là 0-3)."""
    out = df.copy()
    out["ca"] = out["ca"].astype(float)
    out.loc[out["ca"] == 4, "ca"] = np.nan
    return out

def impute_mode(df: pd.DataFrame, cols=("ca",)) -> tuple[pd.DataFrame, dict]:
    """Điền NaN bằng giá trị phổ biến nhất (mode) của từng cột phân loại.
 
    K-Medoids/Gower/StandardScaler không nhận NaN nên cần bước này sau khi
    đã đặt ca == 4 thành NaN. Trả (bảng đã điền, {cột: giá trị đã điền}).
    """
    out = df.copy()
    fills = {}
    for col in cols:
        mode = out[col].mode(dropna=True).iloc[0]
        out[col] = out[col].fillna(mode).astype(int)
        fills[col] = int(mode)
    return out, fills

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách X (13 cột) và target. target chỉ dùng để kiểm tra sau phân cụm."""
    return df[FEATURE_COLS].copy(), df[TARGET_COL].copy()


def data_quality_report(df: pd.DataFrame) -> dict:
    """Tóm tắt chất lượng dữ liệu: kích thước, thiếu, trùng, giá trị bất thường."""
    unexpected = {}
    for col, allowed in EXPECTED_VALUES.items():
        bad = df.loc[~df[col].isin(allowed), col]
        if len(bad):
            unexpected[col] = bad.value_counts().sort_index().to_dict()
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "unexpected_values": unexpected,
    }
