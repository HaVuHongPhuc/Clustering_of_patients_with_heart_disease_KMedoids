from __future__ import annotations
"""Vai trò: tạo adapter mỏng cho triển khai K-Medoids bằng thư viện.

Hướng thực hiện sau:
- Chọn package có API và metric phù hợp với biểu diễn dữ liệu đã quyết định.
- Ghi phiên bản package trong requirements.txt.
- Cấu hình cùng k, metric, seed và tiền xử lý như bản tự cài đặt khi có thể.
- Trả kết quả theo định dạng thống nhất để evaluation.py so sánh hai cách.
"""
"""
Cấu hình cố định: method='pam', init='k-medoids++',
random_state=42. Phương án chính dùng metric='manhattan' trên ma trận đặc
trưng đã tiền xử lý; đối chứng Gower dùng metric='precomputed'.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn_extra.cluster import KMedoids

RANDOM_STATE = 42
METHOD = "pam"
INIT = "k-medoids++"


@dataclass
class KMedoidsResult:
    """Kết quả chạy thống nhất để evaluation.py so sánh các cách chạy."""

    n_clusters: int
    metric: str
    labels: np.ndarray
    medoid_indices: np.ndarray
    inertia: float  # tổng khoảng cách từ mỗi điểm tới medoid của nó
    n_iter: int


def fit_kmedoids(
    data: np.ndarray,
    n_clusters: int,
    metric: str = "manhattan",
    random_state: int = RANDOM_STATE,
) -> KMedoidsResult:
    """Chạy KMedoids (pam, k-medoids++).

    data: ma trận đặc trưng (metric='manhattan') hoặc ma trận khoảng cách
    n x n (metric='precomputed').
    """
    if metric == "precomputed" and data.shape[0] != data.shape[1]:
        raise ValueError("metric='precomputed' cần ma trận khoảng cách vuông n x n")

    with warnings.catch_warnings():
        # sklearn_extra dùng hàm đã bị cảnh báo deprecated trong sklearn mới
        warnings.simplefilter("ignore", FutureWarning)
        model = KMedoids(
            n_clusters=n_clusters,
            metric=metric,
            method=METHOD,
            init=INIT,
            random_state=random_state,
        ).fit(data)

    return KMedoidsResult(
        n_clusters=n_clusters,
        metric=metric,
        labels=model.labels_,
        medoid_indices=model.medoid_indices_,
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )