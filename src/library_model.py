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
from kmedoids import KMedoids

RANDOM_STATE = 42
METHOD = "pam"
INIT = "build"          # kmedoids không có 'k-medoids++'; 'build' là BUILD của PAM
BACKEND = "kmedoids"    # ghi vào run_config


@dataclass
class KMedoidsResult:
    n_clusters: int
    metric: str
    labels: np.ndarray
    medoid_indices: np.ndarray
    inertia: float
    n_iter: int         # kmedoids không expose; giữ -1 để đồng nhất schema


def fit_kmedoids(
    data: np.ndarray,
    n_clusters: int,
    metric: str = "manhattan",
    random_state: int = RANDOM_STATE,
) -> KMedoidsResult:
    """KMedoids (PAM) trên dữ liệu đặc trưng hoặc ma trận khoảng cách precomputed.

    - metric='precomputed' → `data` phải là ma trận vuông n x n.
    - metric khác (manhattan, euclidean, ...) → `data` là ma trận đặc trưng n x d.
    """
    if metric == "precomputed" and data.shape[0] != data.shape[1]:
        raise ValueError("metric='precomputed' cần ma trận khoảng cách vuông n x n")

    # kmedoids dùng `metric='precomputed'` giống sklearn, chỉ khác không có init='k-medoids++'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = KMedoids(
            n_clusters=n_clusters,
            metric=metric,          # 'precomputed' | 'manhattan' | 'euclidean' | ...
            method=METHOD,          # 'pam' | 'fasterpam' | 'alternate'
            init=INIT,              # 'build' | 'random' | 'first' | 'pam'
            random_state=random_state,
        ).fit(data)

    return KMedoidsResult(
        n_clusters=n_clusters,
        metric=metric,
        labels=np.asarray(model.labels_),
        medoid_indices=np.asarray(model.medoid_indices_),
        inertia=float(model.inertia_),
        n_iter=int(getattr(model, "n_iter_", -1)),
    )