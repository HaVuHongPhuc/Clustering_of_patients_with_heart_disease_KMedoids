"""Adapter K-Medoids bằng thư viện `kmedoids`.

Cấu hình hiện dùng PAM với BUILD initialization và seed mặc định 42.
- Manhattan: nhận ma trận đặc trưng đã tiền xử lý.
- Gower: nhận ma trận khoảng cách vuông với metric='precomputed'.

Adapter trả kết quả theo schema thống nhất để evaluation.py lưu labels,
medoids và objective. Chú ý BUILD là khởi tạo xác định; random_state không
đảm bảo tạo cùng medoid ban đầu với bản tự cài đặt khởi tạo ngẫu nhiên.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from kmedoids import KMedoids

RANDOM_STATE = 42
METHOD = "pam"
INIT = "build"
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

    - metric='precomputed' -> `data` phải là ma trận vuông n x n.
    - metric khác (manhattan, euclidean, ...) -> `data` là ma trận đặc trưng n x d.
    """
    if metric == "precomputed" and data.shape[0] != data.shape[1]:
        raise ValueError("metric='precomputed' cần ma trận khoảng cách vuông n x n")

    # kmedoids dùng `metric='precomputed'` giống sklearn, chỉ khác không có init='k-medoids++'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        labels=np.asarray(model.labels_),
        medoid_indices=np.asarray(model.medoid_indices_),
        inertia=float(model.inertia_),
        n_iter=int(getattr(model, "n_iter_", -1)),
    )