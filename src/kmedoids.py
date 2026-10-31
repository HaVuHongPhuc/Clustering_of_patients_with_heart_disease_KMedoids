"""K-Medoids kiểu cập nhật luân phiên (alternate K-Medoids), tự cài đặt.

Dữ liệu đầu vào phải là các đặc trưng số đã được tiền xử lý. Không đưa nhãn
chẩn đoán bệnh vào X nếu nhãn đó chỉ dùng để phân tích sau phân cụm.

CÁCH GIẢI TAY MINH HỌA:

Dùng 6 record một chiều: R1=1, R2=2, R3=3, R4=10, R5=11, R6=20.
Chọn K=3 và medoid ban đầu là R1, R4, R6 (chỉ số Python [0, 3, 5]).
Với một chiều và metric Euclidean, khoảng cách là trị tuyệt đối hiệu hai số.

Vòng 1 - gán vào medoid gần nhất:
  R1: đến (1,10,20) là (0,9,19)  -> cụm 0
  R2: đến (1,10,20) là (1,8,18)  -> cụm 0
  R3: đến (1,10,20) là (2,7,17)  -> cụm 0
  R4: đến (1,10,20) là (9,0,10) -> cụm 1
  R5: đến (1,10,20) là (10,1,9) -> cụm 1
  R6: đến (1,10,20) là (19,10,0)-> cụm 2
  Các cụm: C0=[R1,R2,R3], C1=[R4,R5], C2=[R6].

Vòng 1 - cập nhật medoid:
  Trong C0, tổng khoảng cách lần lượt nếu chọn R1, R2, R3:
      R1: |1-2| + |1-3| = 3
      R2: |2-1| + |2-3| = 2  <- nhỏ nhất, chọn R2
      R3: |3-1| + |3-2| = 3
  Trong C1, R4 và R5 đều có tổng khoảng cách bằng 1; giữ medoid hiện tại R4.
  C2 chỉ có R6 nên medoid vẫn là R6.
  Medoid mới: [R2,R4,R6] (chỉ số Python [1,3,5]).

Vòng 2:
  Gán lại với medoid (2,10,20): các cụm vẫn là [R1,R2,R3], [R4,R5], [R6].
  Cập nhật lại vẫn được [R2,R4,R6], vì vậy medoid không đổi và thuật toán dừng.
  Tổng khoảng cách cuối: (1+0+1) + (0+1) + 0 = 3.

Trong code, nhãn cụm bắt đầu từ 0 và chỉ số record cũng bắt đầu từ 0.
Ví dụ chạy đúng phép tính tay:
    model = KMedoids(3, initial_medoids=[0, 3, 5]).fit(
        [[1], [2], [3], [10], [11], [20]]
    )
Kết quả dự kiến: medoid_indices_=[1,3,5], labels_=[0,0,0,1,1,2],
inertia_=3.0.

Ghi chú: đây là biến thể cập nhật medoid tốt nhất bên trong từng cụm.
Hãy thống nhất biến thể này với Quy trước khi so sánh với bản thư viện.
"""

from math import isfinite, sqrt
from random import Random
from typing import Any, Sequence


class KMedoids:
    """K-Medoids cập nhật luân phiên, không phụ thuộc thư viện ngoài.

    Tham số
    -------
    n_clusters:
        Số cụm K.
    metric:
        "euclidean" hoặc "manhattan". Chọn cùng metric với Quy và dữ liệu
        đã được tiền xử lý phù hợp.
    max_iter:
        Số vòng cập nhật medoid tối đa.
    random_state:
        Seed để chọn medoid ban đầu lặp lại được.
    initial_medoids:
        Có thể truyền chỉ số các dòng dữ liệu làm medoid ban đầu. Dùng tham
        số này để tái hiện phép tính tay; nếu bỏ qua thì chọn ngẫu nhiên có seed.

    Sau fit, các thuộc tính chính:
        labels_: nhãn cụm từ 0 đến K-1 cho từng dòng huấn luyện.
        medoid_indices_: chỉ số các dòng được chọn làm medoid.
        medoids_: giá trị các dòng medoid.
        inertia_: tổng khoảng cách mỗi dòng đến medoid được gán.
        n_iter_: số vòng đã chạy.
        history_: nhật ký medoid, kích thước cụm và objective từng vòng.
    """

    def __init__(
        self,
        n_clusters: int,
        metric: str = "euclidean",
        max_iter: int = 100,
        random_state: int | None = 42,
        initial_medoids: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise TypeError("n_clusters phải là số nguyên.")
        if n_clusters < 1:
            raise ValueError("n_clusters phải >= 1.")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise TypeError("max_iter phải là số nguyên.")
        if max_iter < 1:
            raise ValueError("max_iter phải >= 1.")

        metric = metric.lower().strip()
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric chỉ hỗ trợ 'euclidean' hoặc 'manhattan'.")

        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_medoids = (
            None if initial_medoids is None else list(initial_medoids)
        )

    def fit(self, X: Any) -> "KMedoids":
        """Tìm medoid và nhãn cụm từ ma trận mẫu x đặc trưng."""
        rows = self._to_numeric_rows(X)
        n_samples = len(rows)

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters không thể lớn hơn số record.")

        if self.initial_medoids is None:
            medoids = Random(self.random_state).sample(
                range(n_samples), self.n_clusters
            )
        else:
            medoids = self._validate_initial_medoids(
                self.initial_medoids, n_samples
            )

        # Bảng khoảng cách được tính một lần, rồi dùng lại ở bước gán/cập nhật.
        distances = self._pairwise_distances(rows)
        self.history_ = []
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            # Bước 1: gán mỗi record vào medoid gần nhất.
            labels = self._assign_with_distance_matrix(distances, medoids)
            clusters = self._members_by_cluster(labels)
            objective = self._objective(distances, labels, medoids)

            # Bước 2: trong từng cụm, chọn record có tổng khoảng cách nội cụm nhỏ nhất.
            updated_medoids = self._update_medoids(
                distances, clusters, medoids
            )

            self.history_.append(
                {
                    "iteration": iteration,
                    "medoid_indices": tuple(medoids),
                    "cluster_sizes": tuple(len(members) for members in clusters),
                    "objective": objective,
                    "updated_medoid_indices": tuple(updated_medoids),
                }
            )

            # Bước 3: nếu danh sách medoid không đổi thì đã hội tụ.
            if updated_medoids == medoids:
                self.converged_ = True
                break

            medoids = updated_medoids

        # Tính lại nhãn và objective cho medoid cuối cùng, kể cả khi chạm max_iter.
        final_labels = self._assign_with_distance_matrix(distances, medoids)
        self.labels_ = final_labels
        self.medoid_indices_ = list(medoids)
        self.medoids_ = [list(rows[index]) for index in medoids]
        self.inertia_ = self._objective(distances, final_labels, medoids)
        self.cluster_sizes_ = [
            final_labels.count(cluster_id)
            for cluster_id in range(self.n_clusters)
        ]
        self.n_iter_ = iteration
        self.n_features_in_ = len(rows[0])
        self._training_rows_ = rows

        return self

    def fit_predict(self, X: Any) -> list[int]:
        """Fit mô hình rồi trả nhãn cụm của dữ liệu huấn luyện."""
        return self.fit(X).labels_

    def predict(self, X: Any) -> list[int]:
        """Gán record mới vào medoid gần nhất đã học."""
        if not hasattr(self, "medoids_"):
            raise RuntimeError("Cần gọi fit(X) trước khi predict(X).")

        rows = self._to_numeric_rows(X)
        if len(rows[0]) != self.n_features_in_:
            raise ValueError(
                f"X cần có {self.n_features_in_} đặc trưng, "
                f"nhưng nhận được {len(rows[0])}."
            )

        return [
            min(
                range(self.n_clusters),
                key=lambda cluster_id: (
                    self._distance(row, self.medoids_[cluster_id]),
                    cluster_id,
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _to_numeric_rows(X: Any) -> list[list[float]]:
        """Chấp nhận list, array-like hoặc DataFrame có phương thức to_numpy."""
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(X, "tolist"):
            X = X.tolist()

        try:
            raw_rows = list(X)
        except TypeError as exc:
            raise TypeError("X phải là ma trận các record và đặc trưng.") from exc

        if not raw_rows:
            raise ValueError("X không được rỗng.")

        rows: list[list[float]] = []
        for row_number, raw_row in enumerate(raw_rows, start=1):
            if isinstance(raw_row, (str, bytes)):
                values = [raw_row]
            else:
                try:
                    values = list(raw_row)
                except TypeError:
                    # Cho phép dữ liệu một chiều như [1, 2, 3], xem mỗi số là 1 record.
                    values = [raw_row]

            if not values:
                raise ValueError(f"Record {row_number} không có đặc trưng.")

            try:
                numeric_row = [float(value) for value in values]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Record {row_number} có giá trị không phải số."
                ) from exc

            if not all(isfinite(value) for value in numeric_row):
                raise ValueError(
                    f"Record {row_number} có NaN hoặc giá trị vô hạn; "
                    "hãy xử lý trước khi phân cụm."
                )
            rows.append(numeric_row)

        expected_features = len(rows[0])
        for row_number, row in enumerate(rows, start=1):
            if len(row) != expected_features:
                raise ValueError(
                    f"Record {row_number} có {len(row)} đặc trưng; "
                    f"cần đồng nhất {expected_features}."
                )

        return rows

    def _validate_initial_medoids(
        self, medoids: list[int], n_samples: int
    ) -> list[int]:
        if len(medoids) != self.n_clusters:
            raise ValueError(
                "Số initial_medoids phải bằng n_clusters."
            )
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in medoids
        ):
            raise TypeError("Mỗi initial_medoid phải là chỉ số nguyên.")
        if len(set(medoids)) != len(medoids):
            raise ValueError("Các initial_medoids phải khác nhau.")
        if any(index < 0 or index >= n_samples for index in medoids):
            raise ValueError("Chỉ số initial_medoid nằm ngoài phạm vi X.")
        return medoids.copy()

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        differences = [a - b for a, b in zip(left, right)]
        if self.metric == "manhattan":
            return sum(abs(value) for value in differences)
        return sqrt(sum(value * value for value in differences))

    def _pairwise_distances(
        self, rows: list[list[float]]
    ) -> list[list[float]]:
        n_samples = len(rows)
        matrix = [[0.0] * n_samples for _ in range(n_samples)]

        for left_index in range(n_samples):
            for right_index in range(left_index + 1, n_samples):
                distance = self._distance(
                    rows[left_index], rows[right_index]
                )
                matrix[left_index][right_index] = distance
                matrix[right_index][left_index] = distance

        return matrix

    def _assign_with_distance_matrix(
        self, distances: list[list[float]], medoids: list[int]
    ) -> list[int]:
        labels = [-1] * len(distances)

        # Mỗi medoid được giữ trong cụm của chính nó, kể cả khi dữ liệu trùng.
        for cluster_id, medoid_index in enumerate(medoids):
            labels[medoid_index] = cluster_id

        for sample_index in range(len(distances)):
            if labels[sample_index] != -1:
                continue

            # Nếu hòa khoảng cách, chọn cụm có số thứ tự nhỏ hơn để kết quả ổn định.
            labels[sample_index] = min(
                range(self.n_clusters),
                key=lambda cluster_id: (
                    distances[sample_index][medoids[cluster_id]],
                    cluster_id,
                ),
            )

        return labels

    def _members_by_cluster(self, labels: list[int]) -> list[list[int]]:
        clusters = [[] for _ in range(self.n_clusters)]
        for sample_index, cluster_id in enumerate(labels):
            clusters[cluster_id].append(sample_index)
        return clusters

    @staticmethod
    def _objective(
        distances: list[list[float]],
        labels: list[int],
        medoids: list[int],
    ) -> float:
        return sum(
            distances[sample_index][medoids[cluster_id]]
            for sample_index, cluster_id in enumerate(labels)
        )

    def _update_medoids(
        self,
        distances: list[list[float]],
        clusters: list[list[int]],
        old_medoids: list[int],
    ) -> list[int]:
        new_medoids = []

        for cluster_id, members in enumerate(clusters):
            if not members:
                # Không xảy ra bình thường vì mỗi medoid được gán vào cụm của nó.
                new_medoids.append(old_medoids[cluster_id])
                continue

            costs = {
                candidate: sum(
                    distances[candidate][other] for other in members
                )
                for candidate in members
            }
            best_cost = min(costs.values())
            tolerance = 1e-12 * max(1.0, abs(best_cost))
            best_candidates = [
                candidate
                for candidate, cost in costs.items()
                if abs(cost - best_cost) <= tolerance
            ]

            # Khi nhiều record đồng hạng, giữ medoid cũ nếu có thể;
            # nếu không, chọn record có chỉ số nhỏ nhất.
            old_medoid = old_medoids[cluster_id]
            if old_medoid in best_candidates:
                new_medoids.append(old_medoid)
            else:
                new_medoids.append(min(best_candidates))

        return new_medoids

