"""Kiểm thử K-Medoids tự cài đặt bằng unittest."""

import unittest

from src.kmedoids import KMedoids


class KMedoidsTests(unittest.TestCase):
    def test_assignment_and_medoid_update_by_hand(self):
        # Hai nhóm [0, 2, 3] và [10, 11, 13]. Sau cập nhật, mỗi nhóm có chi phí 3.
        data = [[0], [2], [3], [10], [11], [13]]
        model = KMedoids(2, metric="manhattan", initial_medoids=[0, 3]).fit(data)

        self.assertEqual(model.labels_, [0, 0, 0, 1, 1, 1])
        self.assertEqual(model.medoid_indices_, [1, 4])
        self.assertEqual(model.medoids_, [[2.0], [11.0]])
        self.assertAlmostEqual(model.history_[0]["objective"], 9.0)
        self.assertAlmostEqual(model.inertia_, 6.0)
        self.assertTrue(model.converged_)

    def test_same_seed_repeats_initial_medoids_and_result(self):
        data = [[i, (i * 7) % 11] for i in range(20)]
        first = KMedoids(3, metric="manhattan", random_state=42).fit(data)
        second = KMedoids(3, metric="manhattan", random_state=42).fit(data)

        self.assertEqual(first.history_[0]["medoid_indices"], second.history_[0]["medoid_indices"])
        self.assertEqual(first.medoid_indices_, second.medoid_indices_)
        self.assertEqual(first.labels_, second.labels_)
        self.assertAlmostEqual(first.inertia_, second.inertia_)

    def test_invalid_fit_data_is_rejected(self):
        for data in ([], [[], []], [[1], [2, 3]], [[float("nan")]], [[float("inf")]], [["x"]]):
            with self.subTest(data=data), self.assertRaises((TypeError, ValueError)):
                KMedoids(1, metric="manhattan").fit(data)

    def test_invalid_cluster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            KMedoids(0)
        with self.assertRaises(TypeError):
            KMedoids(True)
        with self.assertRaises(ValueError):
            KMedoids(3).fit([[0], [1]])

    def test_invalid_initial_medoids_are_rejected(self):
        for indices in ([0], [0, 0], [0, 3]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                KMedoids(2, initial_medoids=indices).fit([[0], [1], [2]])

    def test_predict_validates_model_and_feature_count(self):
        model = KMedoids(2, metric="manhattan", initial_medoids=[0, 2])
        with self.assertRaises(RuntimeError):
            model.predict([[0, 0]])
        data = [[0, 0], [1, 0], [10, 10]]
        self.assertEqual(model.fit_predict(data), model.predict(data))
        self.assertEqual(model.predict([[11, 10]]), [1])
        with self.assertRaises(ValueError):
            model.predict([[10]])
