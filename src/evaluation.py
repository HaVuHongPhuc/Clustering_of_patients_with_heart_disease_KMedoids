from __future__ import annotations
"""Đánh giá phân cụm (OOP): chọn k bằng silhouette, kiểm tra hậu nghiệm.

Class chính:
- ClusterEvaluator : silhouette, select_k, posthoc check, profile, compare.
- MetricsRunner    : chạy k=2..8 cho Manhattan & Gower, lưu outputs/metrics/.
"""

import json
import platform
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .data import NUMERIC_COLS, TARGET_COL, HeartDataLoader
from .library_model import INIT, METHOD, RANDOM_STATE, KMedoidsResult, fit_kmedoids
from .preprocessing import DistanceCalculator, FeaturePreprocessor

K_RANGE = range(2, 9)


class ClusterEvaluator:
    """Đánh giá nội tại (silhouette) và ngoại tại (ARI, purity vs target/cp)."""

    @staticmethod
    def silhouette(data: np.ndarray, labels: np.ndarray, metric: str = "manhattan") -> float:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(data, labels, metric=metric))

    @classmethod
    def select_k(
        cls,
        data: np.ndarray,
        metric: str = "manhattan",
        k_range=K_RANGE,
        random_state: int = RANDOM_STATE,
    ) -> tuple[int, pd.DataFrame, dict[int, KMedoidsResult]]:
        rows, results = [], {}
        for k in k_range:
            res = fit_kmedoids(data, k, metric=metric, random_state=random_state)
            results[k] = res
            sizes = np.bincount(res.labels, minlength=k)
            rows.append({
                "k": k,
                "silhouette": cls.silhouette(data, res.labels, metric=metric),
                "inertia": res.inertia,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
            })
        table = pd.DataFrame(rows).set_index("k")
        best_k = int(table["silhouette"].idxmax())
        return best_k, table, results

    @staticmethod
    def cluster_sizes(labels: np.ndarray) -> pd.Series:
        return pd.Series(labels).value_counts().sort_index().rename("n")

    @staticmethod
    def target_crosstab(labels: np.ndarray, target: pd.Series) -> pd.DataFrame:
        ct = pd.crosstab(pd.Series(labels, name="cluster"),
                         pd.Series(target.to_numpy(), name=TARGET_COL))
        ct["tỉ_lệ_bệnh"] = ct.get(1, 0) / ct.sum(axis=1)
        return ct

    @staticmethod
    def cp_crosstab(labels: np.ndarray, cp: pd.Series) -> pd.DataFrame:
        ct = pd.crosstab(pd.Series(labels, name="cluster"),
                         pd.Series(cp.to_numpy(), name="cp"))
        share = ct.div(ct.sum(axis=1), axis=0).add_prefix("tỉ_lệ_cp_")
        return pd.concat([ct, share.round(3)], axis=1)

    @staticmethod
    def posthoc_check(labels: np.ndarray, target: pd.Series) -> dict:
        y = target.to_numpy()
        ct = pd.crosstab(labels, y)
        purity = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
        return {"ARI": float(adjusted_rand_score(y, labels)), "purity": purity}

    @staticmethod
    def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        d = df.copy()
        d["cluster"] = labels
        g = d.groupby("cluster")
        prof = g[NUMERIC_COLS].mean().round(2)
        for col in ["sex", "fbs", "exang"]:
            prof[f"{col}_rate"] = g[col].mean().round(3)
        if "cp" in d.columns:
            for v in sorted(d["cp"].dropna().unique()):
                prof[f"cp{int(v)}_rate"] = g["cp"].apply(lambda s, v=v: (s == v).mean()).round(3)
        if TARGET_COL in d.columns:
            prof["target_rate"] = g[TARGET_COL].mean().round(3)
        prof.insert(0, "n", g.size())
        return prof

    @staticmethod
    def compare_labelings(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
        return float(adjusted_rand_score(labels_a, labels_b))


class MetricsRunner:
    """Chạy k=2..8 cho Manhattan (chính) và Gower (đối chứng), lưu outputs/metrics/."""

    def __init__(self, out_dir, preprocessor: FeaturePreprocessor | None = None,
                 distance_calc: DistanceCalculator | None = None,
                 evaluator: ClusterEvaluator | None = None):
        self.out = Path(out_dir)
        self.pre = preprocessor or FeaturePreprocessor()
        self.dist = distance_calc or DistanceCalculator()
        self.eval = evaluator or ClusterEvaluator()

    @staticmethod
    def _safe_version(pkg_name: str) -> str:
        try:
            return _pkg_version(pkg_name)
        except PackageNotFoundError:
            return "unknown"

    def run(self, df: pd.DataFrame, extra_config: dict | None = None) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)

        X, y = HeartDataLoader.split_features_target(df)
        cp = HeartDataLoader.split_reference(df)["cp"]
        X_mat, _, _ = self.pre.fit_transform(X)
        D_gower = self.dist.gower(X)
        inputs = {"manhattan": (X_mat, "manhattan"), "gower": (D_gower, "precomputed")}

        rows, label_cols, medoid_rows, results_by = [], {}, [], {}
        for name, (data, metric) in inputs.items():
            best_k, table, results = self.eval.select_k(data, metric=metric)
            results_by[name] = results
            for k, res in results.items():
                ext = self.eval.posthoc_check(res.labels, y)
                ext_cp = self.eval.posthoc_check(res.labels, cp)
                row = table.loc[k].to_dict()
                row.update(
                    distance=name, k=k, n_iter=res.n_iter, is_best_k=(k == best_k),
                    ARI_target=ext["ARI"], purity_target=ext["purity"],
                    ARI_cp=ext_cp["ARI"], purity_cp=ext_cp["purity"],
                )
                rows.append(row)
                label_cols[f"{name}_k{k}"] = res.labels
                for c, idx in enumerate(res.medoid_indices):
                    medoid_rows.append({"distance": name, "k": k, "cluster": c,
                                        "medoid_index": int(idx)})

        cols = ["distance", "k", "silhouette", "inertia", "min_cluster_size",
                "max_cluster_size", "n_iter", "ARI_target", "purity_target",
                "ARI_cp", "purity_cp", "is_best_k"]
        ksel = pd.DataFrame(rows)[cols]
        cross = pd.DataFrame({
            "k": list(K_RANGE),
            "ARI_manhattan_vs_gower": [
                self.eval.compare_labelings(results_by["manhattan"][k].labels,
                                            results_by["gower"][k].labels)
                for k in K_RANGE]
        })

        paths = {
            "k_selection": self.out / "k_selection.csv",
            "labels": self.out / "labels.csv",
            "medoids": self.out / "medoids.csv",
            "cross_method_ari": self.out / "cross_method_ari.csv",
            "run_config": self.out / "run_config.json",
        }
        ksel.to_csv(paths["k_selection"], index=False)
        pd.DataFrame(label_cols).to_csv(paths["labels"], index=False)
        pd.DataFrame(medoid_rows).to_csv(paths["medoids"], index=False)
        cross.to_csv(paths["cross_method_ari"], index=False)

        config = {
            "n_rows": int(len(df)),
            "n_features_manhattan": int(X_mat.shape[1]),
            "k_range": list(K_RANGE),
            "random_state": RANDOM_STATE,
            "method": METHOD,
            "init": INIT,
            "n_features_raw": 12,
            "held_out_columns": ["cp", "target"],
            "main_distance": "manhattan (StandardScaler + OneHot + binary giữ nguyên; không có cp)",
            "control_distance": "gower (12 cột gốc, không có cp, KMedoids metric='precomputed')",
            "best_k": {n: int(ksel[(ksel.distance == n) & ksel.is_best_k].k.iloc[0])
                       for n in inputs},
            "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                         "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                         "kmedoids": self._safe_version("kmedoids")},
        }
        if extra_config:
            config.update(extra_config)
        paths["run_config"].write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        return paths