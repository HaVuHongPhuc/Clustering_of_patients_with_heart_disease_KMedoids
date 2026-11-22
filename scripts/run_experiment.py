"""Chạy so sánh K-Medoids trên cấu hình YAML và lưu một bộ kết quả riêng."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from shutil import rmtree
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import FEATURE_COLS, REFERENCE_COLS, TARGET_COL, DataQualityAnalyzer, HeartDataLoader
from src.evaluation import ClusterEvaluator
from src.kmedoids import KMedoids
from src.library_model import INIT, METHOD, fit_kmedoids
from src.preprocessing import DistanceCalculator, FeaturePreprocessor


def read_config(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Cấu hình YAML phải là một bảng khóa và giá trị.")

    expected = {
        "data_path", "feature_columns", "held_out_columns", "n_clusters",
        "random_state", "custom_max_iter", "primary_distance", "control_distance", "output_root",
    }
    missing = expected - settings.keys()
    unknown = settings.keys() - expected
    if missing or unknown:
        raise ValueError(f"Thiếu khóa {sorted(missing)}; khóa lạ {sorted(unknown, key=str)}.")
    if settings["feature_columns"] != FEATURE_COLS:
        raise ValueError("feature_columns phải khớp 12 đặc trưng của HeartDataLoader.")
    if settings["held_out_columns"] != [*REFERENCE_COLS, TARGET_COL]:
        raise ValueError("held_out_columns phải là [cp, target].")
    if (settings["primary_distance"], settings["control_distance"]) != ("manhattan", "gower"):
        raise ValueError("Cấu hình so sánh hỗ trợ Manhattan chính và Gower đối chứng.")
    for key, minimum in [("n_clusters", 2), ("random_state", 0), ("custom_max_iter", 1)]:
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f"{key} phải là số nguyên từ {minimum} trở lên.")
    for key in ("data_path", "output_root"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"{key} phải là đường dẫn không rỗng.")
    return settings


def score_model(data, metric, labels, reference) -> dict:
    result = {"silhouette": ClusterEvaluator.silhouette(data, labels, metric=metric)}
    # Silhouette không xác định nếu đầu ra chỉ có một cụm hữu hiệu.
    if not np.isfinite(result["silhouette"]):
        result["silhouette"] = None
    for column in reference.columns:
        scores = ClusterEvaluator.posthoc_check(labels, reference[column])
        result[f"ARI_{column}"] = scores["ARI"]
        result[f"purity_{column}"] = scores["purity"]
    return result


def fit_models(settings, features, matrix, reference):
    k = settings["n_clusters"]
    seed = settings["random_state"]
    primary = settings["primary_distance"]
    control = settings["control_distance"]

    custom = KMedoids(k, metric=primary, random_state=seed,
                      max_iter=settings["custom_max_iter"]).fit(matrix)
    library = fit_kmedoids(matrix, k, metric=primary, random_state=seed)
    gower = DistanceCalculator.gower(features)
    library_control = fit_kmedoids(gower, k, metric="precomputed", random_state=seed)

    rows = [{
        "model": "custom_manhattan", "method": "alternate", "init": "random",
        "distance": primary, "k": k, "seed": seed, "inertia": custom.inertia_,
        "n_iter": custom.n_iter_, "converged": custom.converged_,
        **score_model(matrix, primary, custom.labels_, reference),
    }]
    for name, distance, result, data, metric in [
        ("library_manhattan", primary, library, matrix, primary),
        ("library_gower", control, library_control, gower, "precomputed"),
    ]:
        rows.append({
            "model": name, "method": METHOD, "init": INIT, "distance": distance,
            "k": k, "seed": seed, "inertia": result.inertia,
            "n_iter": result.n_iter if result.n_iter >= 0 else None,
            "converged": None, **score_model(data, metric, result.labels, reference),
        })
    artifacts = {
        "custom_manhattan": (custom.labels_, custom.medoid_indices_),
        "library_manhattan": (library.labels, library.medoid_indices),
        "library_gower": (library_control.labels, library_control.medoid_indices),
    }
    return rows, artifacts, custom.history_


def write_results(folder, rows, artifacts, history, metadata):
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    agreements = []
    for a, b in [("custom_manhattan", "library_manhattan"),
                 ("library_manhattan", "library_gower")]:
        agreements.append({"model_a": a, "model_b": b, "ARI": ClusterEvaluator.compare_labelings(
            artifacts[a][0], artifacts[b][0])})
    pd.DataFrame(agreements).to_csv(folder / "agreement.csv", index=False)
    for name, (labels, medoids) in artifacts.items():
        destination = folder / name
        destination.mkdir()
        pd.DataFrame({"record_index": np.arange(len(labels)), "cluster": labels}).to_csv(
            destination / "labels.csv", index=False)
        pd.DataFrame({"cluster": np.arange(len(medoids)), "medoid_index": medoids}).to_csv(
            destination / "medoids.csv", index=False)
    pd.DataFrame(history).to_csv(folder / "custom_manhattan/history.csv", index=False)
    (folder / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, output_dir: Path | None) -> Path:
    settings = read_config(ROOT / config_path)
    start = datetime.now(timezone.utc)
    run_id = start.strftime("%Y%m%dT%H%M%S%fZ")
    destination = (ROOT / output_dir if output_dir is not None else
                   ROOT / settings["output_root"] / run_id).resolve()
    if destination.exists():
        raise FileExistsError(f"Thư mục kết quả đã tồn tại: {destination}")

    data_path = ROOT / settings["data_path"]
    data_digest = sha256(data_path.read_bytes()).hexdigest()
    loader = HeartDataLoader(data_path)
    clean = loader.clean()
    if data_digest != sha256(data_path.read_bytes()).hexdigest():
        raise ValueError("Tệp dữ liệu đã thay đổi trong lúc đọc. Hãy chạy lại.")
    features, _ = loader.split_features_target(clean)
    reference = loader.split_reference(clean)
    if settings["n_clusters"] >= len(features):
        raise ValueError("Số cụm phải nhỏ hơn số hồ sơ để tính Silhouette.")
    matrix, names, _ = FeaturePreprocessor().fit_transform(features)
    rows, artifacts, history = fit_models(settings, features, matrix, reference)
    metadata = {
        **settings,
        "run_id": run_id,
        "source_filter": None,
        "data_sha256": data_digest,
        "n_rows": len(clean),
        "n_features_raw": len(features.columns),
        "n_features_manhattan": len(names),
        "manhattan_columns": names,
        "fill_categorical": loader.fill_categorical,
        "fill_numeric": loader.fill_numeric,
        "unexpected_codes_after_harmonization": DataQualityAnalyzer.report(loader.raw)["unexpected_values"],
        "custom_method": "alternate",
        "custom_init": "random",
        "library_method": METHOD,
        "library_init": INIT,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "scikit-learn", "kmedoids", "PyYAML")
        }},
    }

    # Hoàn tất các tệp trước khi công bố thư mục lần chạy.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".experiment-{uuid4().hex}"
    staging.mkdir()
    try:
        write_results(staging, rows, artifacts, history, metadata)
        if destination.exists():
            raise FileExistsError(f"Thư mục kết quả đã tồn tại: {destination}")
        staging.rename(destination)
    finally:
        # Chỉ dọn thư mục tạm do lần gọi này tạo, trong đúng thư mục đầu ra.
        if staging.exists() and staging.resolve().parent == destination.parent:
            rmtree(staging)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"),
                        help="Cấu hình YAML. Đường dẫn tương đối tính từ gốc dự án.")
    parser.add_argument("--output-dir", type=Path, help="Thư mục kết quả mới của lần chạy.")
    args = parser.parse_args()
    try:
        destination = run(args.config, args.output_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(status=2, message=f"Lỗi thực nghiệm: {exc}\n")
    print(f"Đã lưu kết quả thực nghiệm tại: {destination}")


if __name__ == "__main__":
    main()
