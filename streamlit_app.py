"""Dashboard Streamlit cho thực nghiệm K-Medoids trên toàn bộ dữ liệu."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import pairwise_distances

from src.evaluation import ClusterEvaluator
from src.kmedoids import KMedoids
from src.preprocessing import DistanceCalculator


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs/metrics"
EXPERIMENTS = ROOT / "outputs/experiments"
MATRIX = ROOT / "data/processed/X_manhattan.csv"
MODEL_NAMES = {
    "custom_manhattan": "Tự cài đặt · Manhattan",
    "library_manhattan": "Thư viện · Manhattan",
    "library_gower": "Thư viện · Gower",
}


def record(index: int) -> str:
    return f"R{index + 1}"


@st.cache_data(max_entries=4, show_spinner=False)
def fit_full(matrix, seed, initial_medoids):
    model = KMedoids(2, metric="manhattan", random_state=seed,
                     initial_medoids=initial_medoids).fit(matrix)
    return {
        "history": model.history_,
        "labels": model.labels_,
        "medoids": model.medoid_indices_,
        "inertia": model.inertia_,
        "sizes": model.cluster_sizes_,
        "converged": model.converged_,
        "silhouette": ClusterEvaluator.silhouette(matrix, model.labels_),
    }


def show_full_data():
    st.header("Phân cụm theo số hồ sơ")
    st.write("Chạy K-Medoids tự cài đặt với Manhattan và k=2, rồi xem mỗi hồ sơ được gán vào cụm nào.")
    if not MATRIX.is_file():
        st.info("Chạy Notebook 01 để tạo data/processed/X_manhattan.csv.")
        return
    frame = pd.read_csv(MATRIX)
    if frame.shape != (2283, 21) or any(
        c in {"cp", "target", "source"} or c.startswith("cp_") for c in frame.columns
    ):
        st.warning(f"Cần ma trận 2.283 hồ sơ × 21 cột từ Notebook 01. "
                   f"Tệp đang có {len(frame):,} dòng × {frame.shape[1]} cột.")
        return
    all_records = frame.to_numpy(dtype=float)
    if not np.isfinite(all_records).all():
        st.warning("Ma trận Manhattan chứa giá trị còn thiếu hoặc vô hạn.")
        return

    total = len(all_records)
    count = int(st.number_input("Số hồ sơ dùng để phân cụm", min_value=3, max_value=total,
                                value=total, step=1, key="sample_count",
                                help="Dùng các hồ sơ đầu tiên theo thứ tự Notebook 02. Cần ít nhất 3 hồ sơ để tính Silhouette với k=2."))
    st.caption(f"Mẫu chọn gồm {count:,} dòng đầu, từ R1 đến R{count}.")
    mode = st.radio("Cách chọn medoid ban đầu", ("Dùng seed", "Tự chọn hai medoid"),
                    horizontal=True, key="full_init_mode")
    with st.form("full_experiment"):
        if mode == "Dùng seed":
            seed = int(st.number_input("Seed", min_value=0, max_value=2**32-1, value=42,
                                      key="full_seed"))
            initial = None
        else:
            seed = 42
            left, right = st.columns(2)
            for key in ("full_medoid_0", "full_medoid_1"):
                if st.session_state.get(key, 0) >= count:
                    st.session_state.pop(key)
            first = left.selectbox("Medoid ban đầu của cụm 0", range(count),
                                   format_func=record, key="full_medoid_0")
            second = right.selectbox("Medoid ban đầu của cụm 1", range(count), index=min(15, count - 1),
                                     format_func=record, key="full_medoid_1")
            initial = (first, second)
        submitted = st.form_submit_button("Phân cụm", type="primary")
    if submitted:
        if initial is not None and initial[0] == initial[1]:
            st.warning("Chọn hai hồ sơ khác nhau làm medoid.")
        else:
            st.session_state["sample_settings"] = (count, seed, initial)
            st.session_state["full_round"] = "Kết quả cuối"
            st.session_state["full_cluster"] = "Tất cả"

    applied_count, applied_seed, applied_initial = st.session_state.get(
        "sample_settings", (total, 42, None)
    )
    matrix = all_records[:applied_count]
    with st.spinner(f"Đang phân cụm {applied_count:,} hồ sơ…"):
        result = fit_full(matrix, applied_seed, applied_initial)
    history = result["history"]
    initialization = (f"Seed {applied_seed}" if applied_initial is None else
                      "Medoid chỉ định: " + " và ".join(map(record, applied_initial)))
    st.caption(f"Lần chạy đang hiển thị: {initialization} · {len(matrix):,} hồ sơ · {matrix.shape[1]} cột Manhattan")
    before, after, silhouette = st.columns(3)
    before.metric("Tổng khoảng cách lúc đầu", f"{history[0]['objective']:.6f}")
    after.metric("Tổng khoảng cách cuối", f"{result['inertia']:.6f}")
    silhouette.metric("Silhouette", f"{result['silhouette']:.6f}")

    timeline = pd.DataFrame([
        {"Vòng": step["iteration"], "Tổng khoảng cách": step["objective"]}
        for step in history
    ])
    chart, summary = st.columns([2, 1])
    with chart:
        st.line_chart(timeline, x="Vòng", y="Tổng khoảng cách",
                      color="#C2410C", height=260)
    with summary:
        with st.container(border=True):
            st.write("**Medoid cuối**")
            st.write(" · ".join(map(record, result["medoids"])))
            sizes = " / ".join(f"{size:,}" for size in result["sizes"])
            st.caption(f"Kích thước hai cụm: {sizes}")
            status = ("Đã hội tụ" if result["converged"] else "Đã đến số vòng lặp tối đa")
            st.caption(f"{status} · {len(history)} vòng")

    rounds = {step["iteration"]: step for step in history}
    stages = ["Kết quả cuối", *rounds]
    if st.session_state.get("full_round") not in stages:
        st.session_state["full_round"] = stages[0]
    options, filter_col = st.columns(2)
    selected = options.selectbox("Xem kết quả gán cụm", stages, key="full_round",
                                 format_func=lambda value: f"Vòng {value}" if isinstance(value, int) else value)
    cluster_filter = filter_col.selectbox("Lọc theo cụm", ("Tất cả", "Cụm 0", "Cụm 1"),
                                          key="full_cluster")

    step = rounds.get(selected)
    current = list(step["medoid_indices"] if step else result["medoids"])
    to_medoids = pairwise_distances(matrix, matrix[current], metric="manhattan")
    if step:
        updated = " và ".join(map(record, step["updated_medoid_indices"]))
        st.write(f"Medoid sau cập nhật ở vòng **{selected}** là **{updated}**.")
        assignments = to_medoids.argmin(axis=1)
        assignments[current] = np.arange(2)
    else:
        assignments = np.asarray(result["labels"])
    table = pd.DataFrame({
        "Hồ sơ": [record(i) for i in range(len(matrix))],
        **{f"Đến {record(i)}": to_medoids[:, c] for c, i in enumerate(current)},
        "Cụm": assignments,
        "Medoid được gán": [record(current[c]) for c in assignments],
        "Khoảng cách được cộng": to_medoids[np.arange(len(matrix)), assignments],
    })
    if cluster_filter != "Tất cả":
        table = table.loc[table["Cụm"].eq(int(cluster_filter[-1]))]
    st.caption(f"Đang hiển thị {len(table):,}/{len(matrix):,} hồ sơ.")
    st.dataframe(table.round(6), hide_index=True, width="stretch", height=440)

    if step:
        with st.expander("Tổng khoảng cách của từng ứng viên medoid"):
            if st.checkbox("Tính tổng khoảng cách nội cụm", key="full_candidates"):
                candidates = []
                for group in range(2):
                    members = np.flatnonzero(assignments == group)
                    costs = DistanceCalculator.manhattan(matrix[members]).sum(axis=1)
                    candidates.extend({
                        "Cụm": group, "Ứng viên": record(i), "Tổng khoảng cách trong cụm": cost,
                        "Được chọn": i == step["updated_medoid_indices"][group],
                    } for i, cost in zip(members, costs))
                results = pd.DataFrame(candidates).sort_values(["Cụm", "Tổng khoảng cách trong cụm"])
                st.dataframe(results.round(6), hide_index=True, width="stretch", height=370)


def show_sweep():
    st.header("Khảo sát số cụm")
    scores = METRICS / "k_selection.csv"
    if not scores.is_file():
        st.info("Chạy Notebook 01 để tạo kết quả khảo sát số cụm.")
        return
    metrics = pd.read_csv(scores)
    distance = st.radio("Khoảng cách", ("manhattan", "gower"), horizontal=True,
                        format_func=str.capitalize, key="distance")
    rows = metrics.loc[metrics["distance"].eq(distance)].sort_values("k")
    best = rows.loc[rows["silhouette"].idxmax()]
    st.metric("k có Silhouette cao nhất trong dải đã thử", int(best["k"]),
              help=f"Silhouette = {best['silhouette']:.6f}")
    left, right = st.columns(2)
    with left:
        st.write("**Silhouette theo số cụm**")
        st.line_chart(rows, x="k", y="silhouette", color="#C2410C")
    with right:
        st.write("**Tổng khoảng cách đến medoid**")
        st.line_chart(rows, x="k", y="inertia", color="#A12E24")
    st.dataframe(rows[["k", "silhouette", "inertia", "min_cluster_size", "max_cluster_size"]]
                 .rename(columns={
                     "silhouette": "Silhouette", "inertia": "Inertia",
                     "min_cluster_size": "Cụm nhỏ nhất", "max_cluster_size": "Cụm lớn nhất",
                 }).round(6), hide_index=True, width="stretch")
    st.caption("Nguồn kết quả: Notebook 01, K-Medoids thư viện PAM/BUILD.")


def show_comparison():
    st.header("So sánh lần chạy")
    runs = sorted((file.parent.name for file in EXPERIMENTS.glob("*/comparison.csv")
                   if (file.parent / "run_config.json").is_file()), reverse=True)
    if not runs:
        st.info("Chạy mục 13 của Notebook 02 để tạo kết quả so sánh.")
        return
    selected = st.selectbox("Lần chạy đã lưu", runs, key="experiment_run")
    folder = EXPERIMENTS / selected
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(folder / "comparison.csv")
    st.caption(f"{config['n_rows']:,} hồ sơ · k={config['n_clusters']} · seed={config['random_state']}")
    columns = {
        "model": "Mô hình", "silhouette": "Silhouette", "inertia": "Inertia",
        "ARI_target": "ARI target", "purity_target": "Purity target",
        "ARI_cp": "ARI cp", "purity_cp": "Purity cp",
    }
    summary = comparison[list(columns)].rename(columns=columns)
    summary["Mô hình"] = summary["Mô hình"].replace(MODEL_NAMES)
    st.dataframe(summary.round(6), hide_index=True, width="stretch")
    st.caption("So sánh Inertia trong cùng một khoảng cách. Các chỉ số theo cp dùng mã sau làm sạch.")


st.set_page_config(page_title="K-Medoids · Khám phá kết quả", layout="wide")
st.title("Khám phá K-Medoids")
page = st.radio("Nội dung", ("Thử phân cụm", "Khảo sát số cụm", "So sánh lần chạy"),
                horizontal=True, label_visibility="collapsed", key="view")
if page == "Thử phân cụm":
    show_full_data()
elif page == "Khảo sát số cụm":
    show_sweep()
else:
    show_comparison()
