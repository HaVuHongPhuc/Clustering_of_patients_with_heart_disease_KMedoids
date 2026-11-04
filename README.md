# Phân cụm bệnh nhân bị bệnh tim bằng K-Medoids

## Trạng thái hiện tại

Đây là dự án đang triển khai: loader/tiền xử lý, mô hình K-Medoids tự cài đặt, adapter K-Medoids thư viện, pipeline metric thư viện và Notebooks 1–2 đã có mã chạy. Pipeline điều phối so sánh chung và test tự động cho mô hình tự cài đặt vẫn cần hoàn thiện; xem `CHECKLIST.md` để theo dõi trạng thái.

Nhóm đã chốt dùng 13 đặc trưng, giữ `target` riêng để đánh giá hậu nghiệm, Manhattan làm khoảng cách chính, Gower làm đối chứng, `k=2` và seed `42`. Dữ liệu được tải từ [Kaggle — Heart Disease Multi-Source Dataset](https://www.kaggle.com/datasets/alifsathar/heart-disease-multi-source-dataset), đăng bởi Alif Sathar, phiên bản 1 (phát hành 30/04/2026), giấy phép CC BY-SA 4.0. Nhóm xác nhận có bằng chứng về ngày tải; dataset page có mô tả và data dictionary. Quy và Phúc xác nhận giữ tất cả các nguồn trong file tổng hợp; Quy đã xác nhận quy tắc harmonize theo từng `source`.

## Mục tiêu dự án

Phân cụm hồ sơ bệnh nhân bằng K-Medoids, cài đặt thuật toán từ đầu và đối chiếu với một triển khai từ thư viện. Báo cáo cần mô tả nguồn dữ liệu, tiền xử lý, lựa chọn số cụm, thực nghiệm, đánh giá và hạn chế.

## Cấu trúc và vai trò tệp

| Đường dẫn | Vai trò và hướng thực hiện |
|---|---|
| `README.md` | Tài liệu trung tâm: cách tổ chức dự án, kế hoạch triển khai, dữ liệu, thực nghiệm và trạng thái. Cập nhật khi quyết định hoặc cách chạy thay đổi. |
| `HD Thực hiện đồ án cuối kỳ môn Học máy.pdf` | Bản hướng dẫn môn học được cung cấp; dùng để đối chiếu yêu cầu đồ án, không chỉnh sửa. |
| `MauBaoCaoHocMay.docx` | Mẫu báo cáo được cung cấp; dùng làm chuẩn cấu trúc báo cáo, không chỉnh sửa trong phần scaffold. |
| `requirements.txt` | Khai báo dependency runtime và notebook: NumPy, pandas, scikit-learn, kmedoids, Matplotlib, Seaborn, ipykernel. |
| `configs/experiment.yaml` | Mẫu cấu hình; cần chuyển thành cấu hình chạy cuối đã chốt (13 feature, target tách riêng, Manhattan/Gower, k=2, seed=42). |
| `data/raw/` | Có file tổng hợp `heart_disease_whole.csv` từ Kaggle; nhóm xác nhận dùng tất cả nguồn trong file và đã ghi nhận quy tắc harmonize theo từng `source`. File có 2.284 dòng vật lý tính cả header (2.283 bản ghi dữ liệu). |
| `data/processed/` | Đang có `heart_dedup.csv`, `X_manhattan.csv`, `target.csv` và Gower matrix; các artifact được xác nhận cùng 2.283 dòng. |
| `notebooks/01_data_exploration.ipynb` | Đã có EDA, kiểm tra chất lượng, quyết định xử lý và lưu dữ liệu processed/kết quả thư viện. |
| `notebooks/02_experiment_analysis.ipynb` | Đã có phân tích kết quả thư viện, biểu đồ, hồ sơ cụm/medoid và đối chiếu target hậu nghiệm. Chưa có kết quả bản tự cài đặt để so sánh. |
| `src/__init__.py` | Đánh dấu `src` là package Python; hiện chỉ có chú thích. |
| `src/data.py` | Đang đọc/kiểm tra dữ liệu, harmonize encoding, đánh dấu giá trị bất thường và xử lý missing values. |
| `src/preprocessing.py` | Đang tạo biểu diễn Manhattan và ma trận khoảng cách Gower; target tách khỏi feature input. |
| `src/kmedoids.py` | Đã có K-Medoids alternate tự cài đặt; đã kiểm tra ví dụ tính tay, cần bổ sung test tự động và thực nghiệm trên dữ liệu dự án. |
| `src/library_model.py` | Adapter đang chạy với package `kmedoids` (PAM/BUILD); hỗ trợ Manhattan trên ma trận đặc trưng và Gower qua ma trận khoảng cách `precomputed`. |
| `src/evaluation.py` | Đang chạy sweep k, tính silhouette/objective, kích thước cụm và đánh giá target hậu nghiệm; ghi labels, medoids và run config dưới `outputs/metrics/`. |
| `src/visualization.py` | Chưa có hàm vẽ dùng lại; biểu đồ hiện được tạo trong Notebook 2. |
| `scripts/run_experiment.py` | Chưa tích hợp; mục tiêu là điều phối cấu hình, hai mô hình, đánh giá và lưu kết quả. |
| `tests/test_kmedoids.py` | Hiện còn ghi chú; cần viết và chạy test tự động cho thuật toán tự cài đặt. |
| `outputs/figures/` | Đang có ba hình phân tích kết quả thư viện trong Notebook 2. |
| `outputs/metrics/` | Đang có sweep thư viện Manhattan/Gower, k=2..8, seed 42; chưa có kết quả mô hình tự cài đặt. |

## Phân công công việc của Phúc và Quy

Phân công dưới đây theo yêu cầu của nhóm.

| Người | Việc cụ thể | Tệp/thư mục phụ trách | Phần báo cáo |
|---|---|---|---|
| **Phúc** | Tự cài đặt K-Medoids: khởi tạo, gán cụm, cập nhật medoid, điều kiện dừng và trường hợp biên; tích hợp pipeline; chạy cấu hình thực nghiệm, ghi kết quả; phân tích cụm và tạo biểu đồ. | `src/kmedoids.py`, `tests/test_kmedoids.py`, `scripts/run_experiment.py`, `src/visualization.py`, `notebooks/02_experiment_analysis.ipynb`, `configs/experiment.yaml`; kết quả trong `outputs/metrics/` và `outputs/figures/` | K-Medoids tự cài đặt và quy trình chạy ở Chương 4; phân tích cụm, biểu đồ và kết quả thực nghiệm ở Chương 5 |
| **Quy** | Chọn và mô tả nguồn dữ liệu; EDA; tiền xử lý; cài đặt K-Medoids bằng thư viện; tính metric. | `data/raw/` nếu giấy phép cho phép, `src/data.py`, `src/preprocessing.py`, `notebooks/01_data_exploration.ipynb`, `src/library_model.py`, `src/evaluation.py`, `requirements.txt` | Chuẩn bị dữ liệu ở Chương 3; mô hình thư viện ở Chương 4; định nghĩa và cách tính metric ở Chương 5 |
| **Cùng làm** | Thống nhất dữ liệu, đặc trưng, cách xử lý nhãn bệnh, khoảng cách, candidate `k`, seed và quy trình so sánh; rà soát báo cáo, slide và chuẩn bị vấn đáp. | `README.md`, cấu hình thực nghiệm (Phúc cập nhật, Quy kiểm tra), báo cáo và slide | Chương 1, 2, 6; rà soát toàn báo cáo và slide |

### Handoff giữa hai người

1. Quy xác định nguồn dữ liệu, schema, cột đặc trưng và tiền xử lý; Phúc thống nhất với Quy giao diện dữ liệu mà `src/kmedoids.py` cần nhận.
2. Quy triển khai loader/preprocessing và bản thư viện; Phúc có thể phát triển thuật toán bằng dữ liệu nhỏ trước khi dữ liệu thật sẵn sàng.
3. Quy cung cấp dữ liệu đã xử lý và hàm tính metric; Phúc tích hợp pipeline, điền cấu hình đã thống nhất và chạy các thực nghiệm.
4. Phúc lưu kết quả; Quy kiểm tra metric; Phúc dùng kết quả đó phân tích cụm và tạo biểu đồ. Hai phiên bản phải dùng cùng đầu vào, metric khoảng cách, `k` và seed khi thư viện hỗ trợ.

RTX 5070 Ti không tự làm K-Medoids nhanh hơn. Phúc benchmark CPU trước; chỉ dùng GPU nếu quy mô dữ liệu và thư viện/cách cài đặt thật sự tận dụng được.

## Hướng dẫn triển khai theo thứ tự

1. **Chốt dữ liệu và câu hỏi phân cụm.** Chọn bộ dữ liệu có nguồn đáng tin cậy, ghi URL/phiên bản/ngày tải/giấy phép và mô tả chính xác mỗi hàng đại diện cho gì. Ghi số mẫu, số thuộc tính, kiểu biến và cột bị thiếu.
2. **Khám phá dữ liệu.** Dùng `notebooks/01_data_exploration.ipynb` để khảo sát missing values, trùng lặp, phạm vi giá trị, phân phối và biến phân loại. Không biến notebook thành nơi duy nhất chứa logic sẽ cần dùng lại.
3. **Chốt cách biểu diễn khoảng cách.** K-Medoids dựa trên khoảng cách; cần quyết định cách xử lý biến số, biến phân loại, missing values, scaling và metric phù hợp. Ghi rõ các quyết định trong notebook và `configs/experiment.yaml` khi được chốt.
4. **Tách nhãn bệnh khỏi đầu vào phân cụm.** Nếu dữ liệu có nhãn chẩn đoán/đầu ra, mặc định không đưa cột này vào `X` khi fit unsupervised. Có thể giữ nhãn riêng để diễn giải hoặc đánh giá bên ngoài sau khi cụm đã được tạo; nêu rõ giới hạn khi làm vậy.
5. **Cài đặt tiền xử lý dùng chung.** Viết logic tái sử dụng trong `src/preprocessing.py`, đảm bảo hai phiên bản K-Medoids nhận cùng cách biểu diễn và cùng điều kiện thực nghiệm.
6. **Cài đặt bản tự viết.** Triển khai từng bước thuật toán trong `src/kmedoids.py`. Ghi rõ hàm mục tiêu, cách khởi tạo medoid, cách cập nhật, tiêu chí dừng, xử lý trường hợp hòa/điểm trùng và seed.
7. **Cài đặt bản thư viện.** Chọn thư viện sau khi xác minh API và khả năng hỗ trợ kiểu dữ liệu/metric cần dùng. Ghi phiên bản vào `requirements.txt`; đặt phần gọi thư viện trong `src/library_model.py`.
8. **Chọn `k` và đánh giá.** Thử nhiều `k`, lưu cấu hình, objective/dissimilarity và các metric nội bộ phù hợp (ví dụ silhouette nếu định nghĩa khoảng cách cho phép). Không dùng một chỉ số duy nhất làm kết luận nếu không giải thích ý nghĩa và giới hạn.
9. **So sánh và diễn giải.** Dùng `notebooks/02_experiment_analysis.ipynb` để so sánh bản tự viết với thư viện, kiểm tra kích thước cụm và mô tả đặc trưng các medoid/cụm. Nếu dùng nhãn bệnh để đối chiếu thì đây là đánh giá hậu nghiệm, không phải đầu vào fit.
10. **Lưu đầu ra có thể tái lập.** Tạo hình trong `outputs/figures/`, metric/cấu hình chạy trong `outputs/metrics/`. Đảm bảo ghi seed, phiên bản thư viện, tiền xử lý, metric khoảng cách và `k` cho mỗi kết quả.
11. **Hoàn thiện deliverable môn học.** Báo cáo PDF theo mẫu sáu chương, source code và slide là các hạng mục riêng. Có thể tạo `deliverables/` sau nếu muốn lưu báo cáo/slide trong cùng repo; chưa tạo thư mục này.

## Dữ liệu và quyền riêng tư

Không commit dữ liệu bệnh nhân hoặc dữ liệu có thể nhận diện cá nhân nếu chưa xác minh quyền sử dụng và điều kiện chia sẻ. Dữ liệu gốc trong `data/raw/` được giữ nguyên; ghi cách tiền xử lý để kết quả có thể tái lập. `.gitignore` đang loại nội dung dữ liệu và output khỏi Git; chỉ đưa các tệp cần chia sẻ lên Git sau khi kiểm tra giấy phép và quyền riêng tư.

## Môi trường Python

Dự án đã được khởi tạo `.venv/` bằng Python 3.12.10. Tạo lại môi trường trên Windows (PowerShell) nếu cần:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kích hoạt môi trường trong PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trong VS Code, mở Command Palette → **Python: Select Interpreter** → chọn `.venv\Scripts\python.exe`. Với notebook, chọn kernel `.venv` tương ứng. Có thể đăng ký kernel với tên rõ ràng bằng:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name heart-kmedoids --display-name "Python (.venv - Heart K-Medoids)"
```

Thư viện hiện khai báo trong `requirements.txt`: NumPy, pandas, scikit-learn, kmedoids, Matplotlib, Seaborn và ipykernel. `.venv/` được `.gitignore` loại khỏi Git; chỉ commit `requirements.txt`, không commit toàn bộ môi trường.

Notebook 1, Notebook 2, các module loader/preprocessing/library evaluation và thuật toán tự cài đặt hiện có mã chạy. `scripts/run_experiment.py` vẫn là ghi chú/khung; chưa có pipeline chung để chạy và so sánh đầy đủ hai implementation.

Mô hình K-Medoids dùng thư viện **đã được cài đặt** trong `src/library_model.py` và được gọi từ `src/evaluation.py`. Kết quả library sweep hiện có trong `outputs/metrics/` cho Manhattan và Gower với `k=2..8`, seed 42. Pipeline so sánh chung với bản tự cài đặt trong `scripts/run_experiment.py` vẫn chưa hoàn thiện; kết quả library hiện có không phải kết quả so sánh custom-vs-library.
