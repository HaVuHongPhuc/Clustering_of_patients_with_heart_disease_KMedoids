# Phân cụm bệnh nhân bị bệnh tim bằng K-Medoids

## Trạng thái hiện tại

Notebook 01 tạo dữ liệu đã xử lý và khảo sát số cụm. Notebook 02 phân tích kết quả và so sánh ở cùng số cụm. Notebook 03 kiểm tra khả năng tái lập. Script thực nghiệm cũng chạy được trực tiếp từ dữ liệu gốc.

Cấu hình hiện tại dùng 12 đặc trưng trên 2.283 hồ sơ, tương ứng 21 cột Manhattan. Hai cột cp và target dành cho đối chiếu sau phân cụm. So sánh chính dùng Manhattan tại k=2 và seed 42, với Gower làm đối chứng ở cùng k. Dữ liệu lấy từ [Heart Disease Multi-Source Dataset](https://www.kaggle.com/datasets/alifsathar/heart-disease-multi-source-dataset) do Alif Sathar đăng, phiên bản 1 (30/04/2026), giấy phép CC BY-SA 4.0. Trang dữ liệu có mô tả và từ điển dữ liệu. Nhóm lưu thông tin ngày tải và dùng toàn bộ nguồn trong tệp tổng hợp. Quy tắc mã hóa cp theo nguồn đang được xác nhận riêng.

## Mục tiêu dự án

Phân cụm hồ sơ bệnh nhân bằng K-Medoids, cài đặt thuật toán từ đầu và đối chiếu với một triển khai từ thư viện. Báo cáo cần mô tả nguồn dữ liệu, tiền xử lý, lựa chọn số cụm, thực nghiệm, đánh giá và hạn chế.

## Cấu trúc và vai trò tệp

| Đường dẫn | Vai trò và hướng thực hiện |
|---|---|
| `README.md` | Tài liệu trung tâm: cách tổ chức dự án, kế hoạch triển khai, dữ liệu, thực nghiệm và trạng thái. Cập nhật khi quyết định hoặc cách chạy thay đổi. |
| `HD Thực hiện đồ án cuối kỳ môn Học máy.pdf` | Bản hướng dẫn môn học được cung cấp; dùng để đối chiếu yêu cầu đồ án, không chỉnh sửa. |
| `MauBaoCaoHocMay.docx` | Mẫu báo cáo được cung cấp; dùng làm chuẩn cấu trúc báo cáo, không chỉnh sửa trong phần scaffold. |
| `requirements.txt` | Khai báo thư viện chạy mô hình, notebook và đọc YAML (PyYAML). |
| `configs/experiment.yaml` | Cấu hình chạy so sánh hiện tại, gồm đầu vào, đặc trưng, cột tham chiếu, k, seed và thư mục đầu ra. |
| `data/raw/` | Có file tổng hợp `heart_disease_whole.csv` từ Kaggle; nhóm xác nhận dùng tất cả nguồn trong file và đã ghi nhận quy tắc harmonize theo từng `source`. File có 2.284 dòng vật lý tính cả header (2.283 bản ghi dữ liệu). |
| `data/processed/` | Dữ liệu đã xử lý, ma trận Manhattan/Gower và các cột tham chiếu được lưu theo cùng thứ tự hồ sơ. |
| `notebooks/01_data_exploration.ipynb` | Đã có EDA, kiểm tra chất lượng, quyết định xử lý và lưu dữ liệu processed/kết quả thư viện. |
| `notebooks/02_experiment_analysis.ipynb` | So sánh mô hình, phân tích cụm, đối chiếu cp/target và ví dụ 30 hồ sơ. |
| `notebooks/03_TestKmedoids.ipynb` | Kiểm thử tích hợp và đối chiếu ví dụ 30 hồ sơ. |
| `src/__init__.py` | Đánh dấu `src` là package Python; hiện chỉ có chú thích. |
| `src/data.py` | Đang đọc/kiểm tra dữ liệu, harmonize encoding, đánh dấu giá trị bất thường và xử lý missing values. |
| `src/preprocessing.py` | Tạo biểu diễn Manhattan và ma trận khoảng cách Gower từ 12 đặc trưng phân cụm. |
| `src/kmedoids.py` | K-Medoids tự cài đặt theo biến thể Alternate. |
| `src/library_model.py` | Adapter đang chạy với package `kmedoids` (PAM/BUILD); hỗ trợ Manhattan trên ma trận đặc trưng và Gower qua ma trận khoảng cách `precomputed`. |
| `src/evaluation.py` | Khảo sát k và đánh giá nội tại, đối chiếu cp/target sau phân cụm, lưu kết quả thư viện. |
| `src/visualization.py` | Chưa có hàm vẽ dùng lại; biểu đồ hiện được tạo trong Notebook 2. |
| `scripts/run_experiment.py` | Đọc YAML, tiền xử lý dữ liệu gốc, chạy ba mô hình và lưu từng lần chạy. |
| `tests/test_kmedoids.py` | Kiểm thử ví dụ tính tay, seed và đầu vào/điều kiện sử dụng không hợp lệ. |
| `outputs/figures/` | Biểu đồ phân tích từ Notebook 02. |
| `outputs/metrics/` | Khảo sát thư viện Manhattan/Gower tại k=2..8 từ Notebook 01. |
| `outputs/experiments/` | Kết quả so sánh từ script, một thư mục cho mỗi lần chạy. |

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
9. **So sánh và diễn giải.** Dùng `notebooks/02_experiment_analysis.ipynb` để so sánh bản tự viết với thư viện, kiểm tra kích thước cụm và mô tả đặc trưng các medoid/cụm. Dùng cp và nhãn bệnh để đối chiếu sau phân cụm.
10. **Lưu đầu ra có thể tái lập.** Tạo hình trong `outputs/figures/`, kết quả khảo sát trong `outputs/metrics/` và so sánh trong `outputs/experiments/`. Đảm bảo ghi seed, phiên bản thư viện, tiền xử lý, metric khoảng cách và `k` cho mỗi kết quả.
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

## Chạy lại thực nghiệm

Có thể chạy ngay trong Notebook 02, mục 13. Chọn kernel của môi trường dự án, đặt RUN_EXPERIMENT = True và chạy ô ở cuối notebook. Ô gọi cùng pipeline YAML và hiển thị bảng metric, nhãn cụm và medoid. Cách chạy bằng PowerShell ở bên dưới.

Đặt tệp dữ liệu tổng hợp tại `data/raw/heart_disease_whole.csv` và cài các gói trong `requirements.txt` như hướng dẫn trên. Từ thư mục gốc của dự án trên Windows, chạy:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/experiment.yaml
```

Script đọc dữ liệu gốc và dùng chung bước tiền xử lý của Notebook 01. Mặc định, mỗi lần chạy tạo một thư mục có mã thời gian UTC trong `outputs/experiments/`. Để đặt tên rõ hơn, truyền `--output-dir outputs/experiments/manhattan-k2-lan-moi`. Hãy chọn một thư mục chưa tồn tại. Đường dẫn tương đối được tính từ thư mục gốc dự án.

- `comparison.csv` chứa Silhouette, Inertia, ARI và Purity với cp/target cho ba cấu hình tại cùng k, kèm seed và thông tin hội tụ nếu mô hình cung cấp.
- `agreement.csv` chứa ARI giữa hai kết quả Manhattan, và giữa Manhattan với Gower của thư viện.
- Các thư mục `custom_manhattan/`, `library_manhattan/` và `library_gower/` chứa nhãn và medoid theo chỉ số dòng đã xử lý, bắt đầu từ 0. Mô hình tự cài đặt có thêm `history.csv`.
- `run_config.json` ghi cấu hình, mã băm tệp dữ liệu, phiên bản thư viện và các quy tắc điền đã áp dụng. Trạng thái hội tụ của thư viện để trống vì adapter chưa cung cấp.

Các phép đối chiếu cp dùng giá trị sau làm sạch. Khi chưa xác nhận ý nghĩa mã theo từng nguồn, hãy đọc các chỉ số cp theo đúng cấu hình làm sạch ghi trong lần chạy. Đọc Inertia trong cùng một loại khoảng cách. Để khảo sát k từ 2 đến 8, dùng Notebook 01.

### Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Để kiểm thử trên dữ liệu dự án, chạy Notebook 01, 02 rồi 03 trong kernel của môi trường dự án. Notebook 03 đối chiếu ví dụ 30 hồ sơ với đầu ra Notebook 02. Các dòng WARN ghi nhận vấn đề về miền mã hoặc lọc nguồn cần theo dõi riêng.


## Dashboard Streamlit

Mở từ thư mục gốc của dự án. Cài gói dashboard lần đầu nếu môi trường chưa có.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-streamlit.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Mở [dashboard cục bộ](http://127.0.0.1:8501). Ở màn hình **Thử phân cụm**, chọn số hồ sơ từ 3 đến 2.283. Ứng dụng dùng đúng số dòng đầu đã chọn từ ma trận Manhattan 21 cột. Mặc định ứng dụng chạy đủ 2.283 hồ sơ với k=2 và seed 42. Đổi seed hoặc chọn hai medoid rồi bấm **Phân cụm** để cập nhật kết quả. Ví dụ tính tay dùng 30 hồ sơ với R1 và R16 làm medoid ban đầu. Có thể chọn một vòng lặp để xem cách gán, xem tổng khoảng cách của từng ứng viên hoặc lọc bảng theo cụm.

Hai màn hình còn lại đọc khảo sát k từ Notebook 01 và các lần chạy do mục 13 của Notebook 02 lưu. Nhấn Ctrl+C trong terminal để dừng máy chủ.
