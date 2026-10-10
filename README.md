# Phân cụm bệnh nhân bị bệnh tim bằng K-Medoids

## Trạng thái hiện tại

Đây mới là **bộ khung dự án**. Các tệp trong `src/` và `scripts/` chỉ chứa ghi chú hướng dẫn; chưa có mã chạy được, dữ liệu, kết quả, danh sách thư viện hay kiểm thử. Hai notebook chỉ có ô Markdown hướng dẫn, chưa có ô mã.

Tên đề tài và thuật toán dự kiến đã biết. Bộ dữ liệu, nguồn dữ liệu, thành viên nhóm, phạm vi đặc trưng, giá trị `k`, metric và lựa chọn thư viện vẫn cần được xác nhận. Không điền các thông tin này bằng phỏng đoán.

## Mục tiêu dự án

Phân cụm hồ sơ bệnh nhân bằng K-Medoids, cài đặt thuật toán từ đầu và đối chiếu với một triển khai từ thư viện. Báo cáo cần mô tả nguồn dữ liệu, tiền xử lý, lựa chọn số cụm, thực nghiệm, đánh giá và hạn chế.

## Cấu trúc và vai trò tệp

| Đường dẫn | Vai trò và hướng thực hiện |
|---|---|
| `README.md` | Tài liệu trung tâm: cách tổ chức dự án, kế hoạch triển khai, dữ liệu, thực nghiệm và trạng thái. Cập nhật khi quyết định hoặc cách chạy thay đổi. |
| `HD Thực hiện đồ án cuối kỳ môn Học máy.pdf` | Bản hướng dẫn môn học được cung cấp; dùng để đối chiếu yêu cầu đồ án, không chỉnh sửa. |
| `MauBaoCaoHocMay.docx` | Mẫu báo cáo được cung cấp; dùng làm chuẩn cấu trúc báo cáo, không chỉnh sửa trong phần scaffold. |
| `requirements.txt` | Khai báo các thư viện Python sau khi đã chốt lựa chọn và phiên bản. Hiện chỉ có ghi chú, chưa có thư viện nào được thêm. |
| `configs/experiment.yaml` | Lưu cấu hình thực nghiệm như đường dẫn dữ liệu, cột đầu vào, nhãn chỉ dùng để đánh giá, `k`, metric khoảng cách và seed. Chỉ điền sau khi thống nhất quyết định. |
| `data/raw/` | Chứa bản dữ liệu gốc, giữ nguyên không chỉnh sửa. Thư mục đang rỗng và không có `.gitkeep`. |
| `data/processed/` | Chứa dữ liệu sau làm sạch/biến đổi nếu cần lưu trung gian. Thư mục đang rỗng và không có `.gitkeep`. |
| `notebooks/01_data_exploration.ipynb` | Ghi EDA, kiểm tra chất lượng dữ liệu, trực quan hóa và giải thích các quyết định tiền xử lý. Hiện chỉ có hướng dẫn Markdown. |
| `notebooks/02_experiment_analysis.ipynb` | Phân tích kết quả, so sánh cấu hình và mô tả đặc trưng từng cụm. Hiện chỉ có hướng dẫn Markdown. |
| `src/__init__.py` | Đánh dấu `src` là package Python; hiện chỉ có chú thích. |
| `src/data.py` | Sau này cài đặt chức năng đọc dữ liệu, kiểm tra schema và báo lỗi khi thiếu cột cần thiết. |
| `src/preprocessing.py` | Sau này cài đặt làm sạch, mã hóa biến phân loại, chuẩn hóa và tách cột đầu vào khỏi nhãn đánh giá. |
| `src/kmedoids.py` | Sau này cài đặt K-Medoids từ đầu: khởi tạo medoid, gán cụm, cập nhật medoid, dừng và dự đoán/gán nhãn cụm. |
| `src/library_model.py` | Sau này gọi thư viện K-Medoids đã chọn với cùng dữ liệu/metric để so sánh công bằng với bản tự cài đặt. |
| `src/evaluation.py` | Sau này tính các chỉ số phân cụm và bảng tóm tắt kích thước/cấu hình cụm. Chỉ tính metric phù hợp với dữ liệu và cách biểu diễn. |
| `src/visualization.py` | Sau này tập hợp hàm vẽ biểu đồ chọn `k`, phân bố cụm và hồ sơ cụm. |
| `scripts/run_experiment.py` | Điểm chạy pipeline: đọc cấu hình, nạp dữ liệu, tiền xử lý, chạy mô hình, đánh giá và lưu kết quả. Hiện chỉ có ghi chú. |
| `tests/test_kmedoids.py` | Vị trí dự kiến cho kiểm thử thuật toán trên trường hợp nhỏ/giả lập sau khi có implementation. Hiện chưa có kiểm thử. |
| `outputs/figures/` | Chứa hình được sinh ra cho phân tích và báo cáo. Thư mục đang rỗng và không có `.gitkeep`. |
| `outputs/metrics/` | Chứa bảng metric và tóm tắt thực nghiệm được sinh ra. Thư mục đang rỗng và không có `.gitkeep`. |

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

## Chạy dự án

Chưa thể chạy pipeline: các file Python hiện chưa có implementation, `requirements.txt` chưa khai báo thư viện, và chưa có dữ liệu/cấu hình. Sau khi các phần này được bổ sung, cập nhật mục này bằng lệnh cài đặt/chạy chính xác và các tham số cần thiết; không ghi lệnh chạy như thể pipeline đã hoạt động.


