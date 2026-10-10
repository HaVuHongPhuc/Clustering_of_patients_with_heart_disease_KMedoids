"""Vai trò dự kiến: điểm điều phối chạy pipeline thực nghiệm.

Hướng thực hiện sau:
- Đọc configs/experiment.yaml.
- Gọi nạp dữ liệu, tiền xử lý, mô hình tự cài đặt và mô hình thư viện.
- Tính metric, rồi lưu cấu hình và kết quả vào outputs/.
- Cho phép lặp lại cùng thực nghiệm với seed/cấu hình được ghi nhận.

Hiện tại chỉ là ghi chú; chưa có entry point hoặc mã thực thi.
"""
