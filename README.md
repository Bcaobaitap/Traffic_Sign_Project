# Hệ thống Nhận diện Biển báo Giao thông Việt Nam (YOLOv8)

Đề tài xây dựng script tự động nhận diện và phân loại 58 loại biển báo giao thông chuẩn theo **QCVN 41:2019/BGTVT**. Dự án áp dụng mô hình kiến trúc dạng Pipeline, tách biệt hoàn toàn môi trường huấn luyện (Cloud) và môi trường thực thi (Local).

## Cấu trúc Dự án
- `models/best.pt`: Trọng số tối ưu nhất sau quá trình huấn luyện bằng GPU.
- `notebooks/Traffic_Sign_Project.ipynb`: Source code quá trình thu thập dữ liệu và huấn luyện mô hình.
- `src/detect.py`: Logic xử lý hình ảnh và video lõi bằng OpenCV.
- `tests/`: Thư mục chứa dữ liệu đầu vào để kiểm thử.
- `outputs/`: Nơi lưu trữ video/ảnh sau khi hệ thống đóng khung nhận diện.

## Hướng dẫn cài đặt và sử dụng
1. Cài đặt các thư viện lõi: 
   ```bash
   pip install -r requirements.txt