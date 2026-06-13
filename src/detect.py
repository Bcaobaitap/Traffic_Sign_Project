import os
from ultralytics import YOLO
import cv2

VIETNAMESE_LABELS = {
    'DP.135': 'Het tat ca lenh cam', 'P.102': 'Cam di nguoc chieu', 'P.103a': 'Cam o to',
    'P.103b': 'Cam o to re phai', 'P.103c': 'Cam o to re trai', 'P.104': 'Cam mo to',
    'P.106a': 'Cam xe tai', 'P.106b': 'Cam xe tai tren 2.5T', 'P.107a': 'Cam o to khach va tai',
    'P.112': 'Cam nguoi di bo', 'P.115': 'Han che trong luong xe', 'P.117': 'Han che chieu cao',
    'P.123a': 'Cam re trai', 'P.123b': 'Cam re phai', 'P.124a': 'Cam quay dau',
    'P.124b': 'Cam o to quay dau', 'P.124c': 'Cam re trai va quay dau', 'P.125': 'Cam vuot',
    'P.127': 'Toc do toi da cho phep', 'P.128': 'Cam bop coi', 'P.130': 'Cam dung va do xe',
    'P.131a': 'Cam do xe', 'P.137': 'Cam re trai va re phai', 'P.245a': 'Di cham',
    'R.301c': 'Re trai', 'R.301d': 'Re phai', 'R.301e': 'Re trai hoac phai',
    'R.302a': 'Huong phai vong sang phai', 'R.302b': 'Huong phai vong sang trai',
    'R.303': 'Vong xuyen', 'R.407a': 'Duong mot chieu', 'R.409': 'Cho quay dau xe',
    'R.425': 'Duong danh cho o to', 'R.434': 'Ben xe buyt', 'S.509a': 'Chieu cao an toan',
    'W.201a': 'Ngoat ben trai nguy hiem', 'W.201b': 'Ngoat ben phai nguy hiem',
    'W.202a': 'Nhieu cho ngoat (trai truoc)', 'W.202b': 'Nhieu cho ngoat (phai truoc)',
    'W.203b': 'Duong hep ben trai', 'W.203c': 'Duong hep ben phai', 'W.205a': 'Giao nhau duong cung cap',
    'W.205b': 'Giao nhau duong cung cap', 'W.205d': 'Giao nhau duong cung cap',
    'W.207a': 'Giao voi duong khong uu tien', 'W.207b': 'Giao voi duong khong uu tien',
    'W.207c': 'Giao voi duong khong uu tien', 'W.208': 'Giao voi duong uu tien',
    'W.209': 'Giao nhau co den tin hieu', 'W.210': 'Giao voi duong sat co rao',
    'W.219': 'Doc xuong nguy hiem', 'W.221b': 'Duong khong bang phang',
    'W.224': 'Nguoi di bo cat ngang', 'W.225': 'Tre em cat ngang', 'W.227': 'Cong truong',
    'W.233': 'Nguy hiem khac', 'W.235': 'Duong doi', 'W.245a': 'Di cham'
}


def load_model(model_path):
    print(f"[*] Đang nạp mô hình từ: {model_path}...")
    return YOLO(model_path)


def process_image(model, input_path, output_path, conf_thresh=0.5):
    print(f"[*] Đang phân tích ảnh: {input_path}")

    # Chạy nhận diện
    results = model.predict(source=input_path, conf=conf_thresh, verbose=False)
    result = results[0]

    # Ép buộc đổi tên tiếng Việt
    for class_id, class_code in result.names.items():
        if class_code in VIETNAMESE_LABELS:
            result.names[class_id] = VIETNAMESE_LABELS[class_code]

    # Vẽ và lưu ảnh
    annotated_frame = result.plot()
    cv2.imwrite(output_path, annotated_frame)
    print(f"[+] Đã lưu kết quả tại: {output_path}")

    # Hiển thị lên màn hình
    cv2.imshow("Ket qua Anh", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model, input_path, output_path, conf_thresh=0.5):
    print(f"[*] Đang phân tích video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể mở video {input_path}")
        return

    # Lấy thông số kỹ thuật của video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo công cụ ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"\rĐang xử lý frame {frame_count}/{total_frames}...", end="")

        # Nhận diện trên từng frame
        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        result = results[0]

        # Ép buộc đổi tên tiếng Việt
        for class_id, class_code in result.names.items():
            if class_code in VIETNAMESE_LABELS:
                result.names[class_id] = VIETNAMESE_LABELS[class_code]

        # Vẽ bounding box lên frame
        annotated_frame = result.plot()

        # Ghi frame đã vẽ vào video đầu ra
        out.write(annotated_frame)

        # Hiển thị trực tiếp quá trình
        cv2.imshow("Ket qua Video (Bam 'q' de thoat)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Người dùng đã dừng sớm.")
            break

    # Dọn dẹp bộ nhớ
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Đã lưu video kết quả tại: {output_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(root_dir, "models", "best.pt")

    my_model = load_model(model_path)

    #XỬ LÝ ẢNH
    image_in = os.path.join(root_dir, "tests", "sample_image.png")
    image_out = os.path.join(output_dir, "result_image.jpg")
    process_image(my_model, image_in, image_out, conf_thresh=0.5)

    #XỬ LÝ VIDEO
    video_in = os.path.join(root_dir, "tests", "sample_video.mp4")
    video_out = os.path.join(output_dir, "result_video.mp4")
    process_video(my_model, video_in, video_out, conf_thresh=0.5)