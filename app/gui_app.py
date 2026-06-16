import os
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from ultralytics import YOLO
import threading
import queue
from utils.config import VIETNAMESE_LABELS

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TrafficSignGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG - YOLOv8")
        self.geometry("1100x650")

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_dir)
        self.model_path = os.path.join(self.root_dir, "models", "best.pt")

        print(f"[*] Đang nạp mô hình từ: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[!] Lỗi nạp mô hình: {e}")
            self.model = None

        self.cap = None
        self.is_running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="BẢNG ĐIỀU KHIỂN",
                                       font=ctk.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.btn_image = ctk.CTkButton(self.sidebar_frame, text="Chọn ảnh", command=self.start_image,
                                       height=40)
        self.btn_image.grid(row=1, column=0, padx=20, pady=10)

        self.btn_video = ctk.CTkButton(self.sidebar_frame, text="Chọn video", command=self.start_video,
                                       height=40)
        self.btn_video.grid(row=2, column=0, padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="Dừng Xử Lý", command=self.stop_system,
                                      fg_color="crimson", hover_color="#B22222", height=40)
        self.btn_stop.grid(row=3, column=0, padx=20, pady=10)

        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.display_frame, text="Hệ thống đã sẵn sàng\nVui lòng chọn nguồn đầu vào",
                                        fg_color="black", text_color="gray", font=ctk.CTkFont(size=14))
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def start_image(self):
        if self.model is None:
            self.video_label.configure(text="Lỗi: Không tìm thấy file trọng số (best.pt)!", image=None)
            return

        # Dừng luồng video nếu nó đang chạy
        self.stop_system()

        file_path = filedialog.askopenfilename(
            title="Chọn hình ảnh",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not file_path:
            return

        self.video_label.configure(text="Đang xử lý ảnh...", image=None)
        self.update()

        # Đọc ảnh bằng OpenCV dựa trên đường dẫn vừa chọn
        frame = cv2.imread(file_path)
        if frame is None:
            self.video_label.configure(text=f"Lỗi: Không thể đọc dữ liệu ảnh!\n({file_path})", image=None)
            return

        # Nhận diện ảnh qua YOLO
        results = self.model.predict(frame, conf=0.545, device='cpu', verbose=False)
        result = results[0]

        # Ép nhãn tiếng Việt
        for class_id, class_code in result.names.items():
            if class_code in VIETNAMESE_LABELS:
                result.names[class_id] = VIETNAMESE_LABELS[class_code]

        annotated_frame = result.plot()

        # Chuyển đổi màu và hiển thị cho Tkinter
        cv2_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)

        frame_width = self.display_frame.winfo_width() - 20
        frame_height = self.display_frame.winfo_height() - 20

        if frame_width > 0 and frame_height > 0:
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(frame_width, frame_height))
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image

    def start_video(self):
        if self.model is None:
            self.video_label.configure(text="Lỗi: Không tìm thấy file trọng số (best.pt)!", image=None)
            return

        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )

        if not file_path:
            return

        self.setup_stream(file_path)

    def setup_stream(self, source):
        if self.model is None:
            self.video_label.configure(text="Lỗi: Không tìm thấy file trọng số (best.pt)!", image=None)
            return

        self.stop_system()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.video_label.configure(text=f"Lỗi: Không thể mở nguồn dữ liệu!\n({source})", image=None)
            return

        self.is_running = True
        self.stop_event.clear()

        # Kích hoạt Luồng Ngầm (Worker Thread) để chạy AI liên tục cho Video
        self.thread = threading.Thread(target=self.ai_worker, args=(source,), daemon=True)
        self.thread.start()
        self.update_gui_loop()

    # ─── LUỒNG NGẦM: CHUYÊN CHẠY AI (Chỉ dùng cho Video) ───
    def ai_worker(self, source):
        while self.is_running and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.545, device='cpu', verbose=False)
            result= results[0]

            for class_id, class_code in result.names.items():
                if class_code in VIETNAMESE_LABELS:
                    result.names[class_id] = VIETNAMESE_LABELS[class_code]

            annotated_frame = result.plot()

            cv2_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)

            # Đẩy ảnh vào hàng đợi
            if not self.frame_queue.full():
                self.frame_queue.put(pil_image)

        # Giải phóng camera khi thoát vòng lặp
        if self.cap:
            self.cap.release()

    # ─── LUỒNG CHÍNH: RÚT ẢNH TỪ HÀNG ĐỢI ĐỂ VẼ LÊN GIAO DIỆN ───
    def update_gui_loop(self):
        if not self.is_running:
            return

        try:
            pil_image = self.frame_queue.get_nowait()
            frame_width = self.display_frame.winfo_width() - 20
            frame_height = self.display_frame.winfo_height() - 20

            if frame_width > 0 and frame_height > 0:
                ctk_image = ctk.CTkImage(light_image=pil_image, size=(frame_width, frame_height))

                self.video_label.configure(image=ctk_image, text="")
                self.video_label.image = ctk_image
        except queue.Empty:
            pass

        if self.thread and not self.thread.is_alive() and self.frame_queue.empty():
            self.stop_system()
            self.video_label.configure(text="Đã kết thúc luồng Video.", image=None)
            return

        self.after(15, self.update_gui_loop)

    def stop_system(self):
        self.is_running = False
        self.stop_event.set()

        # Làm sạch hàng đợi ảnh
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        empty_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        empty_ctk = ctk.CTkImage(light_image=empty_pil, size=(1, 1))

        self.video_label.configure(image=empty_ctk, text="Hệ thống đã dừng.\nVui lòng chọn nguồn đầu vào mới.")
        self.video_label.image = empty_ctk

if __name__ == "__main__":
    app = TrafficSignGUI()
    app.mainloop()