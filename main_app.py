import os
# Tối ưu hóa hệ thống log và triệt tiêu thông báo rườm rà của TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk # Thêm ImageTk để hiển thị ảnh lên Canvas
import numpy as np
import tensorflow as tf

# Thiết lập cấu hình giao diện chuẩn hiện đại
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

FASHION_LABELS = [
    'Áo thun (T-shirt/top)', 'Quần dài (Trouser)', 'Áo len (Pullover)', 
    'Váy liền (Dress)', 'Áo khoác (Coat)', 'Sandal', 'Áo sơ mi (Shirt)', 
    'Giày thể thao (Sneaker)', 'Túi xách (Bag)', 'Giày bốt (Ankle boot)'
]

class PremiumFashionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Mở rộng kích thước cửa sổ chính để phù hợp với Bảng vẽ to hơn
        self.title("Fashion Recognition System - CNN Model Evaluation")
        self.geometry("950x650")
        self.resizable(False, False)
        
        # --- KIỂM TRA & TẢI MÔ HÌNH AI ---
        try:
            self.model = tf.keras.models.load_model('model_cnn.h5')
            self.status_text = "Hệ thống AI: Đã kết nối thành công mô hình 'model_cnn.h5'"
            self.status_color = "#2ecc71" 
        except Exception as e:
            self.model = None
            self.status_text = "Hệ thống AI: Mất kết nối! Vui lòng kiểm tra lại file 'model_cnn.h5'"
            self.status_color = "#e74c3c" 

        # ==========================================
        # 1. TIÊU ĐỀ ỨNG DỤNG (HEADER BANNER)
        # ==========================================
        self.header_label = ctk.CTkLabel(
            self, 
            text="HỆ THỐNG NHẬN DIỆN TRANG PHỤC THỜI TRANG", 
            font=ctk.CTkFont(family="Arial", size=24, weight="bold")
        )
        self.header_label.pack(pady=(25, 5))
        
        self.subheader_label = ctk.CTkLabel(
            self, 
            text="Phân loại dữ liệu vẽ tay và hình ảnh thực tế dựa trên mạng nơ-ron tích chập (CNN)", 
            font=ctk.CTkFont(family="Arial", size=13, slant="italic"),
            text_color="gray"
        )
        self.subheader_label.pack(pady=(0, 20))

        # ==========================================
        # 2. KHU VỰC NỘI DUNG CHÍNH (MAIN BODY)
        # ==========================================
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=30, pady=0)
        
        # Cấu hình tỉ lệ 2 cột bằng nhau hoàn toàn
        self.main_container.grid_columnconfigure(0, weight=1, uniform="group1")
        self.main_container.grid_columnconfigure(1, weight=1, uniform="group1")
        self.main_container.grid_rowconfigure(0, weight=1)

        # ------------------------------------------
        # CỘT TRÁI: THỂ BẢNG VẼ NÂNG CẤP (LARGE CANVAS CARD)
        # ------------------------------------------
        self.left_card = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.left_card.grid(row=0, column=0, padx=12, pady=10, sticky="nsew")
        
        self.canvas_title = ctk.CTkLabel(
            self.left_card, 
            text="Khung Tương Tác Vẽ / Hiển Thị Ảnh", 
            font=ctk.CTkFont(family="Arial", size=16, weight="bold")
        )
        self.canvas_title.pack(pady=(15, 10))

        # Khung viền tinh tế bao quanh Canvas
        self.canvas_border = ctk.CTkFrame(self.left_card, fg_color="#dcdde1", corner_radius=8)
        self.canvas_border.pack()

        # Đã tăng kích thước lên ĐỒNG BỘ 380x380 tương đương cột bên phải
        self.canvas_width, self.canvas_height = 380, 380
        self.canvas = ctk.CTkCanvas(
            self.canvas_border, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="white", 
            bd=0, 
            highlightthickness=0,
            cursor="pencil"
        )
        self.canvas.pack(padx=2, pady=2) # Thay thế hoàn toàn padding bị lỗi
        
        # Kết nối sự kiện chuột
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.predict_from_canvas)

        # ------------------------------------------
        # CỘT PHẢI: THẺ KẾT QUẢ & ĐIỀU KHIỂN (CONTROL CARD)
        # ------------------------------------------
        self.right_card = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.right_card.grid(row=0, column=1, padx=12, pady=10, sticky="nsew")
        
        self.result_title = ctk.CTkLabel(
            self.right_card, 
            text="Kết Quả Phân Tích Kép", 
            font=ctk.CTkFont(family="Arial", size=16, weight="bold")
        )
        self.result_title.pack(pady=(15, 20))

        # Khung hiển thị tên nhãn nhận diện
        self.display_box = ctk.CTkFrame(self.right_card, fg_color=("#eaeded", "#2c3e50"), corner_radius=10)
        self.display_box.pack(fill="x", padx=25, pady=10)

        self.result_label = ctk.CTkLabel(
            self.display_box, 
            text="SẴN SÀNG NHẬN DỮ LIỆU", 
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color="#2980b9"
        )
        self.result_label.pack(pady=40) # Tăng kích thước đệm dọc cho cân xứng với khung vẽ mới

        # Khu vực chứa nhóm nút điều hướng hành động
        self.button_container = ctk.CTkFrame(self.right_card, fg_color="transparent")
        self.button_container.pack(fill="x", padx=25, pady=(40, 0))

        self.btn_upload = ctk.CTkButton(
            self.button_container, 
            text="Tải Ảnh Thực Tế", 
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            height=45,
            command=self.upload_image
        )
        self.btn_upload.pack(fill="x", pady=8)
        
        self.btn_clear = ctk.CTkButton(
            self.button_container, 
            text="Xóa Bảng Vẽ / Reset", 
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            height=45,
            fg_color="#e74c3c", 
            hover_color="#c0392b",
            command=self.clear_canvas
        )
        self.btn_clear.pack(fill="x", pady=8)

        # ==========================================
        # 3. THANH TRẠNG THÁI DƯỚI CÙNG (STATUS BAR)
        # ==========================================
        self.status_bar = ctk.CTkLabel(
            self, 
            text=self.status_text, 
            font=ctk.CTkFont(family="Arial", size=11),
            fg_color=("#d5dbdb", "#1c2833"),
            text_color=self.status_color,
            anchor="w",
            padx=15
        )
        self.status_bar.pack(side="bottom", fill="x")

        # --- CÁC BIẾN LOGIC HỖ TRỢ VẼ VÀ LƯU ẢNH ---
        self.last_x, self.last_y = None, None
        self.tk_image = None # Biến lưu trữ đối tượng PhotoImage để tránh bị bộ thu gom rác xóa
        self.setup_virtual_image()

    def setup_virtual_image(self):
        # Thiết lập ma trận ảnh ảo nền trắng khớp chuẩn kích thước khung vẽ mới
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def draw_lines(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Tăng nhẹ độ rộng nét vẽ (width=16) để tỷ lệ thuận với kích thước 380x380 mới
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=16, fill="black", capstyle="round", smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=16)
        self.last_x = x
        self.last_y = y

    def predict_from_canvas(self, event):
        self.last_x, self.last_y = None, None
        self.process_and_predict(self.image)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            try:
                # 1. Đọc tệp ảnh thực tế
                img = Image.open(file_path).convert("RGB")
                
                # 2. Thay đổi kích thước ảnh trùng khít với khung vẽ 380x380
                self.image = img.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                self.draw = ImageDraw.Draw(self.image) # Cho phép người dùng vẽ đè trực tiếp lên ảnh nếu muốn
                
                # 3. Hiển thị ảnh trực quan lên màn hình Canvas
                self.tk_image = ImageTk.PhotoImage(self.image)
                self.canvas.delete("all") # Dọn sạch nét vẽ cũ trước đó
                self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
                
                # 4. Đẩy ảnh vào model phân tích ngay lập tức
                self.process_and_predict(self.image)
            except Exception as e:
                self.result_label.configure(text="LỖI ĐỌC TỆP TIN", text_color="#e74c3c")

    def clear_canvas(self):
        # Dọn sạch hoàn toàn mọi hình ảnh hiển thị và nét vẽ trên Canvas
        self.canvas.delete("all")
        # Đặt lại bộ nhớ ảnh ảo về trạng thái bảng vẽ trắng
        self.setup_virtual_image()
        self.tk_image = None
        # Khôi phục trạng thái nhãn hiển thị ban đầu
        self.result_label.configure(text="SẴN SÀNG NHẬN DỮ LIỆU", text_color="#2980b9")

    def process_and_predict(self, pil_image):
        if self.model is None:
            self.result_label.configure(text="MẤT MODEL AI", text_color="#e74c3c")
            return

        # Thực hiện các bước tiền xử lý ảnh đầu vào
        img_gray = ImageOps.grayscale(pil_image)
        img_inverted = ImageOps.invert(img_gray) # Đảo màu nền sang đen nét trắng đồng bộ với tập Fashion MNIST
        img_resized = img_inverted.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1) # Định hình cấu trúc (1, 28, 28, 1) cho lớp Conv2D
        
        # Gọi mô hình CNN dự đoán phân lớp
        predictions = self.model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_name = FASHION_LABELS[class_index]
        
        # Cập nhật kết quả viết hoa lên nhãn nổi bật
        self.result_label.configure(text=predicted_name.upper(), text_color="#27ae60")

if __name__ == "__main__":
    app = PremiumFashionApp()
    app.mainloop()