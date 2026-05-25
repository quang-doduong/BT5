# tensorflow, keras, numpy, matplotlib, opencv-python (Đã được chuyển thành ghi chú)
import matplotlib.pyplot as plt
import os
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape cho mạng CNN (thêm kênh màu 1: xám)
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# One-hot encoding cho nhãn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. XÂY DỰNG MẠNG CNN ĐỀ XUẤT
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) # Tắt ngẫu nhiên 50% nơ-ron để ép model học đặc trưng chung
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. HUẤN LUYỆN VÀ LƯU LỊCH SỬ (Validation data giúp đánh giá thực tế)
print("Bắt đầu huấn luyện mô hình CNN...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 4. ĐÁNH GIÁ VÀ LƯU MÔ HÌNH
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

#model.save(r"C:\Users\LENOVO\Downloads\model_cnn.h5") #r"C:\Users\LENOVO\Downloads\model_cnn.h5"
#print("✅ Đã lưu bộ não AI thành công vào file 'model_cnn.h5'")
# 1. Tự động lấy đường dẫn của thư mục chứa chính file train_model.py này
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Kết hợp thư mục đó với tên file muốn lưu
model_path = os.path.join(current_dir, "model_cnn.h5")

# 3. Ra lệnh cho AI lưu đúng vào đường dẫn vừa tìm được
model.save(model_path)
print(f"✅ Đã lưu bộ não AI thành công vào: {model_path}")

# 5. VẼ BIỂU ĐỒ CHO BÁO CÁO
plt.figure(figsize=(10, 4))

# Biểu đồ Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Độ chính xác (Accuracy)')
plt.legend()

# Biểu đồ Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Hàm mất mát (Loss)')
plt.legend()

plt.tight_layout()
plt.show() # Sẽ bật lên cửa sổ biểu đồ để các bạn chụp màn hình đưa vào Word