import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PKL_PATH = r'D:\GITHUB-NM\anh.pkl'
MODEL_OUT_PATH = r'D:\GITHUB-NM\model_sign_language.sav'

def train_brain():
    # 1. Load dữ liệu từ file pkl bạn đã tạo
    print(" Đang nạp dữ liệu từ file pkl...")
    try:
        with open(PKL_PATH, 'rb') as f:
            X, y, labels = pickle.load(f)
    except FileNotFoundError:
        print(f" Không tìm thấy file {PKL_PATH}. Hãy chạy code tạo Dataset trước!")
        return

    # 2. Chuẩn hóa dữ liệu (Đưa về khoảng 0-1)
    X = X.astype('float32') / 255.0

    # 3. Chia dữ liệu: 80% để học, 20% để thi thử (kiểm tra)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Khởi tạo mô hình MLP (Bộ não)
    # hidden_layer_sizes: các lớp thần kinh ảo
    # max_iter: số lần học đi học lại
    print(f" Đang huấn luyện bộ não với {len(labels)} lớp: {labels}...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu', 
        solver='adam', 
        max_iter=500, 
        verbose=True # Hiện quá trình học ra màn hình
    )

    # 5. Bắt đầu học
    mlp.fit(X_train, y_train)

    # 6. Đánh giá kết quả
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n HUẤN LUYỆN HOÀN TẤT!")
    print(f" Độ chính xác: {acc * 100:.2f}%")

    # 7. Lưu "Bộ não" xuống ổ cứng để code Camera sử dụng
    with open(MODEL_OUT_PATH, 'wb') as f:
        pickle.dump(mlp, f)
    print(f"Đã lưu model tại: {MODEL_OUT_PATH}")

if __name__ == "__main__":
    train_brain()
