import pickle
import numpy as np

def load_dataset(pkl_path):
    # Load dữ liệu từ file pkl
    with open(pkl_path, 'rb') as f:
        X, y, labels = pickle.load(f)

    # Ép kiểu & chuẩn hóa (tối ưu hiệu suất)
    X = X.astype('float32') / 255.0
    y = y.astype('int64')

    return X, y, labels


# ====== GỌI HÀM LOAD ======
PKL_PATH = r'D:\GITHUB-NM\anh.pkl'

X, y, labels = load_dataset(PKL_PATH)

# ====== KIỂM TRA ======
print('✅ Load dữ liệu thành công')
print('X shape:', X.shape)
print('y shape:', y.shape)
print('Labels:', labels)

# ====== TRUY CẬP MẪU ======
i = 0
print('Ảnh thứ 0 thuộc chữ:', labels[y[i]])
