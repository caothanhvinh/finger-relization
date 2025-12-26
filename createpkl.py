import os
import cv2
import pickle
import numpy as np

# ====== CẤU HÌNH ======
DATASET_DIR = r'D:\GITHUB-NM\anh'      # folder ảnh
IMG_SIZE = 64                    # resize ảnh
OUTPUT_PKL = r'D:\GITHUB-NM\anh.pkl'   # file pkl lưu ở ổ D
# ======================

X = []
y = []

# Lấy danh sách folder (A, B, C...)
labels = sorted(os.listdir(DATASET_DIR))

for label_index, label_name in enumerate(labels):
    folder_path = os.path.join(DATASET_DIR, label_name)

    if not os.path.isdir(folder_path):
        continue

    print(f'📂 Đang xử lý folder: {label_name}')

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(label_index)

# Chuyển sang numpy array
X = np.array(X)
y = np.array(y)

# Lưu file pkl vào ổ D
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump((X, y, labels), f)

print('\n✅ HOÀN THÀNH!')
print('📦 File pkl nằm tại:', OUTPUT_PKL)
print('📊 X shape:', X.shape)
print('📊 y shape:', y.shape)
print('🔤 Labels:', labels)
for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)
    print('Đang đọc:', img_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('❌ Không đọc được ảnh')
        continue

    print('✅ Đọc OK')
