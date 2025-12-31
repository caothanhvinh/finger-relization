import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# 1. Cấu hình
model = tf.keras.models.load_model('sign_language_mobilenet.h5')
DATA_PATH = r"C:\Users\PT\PyCharmMiscProject\train_data_6"
labels = sorted(os.listdir(DATA_PATH))

# 2. Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Nhận diện bàn tay
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Lấy tọa độ các điểm để vẽ khung bao quanh tay
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max: x_max = x
                if x < x_min: x_min = x
                if y > y_max: y_max = y
                if y < y_min: y_min = y

            # Mở rộng khung một chút để không bị sát tay quá
            y_min, y_max = max(0, y_min - 20), min(h, y_max + 20)
            x_min, x_max = max(0, x_min - 20), min(w, x_max + 20)

            # Vẽ khung di chuyển theo tay
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Cắt vùng bàn tay và Predict
            try:
                hand_img = frame[y_min:y_max, x_min:x_max]
                img = cv2.resize(hand_img, (224, 224))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]

                if confidence > 0.6:  # Chỉ hiện nếu chắc chắn trên 60%
                    label_text = f"{labels[class_idx]}"
                    cv2.putText(frame, label_text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                pass

            # Vẽ các điểm nối xương tay (optional)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Smart Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()