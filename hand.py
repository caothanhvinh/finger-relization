import cv2
import mediapipe as mp
import numpy as np
import pickle

# ================= PATH =================
PKL_PATH = r'D:\GITHUB-NM\anh.pkl'
MODEL_PATH = r'D:\GITHUB-NM\model_sign_language.sav'

# ================= LOAD DATASET (LABELS) =================
with open(PKL_PATH, 'rb') as f:
    X, y, labels = pickle.load(f)

print(" Labels:", labels)

# ================= LOAD MODEL =================
with open(MODEL_PATH, 'rb') as f:
    mlp = pickle.load(f)

print(" Model loaded successfully!")

# ================= MEDIAPIPE INIT =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

PADDING = 40
ASPECT_RATIO = 1.0
STILLNESS_THRESHOLD = 5

# ================= FUNCTIONS =================
def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb)


def is_right_hand(handedness):
    # MediaPipe bị mirror → Left = tay phải người dùng
    return handedness.classification[0].label == 'Left'


def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)

    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(w, x_max + PADDING)
    y_max = min(h, y_max + PADDING)

    return x_min, y_min, x_max, y_max


def enforce_square(x_min, y_min, x_max, y_max, frame_shape):
    h, w, _ = frame_shape
    bw = x_max - x_min
    bh = y_max - y_min

    if bw > bh:
        diff = bw - bh
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + diff // 2)
    else:
        diff = bh - bw
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + diff // 2)

    return x_min, y_min, x_max, y_max


def is_hand_moving(current, previous, threshold, frame_shape):
    if previous is None:
        return True

    h, w, _ = frame_shape
    total = 0

    for c, p in zip(current, previous):
        cx, cy = int(c.x * w), int(c.y * h)
        px, py = int(p.x * w), int(p.y * h)
        total += np.sqrt((cx - px)**2 + (cy - py)**2)

    return (total / len(current)) > threshold


# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(0)
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, handedness in zip(result.multi_hand_landmarks,
                                           result.multi_handedness):

                if not is_right_hand(handedness):
                    continue

                x_min, y_min, x_max, y_max = calculate_bounding_box(hand_lm, frame.shape)
                x_min, y_min, x_max, y_max = enforce_square(
                    x_min, y_min, x_max, y_max, frame.shape
                )

                color = (0, 0, 255)  # đỏ
                predicted_char = ""

                if not is_hand_moving(hand_lm.landmark, prev_landmarks,
                                      STILLNESS_THRESHOLD, frame.shape):

                    hand_crop = frame[y_min:y_max, x_min:x_max]

                    if hand_crop.size > 0:
                        hand_crop = cv2.resize(hand_crop, (64, 64))
                        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                        hand_crop = hand_crop.astype('float32') / 255.0
                        hand_crop = hand_crop.reshape(1, -1)

                        y_pred = mlp.predict(hand_crop)[0]
                        predicted_char = labels[y_pred]
                        color = (0, 255, 0)  # xanh

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                if predicted_char:
                    cv2.putText(
                        frame,
                        predicted_char,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3
                    )

                prev_landmarks = hand_lm.landmark

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
