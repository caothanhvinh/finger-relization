import cv2
import mediapipe as mp
import numpy as np
import pickle

# ================= PATH =================
PKL_PATH = r'D:\GITHUB-NM\anh.pkl'
MODEL_PATH = r'D:\GITHUB-NM\model_sign_language.sav'

# ================= LOAD DATASET =================
with open(PKL_PATH, 'rb') as f:
    X, y, labels = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    mlp = pickle.load(f)

# ================= MEDIAPIPE INIT =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

PADDING = 20
STILLNESS_THRESHOLD = 5

# ================= FUNCTIONS =================
def is_right_hand(handedness):
    # MediaPipe mirror → Left = tay phải người dùng
    return handedness.classification[0].label == 'Left'


def hand_bbox_from_landmarks(hand_landmarks, frame_shape):
    h, w, _ = frame_shape

    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    x_min = max(0, min(xs) - PADDING)
    y_min = max(0, min(ys) - PADDING)
    x_max = min(w, max(xs) + PADDING)
    y_max = min(h, max(ys) + PADDING)

    return x_min, y_min, x_max, y_max


def is_hand_still(curr, prev, threshold, frame_shape):
    if prev is None:
        return False

    h, w, _ = frame_shape
    dist = 0

    for c, p in zip(curr, prev):
        cx, cy = int(c.x * w), int(c.y * h)
        px, py = int(p.x * w), int(p.y * h)
        dist += np.sqrt((cx - px)**2 + (cy - py)**2)

    return (dist / len(curr)) < threshold


# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(0)
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                if not is_right_hand(handedness):
                    continue

                x1, y1, x2, y2 = hand_bbox_from_landmarks(
                    hand_lm, frame.shape
                )

                if is_hand_still(
                    hand_lm.landmark,
                    prev_landmarks,
                    STILLNESS_THRESHOLD,
                    frame.shape
                ):
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        crop = cv2.resize(crop, (64, 64))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        crop = crop.astype("float32") / 255.0
                        crop = crop.reshape(1, -1)

                        pred = mlp.predict(crop)[0]
                        char = labels[pred]

                        cv2.putText(
                            frame,
                            char,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (255, 255, 255),
                            2
                        )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS
                )

                prev_landmarks = hand_lm.landmark

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
