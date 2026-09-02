import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# connections for drawing the skeleton on screen
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("done")


def fingers_up(landmarks, handedness_label):
    up = {}
    for name in ["index", "middle", "ring", "pinky"]:
        tip_y = landmarks[FINGER_TIPS[name]][1]
        pip_y = landmarks[FINGER_PIPS[name]][1]
        up[name] = tip_y < pip_y

    # thumb is sideways so this needs x not y, and it flips depending on hand
    thumb_tip_x = landmarks[4][0]
    thumb_ip_x = landmarks[3][0]
    if handedness_label == "Right":
        up["thumb"] = thumb_tip_x > thumb_ip_x
    else:
        up["thumb"] = thumb_tip_x < thumb_ip_x

    return up


def classify_gesture(up):
    thumb, index, middle, ring, pinky = up["thumb"], up["index"], up["middle"], up["ring"], up["pinky"]

    if thumb and not index and not middle and not ring and not pinky:
        return "YES"
    if index and middle and not ring and not pinky and not thumb:
        return "PEACE"
    if not thumb and not index and not middle and not ring and not pinky:
        return "STOP"
    if thumb and index and middle and ring and pinky:
        return "HELLO"
    if index and not middle and not ring and not pinky and not thumb:
        return "POINT"
    if thumb and pinky and not index and not middle and not ring:
        return "CALL ME"
    return ""


def draw_hand(frame, landmarks_px):
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start], landmarks_px[end], (0, 200, 0), 2)
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no webcam found")
        return

    sentence = []
    last_commit_time = 0
    COMMIT_COOLDOWN = 0.6
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_index * (1000 / 30))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_index += 1

        current_word = ""

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            handedness_label = result.handedness[0][0].category_name

            h, w, _ = frame.shape
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            draw_hand(frame, landmarks_px)

            up = fingers_up(coords, handedness_label)
            current_word = classify_gesture(up)

        h, w, _ = frame.shape
        overlay_h = 110
        cv2.rectangle(frame, (0, 0), (w, overlay_h), (0, 0, 0), -1)

        cv2.putText(frame, f"Detected: {current_word}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        sentence_text = " ".join(sentence)
        cv2.putText(frame, sentence_text[-60:], (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "[SPACE]=add  [BACKSPACE]=undo  [c]=clear  [q]=quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign to Text", frame)

        if cv2.getWindowProperty("Sign to Text", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == ord("q"):
            break
        elif key == 32:
            if current_word and (now - last_commit_time) > COMMIT_COOLDOWN:
                sentence.append(current_word)
                last_commit_time = now
        elif key == 8:
            if sentence:
                sentence.pop()
        elif key == ord("c"):
            sentence = []

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
