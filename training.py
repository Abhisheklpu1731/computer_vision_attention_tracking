import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# ================= CONFIG =================
MIN_SAMPLES = 100
CAMERA_INDEX = 0

# ================= MediaPipe =================
mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
NOSE = 1

# ================= Webcam =================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

rows = []

print("TRAINING MODE")
print("Press:")
print("  E = ENGAGED")
print("  N = NOT ENGAGED")
print("ESC = finish after 100 samples")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    key = cv2.waitKey(1) & 0xFF
    label = None
    if key == ord('e'):
        label = 1
    elif key == ord('n'):
        label = 0
    elif key == 27 and len(rows) >= MIN_SAMPLES:
        break

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # ================= HEAD FEATURE =================
        head_yaw = lm[NOSE].x - 0.5

        # ================= EYE FEATURES =================
        li_x = np.mean([lm[i].x for i in LEFT_IRIS])
        li_y = np.mean([lm[i].y for i in LEFT_IRIS])

        ri_x = np.mean([lm[i].x for i in RIGHT_IRIS])
        ri_y = np.mean([lm[i].y for i in RIGHT_IRIS])

        le_l = lm[LEFT_EYE[0]].x
        le_r = lm[LEFT_EYE[1]].x
        re_l = lm[RIGHT_EYE[0]].x
        re_r = lm[RIGHT_EYE[1]].x

        gaze_left = (li_x - le_l) / (le_r - le_l + 1e-6)
        gaze_right = (ri_x - re_l) / (re_r - re_l + 1e-6)
        gaze_avg = (gaze_left + gaze_right) / 2.0
        eye_disagree = abs(gaze_left - gaze_right)

        # ================= SAVE SAMPLE =================
        if label is not None:
            rows.append([
                head_yaw,
                gaze_left,
                gaze_right,
                gaze_avg,
                eye_disagree,
                label
            ])

        # ================= DRAW FACE BOX =================
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ================= DRAW EYE BOXES =================
        def draw_box(indices):
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cv2.rectangle(
                frame,
                (min(xs), min(ys)),
                (max(xs), max(ys)),
                (0, 0, 255),
                2
            )

        draw_box(LEFT_EYE + LEFT_IRIS)
        draw_box(RIGHT_EYE + RIGHT_IRIS)

        # ================= DRAW IRIS DOTS =================
        cv2.circle(frame, (int(li_x * w), int(li_y * h)), 4, (0, 0, 255), -1)
        cv2.circle(frame, (int(ri_x * w), int(ri_y * h)), 4, (0, 0, 255), -1)

    # ================= UI OVERLAY =================
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Samples: {len(rows)}/{MIN_SAMPLES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "E=Engaged  N=Not  ESC=Finish", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Training Eye + Head Model", frame)

cap.release()
cv2.destroyAllWindows()
mp_face.close()

# ================= TRAIN MODEL =================
df = pd.DataFrame(rows, columns=[
    "head_yaw",
    "gaze_left",
    "gaze_right",
    "gaze_avg",
    "eye_disagree",
    "label"
])

X = df.drop(columns=["label"])
y = df["label"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X, y)
dump(model, "eye_head_model.joblib")

print("\n✔ Training completed")
print("✔ Model saved as eye_head_model.joblib")
print("✔ Total samples:", len(df))
