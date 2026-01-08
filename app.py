import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from joblib import load
import time

st.set_page_config(page_title="Eye + Head Engagement Detection", layout="wide")

st.title("👁️ Eye + Head Engagement Detection (ML)")
st.write("Real-time engagement detection using trained Random Forest model")

# ================= LOAD MODEL =================
model = load("eye_head_model.joblib")

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

# ================= UI =================
run = st.checkbox("▶ Start Camera")

FRAME_WINDOW = st.image([])
status_box = st.empty()

# ================= Webcam =================
cap = None

def draw_box(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (0,0,255), 2)

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam")
            break

        h, w = frame.shape[:2]
        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prediction = 0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # ===== FEATURES =====
            head_yaw = lm[NOSE].x - 0.5

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
            gaze_avg = (gaze_left + gaze_right) / 2
            eye_disagree = abs(gaze_left - gaze_right)

            X = [[
                head_yaw,
                gaze_left,
                gaze_right,
                gaze_avg,
                eye_disagree
            ]]

            prediction = model.predict(X)[0]

            # ===== DRAW UI =====
            # Face box
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            cv2.rectangle(
                frame,
                (int(min(xs)), int(min(ys))),
                (int(max(xs)), int(max(ys))),
                (0,0,255),
                2
            )

            draw_box(frame, lm, LEFT_EYE + LEFT_IRIS, w, h)
            draw_box(frame, lm, RIGHT_EYE + RIGHT_IRIS, w, h)

            cv2.circle(frame, (int(li_x*w), int(li_y*h)), 4, (0,0,255), -1)
            cv2.circle(frame, (int(ri_x*w), int(ri_y*h)), 4, (0,0,255), -1)

        label = "ENGAGED" if prediction == 1 else "NOT ENGAGED"
        color = (0,200,0) if prediction == 1 else (0,0,255)

        cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
        cv2.putText(
            frame,
            f"STATUS: {label}",
            (10,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status_box.markdown(f"### 🧠 Prediction: **{label}**")

        time.sleep(0.03)

    cap.release()
