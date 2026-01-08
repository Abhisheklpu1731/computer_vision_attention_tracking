import streamlit as st
import os
from joblib import load

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Eye + Head Engagement Detection",
    layout="wide"
)

st.title("👁️ Eye + Head Engagement Detection (ML)")
st.write("Streamlit Cloud–safe deployment")

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "eye_head_model.joblib")

if not os.path.exists(MODEL_PATH):
    st.error("❌ eye_head_model.joblib not found in repository")
    st.stop()

model = load(MODEL_PATH)

st.success("✅ ML model loaded successfully")

# ================= MEDIAPIPE CHECK =================
MEDIAPIPE_OK = False
MEDIAPIPE_ERROR = None

try:
    import mediapipe as mp

    # MediaPipe installs on Streamlit Cloud but backend is missing
    if hasattr(mp, "solutions"):
        MEDIAPIPE_OK = True
    else:
        MEDIAPIPE_ERROR = "MediaPipe backend not available (Python 3.13)"

except Exception as e:
    MEDIAPIPE_ERROR = str(e)

# ================= UI =================
st.divider()

if not MEDIAPIPE_OK:
    st.error("🚫 MediaPipe is NOT supported on Streamlit Cloud")

    st.markdown(
        """
### Why this happens
- Streamlit Cloud runs **Python 3.13**
- MediaPipe **does not support Python 3.13**
- `mediapipe.solutions` is missing → runtime failure

### What is working
- ✅ Your ML model
- ✅ Your code logic
- ❌ Real-time face / eye tracking

### How to run this project correctly
**You must use one of the following:**
- Local machine
- Docker (Python 3.10)
- AWS EC2 / Fly.io / Render

This is a **platform limitation**, not a bug in your code.
"""
    )

    st.info(
        "If you want, I can give you:\n"
        "- Docker + EC2 deployment\n"
        "- Local real-time version\n"
        "- WebRTC-based webcam app\n"
        "- Streamlit demo without MediaPipe"
    )

    st.stop()

# ================= THIS PART NEVER RUNS ON STREAMLIT CLOUD =================
# It WILL run locally / Docker / EC2

import cv2
import numpy as np
import time

st.success("✅ MediaPipe backend detected")

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
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

st.warning("This section runs ONLY outside Streamlit Cloud")

