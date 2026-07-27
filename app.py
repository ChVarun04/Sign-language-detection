import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

from backend.hand_landmarker import create_hand_landmarker
from backend.preprocess import prepare_hand
from backend.predictor import predict_gesture
from backend.drawing_utils import (
    draw_landmarks,
    draw_bounding_box_and_label
)

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ============================================
# CSS
# ============================================

st.markdown("""
<style>

.stApp{
background:#07111F;
color:white;
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#00E5FF;
}

.subtitle{
text-align:center;
font-size:18px;
color:#B0BEC5;
margin-bottom:30px;
}

.card{

background:#0E1B2A;

padding:18px;

border-radius:12px;

text-align:center;

border:1px solid #263238;

}

.metric{

font-size:32px;

font-weight:bold;

color:#00E676;

}

</style>
""",unsafe_allow_html=True)

# ============================================
# TITLE
# ============================================

st.markdown(
'<p class="title">🤟 Sign Language Recognition</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Machine Learning + MediaPipe + Streamlit</p>',
unsafe_allow_html=True
)

# ============================================
# INFORMATION CARDS
# ============================================

c1,c2,c3=st.columns(3)

with c1:

    st.markdown("""
<div class="card">
<h4>Dataset</h4>
<p class="metric">55,500</p>
Images
</div>
""",unsafe_allow_html=True)

with c2:

    st.markdown("""
<div class="card">
<h4>Classes</h4>
<p class="metric">37</p>
0-9, A-Z, _
</div>
""",unsafe_allow_html=True)

with c3:

    st.markdown("""
<div class="card">
<h4>Input Size</h4>
<p class="metric">64×64</p>
Grayscale
</div>
""",unsafe_allow_html=True)

st.divider()

# ============================================
# LOAD MEDIAPIPE
# ============================================

@st.cache_resource
def load_landmarker():

    return create_hand_landmarker()

landmarker=load_landmarker()

# ============================================
# CAMERA
# ============================================

camera_image=st.camera_input(
"📷 Capture Hand Gesture"
)

if camera_image is None:

    st.info("Capture an image to start prediction.")

    st.stop()

# ============================================
# READ IMAGE
# ============================================

bytes_data=camera_image.getvalue()

np_array=np.frombuffer(
bytes_data,
np.uint8
)

image_bgr=cv2.imdecode(
np_array,
cv2.IMREAD_COLOR
)

image_rgb=cv2.cvtColor(
image_bgr,
cv2.COLOR_BGR2RGB
)

display=image_bgr.copy()

# ============================================
# MEDIAPIPE
# ============================================

mp_image=mp.Image(
image_format=mp.ImageFormat.SRGB,
data=image_rgb
)

results=landmarker.detect(
mp_image
)

if not results.hand_landmarks:

    st.error("❌ No hand detected")

    st.image(image_rgb)

    st.stop()

hand=results.hand_landmarks[0]

# ============================================
# DRAW LANDMARKS
# ============================================

draw_landmarks(
    display,
    hand
)

# ============================================
# PREPROCESS
# ============================================

try:

    hand_crop, processed, features, bbox = prepare_hand(
        image_rgb,
        hand
    )

except Exception as e:

    st.error(f"Preprocessing Error : {e}")

    st.stop()

# ============================================
# PREDICTION
# ============================================

prediction, confidence = predict_gesture(
    features
)

# ============================================
# DRAW BOUNDING BOX
# ============================================

draw_bounding_box_and_label(
    display,
    bbox,
    prediction,
    confidence
)

# ============================================
# DISPLAY IMAGES
# ============================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("📷 Original Image")

    st.image(
        image_rgb,
        use_container_width=True
    )

with col2:

    st.subheader("✋ Detected Hand")

    st.image(
        cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

with col3:

    st.subheader("🖼️ Cropped Hand")

    st.image(
        hand_crop,
        use_container_width=True
    )

st.divider()

# ============================================
# RESULTS
# ============================================

st.subheader("🎯 Prediction Result")

left, right = st.columns([1, 1])

with left:

    st.success(f"### Gesture : {prediction}")

    if confidence is not None:

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

with right:

    st.info(
        """
### Model Information

**Model :** Logistic Regression / KNN Pipeline

**Input :**
- RGB Image
- 64 × 64
- Grayscale

**Feature Size :**
4096 Pixels
"""
    )

st.divider()

# ============================================
# PIPELINE
# ============================================

st.subheader("⚙️ Recognition Pipeline")

st.markdown("""
1. 📷 Capture Image
2. ✋ Detect Hand using MediaPipe
3. ✂ Crop Hand Region
4. ⚫ Convert to Grayscale
5. 📏 Resize to **64 × 64**
6. 🔢 Normalizstreamlit rue Pixel Values
7. 📊 Flatten Image (4096 Features)
8. 🤖 Predict Gesture using Trained Model
""")

st.divider()

# ============================================
# SHOW PREPROCESSED IMAGE
# ============================================

st.subheader("🖤 Preprocessed Image")

st.image(
    processed,
    clamp=True,
    channels="GRAY",
    width=250
)

st.divider()

# ============================================
# USER GUIDE
# ============================================

st.info("""
### Tips for Better Accuracy

✅ Keep the entire hand visible.

✅ Use a plain background.

✅ Maintain a distance of approximately 20–30 cm from the camera.

✅ Ensure good lighting.

✅ Keep fingers fully inside the camera frame.
""")

st.success("✅ System Ready for Prediction")