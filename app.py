import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    with open("digits_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="AI Digit Recogniser", page_icon="✍️", layout="wide")

st.title("🤖 AI Handwritten Digit Recogniser")
st.markdown("Draw a digit (0-9) and AI will predict it!")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("🎨 Settings")
brush_size = st.sidebar.slider("Brush Size", 5, 30, 15)
brush_color = st.sidebar.color_picker("Brush Color", "#000000")
bg_color = st.sidebar.color_picker("Background", "#FFFFFF")

st.sidebar.markdown("---")
st.sidebar.write("🧠 Model: Gradient Boosting")
st.sidebar.write("📊 Dataset: sklearn digits")

# ------------------------------
# Canvas Reset
# ------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([3, 2])

with col1:
    canvas_result = st_canvas(
        stroke_width=brush_size,
        stroke_color=brush_color,
        background_color=bg_color,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    if st.button("🗑️ Clear"):
        st.session_state.canvas_key += 1
        st.rerun()

# ------------------------------
# Prediction
# ------------------------------
with col2:
    st.subheader("📊 Result")

    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGB")
        img = img.convert("L")
        img = img.resize((8, 8))

        pixels = np.array(img).reshape(64)
        pixels = 16 - (pixels / 255.0 * 16)
        pixels = np.clip(pixels, 0, 16)

        pred = model.predict([pixels])[0]
        proba = model.predict_proba([pixels])[0]

        st.success(f"🎯 Prediction: {pred}")
        st.progress(float(proba[pred]))

        st.markdown("### 🔝 Top 3 Predictions")
        top3 = np.argsort(proba)[-3:][::-1]
        for i in top3:
            st.write(f"{i} → {proba[i]*100:.2f}%")

        st.bar_chart(proba)

        st.markdown("### 👁️ Model View")
        small_img = (pixels / 16.0 * 255).astype(np.uint8).reshape(8, 8)
        st.image(small_img, width=150)

    else:
        st.info("Draw a digit ✏️")

st.markdown("---")
st.markdown("Made by Nikhil Dongare 🚀")
