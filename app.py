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
        model = pickle.load(f)
    return model

model = load_model()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="🔥 AI Digit Recogniser",
    page_icon="✍️",
    layout="wide"
)

st.title("🤖 AI Handwritten Digit Recogniser")
st.markdown("Draw a digit (0-9) and let Gradient Boosting predict it 🚀")

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("🎨 Drawing Settings")

brush_size = st.sidebar.slider("Brush Size", 5, 30, 15)
brush_color = st.sidebar.color_picker("Brush Color", "#000000")
bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Info")
st.sidebar.write("Model: Gradient Boosting")
st.sidebar.write("Dataset: sklearn digits (8x8)")
st.sidebar.write("Classes: 0 - 9")

# ------------------------------
# Canvas Reset
# ------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([3, 2])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=brush_size,
        stroke_color=brush_color,
        background_color=bg_color,
        width=300,
        height=300,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    if st.button("🗑️ Clear Canvas"):
        st.session_state.canvas_key += 1
        st.rerun()

# ------------------------------
# Prediction Section
# ------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGB")
        img = img.convert("L")
        img = img.resize((8, 8))

        pixels = np.array(img).reshape(64)
        pixels = 16 - (pixels / 255.0 * 16)
        pixels = np.clip(pixels, 0, 16)

        pred = model.predict([pixels])[0]
        proba = model.predict_proba([pixels])[0]

        # 🔮 Main Prediction
        st.success(f"🎯 Predicted Digit: {pred}")
        st.progress(float(proba[pred]))

        # 🔝 Top 3 Predictions
        st.markdown("### 🔝 Top Predictions")
        top3 = np.argsort(proba)[-3:][::-1]

        for i in top3:
            st.write(f"Digit {i} → {proba[i]*100:.2f}%")

        # 📈 Confidence Chart
        st.markdown("### 📈 Confidence Chart")
        st.bar_chart(proba)

        # 👁️ What model sees
        st.markdown("### 👁️ Model View (8x8)")
        small_img = (pixels / 16.0 * 255).astype(np.uint8).reshape(8, 8)
        st.image(small_img, width=150)

    else:
        st.info("✏️ Draw something to see prediction")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Made with ❤️ by Nikhil Dongare")
