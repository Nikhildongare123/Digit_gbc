
import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------------------
# 1. Load the trained model
# ------------------------------
@st.cache_resource
def load_model():
    with open("digits_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ------------------------------
# 2. App title & description
# ------------------------------
st.set_page_config(page_title="Handwritten Digit Recogniser by Pratik The Great 💪🏼😎", page_icon="✍️")
st.title("✍️ Handwritten Digit Recogniser")
st.markdown("Draw a digit (0‑9) in the box below. The model will predict it instantly.")

# ------------------------------
# 3. Canvas for drawing
# ------------------------------
# Use a session state key to reset the canvas when "Clear" is pressed
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Drawing canvas settings
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",   # white fill
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=280,          # canvas width in pixels
    height=280,         # canvas height in pixels
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
    update_streamlit=True,
)

# Clear button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🗑️ Clear"):
        st.session_state.canvas_key += 1
        st.rerun()

# ------------------------------
# 4. Preprocess and predict
# ------------------------------
if canvas_result.image_data is not None:
    # Convert canvas image (RGB) to grayscale PIL image
    img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGB")
    img_gray = img.convert("L")   # 8‑bit grayscale, 0 = black, 255 = white

    # Resize to 8×8 (same as original digits dataset)
    img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to numpy array and flatten
    pixels = np.array(img_resized, dtype=np.float32).reshape(64)

    # Invert and scale to match the original range (0 = background, 16 = digit)
    # Original: white=0, black=16
    # Canvas:   white=255, black=0
    pixels = 16 - (pixels / 255.0 * 16)
    # Clip to [0,16] for safety
    pixels = np.clip(pixels, 0, 16)

    # Predict
    pred = model.predict([pixels])[0]
    proba = model.predict_proba([pixels])[0]

    # Show prediction
    st.subheader(f"🔮 Prediction: **{pred}**")
    st.progress(float(proba[pred]), text=f"Confidence: {proba[pred]:.2%}")

    # Optional: display the small 8×8 image that the model actually sees
    st.markdown("**What the model sees (8×8 resized & scaled)**")
    small_img = (pixels / 16.0 * 255).astype(np.uint8).reshape(8, 8)
    st.image(small_img, caption="8×8 input to the model", width=150, clamp=True)
else:
    st.info("✏️ Draw a digit on the canvas above – the prediction will appear here.")
