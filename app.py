import streamlit as st
import numpy as np
import pickle
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time

# ------------------------------
# Page Config (MUST be first Streamlit command)
# ------------------------------
st.set_page_config(
    page_title="AI Digit Recogniser", 
    page_icon="✍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    try:
        with open("digits_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'digits_model.pkl' not found! Please train and save the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ------------------------------
# Title and Description
# ------------------------------
st.title("🤖 AI Handwritten Digit Recogniser")
st.markdown("Draw a digit (0-9) and the AI will predict it in real-time!")

# ------------------------------
# Sidebar Settings
# ------------------------------
st.sidebar.header("🎨 Drawing Settings")
brush_size = st.sidebar.slider("Brush Size", 5, 30, 15)
brush_color = st.sidebar.color_picker("Brush Color", "#000000")
bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.write("🧠 **Model:** Gradient Boosting Classifier")
st.sidebar.write("📊 **Dataset:** Scikit-learn digits (8x8 images)")
st.sidebar.write("🎯 **Accuracy:** ~96% on test set")
st.sidebar.write("✏️ **Tip:** Draw clear, centered digits for best results")

st.sidebar.markdown("---")
st.sidebar.write("💡 **How to use:**")
st.sidebar.write("1. Draw a digit in the canvas")
st.sidebar.write("2. AI automatically predicts")
st.sidebar.write("3. Click 'Clear' to start over")

# ------------------------------
# Canvas Reset Logic
# ------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ------------------------------
# Main Layout
# ------------------------------
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### ✏️ Drawing Canvas")
    canvas_result = st_canvas(
        stroke_width=brush_size,
        stroke_color=brush_color,
        background_color=bg_color,
        height=350,
        width=350,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        display_toolbar=True,
    )
    
    col_clear, col_info = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.last_prediction = None
            st.rerun()
    with col_info:
        st.caption("💡 Use mouse/touch to draw. Click Clear to erase.")

with col2:
    st.markdown("### 📊 Prediction Result")
    
    # Create placeholder for real-time updates
    result_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    if canvas_result.image_data is not None:
        # Check if canvas has any drawing (not empty)
        if np.sum(canvas_result.image_data[:, :, :3]) > 0:
            
            # Process image for model
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGB")
            img_gray = img.convert("L")
            img_resized = img_gray.resize((8, 8))
            
            # Convert to model format (invert colors for black background)
            pixels = np.array(img_resized).reshape(64)
            pixels = 16 - (pixels / 255.0 * 16)
            pixels = np.clip(pixels, 0, 16)
            
            # Make prediction
            pred = model.predict([pixels])[0]
            proba = model.predict_proba([pixels])[0]
            
            # Update session state
            st.session_state.last_prediction = pred
            
            # Display results
            with result_placeholder.container():
                # Main prediction
                st.markdown(f"## 🎯 **Prediction: {pred}**")
                
                # Confidence bar
                confidence = proba[pred]
                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                st.progress(float(confidence))
                
                # Top 3 predictions
                st.markdown("### 🔝 Top 3 Predictions")
                top3_indices = np.argsort(proba)[-3:][::-1]
                
                for idx, i in enumerate(top3_indices):
                    if i == pred:
                        st.markdown(f"**{i}** → {proba[i]*100:.2f}% ✨")
                    else:
                        st.markdown(f"{i} → {proba[i]*100:.2f}%")
                
                # Confidence chart
                st.markdown("### 📈 Confidence Distribution")
                chart_data = {str(i): float(proba[i]) for i in range(10)}
                st.bar_chart(chart_data, height=300)
                
                # Model's view of the digit
                st.markdown("### 👁️ How the AI sees your digit")
                small_img = (pixels / 16.0 * 255).astype(np.uint8).reshape(8, 8)
                st.image(small_img, width=150, caption="8x8 pixel representation")
        else:
            result_placeholder.info("✏️ Start drawing a digit above...")
    else:
        result_placeholder.info("✏️ Draw a digit on the canvas to begin")

# ------------------------------
# Tips Section
# ------------------------------
st.markdown("---")
st.markdown("### 💡 Tips for Better Recognition")
col_tip1, col_tip2, col_tip3, col_tip4 = st.columns(4)

with col_tip1:
    st.markdown("**✅ Do's**")
    st.markdown("- Draw large and clear")
    st.markdown("- Center the digit")
    st.markdown("- Use thick strokes")

with col_tip2:
    st.markdown("**❌ Don'ts**")
    st.markdown("- Don't draw too small")
    st.markdown("- Avoid overlapping")
    st.markdown("- No extra dots/lines")

with col_tip3:
    st.markdown("**🎯 Best Practices**")
    st.markdown("- Similar to MNIST style")
    st.markdown("- Black on white background")
    st.markdown("- Fill the canvas area")

with col_tip4:
    st.markdown("**🔍 Common Issues**")
    st.markdown("- '0' vs '6' vs '8'")
    st.markdown("- '1' vs '7'")
    st.markdown("- '3' vs '5' vs '8'")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <p>Made with ❤️ by <strong>Nikhil Dongare</strong> 🚀</p>
        <p style='font-size: 12px;'>Powered by Streamlit, Scikit-learn, and Streamlit Drawable Canvas</p>
    </div>
    """, 
    unsafe_allow_html=True
)
