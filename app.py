import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
import cv2

# Set random seed for reproducibility
SEED = 42

# Load and train the model
@st.cache_resource
def train_model():
    # Load digits dataset
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    
    # Train Gradient Boosting model
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=SEED
    )
    gb_model.fit(x_train, y_train)
    
    # Calculate accuracy
    train_score = gb_model.score(x_train, y_train)
    test_score = gb_model.score(x_test, y_test)
    
    return gb_model, train_score, test_score

# Preprocess the drawn image to match digit dataset format
def preprocess_drawing(image_data):
    """
    Convert canvas drawing to the same format as load_digits (8x8 images)
    """
    # Convert to PIL Image
    if image_data is None:
        return None
    
    # The canvas returns RGBA image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 8x8 (same as digits dataset)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert if needed (digits dataset has white background, dark digits)
    # Canvas has dark background, white drawing by default
    # So we need to invert: white drawing becomes dark, dark background becomes white
    img_array = 255 - img_array
    
    # Normalize to 0-16 range (digits dataset uses 0-16)
    img_array = (img_array / 255.0) * 16
    
    # Flatten to 64 pixels (8x8 = 64)
    img_flattened = img_array.flatten()
    
    return img_flattened

# Main app
def main():
    st.set_page_config(
        page_title="Handwritten Digit Recognizer",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Handwritten Digit Recognizer")
    st.markdown("### Draw a digit (0-9) and let the AI recognize it!")
    
    # Train the model
    with st.spinner("Training Gradient Boosting Model..."):
        model, train_acc, test_acc = train_model()
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"**Algorithm:** Gradient Boosting Classifier")
        st.markdown(f"**Training Accuracy:** {train_acc:.2%}")
        st.markdown(f"**Test Accuracy:** {test_acc:.2%}")
        st.markdown("---")
        st.markdown("**Dataset:** Load Digits (8x8 images)")
        st.markdown("**Number of classes:** 10 (0-9)")
        st.markdown("**Features:** 64 (8x8 pixels)")
        st.markdown("---")
        st.markdown("### Instructions:")
        st.markdown("1. Draw a digit in the canvas below")
        st.markdown("2. Click 'Predict' to recognize the digit")
        st.markdown("3. Use 'Clear' to start over")
        st.markdown("4. Adjust brush size and color as needed")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Draw Your Digit Here")
        
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Add buttons for actions
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn3:
            if st.button("ℹ️ About", use_container_width=True):
                st.info("This model recognizes handwritten digits from 0-9. Try writing clearly for best results!")
        
        # Clear canvas functionality
        if clear_btn:
            st.rerun()
    
    with col2:
        st.subheader("🤖 Prediction Result")
        
        if predict_btn and canvas_result.image_data is not None:
            # Preprocess the drawn image
            processed_digit = preprocess_drawing(canvas_result.image_data)
            
            if processed_digit is not None:
                # Make prediction
                processed_digit_reshaped = processed_digit.reshape(1, -1)
                prediction = model.predict(processed_digit_reshaped)[0]
                prediction_proba = model.predict_proba(processed_digit_reshaped)[0]
                
                # Display prediction with confidence
                st.markdown(f"## 🎯 Predicted Digit: **{prediction}**")
                
                # Show confidence bar
                confidence = prediction_proba[prediction] * 100
                st.markdown(f"### Confidence: {confidence:.1f}%")
                st.progress(confidence / 100)
                
                # Show top 3 predictions
                st.markdown("---")
                st.markdown("### Top 3 Predictions:")
                top_3_idx = np.argsort(prediction_proba)[-3:][::-1]
                for idx in top_3_idx:
                    prob = prediction_proba[idx] * 100
                    st.markdown(f"**Digit {idx}:** {prob:.1f}%")
                    st.progress(prob / 100)
                
                # Display preprocessed image (for debugging/info)
                with st.expander("See preprocessed image (8x8)"):
                    preprocessed_img = processed_digit.reshape(8, 8)
                    # Scale up for better visibility
                    scaled_img = cv2.resize(preprocessed_img, (160, 160), interpolation=cv2.INTER_NEAREST)
                    st.image(scaled_img, caption="Preprocessed 8x8 image", use_container_width=True)
                    st.caption("This is how the model sees your drawing")
                    
        elif predict_btn and canvas_result.image_data is None:
            st.warning("Please draw a digit first!")
    
    # Add some styling
    st.markdown("""
    <style>
    .stButton button {
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center>Built with Streamlit | Gradient Boosting Classifier | Digits Dataset</center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
