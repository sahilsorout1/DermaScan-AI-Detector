import os

# --- 1. MAC SAFETY LOCKS (MUST BE AT THE VERY TOP) ---
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# --- 2. IMPORT LIBRARIES ---
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- 3. DISABLE GPU (PREVENTS CRASH) ---
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass

# --- 4. MATPLOTLIB FIX ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 5. LOAD MODEL GLOBAL (PREVENTS CRASH) ---
@st.cache_resource
def get_model():
    # Load model once at startup
    model = tf.keras.models.load_model('models/skin_cancer_model.h5')
    return model

# Trigger load now
model = get_model()

# --------------------------------------------------------
# --- PASTE THIS AT THE BOTTOM OF APP.PY (REPLACING MAIN APP) ---
# --------------------------------------------------------

# --- PAGE CONFIG (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="wide",  # This makes it fill the whole screen!
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Styles to make it look pretty) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Left Panel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("DermaScan Pro")
    st.caption("AI-Powered Skin Lesion Analyzer")
    st.write("---")
    st.write("### 📖 How to use:")
    st.write("1. Take a clear photo of the skin spot.")
    st.write("2. Upload it on the main screen.")
    st.write("3. Click **Analyze** to see the diagnosis.")
    st.write("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. Consult a doctor for real diagnosis.")

# --- MAIN HEADER ---
st.markdown('<div class="main-header">🩺 DermaScan: Skin Cancer Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Advanced Deep Learning Model for Dermatoscopy Analysis</div>', unsafe_allow_html=True)

# --- LAYOUT: TWO COLUMNS (Left: Image, Right: Results) ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.info("📸 **Step 1: Upload Image**")
    uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open and Fix Image
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Display Image with rounded corners
        st.image(image, caption='Uploaded Patient Scan', use_container_width=True)

with col2:
    st.info("🧠 **Step 2: AI Analysis**")
    
    if uploaded_file is not None:
        # Add a big "Analyze" button
        if st.button('🔍 Run AI Diagnosis'):
            
            with st.spinner('Scanning lesion features...'):
                # Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Prediction
                predictions = model(img_array, training=False).numpy()
                
                # Softmax
                def softmax(x):
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()
                
                score = softmax(predictions[0])
                confidence = np.max(score) * 100
                
                class_names = ['Melanoma', 'Nevi', 'Carcinoma', 'Keratoses', 'Benign', 'Dermatofibroma', 'Vascular']
                predicted_class = class_names[np.argmax(score)]

            # --- SHOW RESULTS (Dashboard Style) ---
            st.success(f"### Diagnosis: **{predicted_class}**")
            
            # Confidence Meter
            if confidence > 70:
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%", delta="High Certainty")
            elif confidence > 40:
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%", delta="Moderate", delta_color="off")
            else:
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%", delta="Low/Unsure", delta_color="inverse")

            # Bar Chart
            st.write("---")
            st.write("**Detailed Probability Distribution:**")
            st.bar_chart(dict(zip(class_names, score)))

    else:
        # Placeholder when no image is uploaded
        st.write("👈 *Please upload an image on the left to begin analysis.*")
