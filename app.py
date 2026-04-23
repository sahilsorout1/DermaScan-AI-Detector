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

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import ImageOps  # Make sure this is imported too


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

import numpy as np
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    # 1. Resize to 224x224
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to Array
    img_array = np.asarray(image)
    
    # 3. NORMALIZE (Critical: Matches the training rescale=1./255)
    img_array = img_array / 255.0  
    
    # 4. Predict
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction



# Trigger load now
model = get_model()



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
        
        # 1. DEFINE DATA FIRST (Safe to exist before button click)
        class_names = ['Benign', 'Carcinoma', 'Dermatofibroma', 'Keratoses', 'Melanoma', 'Nevi', 'Vascular']
        
        medical_info = {
            'Benign': {
                'description': 'A non-cancerous benign keratosis or similar epidermal growth. These lesions are typically localized and do not spread to other tissues.',
                'risk': '🟢 Low Risk (Benign)',
                'advice': 'No immediate medical intervention is required. However, continue routine self-examinations and consult a dermatologist if the lesion begins to bleed, itch, or change rapidly.'
            },
            'Carcinoma': {
                'description': 'A malignant neoplasm of the skin, most commonly Basal Cell Carcinoma (BCC) or Squamous Cell Carcinoma (SCC). Often linked to cumulative UV/sun exposure.',
                'risk': '🔴 High Risk (Malignant)',
                'advice': 'Requires prompt medical evaluation. A dermatologist should perform a clinical assessment and biopsy to confirm the diagnosis and determine appropriate surgical treatment.'
            },
            'Dermatofibroma': {
                'description': 'A benign, slow-growing dermal nodule composed of fibrous tissue. It often feels like a firm lump under the skin and may develop after minor trauma (e.g., a bug bite).',
                'risk': '🟢 Low Risk (Benign)',
                'advice': 'Generally asymptomatic and requires no medical treatment. Excision is only recommended if the lesion becomes painful, continually irritated, or cosmetically concerning.'
            },
            'Keratoses': {
                'description': 'Actinic keratoses are rough, scaly patches caused by chronic UV damage. They are considered precancerous and carry a risk of progressing to Squamous Cell Carcinoma.',
                'risk': '🟡 Moderate Risk (Pre-cancerous)',
                'advice': 'Schedule a dermatological consultation. Proactive treatments, such as cryotherapy or topical immunomodulators, are often recommended to prevent malignant progression.'
            },
            'Melanoma': {
                'description': 'An aggressive and potentially life-threatening malignant tumor originating from melanocytes (pigment-producing cells). Early detection and intervention are critical.',
                'risk': '🚨 Critical Risk (Malignant)',
                'advice': 'URGENT MEDICAL ATTENTION REQUIRED. Seek immediate evaluation by a dermatologist or oncologist for an urgent biopsy and comprehensive treatment planning.'
            },
            'Nevi': {
                'description': 'A benign proliferation of melanocytes, commonly known as a mole. They typically present as uniform, well-defined brown or pink macules or papules.',
                'risk': '🟢 Low Risk (Benign)',
                'advice': 'Standard observation is recommended. Perform monthly skin checks using the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving) and report any deviations to a doctor.'
            },
            'Vascular': {
                'description': 'A benign vascular anomaly resulting from an overgrowth or clustering of blood vessels. Common presentations include cherry angiomas or angiokeratomas.',
                'risk': '🟢 Low Risk (Benign)',
                'advice': 'Completely harmless and requires no intervention. If the lesion is frequently traumatized and bleeds, a dermatologist can easily treat it via laser therapy or electrocautery.'
            }
        }

        # 2. THE BUTTON LOGIC
        if st.button('Run AI Diagnosis'):
            
            with st.spinner('Scanning lesion features...'):
                # Calculate scores
                prediction = import_and_predict(image, model)
                score = prediction[0]
                confidence = np.max(score) * 100
                
                # Get the specific class
                predicted_class_index = np.argmax(score)
                predicted_class = class_names[predicted_class_index]
            
            # 3. SHOW RESULTS (Notice how this is indented under 'if st.button'!)
            st.success(f"### Diagnosis: **{predicted_class}**")
            
            info = medical_info[predicted_class]
            st.markdown(f"**Risk Level:** {info['risk']}")
            st.write(f"**What is it?** {info['description']}")
            st.info(f"**Recommendation:** {info['advice']}")
            
            st.write("---")
            
            # 4. SHOW METRICS & CHARTS
            if confidence > 70:
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%", delta="High Certainty")
            elif confidence > 40:
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%", delta="Moderate", delta_color="off")
            else:
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%", delta="Low/Unsure", delta_color="inverse")
                
            st.write("---")
            st.write("### Detailed Probability Distribution:")
            
            import pandas as pd
            chart_data = pd.DataFrame(
                {"Probability": score},
                index=class_names
            )
            st.bar_chart(chart_data)
            
    else:
        # Placeholder when no image is uploaded
        st.write("👉 Please upload an image on the left to begin analysis.")
