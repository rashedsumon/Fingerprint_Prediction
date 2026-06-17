import streamlit as st
import torch
from PIL import Image
import os

# Internal project imports
from data_loader import load_dataset
from model import PalmVeinNet, get_transforms

st.set_page_config(page_title="Fingerprint Prediction", layout="centered")

st.title("🛡️ Fingerprint Prediction")
st.write("An AI Biometric prototype parsing subcutaneous vascular system layouts.")

# 1. Background Data Fetching (Hidden from UI)
@st.cache_resource
def init_data():
    try:
        return load_dataset()
    except Exception as e:
        return None

dataset_path = init_data()

# 2. Instantiate Model
@st.cache_resource
def load_cached_model():
    # Placeholder initialization. For production deployment, load your trained state dictionary:
    # model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    model = PalmVeinNet(num_classes=2) 
    model.eval()
    return model

model = load_cached_model()
transformations = get_transforms()

# 3. Clean Web UI for Verification
st.write("---")

# Use a generic file uploader label and hide the default label text using custom CSS
st.markdown("""
    <style>
    /* Hides the default "Drag and drop file here" and "200MB limit" text */
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Place your hand scan onto the active reader zone:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Active Scan Capture Target', width=300)
    
    # Preprocessing pipeline
    input_tensor = transformations(image).unsqueeze(0)
    
    with st.spinner("Processing structural vascular maps..."):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

    # Arbitrary assignment mapping logic representation
    classes = {0: "Authorized Match Profile (ID: 4409)", 1: "Unknown Profile / Access Denied"}
    
    st.write("### Diagnostics Summary:")
    if prediction == 0:
        st.success(f"🔓 **Status:** {classes[prediction]} ({confidence:.2f}% Match)")
    else:
        st.error(f"🔒 **Status:** {classes[prediction]} ({confidence:.2f}% Confidence Verification Score)")
