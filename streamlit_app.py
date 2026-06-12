import streamlit as st
import torch
from PIL import Image
import os

# Internal project imports
from data_loader import load_dataset
from model import PalmVeinNet, get_transforms

st.set_page_config(page_title="SASH-VPV Palm Vein Authenticator", layout="centered")

st.title("🛡️ Subcutaneous Vascular Palm Vein Auth")
st.write("An AI Biometric prototype parsing subcutaneous vascular system layouts.")

# 1. Trigger Data Fetching inside Streamlit Cached framework
@st.cache_resource
def init_data():
    try:
        path = load_dataset()
        return path
    except Exception as e:
        st.error(f"Failed to access KaggleHub Dataset: {e}")
        return None

dataset_path = init_data()

if dataset_path:
    st.success("✅ SASH-VPV Biometric Dataset Loaded via KaggleHub!")
    with st.expander("Show System Storage Path"):
        st.code(dataset_path)
else:
    st.warning("⚠️ Running app in offline mode without primary dataset hooks.")

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

# 3. Web UI for File Uploading / Prediction Testing
st.write("---")
st.subheader("Simulate Authentication Engine Scan")
uploaded_file = st.file_uploader("Upload an infrared or subcutaneous palm scan profile image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Live Palm Feed Input Target.', width=300)
    
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