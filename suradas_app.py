# suradas_app.py
import streamlit as st
import requests
from PIL import Image
import geocoder
import base64
import os

# Detect if running on Streamlit Cloud
ON_CLOUD = os.getenv("IS_STREAMLIT_CLOUD", "true") == "true"

# Hugging Face Token
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

st.set_page_config(page_title="Suradas AI Agent", layout="centered")
st.title("🦾 Suradas: AI Agent for the Visually Impaired")

# 📍 GPS Location

def get_location():
    g = geocoder.ip('me')
    return f"{g.city}, {g.country}"

# 🔊 Text-to-Speech (Hugging Face)
def text_to_speech(text):
    tts_url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    payload = {"inputs": text}
    response = requests.post(tts_url, headers=headers, json=payload)
    if response.status_code == 200:
        audio_bytes = response.content
        b64 = base64.b64encode(audio_bytes).decode()
        md = f"""
            <audio autoplay controls>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    else:
        st.error("TTS failed.")

# 🖼️ Object Detection + Captioning
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🧠 Object Detection
    with st.spinner("Detecting objects..."):
        od_url = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
        od_response = requests.post(od_url, headers=headers, files={"file": uploaded_file})
        if od_response.status_code == 200:
            objects = od_response.json()
            st.subheader("Detected Objects")
            for obj in objects:
                st.write(f"- {obj['label']} ({round(obj['score'] * 100, 2)}%)")
        else:
            st.error("Object detection failed.")

    # 🧠 Image Captioning
    with st.spinner("Generating image caption..."):
        cap_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
        cap_response = requests.post(cap_url, headers=headers, files={"file": uploaded_file})
        if cap_response.status_code == 200:
            caption = cap_response.json()[0]['generated_text']
            st.success(f"Caption: {caption}")
            st.markdown("🔊 Audio feedback:")
            text_to_speech(caption)
        else:
            st.error("Captioning failed.")

# 📍 Location
st.subheader("📍 Your Location")
location = get_location()
st.write(location)

# 🌐 Cloud Notice
if ON_CLOUD:
    st.info("Note: Voice input is disabled on Streamlit Cloud. Run locally for full features.")
