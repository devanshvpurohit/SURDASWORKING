import streamlit as st
import cv2
import requests
import tempfile
from PIL import Image
import geocoder
import base64
import os
import time

API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# For IP-based location
def get_location():
    g = geocoder.ip("me")
    return f"{g.city}, {g.country}"

# Convert OpenCV frame to PIL
def get_frame_from_camera():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        st.error("Failed to access camera.")
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# Use Hugging Face LLaVA API for vision-language response
def query_llava(image: Image, prompt: str):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        with open(tmp.name, "rb") as f:
            response = requests.post(
                "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf",
                headers=headers,
                files={"image": f},
                data={"inputs": prompt}
            )
    return response.json()

# Text-to-Speech using Hugging Face API
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

# App UI
st.set_page_config(page_title="Suradas AI Live", layout="wide")
st.title("🦾 Suradas Live - 24/7 Object Understanding")

st.subheader("🌍 Your Location")
location = get_location()
st.write(location)

# Text input for manual prompt
user_prompt = st.text_input("Ask the AI about your environment:", value="What do you see in this image?")

# Interval in seconds
interval = st.number_input("Detection Interval (seconds):", min_value=1, max_value=60, value=10)

# Start looping live detection
if st.button("Start Continuous Object Detection"):
    st.warning("Streaming live camera... this works only locally.")
    run = True
    detection_placeholder = st.empty()
    image_placeholder = st.empty()

    while run:
        image = get_frame_from_camera()
        if image:
            image_placeholder.image(image, caption="Live Camera Feed", use_column_width=True)
            with st.spinner("Analyzing using LLaVA..."):
                result = query_llava(image, user_prompt)
                if isinstance(result, dict) and "generated_text" in result:
                    output = result['generated_text']
                    detection_placeholder.success(f"LLaVA: {output}")
                    text_to_speech(output)
                else:
                    detection_placeholder.error("LLaVA response error.")
        time.sleep(interval)
