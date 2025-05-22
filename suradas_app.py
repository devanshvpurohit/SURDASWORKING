import streamlit as st
import requests
import tempfile
from PIL import Image
import geocoder
import base64

API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# For IP-based location
def get_location():
    g = geocoder.ip("me")
    return f"{g.city}, {g.country}"

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
st.title("🦾 Suradas Live - Object Understanding")

st.subheader("🌍 Your Location")
location = get_location()
st.write(location)

# Upload image instead of live camera
uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])
user_prompt = st.text_input("Ask the AI about your environment:", value="What do you see in this image?")

if uploaded_file and user_prompt:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Analyze Image"):
        with st.spinner("Analyzing using LLaVA..."):
            result = query_llava(image, user_prompt)
            if isinstance(result, dict) and "generated_text" in result:
                output = result['generated_text']
                st.success(f"LLaVA: {output}")
                text_to_speech(output)
            else:
                st.error("LLaVA response error.")
