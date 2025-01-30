import os
import streamlit as st
from transformers import pipeline
import soundfile as sf
import tempfile
import torch
import wave

# Hugging Face Token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
def load_pipeline():
    return pipeline("audio-classification", model="HamzaSidhu786/speech-accent-detection", device=device, token=HF_TOKEN)

pipe = load_pipeline()

st.title("Speech Accent Detection 🎤")
st.write("Welcome! Record your voice and check your accent.")

scripts = [
    "Please call Stella.",
    "Ask her to bring these things with her from the store.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
]
selected_script = st.selectbox("Select a script:", scripts)
st.info(selected_script)

# File uploader instead of recording
st.write("### Upload your voice recording (.wav)")
audio_data = st.file_uploader("Upload your audio file", type=["wav"])

if audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data.read())
        temp_audio_path = temp_audio.name

    # Read WAV file properties
    with wave.open(temp_audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
    
    # Load the audio using soundfile
    audio_waveform, sample_rate = sf.read(temp_audio_path)

    # Play the uploaded audio
    st.audio(temp_audio_path, format="audio/wav")

    # Predict the accent
    st.write("### Predicting your accent...")
    result = pipe(temp_audio_path)

    # Show prediction
    st.success(f"Your accent is: **{result[0]['label']}**")
    st.write(f"Confidence: **{result[0]['score']:.2f}**")

    # Show all results
    st.write("### Full Results:")
    for r in result:
        st.write(f"- **{r['label']}**: {r['score']:.2f}")

st.write("---")
st.write("Developed by Hamza Sidhu 🚀")
