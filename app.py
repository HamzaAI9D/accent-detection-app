import streamlit as st
from transformers import pipeline
import torchaudio
import tempfile
import torch
import os

# Hugging Face Token from Secrets
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline():
    return pipeline("audio-classification", model="HamzaSidhu786/speech-accent-detection",device=device,token=HF_TOKEN)

pipe = load_pipeline()

# Streamlit UI
st.title("Speech Accent Detection 🎤")
st.write(
    "Welcome to the Speech Accent Detection App! "
    "Read the given script, record your voice, and find out your accent."
)

# Display a random script for the user to read
scripts = [
    "Please call Stella.",
    "Ask her to bring these things with her from the store.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
    "We also need a small plastic snake and a big toy frog for the kids.",
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
]
selected_script = st.selectbox("Select a script to read:", scripts)

st.write("### Please read the following script:")
st.info(selected_script)

# Audio recording input
st.write("### Record your voice:")
audio_data = st.audio_input("Record your voice message and check your accent.")

# Process and classify the audio
if audio_data is not None:
    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data.getvalue())
        temp_audio_path = temp_audio.name

    # Load the audio for preprocessing
    waveform, sample_rate = torchaudio.load(temp_audio_path)
    st.audio(audio_data, format="audio/wav")

    # Classify the audio using the pipeline
    st.write("### Predicting your accent...")
    result = pipe(temp_audio_path)

    # Display the top prediction
    st.success(f"Your accent is: **{result[0]['label']}**")
    st.write(f"Confidence: **{result[0]['score']:.2f}**")

    # Display all predictions
    st.write("### Full Results:")
    for r in result:
        st.write(f"- **{r['label']}**: {r['score']:.2f}")

# Footer
st.write("---")
st.write("Developed by Hamza Sidhu 🚀")
