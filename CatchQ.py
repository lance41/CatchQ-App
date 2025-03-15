import streamlit as st
import numpy as np
import whisper  # Only STT dependency (installs torch automatically)
from io import BytesIO
import sounddevice as sd  # For audio recording
from scipy.io.wavfile import write
import tempfile

# Hardcoded Taxonomy (customize keywords)
TAXONOMY = {
    "Content": ["what is", "define", "explain", "describe"],
    "Context": ["why is", "background", "history", "related to"],
    "Contest": ["challenge", "disagree", "alternative", "critique"]
}

# Minimal Whisper model (tiny.en ~75MB)
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny.en")

# Audio recording
def record_audio(duration=5, fs=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return fs, recording

# Categorization (keyword-based)
def categorize_question(question):
    question_lower = question.lower()
    for cat, keywords in TAXONOMY.items():
        if any(kw in question_lower for kw in keywords):
            return cat
    return "Uncategorized"

# --- UI ---
st.title("CatchQ Prototype")

# Audio input choice
input_type = st.radio("Input type:", ["Record Audio (5s)", "Upload WAV"])

# Audio handling
audio = None
if input_type == "Record Audio (5s)":
    if st.button("Start Recording"):
        fs, audio = record_audio()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, fs, audio)
            st.session_state.audio_path = f.name
elif input_type == "Upload WAV":
    uploaded = st.file_uploader("Upload WAV", type=["wav"])
    if uploaded:
        st.session_state.audio_path = uploaded.name
        with open(uploaded.name, "wb") as f:
            f.write(uploaded.getvalue())

# Transcribe & Process
if "audio_path" in st.session_state:
    model = load_whisper()
    result = model.transcribe(st.session_state.audio_path)
    text = result["text"]
    
    st.subheader("Transcript")
    st.write(text)
    
    # Generate mock questions (same simple logic)
    questions = [f"What is {text.split()[0]}?", 
                 f"Why is {text.split()[0]} important?", 
                 f"Challenge: {text.split()[0]}"]
    
    st.subheader("Categorized Questions")
    for q in questions:
        cat = categorize_question(q)
        st.write(f"**{cat}**: {q}")
