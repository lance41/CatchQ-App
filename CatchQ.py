import streamlit as st
import whisper
from io import BytesIO
import tempfile

# Hardcoded taxonomy (customize as needed)
TAXONOMY = {
    "Content": ["what is", "define", "explain", "describe"],
    "Context": ["why is", "background", "history", "related to"],
    "Contest": ["challenge", "disagree", "alternative", "critique"]
}

# Load Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny.en")

# Categorization (keyword-based)
def categorize_question(question):
    question_lower = question.lower()
    for cat, keywords in TAXONOMY.items():
        if any(kw in question_lower for kw in keywords):
            return cat
    return "Uncategorized"

# --- UI ---
st.title("CatchQ Prototype")

# Audio file upload
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(uploaded_file.getvalue())
        st.session_state.audio_path = f.name

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
