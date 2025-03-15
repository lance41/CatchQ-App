import streamlit as st
import re
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

# Extract questions from text
def extract_questions(text):
    sentences = re.split(r'[.!?]', text)
    questions = [s.strip() for s in sentences if re.search(r'\b(what|why|how|who|when|where|is|are|can|do)\b', s.lower())]
    return questions

# Categorize questions
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
    
    # Extract questions
    questions = extract_questions(text)
    if questions:
        st.subheader("Extracted Questions")
        categories = [categorize_question(q) for q in questions]
        df = pd.DataFrame({"Question": questions, "Category": categories})
        st.table(df)
    else:
        st.write("No questions found in the transcript.")
