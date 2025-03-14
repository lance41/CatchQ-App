import streamlit as st
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
import re
import pandas as pd

# Hardcoded taxonomy (customize as needed)
TAXONOMY = {
    "Content": ["what is", "define", "explain", "describe"],
    "Context": ["why is", "background", "history", "related to"],
    "Contest": ["challenge", "disagree", "alternative", "critique"]
}

# Initialize recognizer
recognizer = sr.Recognizer()

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
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file:
    with BytesIO(uploaded_file.getvalue()) as audio_file:
        try:
            # Convert MP3 to WAV if necessary
            if uploaded_file.type == "audio/mp3":
                audio = AudioSegment.from_mp3(audio_file)
                wav_file = BytesIO()
                audio.export(wav_file, format="wav")
                wav_file.seek(0)
                audio_file = wav_file

            # Process the audio file
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                try:
                    # Use Google Web Speech API for transcription
                    text = recognizer.recognize_google(audio)
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
                except sr.UnknownValueError:
                    st.error("Google Web Speech API could not understand the audio.")
                except sr.RequestError:
                    st.error("Could not request results from Google Web Speech API.")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
