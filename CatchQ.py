import streamlit as st
import whisper
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import datetime
import os
import tempfile

# Streamlit Title
st.title("CatchQ: AI-Powered Question Analyzer")

# ✅ Caching AI Models for Faster Performance
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def load_question_generation_model():
    model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qa-qg-hl")
    tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
    return model, tokenizer

@st.cache_resource
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load models
whisper_model = load_whisper_model()
qg_model, qg_tokenizer = load_question_generation_model()
classifier = load_zero_shot_classifier()
sbert_model = load_sentence_transformer()

# ✅ Upload Audio File
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # ✅ Transcribe Audio
    st.write("Transcribing...")
    try:
        result = whisper_model.transcribe(temp_audio_path)
        transcribed_text = st.text_area("Edit Transcribed Text", result["text"])
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

    # ✅ Extract Questions
    st.subheader("Extracted Questions")
    def extract_questions(text):
        try:
            inputs = qg_tokenizer.encode("generate questions: " + text, return_tensors="pt", max_length=512, truncation=True)
            outputs = qg_model.generate(inputs, max_length=50, num_return_sequences=5)
            return [qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            st.error(f"Question generation failed: {e}")
            return []

    questions = extract_questions(transcribed_text)
    for q in questions:
        st.write(f"- {q}")

    # ✅ Categorize Questions
    st.subheader("Categorized Questions")
    taxonomy = ["Conceptual", "Technical", "Application", "Miscellaneous"]
    def categorize_question(question, taxonomy):
        try:
            result = classifier(question, taxonomy)
            return result["labels"][0]  # Return the top category
        except Exception as e:
            st.warning(f"Categorization failed for question: {question}")
            return "Uncategorized"

    categories = [categorize_question(q, taxonomy) for q in questions]
    for q, c in zip(questions, categories):
        st.write(f"**{q}** → {c}")

    # ✅ Save Questions for Week-by-Week Comparison
    def save_questions(questions, categories):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        data = {"question": questions, "category": categories, "timestamp": [timestamp] * len(questions)}
        df = pd.DataFrame(data)

        # Ensure file exists before appending
        csv_path = "questions.csv"
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    save_questions(questions, categories)

    # ✅ Trend Analysis
    st.subheader("Trend Analysis")
    if os.path.exists("questions.csv"):
        df = pd.read_csv("questions.csv")

        # Process questions for trend analysis
        all_questions = df["question"].tolist()
        embeddings = sbert_model.encode(all_questions)
        similarity_matrix = cosine_similarity(embeddings)

        # 📊 Plot Similarity Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(similarity_matrix, cmap="coolwarm", aspect="auto")
        plt.colorbar(cax)
        ax.set_xticks(range(len(all_questions)))
        ax.set_yticks(range(len(all_questions)))
        ax.set_xticklabels([f"Q{i}" for i in range(len(all_questions))], rotation=90)
        ax.set_yticklabels([f"Q{i}" for i in range(len(all_questions))])
        ax.set_title("Question Similarity Over Time")
        st.pyplot(fig)

        # 📅 Week-by-Week Comparison
        st.subheader("Week-by-Week Comparison")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        weekly_questions = df.groupby(df["timestamp"].dt.strftime("%Y-%U"))["question"].apply(list).reset_index()
        st.dataframe(weekly_questions)

    # ✅ Download Results
    st.subheader("Download Results")
    results_df = pd.DataFrame({"Questions": questions, "Categories": categories})
    st.download_button("Download as CSV", results_df.to_csv(index=False), file_name="results.csv", mime="text/csv")
