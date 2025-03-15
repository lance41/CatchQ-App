import streamlit as st
import speech_recognition as sr
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import datetime
import os

# Manual cosine similarity implementation
def cosine_similarity(embeddings):
    """
    Compute the cosine similarity matrix for a set of embeddings.
    :param embeddings: A 2D numpy array of shape (n_samples, n_features).
    :return: A 2D numpy array of shape (n_samples, n_samples) containing pairwise cosine similarities.
    """
    # Normalize the embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    return similarity_matrix

# Load SpeechRecognition
st.title("CatchQ: AI-Powered Question Analyzer")

@st.cache_resource
def load_speech_recognizer():
    return sr.Recognizer()

@st.cache_resource
def load_question_generation_model():
    model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qa-qg-hl")
    tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
    return model, tokenizer

@st.cache_resource
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_embeddings_model():
    from gensim.models import KeyedVectors
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-100")  # Lightweight embeddings model

recognizer = load_speech_recognizer()
qg_model, qg_tokenizer = load_question_generation_model()
classifier = load_zero_shot_classifier()
embeddings_model = load_embeddings_model()

# Upload Audio File
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
if audio_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())

    # Transcribe Audio
    st.write("Transcribing...")
    try:
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio)
            transcribed_text = st.text_area("Edit Transcribed Text", transcribed_text)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

    # Extract Questions
    st.subheader("Extracted Questions")
    def extract_questions(text):
        inputs = qg_tokenizer.encode("generate questions: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = qg_model.generate(inputs, max_length=50, num_return_sequences=5)
        return [qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    questions = extract_questions(transcribed_text)
    for q in questions:
        st.write(f"- {q}")

    # Categorize Questions
    st.subheader("Categorized Questions")
    taxonomy = ["Conceptual", "Technical", "Application", "Miscellaneous"]
    def categorize_question(question, taxonomy):
        result = classifier(question, taxonomy)
        return result["labels"][0]  # Return the top category

    categories = [categorize_question(q, taxonomy) for q in questions]
    for q, c in zip(questions, categories):
        st.write(f"**{q}** → {c}")

    # Save Questions for Week-by-Week Comparison
    def save_questions(questions, categories):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        data = {"question": questions, "category": categories, "timestamp": [timestamp] * len(questions)}
        df = pd.DataFrame(data)
        df.to_csv("questions.csv", mode="a", header=not os.path.exists("questions.csv"), index=False)

    save_questions(questions, categories)

    # Trend Analysis
    st.subheader("Trend Analysis")
    if os.path.exists("questions.csv"):
        df = pd.read_csv("questions.csv")
        all_questions = df["question"].tolist()

        # Compute embeddings using GloVe
        embeddings = np.array([np.mean([embeddings_model[word] for word in question.split() if word in embeddings_model] or [np.zeros(100)], axis=0) for question in all_questions])
        similarity_matrix = cosine_similarity(embeddings)

        # Plot Similarity Matrix using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=all_questions,
            y=all_questions,
            colorscale="Viridis"
        ))
        fig.update_layout(
            xaxis=dict(tickangle=90),
            yaxis=dict(autorange="reversed"),
            title="Question Similarity Matrix"
        )
        st.plotly_chart(fig)

        # Week-by-Week Comparison
        st.subheader("Week-by-Week Comparison")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        weekly_questions = df.groupby(df["timestamp"].dt.strftime("%Y-%U"))["question"].apply(list).reset_index()
        st.write(weekly_questions)

    # Download Results
    st.subheader("Download Results")
    results_df = pd.DataFrame({"Questions": questions, "Categories": categories})
    st.download_button("Download as CSV", results_df.to_csv(index=False), file_name="results.csv", mime="text/csv")
