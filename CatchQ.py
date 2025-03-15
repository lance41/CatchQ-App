import streamlit as st
import pandas as pd
from datetime import datetime
import difflib  # Built-in, no extra dependency
import numpy as np  # Lightweight for trends

# Hardcoded taxonomy (customize as needed)
TAXONOMY = {
    "Definition": ["what is", "define", "meaning"],
    "Procedure": ["how to", "steps", "process"],
    "Comparison": ["difference", "similarity", "vs"],
}

# Mock data storage (replace with CSV later)
if "questions" not in st.session_state:
    st.session_state.questions = []

# -------------------------
# Core Functions (Hardcoded)
# -------------------------
def generate_questions(text):
    """Rule-based question generation (minimal logic)."""
    sentences = [s.strip() for s in text.split(". ") if s]
    return [f"What is {s}?" for s in sentences[:3]]  # Simple template

def categorize_question(question):
    """Keyword-based categorization (no ML)."""
    for category, keywords in TAXONOMY.items():
        if any(kw in question.lower() for kw in keywords):
            return category
    return "Uncategorized"

def compare_questions(new_questions, past_questions):
    """Basic similarity check (built-in difflib)."""
    similar = []
    for new_q in new_questions:
        for past_q in past_questions:
            ratio = difflib.SequenceMatcher(None, new_q, past_q).ratio()
            if ratio > 0.7:  # Threshold adjustable
                similar.append((new_q, past_q, ratio))
    return similar

# -------------------------
# Streamlit UI
# -------------------------
st.title("CatchQ Prototype")

# 1. Mock "Audio Upload" → Direct text input (skip STT for now)
text = st.text_area("Paste discussion text (simulates speech-to-text):")

if text:
    # 2. Generate questions
    questions = generate_questions(text)
    st.subheader("Generated Questions")
    for q in questions:
        st.write(f"- {q}")

    # 3. Categorize questions
    st.subheader("Categories")
    categories = [categorize_question(q) for q in questions]
    df = pd.DataFrame({"Question": questions, "Category": categories})
    st.table(df)

    # 4. Compare with past questions (mock persistence)
    if st.button("Save for weekly comparison"):
        st.session_state.questions.extend([
            {"question": q, "category": c, "date": datetime.now()}
            for q, c in zip(questions, categories)
        ])

# 5. Show trends (mock data)
if st.session_state.questions:
    st.subheader("Trends")
    history = pd.DataFrame(st.session_state.questions)
    history["date"] = pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d")
    trend_data = history.groupby(["date", "category"]).size().unstack(fill_value=0)
    st.line_chart(trend_data)

# 6. Similarity check (compare all saved questions)
if len(st.session_state.questions) > 1:
    st.subheader("Similar Questions")
    past_questions = [q["question"] for q in st.session_state.questions[:-1]]
    new_questions = questions
    similar = compare_questions(new_questions, past_questions)
    if similar:
        for new_q, past_q, ratio in similar:
            st.write(f"**New:** {new_q}\n\n**Past:** {past_q}\n\nSimilarity: {ratio:.2f}")
