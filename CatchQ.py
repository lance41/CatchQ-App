import streamlit as st
import pandas as pd
from datetime import datetime
import difflib
import re

# Updated taxonomy with expanded keywords
TAXONOMY = {
    "Content": [
        "what is", "define", "explain", "describe", "definition", 
        "explanation", "meaning", "what are", "what does"
    ],
    "Context": [
        "why is", "background", "history", "related to", "cause of", 
        "reason for", "origin of", "relate to", "connection", "context"
    ],
    "Contest": [
        "challenge", "disagree", "alternative", "critique", "problem with",
        "issue with", "limitation", "objection", "argument against", "learned",
        "innate", "nature vs nurture", "alternative to", "replace", "improve"
    ]
}

# Mock data storage
if "questions" not in st.session_state:
    st.session_state.questions = []

# -------------------------
# Core Functions
# -------------------------
def extract_questions(text):
    """Extract questions using regex."""
    sentences = re.split(r'[.!?]', text)
    question_pattern = re.compile(
        r'(?i)^\s*(what|why|how|who|when|where|is|are|can|do)\b.*'
    )
    questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if re.match(question_pattern, sentence) or sentence.endswith("?"):
            questions.append(sentence)
    return questions

def categorize_question(question):
    """Categorize questions with regex and custom rules."""
    question_lower = question.lower()
    
    # Custom rule for "learned vs innate" questions
    if "learned" in question_lower and ("innate" in question_lower or "nature" in question_lower):
        return "Contest"
    
    # Regex-based keyword matching
    for cat, keywords in TAXONOMY.items():
        pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b')
        if pattern.search(question_lower):
            return cat
    return "Uncategorized"

def compare_questions(new_questions, past_questions):
    """Similarity check using difflib."""
    similar = []
    for new_q in new_questions:
        for past_q in past_questions:
            ratio = difflib.SequenceMatcher(None, new_q, past_q).ratio()
            if ratio > 0.7:
                similar.append((new_q, past_q, ratio))
    return similar

# -------------------------
# Streamlit UI
# -------------------------
st.title("CatchQ Prototype")

# Text input
text = st.text_area("Paste discussion text:")

if text:
    # Extract questions
    questions = extract_questions(text)
    st.subheader("Extracted Questions")
    for q in questions:
        st.write(f"- {q}")

    # Categorize questions
    st.subheader("Categories")
    categories = [categorize_question(q) for q in questions]
    df = pd.DataFrame({"Question": questions, "Category": categories})
    st.table(df)

    # Save for comparison
    if st.button("Save for weekly comparison"):
        st.session_state.questions.extend([
            {"question": q, "category": c, "date": datetime.now()}
            for q, c in zip(questions, categories)
        ])

# Show trends
if st.session_state.questions:
    st.subheader("Trends")
    history = pd.DataFrame(st.session_state.questions)
    history["date"] = pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d")
    trend_data = history.groupby(["date", "category"]).size().unstack(fill_value=0)
    st.line_chart(trend_data)

# Similarity check
if len(st.session_state.questions) > 1:
    st.subheader("Similar Questions")
    past_questions = [q["question"] for q in st.session_state.questions[:-1]]
    new_questions = questions
    similar = compare_questions(new_questions, past_questions)
    if similar:
        for new_q, past_q, ratio in similar:
            st.write(f"**New:** {new_q}\n\n**Past:** {past_q}\n\nSimilarity: {ratio:.2f}")
