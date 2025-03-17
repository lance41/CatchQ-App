import streamlit as st
import pandas as pd
from datetime import datetime
import difflib
import re
import io

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

# **Step 1: Paste last week's questions (CSV format)**
st.subheader("Paste Last Week's Questions (CSV Format)")

last_week_questions_text = st.text_area("Paste last week's questions here in CSV format:")

past_questions = []
if last_week_questions_text:
    try:
        past_df = pd.read_csv(io.StringIO(last_week_questions_text))
        if "Question" in past_df.columns:
            past_questions = past_df["Question"].tolist()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# **Step 2: Enter this week's discussion text**
st.subheader("Paste This Week's Discussion Text")
text = st.text_area("Paste discussion text:")

this_week_questions = []
this_week_categories = []

if text:
    # Extract and categorize questions
    this_week_questions = extract_questions(text)
    this_week_categories = [categorize_question(q) for q in this_week_questions]

    # Display extracted questions
    st.subheader("Extracted Questions")
    df = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    st.table(df)

# **Step 3: Compare This Week's Questions with Last Week's**
if past_questions and this_week_questions:
    st.subheader("Similar Questions from Last Week and This Week")
    similar_questions = compare_questions(this_week_questions, past_questions)

    if similar_questions:
        for new_q, past_q, ratio in similar_questions:
            st.write(f"**New:** {new_q}\n\n**Past:** {past_q}\n\nSimilarity: {ratio:.2f}")
    else:
        st.write("No similar questions found.")

# **Step 4: Download This Week's Questions for Next Week's Use**
if this_week_questions:
    st.subheader("Download This Week's Questions for Next Week")
    this_week_df = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    
    csv = this_week_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "this_week_questions.csv", "text/csv")
