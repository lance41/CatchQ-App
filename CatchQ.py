import streamlit as st
import pandas as pd
from datetime import datetime
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

def count_categories(questions, categories):
    """Count the number of questions in each category."""
    df = pd.DataFrame({"Question": questions, "Category": categories})
    return df["Category"].value_counts().to_dict()

# -------------------------
# Streamlit UI
# -------------------------

st.title("CatchQ: Weekly Question Comparison")

# **Step 1: Paste This Week's Discussion Text**
st.subheader("Paste This Week's Discussion Text")
text = st.text_area("Paste discussion text:")

this_week_questions = []
this_week_categories = []

if text:
    # Extract and categorize questions
    this_week_questions = extract_questions(text)
    this_week_categories = [categorize_question(q) for q in this_week_questions]

    # Display extracted questions
    st.subheader("This Week's Extracted Questions")
    df_this_week = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    st.table(df_this_week)

# **Step 2: Paste Last Week's Questions (CSV Format)**
st.subheader("Paste Last Week's Questions (CSV Format)")

last_week_questions_text = st.text_area("Paste last week's questions here in CSV format:")

past_questions = []
past_categories = []

if last_week_questions_text:
    try:
        past_df = pd.read_csv(io.StringIO(last_week_questions_text))
        if "Question" in past_df.columns and "Category" in past_df.columns:
            past_questions = past_df["Question"].tolist()
            past_categories = past_df["Category"].tolist()
            
            # Display last week's questions below this week's
            st.subheader("Last Week's Questions")
            st.table(past_df)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# **Step 3: Compare Question Category Counts**
if past_questions and this_week_questions:
    st.subheader("Category Comparison: Last Week vs This Week")
    
    past_counts = count_categories(past_questions, past_categories)
    this_week_counts = count_categories(this_week_questions, this_week_categories)

    # Convert to DataFrame for better visualization
    categories = ["Content", "Context", "Contest"]
    comparison_df = pd.DataFrame({
        "Category": categories,
        "Last Week": [past_counts.get(cat, 0) for cat in categories],
        "This Week": [this_week_counts.get(cat, 0) for cat in categories]
    })

    # Display comparison table
    st.table(comparison_df)

    # Display bar chart
    st.subheader("Category Distribution Comparison")
    st.bar_chart(comparison_df.set_index("Category"))

# **Step 4: Download This Week's Questions for Next Week's Use**
if this_week_questions:
    st.subheader("Download This Week's Questions for Next Week")
    this_week_df = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    
    csv = this_week_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "this_week_questions.csv", "text/csv")

