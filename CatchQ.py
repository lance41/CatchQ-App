import streamlit as st
import pandas as pd
from datetime import datetime
import re
import io

# Function to generate week options dynamically
def generate_week_options():
    current_year = datetime.now().year
    return [f"{current_year}-Week-{str(i).zfill(2)}" for i in range(1, 53)]

# Updated taxonomy with expanded keywords
TAXONOMY = {
    "Content": ["what is", "define", "explain", "describe", "definition", "explanation", "meaning", "what are", "what does"],
    "Context": ["why is", "background", "history", "related to", "cause of", "reason for", "origin of", "relate to", "connection", "context"],
    "Contest": ["challenge", "disagree", "alternative", "critique", "problem with", "issue with", "limitation", "objection", "argument against", "learned", "innate", "nature vs nurture", "alternative to", "replace", "improve"]
}

# -------------------------
# Core Functions
# -------------------------

def extract_questions(text):
    """Extract questions using regex."""
    sentences = re.split(r'[.!?]', text)
    question_pattern = re.compile(r'(?i)^\s*(what|why|how|who|when|where|is|are|can|do)\b.*')
    questions = [s.strip() for s in sentences if re.match(question_pattern, s.strip()) or s.strip().endswith("?")]
    return questions

def categorize_question(question):
    """Categorize questions using regex and custom rules."""
    question_lower = question.lower()
    
    if "learned" in question_lower and ("innate" in question_lower or "nature" in question_lower):
        return "Contest"

    for cat, keywords in TAXONOMY.items():
        if re.search(r'\b(' + '|'.join(keywords) + r')\b', question_lower):
            return cat
    return "Uncategorized"

def count_categories(questions, categories):
    """Count the number of questions in each category."""
    df = pd.DataFrame({"Question": questions, "Category": categories})
    return df["Category"].value_counts().to_dict()

# -------------------------
# Streamlit UI
# -------------------------

st.title("CatchQ: Weekly Question Comparison & Trends")

# **Step 1: Paste This Week's Discussion Text**
st.subheader("Paste This Week's Discussion Text")
text = st.text_area("Paste discussion text:")

this_week_questions = []
this_week_categories = []
this_week_counts = {}

if text:
    # Extract and categorize questions
    this_week_questions = extract_questions(text)
    this_week_categories = [categorize_question(q) for q in this_week_questions]
    this_week_counts = count_categories(this_week_questions, this_week_categories)  # Ensure this is defined

    # Display extracted questions
    st.subheader("This Week's Extracted Questions")
    df_this_week = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    st.table(df_this_week)

# **Step 2: Upload Last Week's Questions (CSV Format)**
st.subheader("Upload Last Week's Questions (CSV Format)")
last_week_file = st.file_uploader("Upload last week's CSV file", type=["csv"])

past_questions = []
past_categories = []

if last_week_file:
    try:
        past_df = pd.read_csv(last_week_file)
        if "Question" in past_df.columns and "Category" in past_df.columns:
            past_questions = past_df["Question"].tolist()
            past_categories = past_df["Category"].tolist()
            
            # Display last week's questions
            st.subheader("Last Week's Questions")
            st.table(past_df)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# **Step 3: Compare Question Category Counts**
if past_questions and this_week_questions:
    st.subheader("Category Comparison: Last Week vs This Week")
    
    past_counts = count_categories(past_questions, past_categories)

    # Convert to DataFrame for better visualization
    categories = ["Content", "Context", "Contest"]
    comparison_df = pd.DataFrame({
        "Category": categories,
        "Last Week": [past_counts.get(cat, 0) for cat in categories],
        "This Week": [this_week_counts.get(cat, 0) for cat in categories]
    })

    # Calculate Percentage Change
    comparison_df["Change (%)"] = ((comparison_df["This Week"] - comparison_df["Last Week"]) / comparison_df["Last Week"] * 100).replace([float("inf"), float("-inf")], "New").fillna(0)

    # Display comparison table
    st.table(comparison_df)

    # Display bar chart
    st.subheader("Category Distribution Comparison")
    st.bar_chart(comparison_df.set_index("Category"))

# **Step 4: Download This Week's Questions & Counts for Trend Analysis**
st.subheader("Download This Week's Data")

# Dropdown for selecting the week
week_options = generate_week_options()
selected_week = st.selectbox("Select Week", week_options)

if this_week_questions:
    # Create DataFrames
    this_week_df = pd.DataFrame({"Question": this_week_questions, "Category": this_week_categories})
    this_week_counts_df = pd.DataFrame({
        "Week": [selected_week],
        "Content": [this_week_counts.get("Content", 0)],
        "Context": [this_week_counts.get("Context", 0)],
        "Contest": [this_week_counts.get("Contest", 0)]
    })

    # Generate filenames with the selected week
    questions_filename = f"questions_{selected_week.replace(' ', '-')}.csv"
    counts_filename = f"category_counts_{selected_week.replace(' ', '-')}.csv"

    # Convert to CSV
    csv_questions = this_week_df.to_csv(index=False)
    csv_counts = this_week_counts_df.to_csv(index=False)

    # Download buttons
    st.download_button(f"Download {selected_week} Questions CSV", csv_questions, questions_filename, "text/csv")
    st.download_button(f"Download {selected_week} Category Counts CSV", csv_counts, counts_filename, "text/csv")

# **Step 5: Upload Multiple Weekly CSVs for Trend Analysis**
st.subheader("Upload Multiple CSVs for Trend Analysis")

uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    all_weeks_data = []
    
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)

            # Ensure the file contains valid columns
            if "Week" in df.columns and "Content" in df.columns and "Context" in df.columns and "Contest" in df.columns:
                all_weeks_data.append(df)
            else:
                st.error(f"Incorrect format in {uploaded_file.name}. Required columns: 'Week', 'Content', 'Context', 'Contest'")
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")

    if all_weeks_data:
        # Merge all CSV files into one DataFrame
        historical_df = pd.concat(all_weeks_data, ignore_index=True)

        # Sort by Week
        historical_df = historical_df.sort_values(by="Week")

        st.subheader("Merged Historical Data")
        st.table(historical_df)

        # Plot trend chart
        st.subheader("Trend Analysis Over Multiple Weeks")
        historical_df.set_index("Week", inplace=True)
        st.line_chart(historical_df)
