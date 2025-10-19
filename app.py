import streamlit as st
import joblib
from resume_parser import parse_resume
from utils.text_cleaner import clean_text
from utils.score_calculator import calculate_completeness, match_skills
import plotly.graph_objects as go
from fpdf import FPDF
import google.generativeai as genai

# --- Gemini API Setup ---
genai.configure(api_key="AIzaSyBVhYt5t0JfCj0djRnYF1SeEt8V6R9rBtw")  # replace with your key
gpt_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Streamlit Setup ---
st.set_page_config(page_title="AI Resume Checker Pro", page_icon="", layout="wide")
st.title("AI Resume Checker Pro")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_desc = st.text_area("Paste Job Description (optional)")

if uploaded_file:
    # --- Resume Parsing ---
    text = parse_resume(uploaded_file)
    clean = clean_text(text)

    # --- Load trained model ---
    vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    ml_model = joblib.load('model/resume_quality_model.joblib')
    features = vectorizer.transform([clean])
    pred = ml_model.predict(features)[0]
    prob = ml_model.predict_proba(features).max()

    # --- Calculate Scores ---
    completeness = calculate_completeness(text)
    with open('model/skill_keywords.txt') as f:
        skill_list = [line.strip() for line in f.readlines()]
    skill_score, found_skills = match_skills(text, skill_list)

    exp_proj_score = 100 if any(k in text.lower() for k in ["experience", "project", "internship"]) else 0
    edu_score = 100 if any(k in text.lower() for k in ["education", "degree", "university", "college"]) else 0

    overall_score = round((skill_score + completeness + exp_proj_score + edu_score) / 4, 2)

    # --- Score Breakdown Chart ---
    st.subheader("📊 Resume Score Breakdown")
    score_data = {
        "Skills": skill_score,
        "Experience/Projects": exp_proj_score,
        "Education": edu_score,
        "Completeness": completeness,
    }

    fig = go.Figure(
        go.Bar(
            x=list(score_data.keys()),
            y=list(score_data.values()),
            marker_color=["#FF7F50", "#6A5ACD", "#00CED1", "#32CD32"],
            text=list(score_data.values()),
            textposition="auto",
        )
    )
    fig.update_layout(
        height=300,
        width=500,
        title="Score Breakdown",
        title_font=dict(size=16, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Resume Analysis ---
    st.subheader("📈 Resume Analysis Result")
    st.write(f"**Predicted Quality:** {pred} ({prob*100:.1f}%)")
    st.write(f"**Detected Skills:** {', '.join(found_skills)}")
    st.write(f"**Overall Score:** {overall_score}%")

    missing_sections = [sec for sec in ["Summary", "Education", "Experience/Projects", "Skills"]
                        if sec.lower() not in text.lower()]
    if missing_sections:
        st.warning(f"⚠️ Missing Sections: {', '.join(missing_sections)}")

    # --- AI Feedback via Gemini ---
    st.subheader("💬 AI-Powered Feedback")

    try:
        prompt = (
            f"Analyze this resume and job description if provided. "
            f"Give short, personalized feedback on strengths and what to improve.\n\n"
            f"Resume Text:\n{text}\n\n"
            f"Job Description:\n{job_desc}"
        )
        response = gpt_model.generate_content(prompt)
        feedback = response.text
        st.success("✅ AI Feedback Generated!")
        st.write(feedback)
    except Exception as e:
        st.error(f"AI feedback unavailable: {e}")
        feedback = "No AI feedback due to API issue."
