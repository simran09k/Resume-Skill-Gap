import streamlit as st
import PyPDF2
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Extract Text from PDF
# -------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -------------------------
# Extract Text from DOCX
# -------------------------
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# -------------------------
# Clean Text
# -------------------------
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# -------------------------
# Skill Extraction
# -------------------------
SKILLS_DB = [
    "python","java","c++","sql","machine learning","deep learning",
    "data analysis","pandas","numpy","excel","power bi","tableau",
    "html","css","javascript","react","node","mongodb",
    "linux","aws","cyber security","dbms"
]

def extract_skills(text):
    found_skills = []
    text = text.lower()
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return found_skills

# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 AI Resume Skill Gap Analyzer")

st.write("Upload Resume and Paste Job Description")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if resume_file and job_description:

        # Extract resume text
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)

        # Preprocess
        resume_clean = preprocess(resume_text)
        jd_clean = preprocess(job_description)

        # Similarity Score
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(vectors)[0][1]
        match_percentage = round(similarity * 100, 2)

        # Skill Extraction
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description)

        missing_skills = list(set(jd_skills) - set(resume_skills))

        # Display Results
        st.subheader("📊 Match Score")
        st.success(f"Resume matches {match_percentage}% with the job description")

        st.subheader("✅ Skills Found in Resume")
        st.write(resume_skills)

        st.subheader("❌ Missing Skills (Skill Gap)")
        st.write(missing_skills)

        # Recommendations
        if missing_skills:
            st.subheader("📚 Recommended Skills to Learn")
            for skill in missing_skills:
                st.write(f"- Learn {skill} from platforms like Coursera, Udemy, YouTube")

    else:
        st.warning("Please upload resume and enter job description")
