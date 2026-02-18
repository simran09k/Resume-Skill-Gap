
import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def preprocess(text):
    return text.lower()

SKILLS_DB = [
    "python","java","c++","sql","machine learning",
    "data analysis","html","css","javascript",
    "react","node","mongodb","linux","aws","dbms"
]

def extract_skills(text):
    found_skills = []
    text = text.lower()
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return found_skills

st.title("📄 AI Resume Skill Gap Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and job_description:

        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_docx(resume_file)

        resume_clean = preprocess(resume_text)
        jd_clean = preprocess(job_description)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(vectors)[0][1]
        match_percentage = round(similarity * 100, 2)

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description)

        missing_skills = list(set(jd_skills) - set(resume_skills))

        st.subheader("📊 Match Score")
        st.success(f"Resume matches {match_percentage}% with the job description")

        st.subheader("✅ Skills Found")
        st.write(resume_skills)

        st.subheader("❌ Missing Skills")
        st.write(missing_skills)

    else:
        st.warning("Upload resume and enter job description.")
