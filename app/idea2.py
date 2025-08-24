import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

nlp = spacy.load("en_core_web_sm")

def extract_resume_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def extract_skills(text, skill_list):
    text = text.lower()
    found_skills = []
    for skill in skill_list:
        canonical = skill.lower().strip()
        # Optionally: check for skill in resume text OR allow fuzzy/alias match logic
        if canonical in text:
            found_skills.append(skill)
    return list(set(found_skills))

def extract_requirements(jd_text):
    lines = re.split(r'[\n•\-\,]+', jd_text)
    requirements = [line.strip() for line in lines if len(line.strip()) > 4]
    # Use spaCy noun chunks for more coverage
    key_phrases = []
    for req in requirements:
        doc = nlp(req)
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
    return list(set(requirements + key_phrases))

def calculate_match_score(resume_text, jd_text, extracted_skills, required_skills):
    # TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Jaccard skill overlap
    set_resume = set([s.lower().strip() for s in extracted_skills])
    set_required = set([s.lower().strip() for s in required_skills])
    jaccard = len(set_resume & set_required) / len(set_resume | set_required) if (set_resume | set_required) else 0

    return round(tfidf_score * 100, 2), round(jaccard * 100, 2)

# ---- User Inputs ----

pdf_path = input("Enter the path to the resume PDF: ").strip()
job_desc = input("Paste the job description: ").strip()
print("Paste or type required skills (comma-separated OR one per line, end with blank line):")
req_lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    req_lines.append(line)

if req_lines:
    if len(req_lines) == 1 and ',' in req_lines[0]:
        skills = [s.strip() for s in req_lines[0].split(",") if s.strip()]
    else:
        skills = [re.sub(r'^- ', '', s).strip() for s in req_lines if s.strip()]
else:
    skills = []

required_skills = list(set([s.lower().strip() for s in skills if s]))  # canonicalize

# --------------------------------------------------------

if not pdf_path or not required_skills:
    print("ERROR: Resume PDF and at least one required skill must be provided!")
    exit()

resume_text = extract_resume_text_from_pdf(pdf_path)
resume_skills = extract_skills(resume_text, required_skills)
jd_requirements = extract_requirements(job_desc)
tfidf_match, jaccard_match = calculate_match_score(resume_text, job_desc, resume_skills, required_skills)

print("\n--- Results ---")
print("Extracted Resume Skills:", resume_skills)
print("JD Requirements/Noun Phrases:", jd_requirements)
print("TF-IDF Match Score (%):", tfidf_match)
print("Jaccard Skill Overlap (%):", jaccard_match)
