from flask import Flask, request, render_template_string
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

import csv
import os

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

CSV_FILENAME = "resume_match_results.csv"

def extract_resume_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def extract_skills(text, skill_list):
    text = text.lower()
    found_skills = [skill for skill in skill_list if skill.lower() in text]
    return list(set(found_skills))

def extract_requirements(jd_text):
    lines = re.split(r'[\n•\-]+', jd_text)
    requirements = [line.strip() for line in lines if len(line.strip()) > 4]
    key_phrases = []
    for req in requirements:
        doc = nlp(req)
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
    return list(set(requirements + key_phrases))

def calculate_match_score(resume_text, jd_text, extracted_skills, required_skills):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    set_resume = set([s.lower() for s in extracted_skills])
    set_required = set([s.lower() for s in required_skills])
    jaccard = len(set_resume & set_required) / len(set_resume | set_required) if (set_resume | set_required) else 0

    return round(tfidf_score * 100, 2), round(jaccard * 100, 2)

def save_result_to_csv(resume_name, extracted_skills, job_description,
                       required_skills, tfidf_score, jaccard_score, filename=CSV_FILENAME):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "resume_filename",
            "extracted_skills",
            "job_description",
            "required_skills",
            "tfidf_match_score",
            "jaccard_match_score"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "resume_filename": resume_name,
            "extracted_skills": ", ".join(extracted_skills),
            "job_description": job_description,
            "required_skills": ", ".join(required_skills),
            "tfidf_match_score": tfidf_score,
            "jaccard_match_score": jaccard_score
        })

HTML_TEMPLATE = '''
<!doctype html>
<title>Resume Matcher</title>
<h1>Resume Matcher</h1>
<form method=post enctype=multipart/form-data>
  <label>Resume PDF:</label>
  <input type=file name=resume_file required><br><br>
  <label>Job Description:</label><br>
  <textarea name=job_desc rows=5 cols=60 required>{{request.form.get('job_desc', '')}}</textarea><br><br>
  <label>Required Skills (comma separated):</label><br>
  <input type=text name=required_skills size=60 value="{{request.form.get('required_skills', '')}}" required><br><br>
  <input type=submit value=Match>
</form>

{% if results %}
  <h2>Results</h2>
  <p><b>Extracted Skills:</b> {{ results.extracted_skills }}</p>
  <p><b>TF-IDF Match Score:</b> {{ results.tfidf_match }}%</p>
  <p><b>Jaccard Skill Overlap:</b> {{ results.jaccard_match }}%</p>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        resume_file = request.files.get('resume_file')
        job_desc = request.form.get('job_desc', '')
        required_skills_str = request.form.get('required_skills', '')
        required_skills = [s.strip() for s in required_skills_str.split(",") if s.strip()]

        if resume_file and job_desc and required_skills:
            resume_text = extract_resume_text_from_pdf(resume_file.stream)
            resume_skills = extract_skills(resume_text, required_skills)
            tfidf_match, jaccard_match = calculate_match_score(resume_text, job_desc, resume_skills, required_skills)

            save_result_to_csv(
                resume_name=resume_file.filename,
                extracted_skills=resume_skills,
                job_description=job_desc,
                required_skills=required_skills,
                tfidf_score=tfidf_match,
                jaccard_score=jaccard_match
            )

            results = {
                "extracted_skills": ", ".join(resume_skills),
                "tfidf_match": tfidf_match,
                "jaccard_match": jaccard_match
            }

    return render_template_string(HTML_TEMPLATE, results=results, request=request)

if __name__ == '__main__':
    app.run(debug=True)
