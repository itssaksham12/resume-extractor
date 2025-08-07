from flask import Flask, request, render_template_string
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import os

app = Flask(__name__)

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
    """Simple keyword extraction without NLTK dependencies"""
    # Split by common delimiters
    lines = re.split(r'[\n•\-]+', jd_text)
    
    # Extract meaningful phrases (3+ words)
    requirements = []
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Only lines with substantial content
            # Split into words and find phrases
            words = line.split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 5 and not phrase.lower() in ['the', 'and', 'or', 'but', 'for', 'with']:
                    requirements.append(phrase)
    
    # Also extract single important words (likely skills)
    important_words = []
    for line in lines:
        words = line.split()
        for word in words:
            word = word.strip('.,!?()').lower()
            if len(word) > 3 and word not in ['the', 'and', 'or', 'but', 'for', 'with', 'have', 'will', 'must', 'should', 'need', 'want', 'like', 'good', 'great', 'excellent']:
                important_words.append(word)
    
    return list(set(requirements + important_words))

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
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resume Matcher</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 30px;
      background: #f5f7fa;
      color: #333;
    }
    h1 {
      color: #2c3e50;
    }
    form {
      background: white;
      padding: 25px;
      border-radius: 8px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      max-width: 700px;
    }
    label {
      font-weight: bold;
      display: block;
      margin-top: 15px;
      margin-bottom: 6px;
    }
    input[type="file"], input[type="text"], textarea {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 14px;
      box-sizing: border-box;
    }
    textarea {
      resize: vertical;
      min-height: 120px;
    }
    input[type="submit"] {
      background: #3498db;
      color: white;
      padding: 12px 24px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 16px;
      margin-top: 20px;
    }
    input[type="submit"]:hover {
      background: #2980b9;
    }
    .progress {
      display: none;
      background: #f8f9fa;
      padding: 15px;
      border-radius: 4px;
      margin-top: 15px;
      text-align: center;
      color: #6c757d;
    }
    .results {
      background: white;
      padding: 25px;
      border-radius: 8px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      margin-top: 20px;
      max-width: 700px;
    }
    .result-row {
      margin: 10px 0;
      padding: 10px;
      background: #f8f9fa;
      border-radius: 4px;
    }
    .badge {
      background: #3498db;
      color: white;
      padding: 4px 8px;
      border-radius: 3px;
      font-size: 12px;
      font-weight: bold;
    }
  </style>
  <script>
    function showProgress() {
      document.getElementById('progress').style.display = 'block';
    }
  </script>
</head>
<body>
  <h1>Resume Matcher</h1>
  <form method="post" enctype="multipart/form-data" onsubmit="showProgress()">
    <label for="resume_file">Resume PDF</label>
    <input type="file" name="resume_file" id="resume_file" accept="application/pdf" required />
    
    <label for="job_desc">Job Description</label>
    <textarea name="job_desc" id="job_desc" rows="6" placeholder="Paste the job description here..." required>{{request.form.get('job_desc', '')}}</textarea>
    
    <label for="required_skills">Required Skills (comma separated)</label>
    <input type="text" name="required_skills" id="required_skills" placeholder="Python, machine learning, communication, etc." value="{{request.form.get('required_skills', '')}}" required />
    
    <input type="submit" value="Match Resume" />
    <div id="progress" class="progress">Processing your resume and job data, please wait...</div>
  </form>
  
  {% if results %}
    <div class="results">
      <h2>Match Results</h2>
      <div class="result-row"><span class="badge">Skills Matched:</span> {{ results.extracted_skills }}</div>
      <div class="result-row"><span class="badge" style="background-color: #27ae60;">TF-IDF Similarity:</span> {{ results.tfidf_match }}%</div>
      <div class="result-row"><span class="badge" style="background-color: #e67e22;">Jaccard Overlap:</span> {{ results.jaccard_match }}%</div>
    </div>
  {% endif %}
</body>
</html>
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