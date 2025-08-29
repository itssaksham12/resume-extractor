from flask import Flask, request, render_template_string
import os
import io
import traceback
import PyPDF2

# LSTM predictor
from resume_matcher_predictor import ResumeMatcherPredictor

# BERT skills extractor (with fallback to rule-based if model not present)
from bert_skills_extractor import SkillsExtractor, BERTSkillsModel

app = Flask(__name__)

# Initialize global components once
predictor = None
skills_extractor = None
bert_skills_model = None


def init_components():
    global predictor, skills_extractor, bert_skills_model

    if predictor is None:
        predictor = ResumeMatcherPredictor()  # auto-load latest model and components

    if skills_extractor is None:
        skills_extractor = SkillsExtractor()

    # Try loading trained BERT skills model if available; otherwise fallback to rule-based
    if bert_skills_model is None:
        model_path = os.path.join(os.getcwd(), "bert_skills_model.pth")
        if os.path.exists(model_path):
            try:
                bert_skills_model = BERTSkillsModel(skills_extractor)
                # Need number of classes to load correctly
                # We'll infer it from the saved mlb in the checkpoint
                import torch
                ckpt = torch.load(model_path, map_location=skills_extractor.device)
                n_classes = len(ckpt["mlb"].classes_)
                bert_skills_model.load_model(model_path, n_classes=n_classes)
            except Exception:
                # If loading fails, keep None and use rule-based skills extraction
                bert_skills_model = None


def extract_text_from_pdf(file_storage) -> str:
    try:
        reader = PyPDF2.PdfReader(file_storage)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception:
        # Some PDFs may be scanned or encrypted; return empty string on failure
        return ""


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resume Matcher (BERT + LSTM)</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 30px; background: #f5f7fa; color: #333; }
    h1 { color: #2c3e50; }
    .container { display: grid; grid-template-columns: 1fr; gap: 20px; max-width: 1000px; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    label { font-weight: bold; display: block; margin: 10px 0 6px 0; }
    textarea, input[type="text"], input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
    textarea { min-height: 140px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .btn { background: #3498db; color: #fff; padding: 12px 18px; border: none; border-radius: 4px; cursor: pointer; font-size: 15px; }
    .btn:hover { background: #2d83bd; }
    .badges span { display: inline-block; background: #ecf0f1; color: #34495e; padding: 6px 10px; border-radius: 999px; margin: 4px 6px 0 0; font-size: 12px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .metric { background: #f8f9fa; padding: 12px; border-radius: 8px; text-align: center; }
    .metric h3 { margin: 6px 0; }
    .error { color: #c0392b; white-space: pre-wrap; }

    /* JD highlight styles */
    .jd-card { background: #eef6ff; border: 1px solid #cfe6ff; }
    .jd-header { display: flex; align-items: baseline; gap: 10px; }
    .count-pill { background: #2d83bd; color: #fff; padding: 4px 10px; border-radius: 999px; font-size: 12px; }
    .badges-jd span { display: inline-block; background: #2d83bd; color: #fff; padding: 7px 12px; border-radius: 999px; margin: 6px 8px 0 0; font-size: 13px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Resume Matcher (BERT skills + LSTM scoring)</h1>
      <form method="post" enctype="multipart/form-data">
        <label for="jd">Job Description</label>
        <textarea id="jd" name="jd" placeholder="Paste the job description here..." required>{{ request.form.get('jd', '') }}</textarea>

        <div class="row">
          <div>
            <label for="resume_pdf">Resume PDF (preferred)</label>
            <input id="resume_pdf" name="resume_pdf" type="file" accept="application/pdf" />
          </div>
          <div>
            <label for="resume_text">OR Resume Text</label>
            <textarea id="resume_text" name="resume_text" placeholder="Paste resume text if no PDF"></textarea>
          </div>
        </div>

        <button class="btn" type="submit">Analyze Match</button>
      </form>
    </div>

    {% if error %}
    <div class="card">
      <h2>Error</h2>
      <div class="error">{{ error }}</div>
    </div>
    {% endif %}

    {% if results %}
    <div class="card jd-card">
      <div class="jd-header">
        <h2>JD Skills (extracted)</h2>
        <span class="count-pill">{{ results.jd_extracted_skills|length }} skills</span>
      </div>
      <div class="badges-jd">
        {% for s in results.jd_extracted_skills %}<span>{{ s }}</span>{% endfor %}
      </div>
    </div>

    <div class="card">
      <h2>Results</h2>
      <div class="metrics">
        <div class="metric"><h3>{{ results.overall_match_score }}%</h3><div>Overall Match</div></div>
        <div class="metric"><h3>{{ results.skill_match_score }}%</h3><div>Skill Match</div></div>
        <div class="metric"><h3>{{ results.experience_match_score }}%</h3><div>Experience Match</div></div>
      </div>

      <h3>Recommendation</h3>
      <p>{{ results.recommendation }}</p>

      <h3>Matching Skills</h3>
      <div class="badges">
        {% for s in results.matching_skills %}<span>{{ s }}</span>{% endfor %}
      </div>

      <h3>Missing Skills</h3>
      <div class="badges">
        {% for s in results.missing_skills %}<span>{{ s }}</span>{% endfor %}
      </div>

      <h3>Extra Skills</h3>
      <div class="badges">
        {% for s in results.extra_skills %}<span>{{ s }}</span>{% endfor %}
      </div>

      <h3>Details</h3>
      <ul>
        <li>Candidate Experience: {{ results.candidate_experience }} years</li>
        <li>Required Experience: {{ results.required_experience }} years</li>
        <li>Projects: {{ results.projects_count }}</li>
        <li>Education Level: {{ results.education_level }}</li>
      </ul>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    init_components()

    error = None
    results = None

    if request.method == 'POST':
        try:
            jd_text = request.form.get('jd', '').strip()
            resume_text = request.form.get('resume_text', '').strip()
            resume_pdf = request.files.get('resume_pdf')

            if not jd_text:
                raise ValueError('Job description is required')

            if not resume_text and resume_pdf and resume_pdf.filename:
                resume_text = extract_text_from_pdf(resume_pdf)

            if not resume_text:
                raise ValueError('Provide either a resume PDF or paste resume text')

            # Extract JD skills via BERT if available, else fallback to rule-based extractor
            if bert_skills_model is not None:
                try:
                    jd_skills = bert_skills_model.predict_skills(jd_text, threshold=0.3)
                except Exception:
                    jd_skills = skills_extractor.extract_skills_from_text(jd_text)
            else:
                jd_skills = skills_extractor.extract_skills_from_text(jd_text)

            # Use LSTM predictor to compute match
            match = predictor.predict_single_match(resume_text, jd_text)

            # Inject JD skills into the results for display
            match['jd_extracted_skills'] = sorted(list(set(jd_skills)))[:50]

            results = match
        except Exception as ex:
            error = f"{ex}\n\n{traceback.format_exc()}"

    return render_template_string(HTML_TEMPLATE, results=results, error=error, request=request)


if __name__ == '__main__':
    # Host on localhost:5001 to avoid AirPlay conflict on macOS
    app.run(host='127.0.0.1', port=5001, debug=True)