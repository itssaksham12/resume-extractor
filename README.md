Resume Matcher Web App
This project is a minimal, open-source app for screening resume PDFs against job descriptions and required skills.
It extracts skills from resumes, computes match scores (semantic similarity and skills overlap), and logs every evaluation to a CSV database for easy tracking or further analysis.

Features
Upload a Resume PDF: Extracts text and matches against skills/requirements.

Custom Job Description Input: Paste or type any JD.

Flexible Skill Input: Enter comma-separated or bulleted list.

Automatic Skill Extraction: Find and display matching skills from the resume.

Two Match Scores:

Semantic similarity (TF-IDF/cosine)

Jaccard skill overlap

CSV Results Log: Each screening is recorded for HR review or analytics.

Minimal Web Interface: Built with Flask for rapid, local deployment.

Installation
Clone this repo:

bash
git clone https://github.com/itssaksham12/resume-extractor
cd resume-matcher
Install dependencies (use a virtualenv if you wish):

bash
pip install flask spacy PyPDF2 scikit-learn
python -m spacy download en_core_web_sm
Usage
Start the app:

bash
python app.py
Open your browser to http://localhost:5000

Upload a PDF resume, paste in a job description and required skills, and click “Match.”

View match results on the page.
Each screening is also saved to resume_match_results.csv in the project directory.

Example Output
text
Extracted Skills: python, machine learning, nlp
TF-IDF Match Score: 82.34%
Jaccard Skill Overlap: 57.14%
Result saved to resume_match_results.csv
