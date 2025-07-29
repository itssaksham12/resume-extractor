AI Resume Screening with Deep Learning
This project implements an AI-powered pipeline for resume screening, extracting candidate skills, education, and other attributes using modern transformer embeddings (Sentence-BERT), and predicting recruiter decisions with a feed-forward neural network.
It supports robust feature engineering, model training, and AUC-ROC evaluation, making it suitable for realistic HR automation workflows.

Features
Automated skill, education, and certification embedding using all-mpnet-base-v2 transformer.

Jaccard skill overlap and semantic similarity feature engineering between candidate and job description.

Structured features: years of experience, projects, salary expectation.

Binary classifier neural network to predict “hire”/“no hire” labels and AUC-ROC on validation data.

Handles missing/noisy data robustly.

Data Requirements
Place your CSV files in the indicated directory:

AI_Resume_Screening.csv: Candidate features, skills, education, recruiter decision, job role

processed_job_data.csv: Job Title and Job Description columns

UpdatedResumeDataSet.csv: Additional resume content (optional for augmentation)

Setup
Clone the repo and activate Python 3.7–3.11 environment.

Install dependencies:

text
pip install torch sentence-transformers pandas scikit-learn numpy
Edit paths in the script to point to your CSVs if necessary.

How To Run
bash
python resume-extractor.py
The script will:

Merge the datasets

Extract all embeddings and structured features (may take several minutes for the first run)

Remove or fix rows with missing/invalid data

Train a neural network with early stopping

Display training and validation AUC-ROC after each epoch

Key Files
resume-extractor.py: Main Python script (all logic).

Your CSVs: As described above.

Notes
Assumes CSV columns and schemas as supplied in Kaggle/your project.

Large CSVs and transformer embedding extraction may consume several GB of RAM and take minutes per 10k entries on CPU.

For higher accuracy, tune model, add cross-validation, or use a GPU.

Output
Prints final AUC-ROC, sample class balance, and debug information during run.
