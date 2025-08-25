import csv
import os

CSV_FILENAME = "resume_match_results.csv"

def save_result_to_csv(
    resume_filepath,
    extracted_skills,
    job_description,
    required_skills,
    tfidf_score,
    jaccard_score,
    filename=CSV_FILENAME
):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "resume_filepath",
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
            "resume_filepath": resume_filepath,
            "extracted_skills": ", ".join(extracted_skills),
            "job_description": job_description,
            "required_skills": ", ".join(required_skills),
            "tfidf_match_score": tfidf_score,
            "jaccard_match_score": jaccard_score
        })

    print(f"Result saved to {filename}")
