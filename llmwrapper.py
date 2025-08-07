import ollama
import PyPDF2
import numpy as np

# 1. Gather user input
job_desc = input("Paste the full job description:\n")
skills_required = input("Enter the required skills (comma-separated):\n").lower().split(",")

# 2. Load resume from PDF
resume_file = input("Enter path to the resume PDF file:\n").strip()
with open(resume_file, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    resume_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
resume_text_lower = resume_text.lower()

# 3. Quick direct skill match (case-insensitive, substring)
direct_matches = set()
for skill in map(str.strip, skills_required):
    if skill and skill in resume_text_lower:
        direct_matches.add(skill)

# 4. Embedding-based similarity (semantic matching)
query = " ".join(skills_required) + " " + job_desc

# Generate embeddings
embed_model = "nomic-embed-text"  # Make sure it's pulled
resume_emb_resp = ollama.embed(model=embed_model, input=resume_text)
jd_emb_resp = ollama.embed(model=embed_model, input=query)

# Find correct key for the embedding
def get_vec(resp):
    # Check for possible keys
    return next((resp.get(k) for k in ["embedding", "embeddings"] if k in resp), None)

resume_vec = get_vec(resume_emb_resp)
jd_vec = get_vec(jd_emb_resp)

if resume_vec is not None and jd_vec is not None:
    v1 = np.array(resume_vec).reshape(-1)
    v2 = np.array(jd_vec).reshape(-1)
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"\nCosine similarity (semantic match): {cosine_sim:.3f} (1.0 = perfect match, 0.0 = none)")
else:
    print("Error: Could not extract embeddings for resume or JD.")

# 5. Output results
print("\n---- Matching result ----")
print(f"Direct (exact word) skill matches in resume: {sorted(direct_matches)}")
print("Consider a cosine similarity above 0.7 as a strong semantic match.")
