import chromadb
import ollama
import PyPDF2
import uuid
import time
import os

# Initialize ChromaDB
client = chromadb.PersistentClient(path="resume_db")
collection = client.get_or_create_collection("resumes")

def add_resume_from_pdf(pdf_path, job_desc, skills_list):
    """Add a resume from PDF to ChromaDB with proper embedding format handling"""
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found!")
        return None
    
    # Extract text from PDF using PyPDF2
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            resume_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        print(f"Successfully extracted text from {pdf_path}")
        print(f"Text length: {len(resume_text)} characters")
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    
    # Generate embedding for the resume
    try:
        emb_resp = ollama.embed(model="nomic-embed-text", input=resume_text)
        embedding = emb_resp.get("embedding") or emb_resp.get("embeddings")
        if embedding is None:
            print("No embedding found, skipping.")
            return None
        
        # Fix: Flatten the embedding to remove the extra nesting level
        # If embedding is [[[...]]], we want [[...]]
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list) and len(embedding[0]) > 0:
                if isinstance(embedding[0][0], list):
                    # Triple nested: [[[...]]] -> [[...]]
                    embedding = embedding[0]
                    print("Fixed embedding format (removed extra nesting)")
        
        print(f"Generated embedding with {len(embedding)} dimensions")
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

    # Build metadata
    properties = {
        "skills": skills_list,
        "job_desc": job_desc,
        "resume_file": os.path.basename(pdf_path),
        "hired": None,
        "timestamp": time.time()
    }
    unique_id = str(uuid.uuid4())
    
    # Add to Chroma collection
    try:
        collection.add(
            ids=[unique_id],
            embeddings=[embedding],
            documents=[resume_text],
            metadatas=[properties]
        )
        print(f"Successfully added resume with ID: {unique_id}")
        return unique_id
        
    except Exception as e:
        print(f"Error adding to ChromaDB: {e}")
        return None

if __name__ == "__main__":
    # Configuration
    job_description = "Machine learning engineer. Required: Python, statistics, machine learning, artificial intelligence."
    skills_required = ["python", "statistics", "machine learning", "artificial intelligence"]
    resume_pdf_path = "Saksham_resume.pdf"  # Use the actual PDF file in your workspace
    
    print("Starting resume processing...")
    print(f"PDF path: {resume_pdf_path}")
    print(f"Job description: {job_description}")
    print(f"Required skills: {skills_required}")
    print("-" * 50)
    
    # Process the resume
    resume_id = add_resume_from_pdf(resume_pdf_path, job_description, skills_required)
    
    if resume_id:
        print(f"\n✅ Success! Resume processed and added to ChromaDB with ID: {resume_id}")
    else:
        print("\n❌ Failed to process resume") 