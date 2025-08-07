#!/usr/bin/env python3
"""
Simple test script for resume processing
"""

from resume_processor import add_resume_from_pdf

if __name__ == "__main__":
    # Configuration
    job_description = "Machine learning engineer. Required: Python, statistics, machine learning, artificial intelligence."
    skills_required = ["python", "statistics", "machine learning", "artificial intelligence"]
    resume_pdf_path = "Saksham_resume.pdf"  # Make sure this file exists in your workspace
    
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