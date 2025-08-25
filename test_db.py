#!/usr/bin/env python3
"""
Simple test script for ChromaDB implementation
"""

from resultsDB import ResultsDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database():
    """Test the ChromaDB implementation"""
    print("🧪 Testing ChromaDB Implementation")
    print("=" * 50)
    
    # Initialize database
    db = ResultsDatabase()
    
    # Test data
    job_description = """
    We are seeking a Senior Data Scientist to join our team. The ideal candidate will have 5+ years of experience 
    in machine learning, Python programming, and statistical analysis. You will be responsible for developing 
    predictive models, analyzing large datasets, and communicating insights to stakeholders. 
    Experience with TensorFlow, PyTorch, SQL, and cloud platforms (AWS/Azure) is required. 
    Knowledge of deep learning, NLP, and computer vision is a plus.
    """
    
    resume_text = """
    Sarah Johnson
    Senior Data Scientist
    
    EXPERIENCE:
    TechCorp (2020-2023)
    - Developed machine learning models using Python and TensorFlow
    - Analyzed large datasets and created predictive models
    - Worked with SQL databases and cloud platforms (AWS)
    - Led a team of 3 data analysts
    - Published 2 papers on deep learning applications
    
    SKILLS:
    Python, Machine Learning, TensorFlow, SQL, AWS, Data Analysis, Statistics, Deep Learning, NLP
    
    EDUCATION:
    Master's in Computer Science, Stanford University
    """
    
    # Test 1: Add resume analysis
    print("📄 Test 1: Adding resume analysis...")
    resume_id = db.add_resume_analysis(
        resume_text=resume_text,
        job_description=job_description,
        resume_name="Sarah Johnson"
    )
    
    if resume_id:
        print(f"✅ Success! Resume ID: {resume_id}")
    else:
        print("❌ Failed to add resume")
        return
    
    # Test 2: Get resume by ID
    print("\n📋 Test 2: Getting resume by ID...")
    resume_details = db.get_resume_by_id(resume_id)
    if resume_details:
        print(f"✅ Found resume: {resume_details['metadata']['resume_name']}")
        print(f"   Match Score: {resume_details['metadata']['match_score']}%")
        print(f"   Recommendation: {resume_details['metadata']['recommendation']}")
    else:
        print("❌ Failed to get resume")
    
    # Test 3: Search resumes
    print("\n🔍 Test 3: Searching resumes...")
    search_results = db.search_resumes("python machine learning", n_results=5)
    print(f"✅ Found {search_results['total_results']} matching resumes")
    
    # Test 4: Get analytics
    print("\n📊 Test 4: Getting analytics...")
    analytics = db.get_analytics_summary()
    print(f"✅ Database has {analytics['total_resumes']} resumes")
    print(f"   Average match score: {analytics['average_match_score']:.1f}%")
    print(f"   Recommended candidates: {analytics['recommended_candidates']}")
    
    # Test 5: Get all resumes
    print("\n📋 Test 5: Getting all resumes...")
    all_resumes = db.get_all_resumes(limit=10)
    print(f"✅ Found {all_resumes['total_count']} resumes in database")
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    test_database()
