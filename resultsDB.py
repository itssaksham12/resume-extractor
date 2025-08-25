#!/usr/bin/env python3
"""
ChromaDB Results Database for Resume Analysis
Stores comprehensive resume analysis results with the following schema:
- resume_name
- jd_comparison_result
- job_description
- matching_skills
- missing_skills
- resume_summary
"""

import chromadb
import uuid
import time
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import PyPDF2

# Import existing models from the project
from bert_summarizer import BERTSummarizerTrainer, TextPreprocessor
from bert_skills_extractor import SkillsExtractor
from resume_matcher_predictor import ResumeMatcherPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsDatabase:
    """ChromaDB-based results database for storing resume analysis results"""
    
    def __init__(self, db_path: str = "resume_results_db"):
        """
        Initialize the results database
        
        Args:
            db_path: Path to store the ChromaDB database
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.models_initialized = False
        
        # Initialize database and models
        self._initialize_database()
        self._initialize_models()
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Create or get the results collection
            self.collection = self.client.get_or_create_collection(
                name="resume_analysis_results",
                metadata={
                    "description": "Resume analysis results with comprehensive schema",
                    "created_at": datetime.now().isoformat(),
                    "schema_version": "1.0"
                }
            )
            
            logger.info(f"✅ ChromaDB initialized at: {self.db_path}")
            logger.info(f"✅ Collection: {self.collection.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize AI models for analysis"""
        try:
            # Initialize summarizer
            self.summarizer = BERTSummarizerTrainer()
            summarizer_path = "bert_summarizer_model.pth"
            if os.path.exists(summarizer_path):
                try:
                    self.summarizer.load_model(summarizer_path)
                    logger.info("✅ BERT Summarizer loaded")
                except Exception as e:
                    logger.warning(f"⚠️ BERT Summarizer failed to load: {e}")
            else:
                logger.warning("⚠️ BERT Summarizer model not found, using rule-based")
            
            # Initialize skills extractor
            self.skills_extractor = SkillsExtractor()
            logger.info("✅ Skills Extractor initialized")
            
            # Initialize resume matcher
            try:
                self.resume_matcher = ResumeMatcherPredictor()
                logger.info("✅ Resume Matcher initialized")
            except Exception as e:
                logger.warning(f"⚠️ Resume Matcher failed to initialize: {e}")
                self.resume_matcher = None
            
            self.models_initialized = True
            logger.info("✅ All models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            self.models_initialized = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def analyze_skills_match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze skills matching between resume and job description"""
        try:
            # Extract skills from both texts
            resume_skills = self.skills_extractor.extract_skills_from_text(resume_text)
            jd_skills = self.skills_extractor.extract_skills_from_text(job_description)
            
            # Calculate matching and missing skills
            resume_skills_set = set([skill.lower() for skill in resume_skills])
            jd_skills_set = set([skill.lower() for skill in jd_skills])
            
            matching_skills = list(resume_skills_set & jd_skills_set)
            missing_skills = list(jd_skills_set - resume_skills_set)
            extra_skills = list(resume_skills_set - jd_skills_set)
            
            # Calculate match percentage
            match_percentage = (len(matching_skills) / max(len(jd_skills), 1)) * 100
            
            return {
                "resume_skills": resume_skills,
                "jd_skills": jd_skills,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "extra_skills": extra_skills,
                "match_percentage": round(match_percentage, 2),
                "extraction_method": "Rule-based Skills Extractor"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skills match: {e}")
            return {
                "resume_skills": [],
                "jd_skills": [],
                "matching_skills": [],
                "missing_skills": [],
                "extra_skills": [],
                "match_percentage": 0.0,
                "extraction_method": "Error",
                "error": str(e)
            }
    
    def generate_resume_summary(self, resume_text: str, max_sentences: int = 3) -> str:
        """Generate resume summary using BERT summarizer or rule-based fallback"""
        try:
            if hasattr(self.summarizer, 'model_loaded') and self.summarizer.model_loaded:
                return self.summarizer.summarize(resume_text, max_sentences)
            else:
                # Fallback to rule-based summarization
                preprocessor = TextPreprocessor()
                from nltk.tokenize import sent_tokenize
                
                sentences = sent_tokenize(resume_text)
                if len(sentences) <= max_sentences:
                    return resume_text
                
                word_freq_list = preprocessor.word_frequency(sentences)
                sentence_scores = preprocessor.calculate_sentence_scores(sentences, word_freq_list)
                top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
                return '. '.join([sent for sent, _ in top_sentences])
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def get_jd_comparison_result(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Get comprehensive comparison result using LSTM matcher or fallback"""
        try:
            if self.resume_matcher:
                result = self.resume_matcher.predict_single_match(resume_text, job_description)
                match_score = result.get("match_score", 0.0)
                return {
                    "match_score": match_score,
                    "prediction_confidence": result.get("confidence", 0.8),
                    "recommendation": self._get_recommendation(match_score),
                    "analysis_method": "LSTM AI Model",
                    "detailed_analysis": result
                }
            else:
                # Fallback to skills-based matching
                skills_analysis = self.analyze_skills_match(resume_text, job_description)
                match_score = skills_analysis["match_percentage"]
                return {
                    "match_score": match_score,
                    "prediction_confidence": 0.8,
                    "recommendation": self._get_recommendation(match_score),
                    "analysis_method": "Skills-based Analysis",
                    "detailed_analysis": skills_analysis
                }
        except Exception as e:
            logger.error(f"Error in JD comparison: {e}")
            return {
                "match_score": 0.0,
                "prediction_confidence": 0.0,
                "recommendation": "Analysis failed",
                "analysis_method": "Error",
                "error": str(e)
            }
    
    def _get_recommendation(self, match_score: float) -> str:
        """Get recommendation based on match score"""
        if match_score >= 80:
            return "Excellent Match - Highly Recommended"
        elif match_score >= 65:
            return "Good Match - Recommended"
        elif match_score >= 50:
            return "Fair Match - Consider with training"
        elif match_score >= 30:
            return "Poor Match - Significant gaps"
        else:
            return "Very Poor Match - Not recommended"
    
    def add_resume_analysis(self, 
                          resume_text: str,
                          job_description: str,
                          resume_name: str,
                          pdf_path: Optional[str] = None) -> Optional[str]:
        """
        Add comprehensive resume analysis to ChromaDB
        
        Args:
            resume_text: Text content of the resume
            job_description: Job description to compare against
            resume_name: Name of the candidate/resume
            pdf_path: Optional path to the PDF file
            
        Returns:
            Unique ID of the stored analysis or None if failed
        """
        try:
            # Generate all analysis components
            logger.info(f"🔍 Analyzing resume: {resume_name}")
            
            # Skills analysis
            skills_analysis = self.analyze_skills_match(resume_text, job_description)
            
            # JD comparison
            jd_comparison = self.get_jd_comparison_result(resume_text, job_description)
            
            # Resume summary
            resume_summary = self.generate_resume_summary(resume_text)
            
            # Generate embedding for semantic search
            embedding = self._generate_embedding(resume_text)
            if not embedding:
                logger.warning("Failed to generate embedding, using zero vector")
                embedding = [0.0] * 384  # Default embedding size
            
            # Build metadata following the requested schema (flattened for ChromaDB)
            metadata = {
                # Core schema fields
                "resume_name": resume_name,
                "job_description": job_description,
                "matching_skills": ", ".join(skills_analysis["matching_skills"]),
                "missing_skills": ", ".join(skills_analysis["missing_skills"]),
                "resume_summary": resume_summary,
                
                # Flattened JD comparison results
                "match_score": jd_comparison["match_score"],
                "prediction_confidence": jd_comparison["prediction_confidence"],
                "recommendation": jd_comparison["recommendation"],
                "analysis_method": jd_comparison["analysis_method"],
                
                # Additional useful metadata
                "extra_skills": ", ".join(skills_analysis["extra_skills"]),
                "skills_match_percentage": skills_analysis["match_percentage"],
                "total_resume_skills": len(skills_analysis["resume_skills"]),
                "total_jd_skills": len(skills_analysis["jd_skills"]),
                "extraction_method": skills_analysis["extraction_method"],
                
                # File and processing metadata
                "pdf_path": pdf_path or "",
                "resume_text_length": len(resume_text),
                "processed_timestamp": time.time(),
                "processed_date": datetime.now().isoformat(),
                
                # Analysis flags
                "is_recommended": jd_comparison["match_score"] >= 65,
                "needs_training": 50 <= jd_comparison["match_score"] < 65,
                "has_critical_gaps": len(skills_analysis["missing_skills"]) > 5,
                "analysis_status": "completed"
            }
            
            # Generate unique ID
            unique_id = str(uuid.uuid4())
            
            # Add to ChromaDB
            self.collection.add(
                ids=[unique_id],
                embeddings=[embedding],
                documents=[resume_text],
                metadatas=[metadata]
            )
            
            logger.info(f"✅ Successfully stored analysis with ID: {unique_id}")
            self._print_analysis_summary(metadata)
            return unique_id
            
        except Exception as e:
            logger.error(f"Error adding resume analysis: {e}")
            return None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for semantic search"""
        try:
            # Try to use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(text).tolist()
                return embedding
            except ImportError:
                logger.warning("sentence-transformers not available, using simple embedding")
                # Simple fallback embedding (not recommended for production)
                return [hash(text) % 1000 / 1000.0] * 384
            except Exception as e:
                logger.warning(f"sentence-transformers failed: {e}, using simple embedding")
                # Simple fallback embedding
                return [hash(text) % 1000 / 1000.0] * 384
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a simple hash-based embedding as last resort
            try:
                return [hash(text) % 1000 / 1000.0] * 384
            except:
                return [0.0] * 384
    
    def _print_analysis_summary(self, metadata: Dict[str, Any]):
        """Print a nice summary of the analysis"""
        print("\n" + "="*60)
        print("📊 RESUME ANALYSIS SUMMARY")
        print("="*60)
        print(f"👤 Candidate: {metadata['resume_name']}")
        print(f"🎯 Match Score: {metadata['match_score']:.1f}%")
        print(f"💡 Recommendation: {metadata['recommendation']}")
        
        matching_skills = metadata['matching_skills'].split(", ") if metadata['matching_skills'] else []
        missing_skills = metadata['missing_skills'].split(", ") if metadata['missing_skills'] else []
        
        print(f"✅ Matching Skills ({len(matching_skills)}): {', '.join(matching_skills[:5])}{'...' if len(matching_skills) > 5 else ''}")
        print(f"❌ Missing Skills ({len(missing_skills)}): {', '.join(missing_skills[:5])}{'...' if len(missing_skills) > 5 else ''}")
        print(f"📝 Summary: {metadata['resume_summary'][:100]}...")
        print("="*60)
    
    def search_resumes(self, 
                      query: str, 
                      n_results: int = 5,
                      filter_criteria: Optional[Dict] = None) -> Dict[str, Any]:
        """Search resumes using semantic search"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if filter_criteria:
                if filter_criteria.get("min_match_score"):
                    where_clause["match_score"] = {"$gte": filter_criteria["min_match_score"]}
                if filter_criteria.get("is_recommended"):
                    where_clause["is_recommended"] = True
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            return {
                "query": query,
                "total_results": len(results["ids"][0]),
                "results": [
                    {
                        "id": results["ids"][0][i],
                        "resume_name": results["metadatas"][0][i]["resume_name"],
                        "match_score": results["metadatas"][0][i]["match_score"],
                        "recommendation": results["metadatas"][0][i]["recommendation"],
                        "matching_skills": results["metadatas"][0][i]["matching_skills"].split(", ") if results["metadatas"][0][i]["matching_skills"] else [],
                        "missing_skills": results["metadatas"][0][i]["missing_skills"].split(", ") if results["metadatas"][0][i]["missing_skills"] else [],
                        "summary": results["metadatas"][0][i]["resume_summary"],
                        "similarity_score": results["distances"][0][i] if results["distances"] else None
                    }
                    for i in range(len(results["ids"][0]))
                ]
            }
            
        except Exception as e:
            logger.error(f"Error searching resumes: {e}")
            return {"query": query, "total_results": 0, "results": [], "error": str(e)}
    
    def get_resume_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Get complete resume analysis by ID"""
        try:
            results = self.collection.get(
                ids=[resume_id],
                include=["metadatas", "documents"]
            )
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0],
                    "full_text": results["documents"][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting resume by ID: {e}")
            return None
    
    def get_all_resumes(self, limit: int = 100) -> Dict[str, Any]:
        """Get all resumes from the database"""
        try:
            results = self.collection.get(
                limit=limit,
                include=["metadatas", "documents"]
            )
            
            return {
                "total_count": len(results["ids"]),
                "resumes": [
                    {
                        "id": results["ids"][i],
                        "resume_name": results["metadatas"][i]["resume_name"],
                        "match_score": results["metadatas"][i]["match_score"],
                        "recommendation": results["metadatas"][i]["recommendation"],
                        "processed_date": results["metadatas"][i]["processed_date"]
                    }
                    for i in range(len(results["ids"]))
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting all resumes: {e}")
            return {"total_count": 0, "resumes": [], "error": str(e)}
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary of all resumes in database"""
        try:
            all_resumes = self.collection.get(include=["metadatas"])
            
            if not all_resumes["metadatas"]:
                return {"total_resumes": 0}
            
            metadatas = all_resumes["metadatas"]
            
            # Calculate statistics
            match_scores = [m["match_score"] for m in metadatas]
            
            return {
                "total_resumes": len(metadatas),
                "average_match_score": sum(match_scores) / len(match_scores),
                "recommended_candidates": sum(1 for m in metadatas if m.get("is_recommended", False)),
                "candidates_needing_training": sum(1 for m in metadatas if m.get("needs_training", False)),
                "high_performers": sum(1 for score in match_scores if score >= 80),
                "most_common_missing_skills": self._get_most_common_missing_skills(metadatas),
                "most_common_matching_skills": self._get_most_common_matching_skills(metadatas)
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}
    
    def _get_most_common_missing_skills(self, metadatas: List[Dict]) -> List[Tuple[str, int]]:
        """Get most common missing skills across all resumes"""
        from collections import Counter
        all_missing = []
        for m in metadatas:
            missing_skills = m.get("missing_skills", "")
            if missing_skills:
                all_missing.extend([skill.strip() for skill in missing_skills.split(",") if skill.strip()])
        return Counter(all_missing).most_common(10)
    
    def _get_most_common_matching_skills(self, metadatas: List[Dict]) -> List[Tuple[str, int]]:
        """Get most common matching skills across all resumes"""
        from collections import Counter
        all_matching = []
        for m in metadatas:
            matching_skills = m.get("matching_skills", "")
            if matching_skills:
                all_matching.extend([skill.strip() for skill in matching_skills.split(",") if skill.strip()])
        return Counter(all_matching).most_common(10)
    
    def delete_resume(self, resume_id: str) -> bool:
        """Delete a resume analysis from the database"""
        try:
            self.collection.delete(ids=[resume_id])
            logger.info(f"✅ Deleted resume analysis: {resume_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting resume: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all data from the database"""
        try:
            self.collection.delete(where={})
            logger.info("✅ Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

# Example usage and testing
def main():
    """Example usage of the ResultsDatabase"""
    print("🚀 Results Database with ChromaDB")
    print("=" * 60)
    
    # Initialize database
    db = ResultsDatabase()
    
    # Example job description
    job_description = """
    We are seeking a Senior Data Scientist to join our team. The ideal candidate will have 5+ years of experience 
    in machine learning, Python programming, and statistical analysis. You will be responsible for developing 
    predictive models, analyzing large datasets, and communicating insights to stakeholders. 
    Experience with TensorFlow, PyTorch, SQL, and cloud platforms (AWS/Azure) is required. 
    Knowledge of deep learning, NLP, and computer vision is a plus.
    """
    
    # Example resume text
    sample_resume = """
    John Doe
    Senior Data Scientist
    
    EXPERIENCE:
    ABC Company (2020-2023)
    - Developed machine learning models using Python and TensorFlow
    - Analyzed large datasets and created predictive models
    - Worked with SQL databases and cloud platforms
    - Led a team of 3 data analysts
    
    SKILLS:
    Python, Machine Learning, TensorFlow, SQL, AWS, Data Analysis, Statistics
    
    EDUCATION:
    Master's in Computer Science, Stanford University
    """
    
    # Add resume analysis
    print("📄 Adding sample resume analysis...")
    resume_id = db.add_resume_analysis(
        resume_text=sample_resume,
        job_description=job_description,
        resume_name="John Doe",
        pdf_path="sample_resume.pdf"
    )
    
    if resume_id:
        print(f"✅ Resume analysis stored with ID: {resume_id}")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        search_results = db.search_resumes("machine learning python data science")
        print(f"Found {search_results['total_results']} matching resumes")
        
        # Test analytics
        print("\n📊 Database analytics:")
        analytics = db.get_analytics_summary()
        print(f"Total resumes: {analytics.get('total_resumes', 0)}")
        print(f"Average match score: {analytics.get('average_match_score', 0):.1f}%")
        
        # Test getting resume by ID
        print(f"\n📋 Getting resume details for ID: {resume_id}")
        resume_details = db.get_resume_by_id(resume_id)
        if resume_details:
            print(f"Resume Name: {resume_details['metadata']['resume_name']}")
            print(f"Match Score: {resume_details['metadata']['match_score']}%")
    
    else:
        print("❌ Failed to add resume analysis")

if __name__ == "__main__":
    main()
