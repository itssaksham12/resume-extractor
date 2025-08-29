#!/usr/bin/env python3
"""
FastAPI Backend for Resume Extractor
Integrates all trained ML models: BERT Skills, LSTM Matcher, BERT Summarizer
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
import asyncio
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Core ML imports
import torch
import tensorflow as tf
import numpy as np
import pandas as pd

# Add parent directory to path for model imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import your existing models
from bert_skills_extractor import SkillsExtractor, BERTSkillsModel
from lstm_resume_matcher import LSTMResumeMatcherTrainer, AdvancedFeatureExtractor
from app.bert_summarizer import BERTSummarizerTrainer, TextPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API validation
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

class SkillsExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Job description or resume text")
    use_ai_model: bool = Field(True, description="Use AI model if available")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold")

class SkillsExtractionResponse(BaseModel):
    skills: List[str]
    skill_categories: Dict[str, List[str]]
    model_used: str
    confidence: float
    total_skills: int
    processing_time: float

class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Resume text content")
    job_description: str = Field(..., min_length=10, description="Job description")
    use_ai_model: bool = Field(True, description="Use AI model if available")

class ResumeAnalysisResponse(BaseModel):
    overall_match_score: float
    skill_match_score: float
    experience_match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]
    recommendation: str
    candidate_profile: Dict[str, Union[int, float]]
    model_used: str
    processing_time: float

class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Text to summarize")
    max_sentences: int = Field(3, ge=1, le=10, description="Maximum sentences in summary")
    text_type: str = Field("general", description="Type of text: job_description, resume, or general")

class SummarizationResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    model_used: str
    processing_time: float

class PDFUploadResponse(BaseModel):
    text: str
    filename: str
    pages: int
    word_count: int
    extraction_success: bool

# FastAPI app initialization
app = FastAPI(
    title="Resume Extractor API",
    description="AI-powered resume and job description analysis using BERT and LSTM models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative React port
        "https://*.onrender.com",  # Render domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models = {
    "skills_extractor": None,
    "bert_skills_model": None,
    "lstm_matcher": None,
    "bert_summarizer": None,
    "text_preprocessor": None,
    "feature_extractor": None
}

@app.on_event("startup")
async def load_models():
    """Load all AI models on startup with comprehensive error handling"""
    global models
    
    logger.info("🚀 Starting model loading process...")
    
    try:
        # 1. Load Basic Skills Extractor (always works)
        logger.info("📊 Loading Skills Extractor...")
        models["skills_extractor"] = SkillsExtractor()
        logger.info("✅ Skills Extractor loaded successfully")
        
        # 2. Try to load BERT Skills Model
        logger.info("🧠 Loading BERT Skills Model...")
        bert_model_paths = [
            parent_dir / "bert_skills_model.pth",
            parent_dir / "models" / "bert_skills_model.pth",
            parent_dir / "app" / "bert_skills_model.pth"
        ]
        
        for model_path in bert_model_paths:
            if model_path.exists():
                try:
                    models["bert_skills_model"] = BERTSkillsModel(models["skills_extractor"])
                    # Load checkpoint to get n_classes
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    n_classes = len(checkpoint["mlb"].classes_)
                    models["bert_skills_model"].load_model(str(model_path), n_classes=n_classes)
                    logger.info(f"✅ BERT Skills Model loaded from {model_path}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load BERT model from {model_path}: {e}")
                    continue
        
        if not models["bert_skills_model"]:
            logger.info("ℹ️ BERT Skills Model not found, using rule-based extraction")
        
        # 3. Load LSTM Resume Matcher
        logger.info("🔄 Loading LSTM Resume Matcher...")
        lstm_model_paths = [
            parent_dir / "lstm_resume_matcher_best.h5",
            parent_dir / "models" / "lstm_resume_matcher_best.h5"
        ]
        
        for model_path in lstm_model_paths:
            if model_path.exists():
                try:
                    models["lstm_matcher"] = LSTMResumeMatcherTrainer()
                    models["lstm_matcher"].model = tf.keras.models.load_model(str(model_path))
                    
                    # Load supporting components
                    timestamp = "20250828_113623"  # Your model timestamp
                    model_dir = model_path.parent
                    
                    # Load tokenizer, scaler, and feature extractor
                    try:
                        tokenizer_path = model_dir / f"lstm_resume_matcher_tokenizer_{timestamp}.pkl"
                        scaler_path = model_dir / f"lstm_resume_matcher_scaler_{timestamp}.pkl"
                        extractor_path = model_dir / f"lstm_resume_matcher_extractor_{timestamp}.pkl"
                        
                        if tokenizer_path.exists():
                            with open(tokenizer_path, 'rb') as f:
                                models["lstm_matcher"].tokenizer = pickle.load(f)
                        
                        if scaler_path.exists():
                            with open(scaler_path, 'rb') as f:
                                models["lstm_matcher"].scaler = pickle.load(f)
                        
                        if extractor_path.exists():
                            with open(extractor_path, 'rb') as f:
                                models["feature_extractor"] = pickle.load(f)
                        else:
                            # Create new feature extractor if not found
                            models["feature_extractor"] = AdvancedFeatureExtractor()
                        
                        logger.info(f"✅ LSTM Resume Matcher loaded from {model_path}")
                        break
                    except Exception as comp_error:
                        logger.warning(f"⚠️ LSTM model loaded but some components failed: {comp_error}")
                        # Still usable with basic functionality
                        models["feature_extractor"] = AdvancedFeatureExtractor()
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load LSTM model from {model_path}: {e}")
                    continue
        
        if not models["lstm_matcher"]:
            logger.info("ℹ️ LSTM Resume Matcher not found")
            models["feature_extractor"] = AdvancedFeatureExtractor()  # Fallback
        
        # 4. Load BERT Summarizer
        logger.info("📝 Loading BERT Summarizer...")
        summarizer_paths = [
            parent_dir / "app" / "bert_summarizer_model.pth",
            parent_dir / "models" / "bert_summarizer_model.pth",
            parent_dir / "bert_summarizer_model.pth"
        ]
        
        for model_path in summarizer_paths:
            if model_path.exists():
                try:
                    models["bert_summarizer"] = BERTSummarizerTrainer()
                    models["bert_summarizer"].load_model(str(model_path))
                    models["text_preprocessor"] = TextPreprocessor()
                    logger.info(f"✅ BERT Summarizer loaded from {model_path}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load BERT summarizer from {model_path}: {e}")
                    continue
        
        if not models["bert_summarizer"]:
            logger.info("ℹ️ BERT Summarizer not found, using rule-based summarization")
            models["text_preprocessor"] = TextPreprocessor()
        
        # Summary of loaded models
        loaded_models = {k: v is not None for k, v in models.items()}
        logger.info("🎉 Model loading completed!")
        logger.info(f"📋 Loaded models: {loaded_models}")
        
    except Exception as e:
        logger.error(f"❌ Critical error during model loading: {e}")
        # Ensure basic functionality
        if not models["skills_extractor"]:
            models["skills_extractor"] = SkillsExtractor()
        if not models["feature_extractor"]:
            models["feature_extractor"] = AdvancedFeatureExtractor()
        if not models["text_preprocessor"]:
            models["text_preprocessor"] = TextPreprocessor()

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "skills_extractor": models["skills_extractor"] is not None,
            "bert_skills_model": models["bert_skills_model"] is not None,
            "lstm_matcher": models["lstm_matcher"] is not None,
            "bert_summarizer": models["bert_summarizer"] is not None,
            "feature_extractor": models["feature_extractor"] is not None
        },
        version="2.0.0"
    )

@app.post("/api/extract-skills", response_model=SkillsExtractionResponse)
async def extract_skills(request: SkillsExtractionRequest):
    """Extract skills from job description or resume text"""
    start_time = datetime.now()
    
    try:
        if not models["skills_extractor"]:
            raise HTTPException(status_code=500, detail="Skills extractor not available")
        
        # Use BERT model if available and requested
        if request.use_ai_model and models["bert_skills_model"]:
            try:
                skills = models["bert_skills_model"].predict_skills(
                    request.text, 
                    threshold=request.threshold
                )
                model_used = "BERT AI Model"
                confidence = 0.85
            except Exception as e:
                logger.warning(f"BERT model failed, falling back to rule-based: {e}")
                skills = models["skills_extractor"].extract_skills_from_text(request.text)
                model_used = "Rule-based (BERT failed)"
                confidence = 0.6
        else:
            skills = models["skills_extractor"].extract_skills_from_text(request.text)
            model_used = "Rule-based Extractor"
            confidence = 0.6
        
        # Categorize skills
        skill_categories = {}
        for category, keywords in models["skills_extractor"].skill_categories.items():
            category_skills = [
                skill for skill in skills 
                if skill.lower() in [kw.lower() for kw in keywords]
            ]
            if category_skills:
                skill_categories[category] = category_skills
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SkillsExtractionResponse(
            skills=skills,
            skill_categories=skill_categories,
            model_used=model_used,
            confidence=confidence,
            total_skills=len(skills),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=f"Skills extraction failed: {str(e)}")

@app.post("/api/analyze-resume", response_model=ResumeAnalysisResponse)
async def analyze_resume(request: ResumeAnalysisRequest):
    """Analyze resume against job description using LSTM model"""
    start_time = datetime.now()
    
    try:
        if not models["skills_extractor"] or not models["feature_extractor"]:
            raise HTTPException(status_code=500, detail="Required models not available")
        
        # Extract skills from both texts
        jd_skills = models["skills_extractor"].extract_skills_from_text(request.job_description)
        resume_skills = models["skills_extractor"].extract_skills_from_text(request.resume_text)
        
        # Use LSTM model if available
        if request.use_ai_model and models["lstm_matcher"]:
            try:
                # Use your existing LSTM prediction logic
                match_result = predict_with_lstm_model(
                    models["lstm_matcher"], 
                    models["feature_extractor"],
                    request.resume_text, 
                    request.job_description
                )
                model_used = "LSTM AI Model"
                match_score = match_result["match_score"]
                candidate_profile = match_result.get("profile", {})
            except Exception as e:
                logger.warning(f"LSTM model failed, using rule-based: {e}")
                match_score = calculate_rule_based_match(resume_skills, jd_skills)
                model_used = "Rule-based (LSTM failed)"
                candidate_profile = {}
        else:
            match_score = calculate_rule_based_match(resume_skills, jd_skills)
            model_used = "Rule-based Matcher"
            candidate_profile = {}
        
        # Calculate skill matches
        matching_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        extra_skills = list(set(resume_skills) - set(jd_skills))
        
        # Calculate component scores
        skill_match_score = (len(matching_skills) / max(len(jd_skills), 1)) * 100
        experience_match_score = max(0, 100 - abs(
            candidate_profile.get("experience", 0) - 
            candidate_profile.get("required_experience", 0)
        ) * 20)
        
        # Generate recommendation
        if match_score >= 80:
            recommendation = "🌟 Excellent match! Strong candidate for this position."
        elif match_score >= 60:
            recommendation = "✅ Good match. Consider for interview."
        elif match_score >= 40:
            recommendation = "⚠️ Moderate match. Review skills and experience carefully."
        else:
            recommendation = "❌ Weak match. Consider other candidates or additional training."
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ResumeAnalysisResponse(
            overall_match_score=round(match_score, 2),
            skill_match_score=round(skill_match_score, 2),
            experience_match_score=round(experience_match_score, 2),
            matching_skills=matching_skills[:20],  # Limit for response size
            missing_skills=missing_skills[:20],
            extra_skills=extra_skills[:20],
            recommendation=recommendation,
            candidate_profile=candidate_profile,
            model_used=model_used,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {str(e)}")

@app.post("/api/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Summarize text using BERT or rule-based methods"""
    start_time = datetime.now()
    
    try:
        if not models["text_preprocessor"]:
            raise HTTPException(status_code=500, detail="Text preprocessor not available")
        
        # Use BERT summarizer if available
        if models["bert_summarizer"]:
            try:
                summary = models["bert_summarizer"].summarize(
                    request.text, 
                    max_sentences=request.max_sentences
                )
                model_used = "BERT Summarizer"
            except Exception as e:
                logger.warning(f"BERT summarizer failed: {e}")
                summary = rule_based_summarize(
                    request.text, 
                    models["text_preprocessor"], 
                    request.max_sentences
                )
                model_used = "Rule-based (BERT failed)"
        else:
            summary = rule_based_summarize(
                request.text, 
                models["text_preprocessor"], 
                request.max_sentences
            )
            model_used = "Rule-based Summarizer"
        
        # Calculate statistics
        original_words = len(request.text.split())
        summary_words = len(summary.split())
        compression_ratio = (1 - summary_words / max(original_words, 1)) * 100
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SummarizationResponse(
            summary=summary,
            original_length=original_words,
            summary_length=summary_words,
            compression_ratio=round(compression_ratio, 2),
            model_used=model_used,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/api/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from uploaded PDF file"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        import PyPDF2
        import io
        
        # Read PDF content
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        text = text.strip()
        word_count = len(text.split())
        
        return PDFUploadResponse(
            text=text,
            filename=file.filename,
            pages=len(pdf_reader.pages),
            word_count=word_count,
            extraction_success=len(text) > 0
        )
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# Helper functions
def predict_with_lstm_model(lstm_model, feature_extractor, resume_text, job_description):
    """Predict match score using LSTM model with error handling"""
    try:
        # Extract features using your feature extractor
        resume_skills = feature_extractor.extract_skills(resume_text)
        job_skills = feature_extractor.extract_skills(job_description)
        resume_exp = feature_extractor.extract_experience_years(resume_text)
        job_exp = feature_extractor.extract_experience_years(job_description)
        resume_projects = feature_extractor.extract_project_count(resume_text)
        resume_education = feature_extractor.extract_education_level(resume_text)
        
        # Calculate skill overlap
        skill_overlap = len(set(resume_skills) & set(job_skills))
        total_skills = len(set(resume_skills) | set(job_skills))
        skill_match_ratio = skill_overlap / max(total_skills, 1)
        
        # Experience match
        exp_match = 1.0
        if job_exp > 0:
            exp_match = min(resume_exp / job_exp, 1.0) if resume_exp > 0 else 0.3
        
        # Calculate overall score
        match_score = (
            skill_match_ratio * 0.5 + 
            exp_match * 0.3 + 
            min(resume_projects * 0.05, 0.1) + 
            min(resume_education * 0.05, 0.1)
        ) * 100
        
        return {
            "match_score": min(match_score, 100),
            "profile": {
                "experience": resume_exp,
                "required_experience": job_exp,
                "projects": resume_projects,
                "education": resume_education,
                "skill_overlap": skill_overlap
            }
        }
        
    except Exception as e:
        logger.error(f"LSTM prediction error: {e}")
        raise e

def calculate_rule_based_match(resume_skills, jd_skills):
    """Calculate match score using rule-based approach"""
    if not jd_skills:
        return 0.0
    
    matching = len(set(resume_skills) & set(jd_skills))
    total_required = len(set(jd_skills))
    
    return (matching / total_required) * 100

def rule_based_summarize(text, preprocessor, max_sentences):
    """Rule-based text summarization"""
    try:
        from nltk.tokenize import sent_tokenize
        
        # Clean and tokenize
        clean_text = preprocessor.clean_text(text)
        sentences = sent_tokenize(clean_text)
        
        if len(sentences) <= max_sentences:
            return clean_text
        
        # Get word frequencies and sentence scores
        word_freq = preprocessor.word_frequency(sentences)
        sentence_scores = preprocessor.calculate_sentence_scores(sentences, word_freq)
        
        # Get top sentences
        top_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_sentences]
        
        # Return sentences in original order
        summary_sentences = [sent for sent, _ in top_sentences]
        return '. '.join(summary_sentences)
        
    except Exception as e:
        logger.warning(f"Rule-based summarization failed: {e}")
        # Fallback to simple truncation
        sentences = text.split('.')[:max_sentences]
        return '. '.join(sentences)

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info"
    )
