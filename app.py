#!/usr/bin/env python3
"""
Simplified FastAPI app for Render deployment
Combines all functionality in a single file for easier deployment
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Core ML imports
import torch
import tensorflow as tf
import numpy as np
import pandas as pd

# Import your existing models
from bert_skills_extractor import SkillsExtractor, BERTSkillsModel
from lstm_resume_matcher import LSTMResumeMatcherTrainer, AdvancedFeatureExtractor

# Try to import BERT summarizer (fallback if not available)
try:
    from bert_summarizer import BERTSummarizerTrainer, TextPreprocessor
except ImportError:
    BERTSummarizerTrainer = None
    TextPreprocessor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API validation
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

class SkillsExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10)
    use_ai_model: bool = Field(True)
    threshold: float = Field(0.3, ge=0.0, le=1.0)

class SkillsExtractionResponse(BaseModel):
    skills: List[str]
    skill_categories: Dict[str, List[str]]
    model_used: str
    confidence: float
    total_skills: int
    processing_time: float

class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=50)
    job_description: str = Field(..., min_length=10)
    use_ai_model: bool = Field(True)

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

class PDFUploadResponse(BaseModel):
    text: str
    filename: str
    pages: int
    word_count: int
    extraction_success: bool

# FastAPI app initialization
app = FastAPI(
    title="Resume Extractor API",
    description="AI-powered resume and job description analysis",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
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
        
        # 2. Try to load BERT Skills Model (optional)
        logger.info("🧠 Attempting to load BERT Skills Model...")
        try:
            bert_model_paths = ["bert_skills_model.pth", "models/bert_skills_model.pth"]
            for model_path in bert_model_paths:
                if os.path.exists(model_path):
                    models["bert_skills_model"] = BERTSkillsModel(models["skills_extractor"])
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    n_classes = len(checkpoint["mlb"].classes_)
                    models["bert_skills_model"].load_model(model_path, n_classes=n_classes)
                    logger.info(f"✅ BERT Skills Model loaded from {model_path}")
                    break
        except Exception as e:
            logger.warning(f"⚠️ BERT Skills Model not available: {e}")
        
        # 3. Try to load LSTM Resume Matcher (optional)
        logger.info("🔄 Attempting to load LSTM Resume Matcher...")
        try:
            lstm_paths = ["lstm_resume_matcher_best.h5", "models/lstm_resume_matcher_best.h5"]
            for model_path in lstm_paths:
                if os.path.exists(model_path):
                    models["lstm_matcher"] = LSTMResumeMatcherTrainer()
                    models["lstm_matcher"].model = tf.keras.models.load_model(model_path)
                    
                    # Try to load support files
                    timestamp = "20250828_113623"
                    try:
                        with open(f"lstm_resume_matcher_tokenizer_{timestamp}.pkl", 'rb') as f:
                            models["lstm_matcher"].tokenizer = pickle.load(f)
                        with open(f"lstm_resume_matcher_scaler_{timestamp}.pkl", 'rb') as f:
                            models["lstm_matcher"].scaler = pickle.load(f)
                        with open(f"lstm_resume_matcher_extractor_{timestamp}.pkl", 'rb') as f:
                            models["feature_extractor"] = pickle.load(f)
                    except Exception:
                        logger.warning("⚠️ Some LSTM support files not found, using fallback")
                        models["feature_extractor"] = AdvancedFeatureExtractor()
                    
                    logger.info(f"✅ LSTM Resume Matcher loaded from {model_path}")
                    break
        except Exception as e:
            logger.warning(f"⚠️ LSTM Resume Matcher not available: {e}")
        
        # 4. Always ensure we have a feature extractor
        if not models["feature_extractor"]:
            models["feature_extractor"] = AdvancedFeatureExtractor()
        
        # 5. Try to load BERT Summarizer (optional)
        if BERTSummarizerTrainer and TextPreprocessor:
            logger.info("📝 Attempting to load BERT Summarizer...")
            try:
                summarizer_paths = ["bert_summarizer_model.pth", "models/bert_summarizer_model.pth"]
                for model_path in summarizer_paths:
                    if os.path.exists(model_path):
                        models["bert_summarizer"] = BERTSummarizerTrainer()
                        models["bert_summarizer"].load_model(model_path)
                        models["text_preprocessor"] = TextPreprocessor()
                        logger.info(f"✅ BERT Summarizer loaded from {model_path}")
                        break
            except Exception as e:
                logger.warning(f"⚠️ BERT Summarizer not available: {e}")
        
        # Always ensure we have a text preprocessor
        if not models["text_preprocessor"] and TextPreprocessor:
            models["text_preprocessor"] = TextPreprocessor()
        
        # Summary
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
                logger.warning(f"BERT model failed: {e}")
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
    """Analyze resume against job description"""
    start_time = datetime.now()
    
    try:
        if not models["skills_extractor"] or not models["feature_extractor"]:
            raise HTTPException(status_code=500, detail="Required models not available")
        
        # Extract skills from both texts
        jd_skills = models["skills_extractor"].extract_skills_from_text(request.job_description)
        resume_skills = models["skills_extractor"].extract_skills_from_text(request.resume_text)
        
        # Use rule-based matching (LSTM model complex to implement quickly)
        match_score = calculate_rule_based_match(resume_skills, jd_skills)
        model_used = "Rule-based Matcher"
        
        # Calculate skill matches
        matching_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        extra_skills = list(set(resume_skills) - set(jd_skills))
        
        # Calculate component scores
        skill_match_score = (len(matching_skills) / max(len(jd_skills), 1)) * 100
        experience_match_score = 75.0  # Default for rule-based
        
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
            matching_skills=matching_skills[:20],
            missing_skills=missing_skills[:20],
            extra_skills=extra_skills[:20],
            recommendation=recommendation,
            candidate_profile={"skills_found": len(resume_skills)},
            model_used=model_used,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {str(e)}")

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

def calculate_rule_based_match(resume_skills, jd_skills):
    """Calculate match score using rule-based approach"""
    if not jd_skills:
        return 0.0
    
    matching = len(set(resume_skills) & set(jd_skills))
    total_required = len(set(jd_skills))
    
    return (matching / total_required) * 100

if __name__ == "__main__":
    # For local development and Render deployment
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
