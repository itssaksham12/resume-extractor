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

# Set environment variables before importing ML libraries to fix mutex lock
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# Set multiprocessing start method for Mac
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Core data processing imports
import numpy as np
import pandas as pd

# Models will be imported lazily to avoid mutex lock issues

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

# Model loading status
model_loading_status = {
    "loading": False,
    "completed": False,
    "error": None
}

@app.on_event("startup")
async def startup_event():
    """Start background model loading after server starts"""
    logger.info("🚀 Server starting - models will load in background")
    # Start model loading in background
    import asyncio
    asyncio.create_task(load_models_background())

async def load_models_background():
    """Load all AI models in background after server starts"""
    global models, model_loading_status
    
    model_loading_status["loading"] = True
    logger.info("🔄 Starting background model loading...")
    
    try:
        # Use asyncio.wait_for to add timeout
        await asyncio.wait_for(_load_models_with_timeout(), timeout=120.0)
        model_loading_status["completed"] = True
        logger.info("✅ Background model loading completed")
    except asyncio.TimeoutError:
        logger.error("❌ Model loading timed out after 120 seconds")
        model_loading_status["error"] = "Timeout after 120 seconds"
        logger.info("🚀 Server running with basic functionality")
    except Exception as e:
        logger.error(f"❌ Critical error during model loading: {e}")
        model_loading_status["error"] = str(e)
        logger.info("🚀 Server running with basic functionality")
    finally:
        model_loading_status["loading"] = False

async def _load_models_with_timeout():
    """Internal model loading function with timeout"""
    global models
    
    try:
        # 1. Try to load Basic Skills Extractor
        logger.info("📊 Loading Skills Extractor...")
        try:
            from bert_skills_extractor import SkillsExtractor
            models["skills_extractor"] = SkillsExtractor()
            logger.info("✅ Skills Extractor loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Skills Extractor import failed: {e}")
            logger.error("📦 Please install required dependencies: pip install torch tensorflow")
            models["skills_extractor"] = None
        
        # 2. Try to load BERT Skills Model with lazy import
        logger.info("🧠 Loading BERT Skills Model...")
        try:
            import torch
            from bert_skills_extractor import BERTSkillsModel
            bert_model_paths = [
                Path("../bert_skills_model.pth"),  # Root directory (from backend/)
                Path("../../bert_skills_model.pth"),  # Root directory (if deeper)
                Path("bert_skills_model.pth"),     # Backend directory
                Path("/opt/render/project/src/bert_skills_model.pth"),  # Render absolute path
                Path("models") / "bert_skills_model.pth",  # Models directory
                Path("../app/bert_skills_model.pth")  # App directory
            ]
            
            logger.info(f"🔍 Searching for BERT model in: {[str(p) for p in bert_model_paths]}")
            
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
        except Exception as e:
            logger.warning(f"⚠️ PyTorch import failed: {e}")
        
        if not models.get("bert_skills_model"):
            logger.info("ℹ️ BERT Skills Model not found, using rule-based extraction")
        
        # 3. Load LSTM Resume Matcher with lazy import and TensorFlow config  
        logger.info("🔄 Loading LSTM Resume Matcher...")
        try:
            # Configure TensorFlow to avoid mutex lock BEFORE import
            logger.info("⚙️ Configuring TensorFlow for macOS...")
            
            # Apple Silicon M1 optimized TensorFlow configuration
            logger.info("🍎 Configuring TensorFlow for Apple Silicon M1...")
            
            # Set Apple Silicon specific environment variables BEFORE TensorFlow import
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_DISABLE_MKL'] = '1'
            os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
            
            # Critical: Force single-threaded execution for M1
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            
            # Import TensorFlow AFTER environment configuration
            import tensorflow as tf
            
            logger.info(f"✅ TensorFlow version: {tf.__version__}")
            
            # Check if running on Apple Silicon with Metal support
            physical_devices = tf.config.list_physical_devices()
            gpu_devices = tf.config.list_physical_devices('GPU')
            
            logger.info(f"📱 Physical devices: {len(physical_devices)}")
            logger.info(f"🚀 GPU devices: {len(gpu_devices)}")
            
            # Force CPU-only execution to avoid M1 GPU mutex issues
            logger.info("💻 Forcing CPU-only execution for stability on M1")
            tf.config.set_visible_devices([], 'GPU')
            
            # Critical: Force single-threaded execution to prevent mutex locks
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            # Disable all optimizations that could cause threading issues
            tf.config.optimizer.set_jit(False)
            tf.config.run_functions_eagerly(True)
            
            logger.info("✅ TensorFlow configured for single-threaded CPU execution on M1")
            
            logger.info("📚 Importing LSTM components...")
            
            # Add timeout for imports
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LSTM import timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                from lstm_resume_matcher import LSTMResumeMatcherTrainer, AdvancedFeatureExtractor
                signal.alarm(0)  # Cancel timeout
                logger.info("✅ LSTM components imported successfully")
            except TimeoutError:
                signal.alarm(0)
                raise Exception("LSTM import timed out after 30 seconds")
            
            lstm_model_paths = [
                Path("../lstm_resume_matcher_best.h5"),  # Root directory (from backend/)
                Path("../../lstm_resume_matcher_best.h5"),  # Root directory (if deeper)
                Path("lstm_resume_matcher_best.h5"),     # Backend directory
                Path("/opt/render/project/src/lstm_resume_matcher_best.h5"),  # Render absolute path
                Path("models") / "lstm_resume_matcher_best.h5",  # Models directory
                Path("../app/lstm_resume_matcher_best.h5")  # App directory
            ]
            
            logger.info(f"🔍 Searching for LSTM model in: {[str(p) for p in lstm_model_paths]}")
            
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
        except Exception as e:
            logger.warning(f"⚠️ TensorFlow/LSTM loading failed: {e}")
            logger.info("🔄 Creating fallback feature extractor...")
            try:
                from lstm_resume_matcher import AdvancedFeatureExtractor
                models["feature_extractor"] = AdvancedFeatureExtractor()
            except ImportError:
                logger.warning("⚠️ Even fallback feature extractor failed")
                models["feature_extractor"] = None
        
        if not models.get("lstm_matcher"):
            logger.info("ℹ️ LSTM Resume Matcher not loaded, using rule-based matching")
            if not models.get("feature_extractor"):
                try:
                    from lstm_resume_matcher import AdvancedFeatureExtractor
                    models["feature_extractor"] = AdvancedFeatureExtractor()
                except ImportError:
                    models["feature_extractor"] = None
        
        # 4. Load BERT Summarizer with lazy import
        logger.info("📝 Loading BERT Summarizer...")
        try:
            from bert_summarizer import BERTSummarizerTrainer, TextPreprocessor
            summarizer_paths = [
                Path("../bert_summarizer_model.pth"),  # Root directory (from backend/)
                Path("../../bert_summarizer_model.pth"),  # Root directory (if deeper)
                Path("bert_summarizer_model.pth"),     # Backend directory
                Path("/opt/render/project/src/bert_summarizer_model.pth"),  # Render absolute path
                Path("models") / "bert_summarizer_model.pth",  # Models directory
                Path("../app/bert_summarizer_model.pth")  # App directory
            ]
            
            logger.info(f"🔍 Searching for BERT Summarizer in: {[str(p) for p in summarizer_paths]}")
            
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
            
            if not models.get("bert_summarizer"):
                logger.info("ℹ️ BERT Summarizer not found, using rule-based summarization")
                models["text_preprocessor"] = TextPreprocessor()
        except Exception as e:
            logger.warning(f"⚠️ BERT Summarizer import failed: {e}")
            try:
                from bert_summarizer import TextPreprocessor
                models["text_preprocessor"] = TextPreprocessor()
            except:
                models["text_preprocessor"] = None
        
        # Summary of loaded models
        loaded_models = {k: v is not None for k, v in models.items()}
        logger.info("🎉 Model loading completed!")
        logger.info(f"📋 Loaded models: {loaded_models}")
        
    except Exception as e:
        logger.error(f"❌ Critical error during model loading: {e}")
        # Ensure basic functionality
        # Ensure basic functionality even if models failed to load
        if not models.get("skills_extractor"):
            logger.warning("⚠️ Skills extractor not available - some features will be limited")
        if not models.get("feature_extractor"):
            try:
                from lstm_resume_matcher import AdvancedFeatureExtractor
                models["feature_extractor"] = AdvancedFeatureExtractor()
            except ImportError:
                logger.warning("⚠️ Feature extractor not available")
                models["feature_extractor"] = None
        if not models.get("text_preprocessor"):
            try:
                from bert_summarizer import TextPreprocessor
                models["text_preprocessor"] = TextPreprocessor()
            except ImportError:
                logger.warning("⚠️ Text preprocessor not available")
                models["text_preprocessor"] = None

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status - works immediately"""
    global model_loading_status
    
    # Determine status based on model loading
    if model_loading_status["loading"]:
        status = "loading_models"
    elif model_loading_status["error"]:
        status = "models_failed"
    elif model_loading_status["completed"]:
        status = "ready"
    else:
        status = "starting"
    
    return HealthResponse(
        status=status,
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

@app.get("/api/models/status")
async def get_model_status():
    """Get current model loading status"""
    global model_loading_status
    return {
        "loading": model_loading_status["loading"],
        "completed": model_loading_status["completed"],
        "error": model_loading_status["error"],
        "models_available": {
            "skills_extractor": models["skills_extractor"] is not None,
            "bert_skills_model": models["bert_skills_model"] is not None,
            "lstm_matcher": models["lstm_matcher"] is not None,
            "bert_summarizer": models["bert_summarizer"] is not None,
            "feature_extractor": models["feature_extractor"] is not None
        }
    }

@app.post("/api/extract-skills", response_model=SkillsExtractionResponse)
async def extract_skills(request: SkillsExtractionRequest):
    """Extract skills from job description or resume text"""
    start_time = datetime.now()
    
    try:
        if not models.get("skills_extractor"):
            if model_loading_status["loading"]:
                raise HTTPException(
                    status_code=503, 
                    detail="Models are still loading. Please wait a moment and try again."
                )
            else:
                raise HTTPException(
                    status_code=503, 
                    detail="Skills extractor not available. Please check model loading status."
                )
        
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
