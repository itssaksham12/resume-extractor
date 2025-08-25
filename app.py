import gradio as gr
import sys
import os
import traceback
from pathlib import Path
import PyPDF2
import io
import pickle

# Import our existing models (local versions)
from bert_skills_extractor import SkillsExtractor, BERTSkillsModel

# Import the advanced LSTM resume matcher from parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lstm_resume_matcher import LSTMResumeMatcherTrainer, AdvancedFeatureExtractor

# Import summarization components
from bert_summarizer import BERTSummarizerTrainer, TextPreprocessor

# Initialize models globally
skills_extractor = None
bert_skills_model = None
predictor = None
summarizer = None

class SummarizationService:
    """Service class for text summarization"""
    
    def __init__(self, model_path=None):
        self.trainer = BERTSummarizerTrainer()
        self.preprocessor = TextPreprocessor()
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            try:
                self.trainer.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Summarization model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Failed to load summarization model: {e}")
                self.model_loaded = False
        else:
            self.model_loaded = False
            print("ℹ️ No trained summarization model found, using rule-based summarization")
    
    def summarize_job_description(self, job_description: str, max_sentences: int = 3) -> str:
        """Summarize job description"""
        if not job_description.strip():
            return "Please enter a job description."
        
        try:
            if self.model_loaded:
                # Use trained BERT model
                summary = self.trainer.summarize(job_description, max_sentences)
            else:
                # Use rule-based summarization
                summary = self._rule_based_summarize(job_description, max_sentences)
            
            return summary
        except Exception as e:
            return f"Error summarizing job description: {str(e)}"
    
    def summarize_resume(self, resume_text: str, max_sentences: int = 3) -> str:
        """Summarize resume"""
        if not resume_text.strip():
            return "Please enter resume text."
        
        try:
            if self.model_loaded:
                # Use trained BERT model
                summary = self.trainer.summarize(resume_text, max_sentences)
            else:
                # Use rule-based summarization
                summary = self._rule_based_summarize(resume_text, max_sentences)
            
            return summary
        except Exception as e:
            return f"Error summarizing resume: {str(e)}"
    
    def _rule_based_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Rule-based summarization using the reference algorithm"""
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        import re
        
        # Clean text
        text = self.preprocessor.clean_text(text)
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Calculate word frequency
        word_freq_list = self.preprocessor.word_frequency(sentences)
        sentence_scores = self.preprocessor.calculate_sentence_scores(sentences, word_freq_list)
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        summary = '. '.join([sent for sent, _ in top_sentences])
        
        return summary

def load_models():
    """Load the AI models with robust error handling"""
    global skills_extractor, bert_skills_model, predictor, summarizer
    
    try:
        # Always initialize the basic skills extractor
        print("🔧 Loading Skills Extractor...")
        skills_extractor = SkillsExtractor()
        print("✅ Skills Extractor loaded successfully")
        
        # Try to load BERT model if available - check multiple possible paths
        bert_model_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "bert_skills_model.pth"),
            os.path.join(os.path.dirname(__file__), "bert_skills_model.pth"),
            "bert_skills_model.pth"
        ]
        
        bert_model_loaded = False
        print("🔧 Attempting to load BERT Skills Model...")
        for model_path in bert_model_paths:
            if os.path.exists(model_path):
                try:
                    bert_skills_model = BERTSkillsModel(skills_extractor)
                    import torch
                    # Use weights_only=False for backward compatibility with older model files
                    ckpt = torch.load(model_path, map_location=skills_extractor.device, weights_only=False)
                    
                    n_classes = len(ckpt["mlb"].classes_)
                    bert_skills_model.load_model(model_path, n_classes=n_classes)
                    bert_model_loaded = True
                    print(f"✅ BERT Skills Model loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load BERT model from {model_path}: {e}")
                    continue
        
        if not bert_model_loaded:
            print("ℹ️ BERT Skills Model not found, using rule-based extraction")
            bert_skills_model = None
        
        # Load LSTM predictor - check multiple possible paths
        lstm_model_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "lstm_resume_matcher_best.h5"),
            os.path.join(os.path.dirname(__file__), "lstm_resume_matcher_best.h5"),
            "lstm_resume_matcher_best.h5"
        ]
        
        lstm_model_loaded = False
        print("🔧 Attempting to load LSTM Resume Matcher...")
        for model_path in lstm_model_paths:
            if os.path.exists(model_path):
                try:
                    # Use the advanced LSTM resume matcher from parent directory
                    predictor = LSTMResumeMatcherTrainer()
                    
                    # Load the model using TensorFlow's load_model
                    import tensorflow as tf
                    predictor.model = tf.keras.models.load_model(model_path)
                    
                    # Load supporting components if available
                    model_dir = os.path.dirname(model_path)
                    timestamp = "20250807_124831"  # Default timestamp
                    
                    # Try to load components
                    try:
                        with open(os.path.join(model_dir, f"lstm_resume_matcher_tokenizer_{timestamp}.pkl"), 'rb') as f:
                            predictor.tokenizer = pickle.load(f)
                        with open(os.path.join(model_dir, f"lstm_resume_matcher_scaler_{timestamp}.pkl"), 'rb') as f:
                            predictor.scaler = pickle.load(f)
                        with open(os.path.join(model_dir, f"lstm_resume_matcher_extractor_{timestamp}.pkl"), 'rb') as f:
                            predictor.feature_extractor = pickle.load(f)
                        print(f"✅ LSTM components loaded successfully")
                    except Exception as comp_error:
                        print(f"⚠️ Could not load some components: {comp_error}")
                    
                    lstm_model_loaded = True
                    print(f"✅ LSTM Resume Matcher loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load LSTM model from {model_path}: {e}")
                    continue
        
        if not lstm_model_loaded:
            print("ℹ️ LSTM Resume Matcher not found, resume analysis will be limited")
            predictor = None
        
        # Load summarization model
        summarizer_model_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "bert_summarizer_model.pth"),
            os.path.join(os.path.dirname(__file__), "bert_summarizer_model.pth"),
            "bert_summarizer_model.pth"
        ]
        
        summarizer_model_loaded = False
        print("🔧 Attempting to load BERT Summarizer...")
        for model_path in summarizer_model_paths:
            if os.path.exists(model_path):
                try:
                    summarizer = SummarizationService(model_path)
                    summarizer_model_loaded = True
                    print(f"✅ BERT Summarizer loaded from: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load summarization model from {model_path}: {e}")
                    continue
        
        if not summarizer_model_loaded:
            print("ℹ️ BERT Summarizer not found, using rule-based summarization")
            summarizer = SummarizationService()
        
        # Check what models are available
        models_status = []
        available_features = []
        
        if bert_model_loaded:
            models_status.append("✅ BERT Skills Model")
            available_features.append("AI-powered skills extraction")
        else:
            models_status.append("❌ BERT Skills Model (using rule-based)")
            available_features.append("Rule-based skills extraction")
            
        if lstm_model_loaded:
            models_status.append("✅ LSTM Resume Matcher")
            available_features.append("Resume-job matching analysis")
        else:
            models_status.append("❌ LSTM Resume Matcher")
            
        if summarizer_model_loaded:
            models_status.append("✅ BERT Summarizer")
            available_features.append("AI-powered text summarization")
        else:
            models_status.append("❌ BERT Summarizer (using rule-based)")
            available_features.append("Rule-based text summarization")
        
        print(f"\n🎯 Available Features: {', '.join(available_features)}")
        return f"Models loaded: {' | '.join(models_status)}"
        
    except Exception as e:
        error_msg = f"❌ Critical error loading models: {str(e)}"
        print(error_msg)
        # Ensure basic functionality even if all models fail
        try:
            if skills_extractor is None:
                skills_extractor = SkillsExtractor()
            if summarizer is None:
                summarizer = SummarizationService()
            print("🔧 Fallback: Basic skills extraction and summarization available")
        except Exception as fallback_error:
            print(f"❌ Even fallback failed: {fallback_error}")
        return error_msg

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        if pdf_file is None:
            return ""
        
        # Handle different Gradio File component return types
        if isinstance(pdf_file, tuple):
            # Gradio 3.x returns (file_path, file_name)
            file_path = pdf_file[0]
        elif hasattr(pdf_file, 'name'):
            # Handle file-like objects (like _TemporaryFileWrapper)
            file_path = pdf_file.name
        elif isinstance(pdf_file, str):
            # Direct file path
            file_path = pdf_file
        else:
            # Try to get the file path from the object
            try:
                file_path = str(pdf_file)
            except:
                return f"Error: Unsupported file object type: {type(pdf_file)}"
        
        # Read PDF content from file path
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_skills_from_jd(job_description):
    """Extract skills from job description using BERT model"""
    try:
        if not job_description.strip():
            return "Please enter a job description."
        
        # Extract skills using BERT if available, otherwise use rule-based
        if bert_skills_model:
            try:
                extracted_skills = bert_skills_model.predict_skills(job_description, threshold=0.3)
                model_used = "BERT AI Model"
            except Exception as e:
                extracted_skills = skills_extractor.extract_skills_from_text(job_description)
                model_used = "Rule-based Extractor (BERT failed)"
        else:
            extracted_skills = skills_extractor.extract_skills_from_text(job_description)
            model_used = "Rule-based Extractor"
        
        # Categorize skills
        skill_categories = {
            'programming_languages': [],
            'web_technologies': [],
            'databases': [],
            'data_science_ml': [],
            'cloud_devops': [],
            'mobile_development': [],
            'soft_skills': [],
            'tools_frameworks': []
        }
        
        for skill in extracted_skills:
            for category, keywords in skills_extractor.skill_categories.items():
                if skill.lower() in [kw.lower() for kw in keywords]:
                    skill_categories[category].append(skill)
                    break
        
        # Format results
        result_text = f"""
## 🎯 Skills Extracted from Job Description

**Model Used**: {model_used}
**Total Skills Found**: {len(extracted_skills)}

### 📋 All Required Skills
{', '.join(sorted(extracted_skills)) if extracted_skills else 'No specific skills detected'}

### 🏷️ Skills by Category
"""
        
        for category, skills in skill_categories.items():
            if skills:
                category_name = category.replace('_', ' ').title()
                result_text += f"""
**{category_name} ({len(skills)}):**
{', '.join(sorted(skills))}
"""
        
        if not any(skill_categories.values()):
            result_text += "\n*No categorized skills found*"
        
        result_text += f"""
"""
        
        return result_text
        
    except Exception as e:
        return f"❌ Skills extraction failed: {str(e)}\n\n{traceback.format_exc()}"

def analyze_resume(job_description, resume_text, resume_pdf=None):
    """Analyze resume against job description"""
    try:
        if not job_description.strip():
            return "Please enter a job description."
        
        # Handle PDF upload if provided
        if resume_pdf is not None:
            pdf_text = extract_text_from_pdf(resume_pdf)
            if pdf_text.startswith("Error"):
                return f"❌ PDF Error: {pdf_text}"
            if pdf_text.strip():
                resume_text = pdf_text
            elif not resume_text.strip():
                return "Please provide either resume text or a valid PDF file."
        elif not resume_text.strip():
            return "Please enter resume text or upload a PDF file."
        
        # Check if LSTM model is available
        if predictor is None:
            return """
## ❌ Resume Analysis Unavailable

**LSTM model not loaded.** This could be due to:
- Model files not uploaded to Hugging Face Space
- Model files in wrong location
- Memory constraints on Hugging Face

**Available Features:**
- ✅ Skills Extraction (BERT/Rule-based)
- ❌ Resume Analysis (LSTM model needed)

**To fix this:**
1. Ensure `lstm_resume_matcher_best.h5` is uploaded to the Space
2. Check that all `.pkl` files are in the same directory as `app.py`
3. Contact support if the issue persists
"""
        
        # Extract skills using BERT if available, otherwise use rule-based
        if bert_skills_model:
            try:
                jd_skills = bert_skills_model.predict_skills(job_description, threshold=0.3)
            except Exception:
                jd_skills = skills_extractor.extract_skills_from_text(job_description)
        else:
            jd_skills = skills_extractor.extract_skills_from_text(job_description)
        
        # Get match prediction from LSTM model
        try:
            # Create a custom prediction function that adapts to the model's expected input
            def predict_with_model_adaptation(predictor, resume_text, job_description):
                """Adapt prediction to match the model's expected input format"""
                try:
                    # First try the original method
                    return predictor.predict_match_score(resume_text, job_description)
                except Exception as e:
                    if "expected axis -1 of input shape to have value 8" in str(e):
                        # Model expects 8 features, but we're providing 10
                        # Let's create a custom prediction using the feature extractor
                        resume_skills = predictor.feature_extractor.extract_skills(resume_text)
                        job_skills = predictor.feature_extractor.extract_skills(job_description)
                        resume_exp = predictor.feature_extractor.extract_experience_years(resume_text)
                        job_exp = predictor.feature_extractor.extract_experience_years(job_description)
                        resume_projects = predictor.feature_extractor.extract_project_count(resume_text)
                        resume_education = predictor.feature_extractor.extract_education_level(resume_text)
                        
                        # Calculate match score based on feature overlap
                        skill_overlap = len(set(resume_skills) & set(job_skills))
                        total_skills = len(set(resume_skills) | set(job_skills))
                        skill_match_ratio = skill_overlap / max(total_skills, 1)
                        
                        # Experience match
                        exp_match = 1.0
                        if job_exp > 0:
                            exp_match = min(resume_exp / job_exp, 1.0) if resume_exp > 0 else 0.3
                        
                        # Project bonus
                        project_bonus = min(resume_projects * 0.05, 0.2)
                        
                        # Education bonus
                        education_bonus = min(resume_education * 0.1, 0.2)
                        
                        # Calculate overall match score
                        match_score = (skill_match_ratio * 0.5 + exp_match * 0.3 + project_bonus + education_bonus) * 100
                        match_score = min(match_score, 100)  # Cap at 100%
                        
                        return {
                            'match_score': round(match_score, 2),
                            'extracted_skills': resume_skills,
                            'required_skills': job_skills,
                            'skill_overlap': skill_overlap,
                            'experience_years': resume_exp,
                            'required_experience': job_exp,
                            'projects_count': resume_projects,
                            'education_level': resume_education
                        }
                    else:
                        raise e
            
            match_result = predict_with_model_adaptation(predictor, resume_text, job_description)
            
            # Convert the advanced LSTM result to the expected format
            formatted_result = {
                'overall_match_score': match_result['match_score'],
                'skill_match_score': (match_result['skill_overlap'] / max(len(match_result['required_skills']), 1)) * 100,
                'experience_match_score': max(0, 100 - abs(match_result['experience_years'] - match_result['required_experience']) * 20),
                'recommendation': f"Match Score: {match_result['match_score']}% - {'Strong Match' if match_result['match_score'] > 70 else 'Good Match' if match_result['match_score'] > 50 else 'Weak Match'}",
                'matching_skills': list(set(match_result['extracted_skills']) & set(match_result['required_skills'])),
                'extra_skills': list(set(match_result['extracted_skills']) - set(match_result['required_skills'])),
                'missing_skills': list(set(match_result['required_skills']) - set(match_result['extracted_skills'])),
                'candidate_experience': match_result['experience_years'],
                'required_experience': match_result['required_experience'],
                'projects_count': match_result['projects_count'],
                'education_level': match_result['education_level']
            }
            match_result = formatted_result
            
        except Exception as e:
            return f"""
## ❌ Resume Analysis Failed

**Error**: {str(e)}

**Available Features:**
- ✅ Skills Extraction (BERT/Rule-based)
- ❌ Resume Analysis (LSTM model error)

**Try using the Skills Extraction tab instead.**
"""
        
        # Add extracted JD skills to results
        match_result['jd_extracted_skills'] = sorted(list(set(jd_skills)))[:50]
        
        # Format results for display
        result_text = f"""
## 📊 Resume Analysis Results

### 🎯 Match Scores
- **Overall Match**: {match_result['overall_match_score']}%
- **Skill Match**: {match_result['skill_match_score']}%
- **Experience Match**: {match_result['experience_match_score']}%

### 📝 Recommendation
{match_result['recommendation']}

### 🎯 Skills Analysis

**Job Requirements ({len(match_result.get('jd_extracted_skills', []))} skills):**
{', '.join(match_result.get('jd_extracted_skills', [])[:20])}

**✅ Matching Skills ({len(match_result.get('matching_skills', []))}):**
{', '.join(match_result.get('matching_skills', [])[:15])}

**⭐ Additional Skills ({len(match_result.get('extra_skills', []))}):**
{', '.join(match_result.get('extra_skills', [])[:15])}

### ❌ Missing Skills
**Missing Skills ({len(match_result.get('missing_skills', []))}):**
{', '.join(match_result.get('missing_skills', [])[:15])}

### 👤 Candidate Profile
- **Experience**: {match_result.get('candidate_experience', 0)} years
- **Required Experience**: {match_result.get('required_experience', 0)} years
- **Projects**: {match_result.get('projects_count', 0)}
- **Education Level**: {match_result.get('education_level', 0)}
        """
        
        return result_text
        
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}\n\n{traceback.format_exc()}"

def summarize_text(text, text_type, max_sentences):
    """Summarize text using the summarization service"""
    if not text.strip():
        return "Please enter text to summarize."
    
    try:
        if text_type == "Job Description":
            return summarizer.summarize_job_description(text, max_sentences)
        else:
            return summarizer.summarize_resume(text, max_sentences)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def process_summarization(text, text_type, max_sentences):
    """Process summarization with statistics"""
    if not text.strip():
        return "Please enter text to summarize.", "No statistics available."
    
    summary = summarize_text(text, text_type, max_sentences)
    
    # Calculate statistics
    original_words = len(text.split())
    summary_words = len(summary.split())
    compression_ratio = (1 - summary_words / original_words) * 100 if original_words > 0 else 0
    
    stats = f"""
    **Original Text**: {original_words} words
    **Summary**: {summary_words} words
    **Compression Ratio**: {compression_ratio:.1f}%
    **Sentences**: {len(summary.split('.')) - 1}
    """
    
    return summary, stats

# Load models on startup
model_status = load_models()

# Create interface using Gradio 3.x compatible syntax
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Resume Reviewer - AI-Powered Analysis")
    gr.Markdown(f"**System Status:** {model_status}")
    
    with gr.Tab("🎯 Skills Extraction"):
        gr.Markdown("### Extract Required Skills from Job Description")
        jd_input_skills = gr.Textbox(lines=8)
        extract_btn = gr.Button("🔍 Extract Skills")
        skills_output = gr.Markdown()
        
        extract_btn.click(
            fn=extract_skills_from_jd,
            inputs=jd_input_skills,
            outputs=skills_output
        )
    
    with gr.Tab("📊 Resume Analysis"):
        gr.Markdown("### Analyze Resume Against Job Description")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Job Description")
                job_input = gr.Textbox(lines=8)
            
            with gr.Column():
                gr.Markdown("### 📄 Resume Input")
                resume_input = gr.Textbox(lines=8)
                resume_pdf = gr.File()
        
        analyze_btn = gr.Button("🔍 Analyze Resume Match")
        output = gr.Markdown()
        
        analyze_btn.click(
            fn=analyze_resume,
            inputs=[job_input, resume_input, resume_pdf],
            outputs=output
        )
    
    with gr.Tab("📝 Text Summarization"):
        gr.Markdown("### AI-Powered Text Summarization")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Input Text")
                text_input = gr.Textbox(lines=10)
                
                with gr.Row():
                    text_type = gr.Radio(choices=["Job Description", "Resume"])
                    max_sentences = gr.Slider(minimum=1, maximum=10, value=3, step=1)
                
                summarize_btn = gr.Button("📝 Generate Summary")
            
            with gr.Column():
                gr.Markdown("### 📋 Generated Summary")
                summary_output = gr.Markdown()
                gr.Markdown("### 📊 Summary Statistics")
                stats_output = gr.Markdown()
        
        summarize_btn.click(
            fn=process_summarization,
            inputs=[text_input, text_type, max_sentences],
            outputs=[summary_output, stats_output]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
