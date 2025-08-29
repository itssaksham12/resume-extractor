# Resume Extractor & Analyzer - Streamlit App

## Overview

This is a modern, user-friendly web application for resume and job description analysis, built with Streamlit. It provides comprehensive AI-powered analysis including skills extraction, text summarization, and resume-job matching.

## Features

### 🎯 Core Features
- **Resume Analysis**: Extract skills, experience, education, and generate summaries
- **Job Description Analysis**: Analyze job postings for required skills and generate summaries
- **Resume-Job Matching**: Calculate compatibility scores between resumes and job descriptions
- **PDF Support**: Upload and analyze PDF resumes directly

### 🤖 AI Models
- **BERT Skills Extractor**: AI-powered skills extraction from text
- **LSTM Resume Matcher**: Deep learning model for resume-job compatibility scoring
- **BERT Summarizer**: AI-powered text summarization
- **Rule-based Extractors**: Fallback methods for basic analysis

## Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the App
```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Resume Analysis
1. Navigate to "Resume Analysis" in the sidebar
2. Upload a PDF resume or paste text directly
3. Click "Analyze Resume" to get:
   - Extracted skills
   - Experience years
   - Education level
   - AI-generated summary

### Job Description Analysis
1. Navigate to "Job Description Analysis"
2. Paste a job description
3. Click "Analyze Job Description" to get:
   - Required skills
   - Required experience
   - AI-generated summary

### Resume-Job Matching
1. Navigate to "Resume-Job Matching"
2. Upload a resume (PDF or text)
3. Enter a job description
4. Click "Calculate Match Score" to get:
   - Compatibility percentage
   - Skill overlap analysis
   - Detailed comparison

## Model Status

The app automatically loads available models and shows their status:

- ✅ **Loaded**: Model is available and working
- ⚠️ **Warning**: Model failed to load, using fallback
- ℹ️ **Info**: Model not found, using rule-based methods

## File Structure

```
resume-extractor-clean/
├── streamlit_app.py          # Main Streamlit application
├── app/                      # Core application modules
│   ├── bert_skills_extractor.py
│   ├── bert_summarizer.py
│   └── ...
├── enhanced_models/          # Trained model files
├── enhanced_data/           # Training data
└── requirements.txt         # Python dependencies
```

## Troubleshooting

### Model Loading Issues
- Ensure all model files are in the correct directories
- Check that TensorFlow and PyTorch versions are compatible
- Verify that model files are not corrupted

### PDF Upload Issues
- Ensure PDF files are not password-protected
- Check that PDF files contain extractable text
- Try copying text directly if PDF extraction fails

### Performance Issues
- The app loads models on startup, which may take a few seconds
- Large PDF files may take longer to process
- AI models require sufficient memory

## Development

### Adding New Features
1. Modify the appropriate function in `streamlit_app.py`
2. Add new pages to the sidebar navigation
3. Update the model loading logic if needed

### Customizing Models
- Models are trained using the scripts in the root directory
- Update model paths in `load_models()` function
- Add new model types as needed

## Dependencies

Key dependencies include:
- `streamlit>=1.49.0` - Web framework
- `tensorflow>=2.10.0` - Deep learning
- `torch>=2.0.0` - PyTorch for BERT models
- `transformers>=4.20.0` - Hugging Face transformers
- `PyPDF2>=3.0.0` - PDF processing
- `nltk>=3.7` - Natural language processing

## License

This project is part of the resume extractor system. See the main README for licensing information.
