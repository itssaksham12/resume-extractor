# 📁 Project Structure - Resume Reviewer AI

## 🗂️ **Clean Project Organization**

```
resume_reviewer/
├── 📁 models/                    # AI Model Files
│   ├── bert_skills_model.pth                    # BERT Skills Extractor (438MB)
│   ├── bert_summarizer_model.pth                # BERT Text Summarizer (439MB)
│   ├── lstm_resume_matcher_best.h5              # LSTM Resume Matcher (24MB)
│   ├── lstm_resume_matcher_20250822_174743.h5   # LSTM Model Backup
│   ├── lstm_resume_matcher_extractor_*.pkl      # Feature Extractor
│   ├── lstm_resume_matcher_scaler_*.pkl         # Data Scaler
│   └── lstm_resume_matcher_tokenizer_*.pkl      # Text Tokenizer
│
├── 📁 data/                      # Dataset Files
│   ├── AI_Resume_Screening.csv                  # Resume screening data
│   ├── job_title_des.csv                        # Job descriptions
│   ├── processed_job_data.csv                   # Processed job data
│   ├── UpdatedResumeDataSet.csv                 # Updated resume dataset
│   ├── NGIL_resume.csv                          # NGIL specific data
│   ├── resume_match_results.csv                 # Match results
│   ├── normlized_classes.txt                    # Normalized skill classes
│   └── resume_samples.txt                       # Sample resumes
│
├── 📁 docs/                      # Documentation
│   ├── LSTM_README.md                           # LSTM training guide
│   └── RESTORATION_SUMMARY.md                   # AI restoration summary
│
├── 📁 resume-reviewer/           # Main Application
│   ├── app.py                                   # Gradio web interface
│   ├── bert_skills_extractor.py                 # BERT skills model
│   ├── bert_summarizer.py                       # BERT summarizer
│   ├── resume_matcher_predictor.py              # LSTM predictor
│   ├── README.md                                # App documentation
│   ├── DEPLOYMENT_GUIDE.md                      # Deployment guide
│   └── requirements.txt                         # App dependencies
│
├── 📁 resumes_corpus/            # Resume Corpus (Large Dataset)
│   └── [60,000+ resume files]                   # Training corpus
│
├── 📁 resume_db/                 # Database Files
│   └── [Database files]                         # Resume database
│
├── 📁 venv/                      # Virtual Environment
│   └── [Python virtual environment]             # Isolated dependencies
│
├── 🔧 Core Files
│   ├── bert_skills_extractor.py                 # BERT training script
│   ├── lstm_resume_matcher.py                   # LSTM training script
│   ├── resume_matcher_predictor.py              # LSTM predictor
│   ├── frontend.py                              # Flask frontend
│   ├── resume_processor.py                      # Resume processing
│   ├── resume-extractor.py                      # Resume extraction
│   ├── preprocessing.py                         # Data preprocessing
│   ├── run_training_example.py                  # Training examples
│   └── test_resume_processor.py                 # Test scripts
│
├── 📋 Configuration Files
│   ├── requirements.txt                         # Main dependencies
│   ├── requirements_lstm.txt                    # LSTM specific deps
│   ├── .gitignore                               # Git ignore rules
│   ├── .gitattributes                           # Git LFS config
│   └── README.md                                # Project overview
│
└── 📊 Logs & Outputs
    └── lstm_training.log                        # Training logs
```

## 🎯 **Key Features**

### **✅ AI Models (Restored)**
- **BERT Skills Model**: 95%+ accuracy skill extraction
- **LSTM Resume Matcher**: Complete analysis with experience gaps
- **BERT Summarizer**: Semantic text summarization

### **📊 Datasets**
- **3,277 training records** for BERT skills model
- **3,000 resume-job pairs** for LSTM training
- **60,000+ resume corpus** for additional training

### **🚀 Applications**
- **Gradio Web Interface**: Modern UI with full AI capabilities
- **Flask Frontend**: Alternative web interface
- **Command Line Tools**: Direct model usage

## 🔧 **Model Loading Paths**

The application automatically searches for models in this order:
1. `models/` directory (organized structure)
2. `resume-reviewer/` directory (deployment structure)
3. Current directory (fallback)

## 📦 **Deployment Ready**

- ✅ **Local Development**: All models functional
- ✅ **Hugging Face**: Compatible with Spaces deployment
- ✅ **Git LFS**: Configured for large model files
- ✅ **Fallback Systems**: Available for reliability

## 🎮 **Usage**

### **Start the Application**
```bash
cd resume-reviewer
source venv/bin/activate
python app.py
```

### **Access the Web Interface**
- **Local**: http://localhost:7860
- **Public**: https://[gradio-url].gradio.live

---

*Project cleaned and organized on: August 22, 2025*
