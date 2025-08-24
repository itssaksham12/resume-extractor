# LSTM Resume Matcher Training System

A comprehensive deep learning system for matching resumes to job descriptions using LSTM neural networks. This system automatically extracts skills, experiences, and projects from resumes and generates numerical match scores.

## 🚀 Features

- **Advanced Feature Extraction**: Automatically extracts skills, experience years, project counts, and education levels
- **Multi-Modal LSTM**: Combines text embeddings with numerical features for better accuracy
- **Comprehensive Matching**: Compares skills, experiences, projects, and education levels
- **Batch Processing**: Handle multiple resume-job pairs efficiently
- **Ranking System**: Rank candidates for specific job positions
- **Gap Analysis**: Identify skill gaps and provide improvement suggestions
- **Detailed Reporting**: Generate comprehensive match reports with recommendations

## 📊 Dataset Requirements

The system uses the following CSV files:
- `AI_Resume_Screening.csv` (1,000 records) - Contains hiring decisions and candidate data
- `UpdatedResumeDataSet.csv` (42,000 records) - Large resume dataset with categories
- `job_title_des.csv` (63,000 records) - Job titles and descriptions
- `processed_job_data.csv` (2,300 records) - Processed job descriptions

**Total Training Data**: ~107,000 records combined

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements_lstm.txt
```

2. **Install additional NLTK data** (if needed):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🎯 Quick Start

### 1. Train the Model

```bash
# Full training (recommended)
python lstm_resume_matcher.py

# Quick training for testing
python run_training_example.py --mode train --quick
```

### 2. Test Predictions

```bash
# Test with sample data
python run_training_example.py --mode test

# Test both training and prediction
python run_training_example.py --mode both --quick
```

### 3. Use Trained Model for Predictions

```python
from resume_matcher_predictor import ResumeMatcherPredictor

# Initialize predictor (automatically finds latest model)
predictor = ResumeMatcherPredictor()

# Single prediction
resume_text = """
Skills: Python, Machine Learning, TensorFlow
Experience: 3 years in data science
Education: M.Sc Computer Science
Projects: 5 ML projects completed
"""

job_description = """
Looking for Data Scientist with 2+ years experience in Python,
Machine Learning, and TensorFlow. Must have project experience.
"""

result = predictor.predict_single_match(resume_text, job_description)
print(f"Match Score: {result['overall_match_score']}%")
print(f"Recommendation: {result['recommendation']}")
```

## 📈 Model Architecture

### LSTM Network Structure:
```
Text Input (Resume + Job Description)
    ↓
Embedding Layer (128 dimensions)
    ↓
Bidirectional LSTM (64 units) + Dropout
    ↓
Bidirectional LSTM (32 units) + Dropout
    ↓
Dense Layer (32 units)

Numerical Features Input (10 features)
    ↓
Dense Layer (32 units) + BatchNorm + Dropout
    ↓
Dense Layer (16 units) + Dropout

Combined Features
    ↓
Dense Layer (64 units) + BatchNorm + Dropout
    ↓
Dense Layer (32 units) + Dropout
    ↓
Output Layer (1 unit, sigmoid) → Match Score (0-1)
```

### Features Extracted:
1. **Text Features**:
   - Combined resume and job description embeddings
   - LSTM-processed sequential information

2. **Numerical Features**:
   - Number of skills in resume
   - Number of required skills
   - Skill overlap count
   - Total unique skills
   - Candidate experience years
   - Required experience years
   - Number of projects
   - Education level (0-4 scale)
   - Resume text length
   - Job description length

## 🎯 Performance Metrics

Expected performance on validation data:
- **Accuracy**: 75-85%
- **Mean Absolute Error**: ~0.1-0.15
- **R² Score**: 0.6-0.8

## 📋 Command Line Usage

### Training:
```bash
# Full training
python lstm_resume_matcher.py

# Quick example training
python run_training_example.py --mode train --quick
```

### Single Prediction:
```bash
python resume_matcher_predictor.py --mode single \
    --resume "path/to/resume.txt" \
    --job "path/to/job_description.txt"
```

### Batch Processing:
```bash
python resume_matcher_predictor.py --mode batch \
    --input_file "resume_job_pairs.csv" \
    --output_file "match_results.csv"
```

Expected CSV format for batch processing:
```csv
resume_text,job_description,candidate_id,job_id
"Skills: Python...",Job description text...,C001,J001
"Skills: Java...",Another job description...,C002,J002
```

## 📊 Output Format

### Single Prediction Result:
```json
{
    "overall_match_score": 78.5,
    "skill_match_score": 85.0,
    "experience_match_score": 75.0,
    "extracted_skills": ["python", "machine learning", "sql"],
    "required_skills": ["python", "sql", "tensorflow"],
    "matching_skills": ["python", "sql"],
    "missing_skills": ["tensorflow"],
    "extra_skills": ["machine learning"],
    "candidate_experience": 3,
    "required_experience": 2,
    "projects_count": 5,
    "education_level": 3,
    "recommendation": "Good Match - Recommended"
}
```

### Skill Gap Analysis:
```json
{
    "current_match_score": 78.5,
    "missing_critical_skills": ["tensorflow"],
    "missing_nice_to_have": ["docker", "aws"],
    "experience_gap_years": 0,
    "improvement_suggestions": [
        "Learn TensorFlow for better ML model development",
        "Gain cloud platform experience (AWS/Azure)",
        "Build more projects to demonstrate practical skills"
    ],
    "potential_score_improvement": 88.5
}
```

## 🔧 Advanced Usage

### Custom Model Training:
```python
from lstm_resume_matcher import LSTMResumeMatcherTrainer

# Custom configuration
trainer = LSTMResumeMatcherTrainer(
    max_features=15000,  # Vocabulary size
    max_len=600         # Maximum sequence length
)

# Load data and train
ai_screening, updated_resumes, all_jobs = trainer.load_and_preprocess_data()
training_df = trainer.create_training_pairs(ai_screening, updated_resumes, all_jobs)
X_text = trainer.prepare_text_data(training_df)
X_numerical = trainer.extract_numerical_features(training_df)
y = training_df['match_score'].values

# Train with custom parameters
history = trainer.train_model(
    X_text, X_numerical, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

### Ranking Candidates:
```python
candidates = [
    {'name': 'John Doe', 'resume_text': 'Skills: Python...'},
    {'name': 'Jane Smith', 'resume_text': 'Skills: Java...'},
]

job_description = "Looking for Python developer..."

# Rank candidates
ranked_results = predictor.rank_candidates(candidates, job_description, top_k=5)

for result in ranked_results:
    print(f"Rank {result['rank']}: {result['candidate_name']} - {result['overall_match_score']}%")
```

## 📁 File Structure

```
├── lstm_resume_matcher.py          # Main training script
├── resume_matcher_predictor.py     # Prediction and inference
├── run_training_example.py         # Example training script
├── requirements_lstm.txt           # Dependencies
├── LSTM_README.md                 # This file
├── training_history.png           # Training plots (generated)
├── lstm_training.log              # Training logs (generated)
└── saved_models/                  # Model artifacts (generated)
    ├── lstm_resume_matcher_YYYYMMDD_HHMMSS.h5
    ├── lstm_resume_matcher_tokenizer_YYYYMMDD_HHMMSS.pkl
    ├── lstm_resume_matcher_scaler_YYYYMMDD_HHMMSS.pkl
    └── lstm_resume_matcher_extractor_YYYYMMDD_HHMMSS.pkl
```

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Error during training**:
   - Reduce `max_features` or `max_len`
   - Use smaller batch size
   - Reduce training data size

2. **Poor model performance**:
   - Increase training epochs
   - Add more training data
   - Adjust learning rate
   - Check data quality

3. **Import errors**:
   - Install all requirements: `pip install -r requirements_lstm.txt`
   - Check TensorFlow installation
   - Update numpy/pandas versions

### Performance Optimization:

1. **GPU Training**:
   ```python
   # Check GPU availability
   import tensorflow as tf
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   ```

2. **Memory Optimization**:
   ```python
   # Reduce memory usage
   trainer = LSTMResumeMatcherTrainer(
       max_features=8000,   # Reduce vocabulary
       max_len=400         # Reduce sequence length
   )
   ```

## 📈 Model Evaluation

The system provides comprehensive evaluation metrics:

- **Training/Validation Loss**: Monitor overfitting
- **Mean Absolute Error**: Average prediction error
- **R² Score**: Explained variance
- **Skill Match Accuracy**: How well skills are matched
- **Experience Match Accuracy**: Experience level matching

## 🔮 Future Enhancements

Potential improvements:
1. **BERT Integration**: Use pre-trained transformers for better text understanding
2. **Multi-task Learning**: Predict both match score and hiring decision
3. **Attention Mechanisms**: Add attention layers for better interpretability
4. **Domain-specific Models**: Train separate models for different job categories
5. **Real-time API**: Deploy as web service for real-time predictions

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data privacy regulations when using with real resume data.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Submit pull request

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Review training logs
3. Verify data format and paths
4. Ensure all dependencies are installed