BERT-based Skills Extraction Model
A machine learning model that automatically extracts technical skills from job descriptions and resumes using BERT (Bidirectional Encoder Representations from Transformers).
📋 Overview
This project implements a multi-label classification system that can identify and extract relevant skills from unstructured text data such as job descriptions and resumes. The model is trained on a comprehensive dataset combining job postings and resume data to provide accurate skill extraction across various technology domains.
🎯 Features

Multi-label Classification: Extracts multiple skills from a single text
Comprehensive Skill Coverage: 135+ skills across 8 major categories
BERT-based Architecture: Leverages pre-trained BERT for superior text understanding
Flexible Confidence Thresholds: Adjustable prediction confidence levels
Training Visualization: Built-in training progress and loss visualization
Model Persistence: Save and load trained models for reuse

🏗️ Architecture
The model consists of:

BERT Base Model: Pre-trained bert-base-uncased for text encoding
Classification Head: Linear layer with dropout for multi-label prediction
Skills Extractor: Pattern-matching component for skill identification
Training Pipeline: Complete end-to-end training with validation

📊 Skill Categories
The model recognizes skills across 8 major categories:
CategoryExamplesProgramming LanguagesPython, Java, JavaScript, C++, SQLWeb TechnologiesReact, Django, Node.js, HTML/CSS, REST APIDatabasesMySQL, MongoDB, PostgreSQL, RedisData Science/MLTensorFlow, PyTorch, pandas, scikit-learnCloud/DevOpsAWS, Docker, Kubernetes, Jenkins, GitMobile DevelopmentAndroid, iOS, React Native, FlutterSoft SkillsLeadership, Communication, Project ManagementTools/FrameworksJira, Tableau, Visual Studio, Figma
📁 Dataset Requirements
The model expects two CSV files:
processed_job_data.csv

Columns: Job Title, Job Description
Content: Job postings with detailed descriptions
Format: Each row represents one job posting

UpdatedResumeDataSet.csv

Columns: Category, Resume
Content: Resume text data across various professional domains
Format: Each row contains full resume text

🚀 Installation
Prerequisites

Python 3.8+
PyTorch
Transformers library

Install Dependencies
bashpip install pandas numpy scikit-learn torch transformers matplotlib seaborn tqdm
💻 Usage
Basic Training

Update file paths in the script:

pythonjob_data_path = "/path/to/your/processed_job_data.csv"
resume_data_path = "/path/to/your/UpdatedResumeDataSet.csv"

Run the training script:

bashpython bert_skills_extractor.py
Training Parameters
You can customize training parameters in the main() function:
pythontrain_losses, val_losses = model.train(
    texts, skill_labels,
    epochs=3,           # Number of training epochs
    batch_size=8,       # Batch size (reduce if memory issues)
    learning_rate=2e-5  # Learning rate for optimizer
)
Prediction Example
python# Load trained model
extractor = SkillsExtractor()
model = BERTSkillsModel(extractor)
model.load_model("bert_skills_model.pth", n_classes=135)

# Extract skills from text
job_description = """
We are looking for a Python developer with experience in Django, 
React, PostgreSQL, and AWS cloud services. Knowledge of machine 
learning and TensorFlow is a plus.
"""

predicted_skills = model.predict_skills(job_description, threshold=0.3)
print("Extracted Skills:", predicted_skills)
📈 Model Performance
Training Results
Based on a typical training run:
MetricEpoch 1Epoch 2Epoch 3Training Loss0.30060.17580.1672Validation Loss0.17610.16140.1570
Performance Characteristics

Training Time: ~54 minutes (CPU) for 3 epochs
Inference Time: 1-3 seconds per prediction
Model Size: ~110MB (BERT base + classification layer)
Skill Classes: 135 unique skills identified

🔧 Configuration
Adjusting Confidence Threshold
python# Higher threshold = more confident predictions (fewer skills)
high_confidence_skills = model.predict_skills(text, threshold=0.5)

# Lower threshold = more inclusive predictions (more skills)
inclusive_skills = model.predict_skills(text, threshold=0.2)
Adding Custom Skills
To add domain-specific skills, modify the skill_categories dictionary:
pythonself.skill_categories['your_domain'] = [
    'custom_skill_1', 'custom_skill_2', 'domain_specific_tool'
]
📂 Output Files
The training process generates:

bert_skills_model.pth: Trained model weights and metadata
training_history.png: Training and validation loss visualization
Console logs: Detailed training progress and sample predictions

🛠️ Troubleshooting
Common Issues
ImportError with AdamW:
The script automatically handles different transformers versions. If issues persist:
bashpip install transformers==4.21.0 torch>=1.12.0
Memory Issues:

Reduce batch_size from 8 to 4 or 2
Use CPU instead of GPU if CUDA memory is insufficient

No Skills Extracted:

Check if your CSV files have the expected column names
Ensure text data is not empty or too short (minimum 50 characters)

Hardware Requirements
ComponentMinimumRecommendedRAM8GB16GB+Storage2GB5GB+CPU4 cores8+ coresGPUOptionalCUDA-compatible (speeds up training)
📊 Sample Results
Example Input
"We need a full-stack developer with React, Node.js, MongoDB, 
and AWS experience. Docker and CI/CD knowledge preferred."
Example Output
python[
    'react', 'nodejs', 'mongodb', 'aws', 'docker', 
    'ci/cd', 'full-stack', 'javascript'
]