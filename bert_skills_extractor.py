import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core ML and NLP libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification, BertModel,
    get_linear_schedule_with_warmup
)
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SkillsExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Predefined skill categories and keywords
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'sql',
                'typescript', 'dart', 'objective-c', 'assembly', 'cobol', 'fortran'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django',
                'flask', 'laravel', 'spring', 'bootstrap', 'jquery', 'ajax', 'json',
                'xml', 'rest api', 'graphql', 'soap', 'microservices', 'webpack'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'cassandra',
                'elasticsearch', 'dynamodb', 'firebase', 'mariadb', 'neo4j', 'couchdb'
            ],
            'data_science_ml': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'data analysis',
                'data visualization', 'statistics', 'neural networks', 'nlp', 'computer vision',
                'keras', 'xgboost', 'random forest', 'svm', 'clustering', 'regression'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'gitlab',
                'github', 'ci/cd', 'terraform', 'ansible', 'puppet', 'chef', 'vagrant',
                'linux', 'unix', 'windows', 'nginx', 'apache', 'load balancing'
            ],
            'mobile_development': [
                'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova',
                'ionic', 'mobile app development', 'app store', 'play store'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'project management',
                'agile', 'scrum', 'analytical thinking', 'creativity', 'adaptability',
                'time management', 'critical thinking', 'collaboration', 'mentoring'
            ],
            'tools_frameworks': [
                'jira', 'confluence', 'slack', 'trello', 'figma', 'sketch', 'photoshop',
                'illustrator', 'tableau', 'power bi', 'excel', 'word', 'powerpoint',
                'visual studio', 'intellij', 'eclipse', 'vim', 'emacs'
            ]
        }
        
        # Create comprehensive skills list
        self.all_skills = set()
        for category_skills in self.skill_categories.values():
            self.all_skills.update([skill.lower() for skill in category_skills])
        
        self.mlb = MultiLabelBinarizer()
        
    def load_and_preprocess_data(self, job_data_path: str, resume_data_path: str) -> pd.DataFrame:
        """Load and preprocess both datasets"""
        print("Loading datasets...")
        
        # Load job data
        job_df = pd.read_csv(job_data_path)
        print(f"Job dataset shape: {job_df.shape}")
        
        # Load resume data
        resume_df = pd.read_csv(resume_data_path)
        print(f"Resume dataset shape: {resume_df.shape}")
        
        # Prepare job descriptions data
        job_texts = []
        if 'Job Description' in job_df.columns:
            job_texts = job_df['Job Description'].fillna('').tolist()
        elif 'job_description' in job_df.columns:
            job_texts = job_df['job_description'].fillna('').tolist()
        else:
            # Try to find description column
            desc_cols = [col for col in job_df.columns if 'desc' in col.lower()]
            if desc_cols:
                job_texts = job_df[desc_cols[0]].fillna('').tolist()
        
        # Prepare resume data
        resume_texts = []
        if 'Resume' in resume_df.columns:
            resume_texts = resume_df['Resume'].fillna('').tolist()
        elif 'resume' in resume_df.columns:
            resume_texts = resume_df['resume'].fillna('').tolist()
        
        # Combine all text data
        all_texts = job_texts + resume_texts
        
        # Create DataFrame
        data = pd.DataFrame({
            'text': all_texts,
            'source': ['job'] * len(job_texts) + ['resume'] * len(resume_texts)
        })
        
        print(f"Combined dataset shape: {data.shape}")
        return data
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.all_skills:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[List[str], List[List[str]]]:
        """Prepare training data with extracted skills"""
        print("Extracting skills from texts...")
        
        texts = []
        skill_labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row['text'])
            if len(text.strip()) < 50:  # Skip very short texts
                continue
                
            extracted_skills = self.extract_skills_from_text(text)
            if len(extracted_skills) > 0:  # Only include texts with skills
                texts.append(text)
                skill_labels.append(extracted_skills)
        
        print(f"Prepared {len(texts)} training samples")
        return texts, skill_labels
    
    def create_skill_labels(self, skill_lists: List[List[str]]) -> np.ndarray:
        """Convert skill lists to binary labels"""
        return self.mlb.fit_transform(skill_lists)

class SkillsDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

class BERTSkillsClassifier(nn.Module):
    def __init__(self, n_classes: int, dropout_rate: float = 0.3):
        super(BERTSkillsClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class BERTSkillsModel:
    def __init__(self, skills_extractor: SkillsExtractor):
        self.extractor = skills_extractor
        self.model = None
        self.device = skills_extractor.device
        
    def train(self, texts: List[str], skill_labels: List[List[str]], 
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the BERT model"""
        
        # Prepare labels
        binary_labels = self.extractor.create_skill_labels(skill_labels)
        n_classes = binary_labels.shape[1]
        
        print(f"Training with {len(texts)} samples and {n_classes} skill classes")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, binary_labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = SkillsDataset(X_train, y_train, self.extractor.tokenizer)
        val_dataset = SkillsDataset(X_val, y_val, self.extractor.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = BERTSkillsClassifier(n_classes)
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Loss function for multi-label classification
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            train_progress = tqdm(train_loader, desc="Training")
            
            for batch in train_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def predict_skills(self, text: str, threshold: float = 0.3) -> List[str]:
        """Predict skills from text"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.extractor.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Get skills above threshold
        skill_indices = np.where(probabilities > threshold)[0]
        predicted_skills = [self.extractor.mlb.classes_[i] for i in skill_indices]
        
        return predicted_skills
    
    def plot_training_history(self, train_losses: List[float], val_losses: List[float]):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mlb': self.extractor.mlb,
            'skill_categories': self.extractor.skill_categories
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str, n_classes: int):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = BERTSkillsClassifier(n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        self.extractor.mlb = checkpoint['mlb']
        self.extractor.skill_categories = checkpoint['skill_categories']
        print(f"Model loaded from {path}")

def main():
    """Main function to run the complete pipeline"""
    
    # File paths (update these with your actual paths)
    job_data_path = "/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/processed_job_data.csv"
    resume_data_path = "/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/UpdatedResumeDataSet.csv"
    
    print("🚀 Starting BERT-based Skills Extraction Model Training")
    print("=" * 60)
    
    # Initialize skills extractor
    extractor = SkillsExtractor()
    
    # Load and preprocess data
    df = extractor.load_and_preprocess_data(job_data_path, resume_data_path)
    
    # Prepare training data
    texts, skill_labels = extractor.prepare_training_data(df)
    
    if len(texts) == 0:
        print("❌ No training data found! Check your datasets.")
        return
    
    # Initialize and train model
    model = BERTSkillsModel(extractor)
    
    print("\n🎯 Starting model training...")
    train_losses, val_losses = model.train(
        texts, skill_labels,
        epochs=3,  # Adjust as needed
        batch_size=8,  # Reduce if you have memory issues
        learning_rate=2e-5
    )
    
    # Save the trained model
    model.save_model("bert_skills_model.pth")
    
    # Test the model with sample texts
    print("\n🧪 Testing the model...")
    
    sample_texts = [
        "We are looking for a Python developer with experience in Django, React, and PostgreSQL.",
        "Required skills: Machine Learning, TensorFlow, Data Analysis, and AWS cloud services.",
        "Frontend developer needed with React, JavaScript, HTML, CSS, and Node.js experience."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📝 Sample {i}:")
        print(f"Text: {text}")
        predicted_skills = model.predict_skills(text, threshold=0.3)
        print(f"Predicted Skills: {predicted_skills}")
    
    print("\n✅ Training completed successfully!")
    print("💾 Model saved as 'bert_skills_model.pth'")

if __name__ == "__main__":
    main()

# Usage example for loading and using the model later:
"""
# To use the trained model later:

# Initialize extractor and model
extractor = SkillsExtractor()
model = BERTSkillsModel(extractor)

# Load the trained model (you need to know n_classes from training)
model.load_model("bert_skills_model.pth", n_classes=YOUR_N_CLASSES)

# Predict skills from new text
job_description = "We need a data scientist with Python, scikit-learn, and AWS experience."
predicted_skills = model.predict_skills(job_description, threshold=0.3)
print("Predicted skills:", predicted_skills)
"""