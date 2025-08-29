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
        # Use weights_only=False for backward compatibility with older model files
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model = BERTSkillsClassifier(n_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        self.extractor.mlb = checkpoint['mlb']
        self.extractor.skill_categories = checkpoint['skill_categories']
        print(f"Model loaded from {path}")
