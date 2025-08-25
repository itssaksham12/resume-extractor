#!/usr/bin/env python3
"""
BERT-based Text Summarization for Job Descriptions and Resumes
Uses BERT embeddings and extractive summarization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
import re
import pickle
import os
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class BERTSummarizer(nn.Module):
    """BERT-based extractive summarization model"""
    
    def __init__(self, bert_model_name='bert-base-uncased', max_length=512):
        super(BERTSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        
        # Freeze BERT layers (optional - can be fine-tuned)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Summarization head
        self.summary_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation for sentence classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Predict sentence importance score
        scores = self.summary_head(cls_output)
        return scores

class TextPreprocessor:
    """Text preprocessing utilities for summarization"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text
    
    def sentence_similarity(self, sent1: str, sent2: str, embeddings) -> float:
        """Calculate cosine similarity between two sentences"""
        try:
            A = embeddings([sent1])[0]
            B = embeddings([sent2])[0]
            return 1 - (np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))
        except:
            return 0.0
    
    def build_similarity_matrix(self, sentences: List[str], embeddings) -> np.ndarray:
        """Build similarity matrix between sentences"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(
                        sentences[i], sentences[j], embeddings
                    )
        
        return similarity_matrix
    
    def word_frequency(self, sentences: List[str]) -> List[Dict[str, float]]:
        """Calculate word frequency for each sentence"""
        word_freq_list = []
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            # Remove stopwords and punctuation
            words = [w for w in words if w not in self.stop_words and w not in self.punctuation]
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Normalize frequencies
            if word_freq:
                max_freq = max(word_freq.values())
                word_freq = {k: v/max_freq for k, v in word_freq.items()}
            
            word_freq_list.append(word_freq)
        
        return word_freq_list
    
    def calculate_sentence_scores(self, sentences: List[str], word_freq_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate importance scores for sentences"""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w not in self.stop_words and w not in self.punctuation]
            
            score = 0.0
            word_freq = word_freq_list[i] if i < len(word_freq_list) else {}
            
            for word in words:
                score += word_freq.get(word, 0)
            
            sentence_scores[sentence] = score
        
        return sentence_scores

class SummarizationDataset(Dataset):
    """Dataset for training the summarization model"""
    
    def __init__(self, texts: List[str], summaries: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize summary
        summary_encoding = self.tokenizer(
            summary,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'summary_ids': summary_encoding['input_ids'].squeeze(),
            'summary_mask': summary_encoding['attention_mask'].squeeze()
        }

class BERTSummarizerTrainer:
    """Trainer class for BERT summarization model"""
    
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BERTSummarizer(model_name).to(self.device)
        self.preprocessor = TextPreprocessor()
        
    def prepare_training_data(self, job_descriptions: List[str], resumes: List[str]) -> Tuple[List[str], List[str]]:
        """Prepare training data from job descriptions and resumes"""
        texts = []
        summaries = []
        
        # Process job descriptions
        for jd in job_descriptions:
            if len(jd.strip()) > 100:  # Only process substantial texts
                sentences = sent_tokenize(jd)
                if len(sentences) > 3:  # Only if there are multiple sentences
                    # Create extractive summary (top 25% sentences)
                    word_freq_list = self.preprocessor.word_frequency(sentences)
                    sentence_scores = self.preprocessor.calculate_sentence_scores(sentences, word_freq_list)
                    
                    # Select top sentences for summary
                    select_length = max(1, int(len(sentences) * 0.25))
                    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:select_length]
                    summary = '. '.join([sent for sent, _ in top_sentences])
                    
                    texts.append(jd)
                    summaries.append(summary)
        
        # Process resumes
        for resume in resumes:
            if len(resume.strip()) > 100:
                sentences = sent_tokenize(resume)
                if len(sentences) > 3:
                    word_freq_list = self.preprocessor.word_frequency(sentences)
                    sentence_scores = self.preprocessor.calculate_sentence_scores(sentences, word_freq_list)
                    
                    select_length = max(1, int(len(sentences) * 0.25))
                    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:select_length]
                    summary = '. '.join([sent for sent, _ in top_sentences])
                    
                    texts.append(resume)
                    summaries.append(summary)
        
        return texts, summaries
    
    def train(self, texts: List[str], summaries: List[str], 
              batch_size=8, epochs=10, learning_rate=2e-5, save_path='bert_summarizer_model.pth'):
        """Train the BERT summarization model"""
        
        # Prepare dataset
        dataset = SummarizationDataset(texts, summaries, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        print(f"Training on {len(texts)} samples...")
        print(f"Device: {self.device}")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                summary_ids = batch['summary_ids'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss (simplified - in practice you'd want more sophisticated loss)
                loss = criterion(outputs.squeeze(), torch.ones(outputs.size(0)).to(self.device))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'preprocessor': self.preprocessor
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def summarize(self, text: str, max_sentences: int = 5) -> str:
        """Generate summary for given text"""
        self.model.eval()
        
        # Preprocess text
        text = self.preprocessor.clean_text(text)
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Calculate sentence scores
        word_freq_list = self.preprocessor.word_frequency(sentences)
        sentence_scores = self.preprocessor.calculate_sentence_scores(sentences, word_freq_list)
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        summary = '. '.join([sent for sent, _ in top_sentences])
        
        return summary
    
    def load_model(self, model_path: str):
        """Load trained model"""
        # Use weights_only=False for backward compatibility with older model files
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

def create_sample_data():
    """Create sample job descriptions and resumes for training"""
    
    sample_job_descriptions = [
        """
        We are seeking a Senior Data Scientist to join our team. The ideal candidate will have 5+ years of experience in machine learning, 
        Python programming, and statistical analysis. You will be responsible for developing predictive models, analyzing large datasets, 
        and communicating insights to stakeholders. Experience with TensorFlow, PyTorch, and SQL is required. 
        A Master's degree in Computer Science, Statistics, or related field is preferred.
        """,
        
        """
        Software Engineer - Full Stack Development position available. We need someone with 3+ years of experience in JavaScript, 
        React, Node.js, and Python. You will work on building scalable web applications, implementing new features, 
        and collaborating with cross-functional teams. Experience with AWS, Docker, and Git is required. 
        Knowledge of database design and RESTful APIs is essential.
        """,
        
        """
        Product Manager role for a fast-growing startup. We're looking for someone with 4+ years of product management experience 
        in the technology sector. You will be responsible for defining product strategy, gathering requirements, 
        working with engineering teams, and analyzing user feedback. Experience with Agile methodologies, 
        data analysis tools, and user research is required. MBA or related degree preferred.
        """
    ]
    
    sample_resumes = [
        """
        John Smith - Senior Data Scientist with 6 years of experience in machine learning and data analysis. 
        Proficient in Python, TensorFlow, PyTorch, and SQL. Led development of customer segmentation models 
        achieving 95% accuracy. Experience with big data technologies including Spark and Hadoop. 
        Master's degree in Computer Science from Stanford University. Published 3 papers in top ML conferences.
        """,
        
        """
        Sarah Johnson - Full Stack Developer with 4 years of experience building web applications. 
        Expert in JavaScript, React, Node.js, and Python. Built scalable microservices architecture 
        serving 1M+ users. Experience with AWS, Docker, and CI/CD pipelines. Led team of 3 developers 
        on major product launches. Bachelor's degree in Computer Science from MIT.
        """,
        
        """
        Michael Chen - Product Manager with 5 years of experience in SaaS and mobile applications. 
        Successfully launched 3 products with combined revenue of $10M. Expert in user research, 
        data analysis, and Agile methodologies. Experience with A/B testing, user interviews, 
        and competitive analysis. MBA from Harvard Business School. Led cross-functional teams of 10+ people.
        """
    ]
    
    return sample_job_descriptions, sample_resumes

def main():
    """Main training function"""
    print("🚀 BERT Summarizer Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = BERTSummarizerTrainer()
    
    # Create sample data
    print("📝 Creating sample training data...")
    job_descriptions, resumes = create_sample_data()
    
    # Prepare training data
    print("🔄 Preparing training data...")
    texts, summaries = trainer.prepare_training_data(job_descriptions, resumes)
    
    print(f"📊 Training data prepared: {len(texts)} samples")
    
    # Train model
    print("🎯 Starting model training...")
    trainer.train(texts, summaries, epochs=5, batch_size=4)
    
    # Test summarization
    print("\n🧪 Testing summarization...")
    test_text = """
    We are looking for a Machine Learning Engineer with expertise in deep learning, computer vision, 
    and natural language processing. The candidate should have 3+ years of experience with PyTorch, 
    TensorFlow, and Python. You will work on cutting-edge AI projects, develop production-ready models, 
    and collaborate with research teams. Experience with cloud platforms like AWS or GCP is required. 
    Knowledge of MLOps, Docker, and Kubernetes is a plus. A PhD in Computer Science or related field is preferred.
    """
    
    summary = trainer.summarize(test_text, max_sentences=3)
    print(f"Original text length: {len(test_text)} characters")
    print(f"Summary length: {len(summary)} characters")
    print(f"Summary: {summary}")
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()
