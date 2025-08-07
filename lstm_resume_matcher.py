#!/usr/bin/env python3
"""
LSTM Resume Matcher - Advanced Training Script
Automatically extracts skills, experiences, and projects from resumes
and generates numerical match scores against job descriptions.
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Embedding, Dropout, Input, 
    Concatenate, BatchNormalization, Bidirectional
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Advanced feature extraction from resumes and job descriptions"""
    
    def __init__(self):
        # Skill keywords (expanded list)
        self.technical_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'angular', 'vue.js',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'data science', 'data analysis', 'statistics', 'algorithms',
            'html', 'css', 'bootstrap', 'jquery', 'php', 'ruby', 'go', 'rust',
            'c++', 'c#', '.net', 'spring boot', 'django', 'flask', 'fastapi',
            'spark', 'hadoop', 'kafka', 'airflow', 'tableau', 'power bi',
            'linux', 'unix', 'bash', 'powershell', 'networking', 'cybersecurity'
        ]
        
        self.soft_skills = [
            'communication', 'leadership', 'teamwork', 'problem solving',
            'project management', 'agile', 'scrum', 'analytical thinking',
            'creativity', 'adaptability', 'time management', 'collaboration'
        ]
        
        # Experience indicators
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*year\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*years?\s*working'
        ]
        
        # Project indicators
        self.project_patterns = [
            r'projects?[:\s]*(\d+)',
            r'(\d+)\+?\s*projects?',
            r'worked\s+on\s+(\d+)\+?\s*projects?',
            r'completed\s+(\d+)\+?\s*projects?',
            r'led\s+(\d+)\+?\s*projects?'
        ]

    def extract_skills(self, text):
        """Extract technical and soft skills from text"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        # Technical skills
        for skill in self.technical_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        # Soft skills
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))

    def extract_experience_years(self, text):
        """Extract years of experience from text"""
        if pd.isna(text) or not isinstance(text, str):
            return 0
        
        years = []
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years.append(int(match))
                except ValueError:
                    continue
        
        return max(years) if years else 0

    def extract_project_count(self, text):
        """Extract number of projects from text"""
        if pd.isna(text) or not isinstance(text, str):
            return 0
        
        projects = []
        for pattern in self.project_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    projects.append(int(match))
                except ValueError:
                    continue
        
        return max(projects) if projects else 0

    def extract_education_level(self, text):
        """Extract education level and encode it numerically"""
        if pd.isna(text) or not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        
        # Education level mapping (higher number = higher education)
        if any(word in text_lower for word in ['phd', 'ph.d', 'doctorate']):
            return 4
        elif any(word in text_lower for word in ['masters', 'master', 'm.tech', 'mba', 'm.sc']):
            return 3
        elif any(word in text_lower for word in ['bachelor', 'b.tech', 'b.sc', 'b.e']):
            return 2
        elif any(word in text_lower for word in ['diploma', 'associate']):
            return 1
        else:
            return 0

class LSTMResumeMatcherTrainer:
    """Main training class for LSTM Resume Matcher"""
    
    def __init__(self, max_features=10000, max_len=500):
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
        self.feature_extractor = AdvancedFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets"""
        logger.info("Loading datasets...")
        
        # Load datasets
        ai_screening = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/AI_Resume_Screening.csv')
        updated_resumes = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/UpdatedResumeDataSet.csv')
        job_titles = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/job_title_des.csv')
        processed_jobs = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/processed_job_data.csv')
        
        logger.info(f"AI Screening: {len(ai_screening)} records")
        logger.info(f"Updated Resumes: {len(updated_resumes)} records")
        logger.info(f"Job Titles: {len(job_titles)} records")
        logger.info(f"Processed Jobs: {len(processed_jobs)} records")
        
        # Combine job descriptions
        all_jobs = pd.concat([
            job_titles[['Job Title', 'Job Description']].rename(columns={'Job Title': 'title', 'Job Description': 'description'}),
            processed_jobs[['Job Title', 'Job Description']].rename(columns={'Job Title': 'title', 'Job Description': 'description'})
        ]).drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Combined jobs: {len(all_jobs)} records")
        
        return ai_screening, updated_resumes, all_jobs
    
    def create_training_pairs(self, ai_screening, updated_resumes, all_jobs, num_pairs=50000):
        """Create training pairs with comprehensive features"""
        logger.info("Creating training pairs...")
        
        training_data = []
        
        # Method 1: Use AI Screening data with hiring decisions
        logger.info("Processing AI Screening data...")
        for _, row in ai_screening.iterrows():
            # Create resume text from available fields
            resume_parts = []
            if pd.notna(row.get('Skills')):
                resume_parts.append(f"Skills: {row['Skills']}")
            if pd.notna(row.get('Experience (Years)')):
                resume_parts.append(f"Experience: {row['Experience (Years)']} years")
            if pd.notna(row.get('Education')):
                resume_parts.append(f"Education: {row['Education']}")
            if pd.notna(row.get('Certifications')) and row['Certifications'] != 'None':
                resume_parts.append(f"Certifications: {row['Certifications']}")
            if pd.notna(row.get('Projects Count')):
                resume_parts.append(f"Projects: {row['Projects Count']} projects")
            
            resume_text = ". ".join(resume_parts)
            
            # Match with random job descriptions from the same role
            job_role = row.get('Job Role', '').lower()
            
            # Find relevant jobs
            relevant_jobs = all_jobs[all_jobs['title'].str.lower().str.contains(job_role, na=False)]
            if len(relevant_jobs) == 0:
                relevant_jobs = all_jobs.sample(min(5, len(all_jobs)))
            
            for _, job in relevant_jobs.head(3).iterrows():
                if pd.notna(job['description']):
                    # Calculate match score based on hiring decision and features
                    base_score = 0.8 if row.get('Recruiter Decision') == 'Hire' else 0.3
                    
                    # Add noise for more realistic scores
                    noise = np.random.normal(0, 0.1)
                    match_score = np.clip(base_score + noise, 0, 1)
                    
                    training_data.append({
                        'resume_text': resume_text,
                        'job_description': job['description'],
                        'match_score': match_score,
                        'source': 'ai_screening'
                    })
        
        # Method 2: Use Updated Resume Dataset with synthetic job matching
        logger.info("Processing Updated Resume Dataset...")
        for _, resume_row in updated_resumes.head(5000).iterrows():  # Limit for processing time
            if pd.notna(resume_row.get('Resume')):
                resume_text = resume_row['Resume']
                category = resume_row.get('Category', '').lower()
                
                # Find jobs that match the resume category
                category_keywords = category.split()
                relevant_jobs = all_jobs
                for keyword in category_keywords:
                    if keyword:
                        relevant_jobs = relevant_jobs[
                            relevant_jobs['title'].str.lower().str.contains(keyword, na=False) |
                            relevant_jobs['description'].str.lower().str.contains(keyword, na=False)
                        ]
                
                if len(relevant_jobs) == 0:
                    relevant_jobs = all_jobs.sample(min(3, len(all_jobs)))
                
                for _, job in relevant_jobs.head(2).iterrows():
                    if pd.notna(job['description']):
                        # Calculate synthetic match score based on text similarity
                        match_score = self.calculate_synthetic_match_score(resume_text, job['description'], category)
                        
                        training_data.append({
                            'resume_text': resume_text,
                            'job_description': job['description'],
                            'match_score': match_score,
                            'source': 'updated_resumes'
                        })
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        logger.info(f"Created {len(training_df)} training pairs")
        
        return training_df
    
    def calculate_synthetic_match_score(self, resume_text, job_description, category):
        """Calculate synthetic match score based on feature overlap"""
        # Extract features from both texts
        resume_skills = self.feature_extractor.extract_skills(resume_text)
        job_skills = self.feature_extractor.extract_skills(job_description)
        
        resume_exp = self.feature_extractor.extract_experience_years(resume_text)
        job_exp = self.feature_extractor.extract_experience_years(job_description)
        
        resume_projects = self.feature_extractor.extract_project_count(resume_text)
        
        # Calculate skill overlap
        skill_overlap = len(set(resume_skills) & set(job_skills)) / max(len(set(resume_skills) | set(job_skills)), 1)
        
        # Experience match (penalize if resume exp is much less than required)
        exp_match = 1.0
        if job_exp > 0:
            exp_match = min(resume_exp / job_exp, 1.0) if resume_exp > 0 else 0.3
        
        # Project bonus
        project_bonus = min(resume_projects * 0.05, 0.2)
        
        # Category match bonus
        category_bonus = 0.1 if category.lower() in job_description.lower() else 0
        
        # Combine scores
        base_score = (skill_overlap * 0.6 + exp_match * 0.3 + project_bonus + category_bonus)
        
        # Add some randomness for more realistic training data
        noise = np.random.normal(0, 0.05)
        final_score = np.clip(base_score + noise, 0.1, 0.95)
        
        return final_score
    
    def extract_numerical_features(self, df):
        """Extract numerical features from text data"""
        logger.info("Extracting numerical features...")
        
        features = []
        
        for _, row in df.iterrows():
            resume_text = str(row['resume_text'])
            job_text = str(row['job_description'])
            
            # Extract features
            resume_skills = self.feature_extractor.extract_skills(resume_text)
            job_skills = self.feature_extractor.extract_skills(job_text)
            
            resume_exp = self.feature_extractor.extract_experience_years(resume_text)
            job_exp = self.feature_extractor.extract_experience_years(job_text)
            
            resume_projects = self.feature_extractor.extract_project_count(resume_text)
            resume_education = self.feature_extractor.extract_education_level(resume_text)
            
            # Calculate feature vector
            feature_vector = [
                len(resume_skills),  # Number of skills in resume
                len(job_skills),     # Number of skills required
                len(set(resume_skills) & set(job_skills)),  # Skill overlap
                len(set(resume_skills) | set(job_skills)),  # Total unique skills
                resume_exp,          # Resume experience years
                job_exp,             # Required experience years
                resume_projects,     # Number of projects
                resume_education,    # Education level
                len(resume_text.split()),  # Resume length
                len(job_text.split()),     # Job description length
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def prepare_text_data(self, df):
        """Prepare text data for LSTM input"""
        logger.info("Preparing text data...")
        
        # Combine resume and job description for tokenization
        combined_texts = (df['resume_text'] + ' [SEP] ' + df['job_description']).astype(str)
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(combined_texts)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(combined_texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded_sequences
    
    def build_model(self, numerical_features_dim):
        """Build the LSTM model with both text and numerical inputs"""
        logger.info("Building LSTM model...")
        
        # Text input branch
        text_input = Input(shape=(self.max_len,), name='text_input')
        embedding = Embedding(self.max_features, 128, input_length=self.max_len)(text_input)
        lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(embedding)
        lstm2 = Bidirectional(LSTM(32, dropout=0.3))(lstm1)
        text_features = Dense(32, activation='relu')(lstm2)
        text_features = Dropout(0.3)(text_features)
        
        # Numerical features input branch
        numerical_input = Input(shape=(numerical_features_dim,), name='numerical_input')
        numerical_dense1 = Dense(32, activation='relu')(numerical_input)
        numerical_dense1 = BatchNormalization()(numerical_dense1)
        numerical_dense1 = Dropout(0.3)(numerical_dense1)
        numerical_dense2 = Dense(16, activation='relu')(numerical_dense1)
        numerical_dense2 = Dropout(0.2)(numerical_dense2)
        
        # Combine both branches
        combined = Concatenate()([text_features, numerical_dense2])
        combined_dense1 = Dense(64, activation='relu')(combined)
        combined_dense1 = BatchNormalization()(combined_dense1)
        combined_dense1 = Dropout(0.3)(combined_dense1)
        
        combined_dense2 = Dense(32, activation='relu')(combined_dense1)
        combined_dense2 = Dropout(0.2)(combined_dense2)
        
        # Output layer (match score between 0 and 1)
        output = Dense(1, activation='sigmoid', name='match_score')(combined_dense2)
        
        # Create model
        model = Model(inputs=[text_input, numerical_input], outputs=output)
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_text, X_numerical, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the LSTM model"""
        logger.info("Starting model training...")
        
        # Split data
        indices = np.arange(len(X_text))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
        X_num_train, X_num_val = X_numerical[train_idx], X_numerical[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale numerical features
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        X_num_val_scaled = self.scaler.transform(X_num_val)
        
        # Build model
        self.model = self.build_model(X_numerical.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'lstm_resume_matcher_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            [X_text_train, X_num_train_scaled],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_text_val, X_num_val_scaled], y_val),
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        train_pred = self.model.predict([X_text_train, X_num_train_scaled])
        val_pred = self.model.predict([X_text_val, X_num_val_scaled])
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        logger.info(f"Training MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        logger.info(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.title('Model MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_components(self, filename_prefix='lstm_resume_matcher'):
        """Save the complete model and preprocessing components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = f"{filename_prefix}_{timestamp}.h5"
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save tokenizer
        tokenizer_path = f"{filename_prefix}_tokenizer_{timestamp}.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save scaler
        scaler_path = f"{filename_prefix}_scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature extractor
        extractor_path = f"{filename_prefix}_extractor_{timestamp}.pkl"
        with open(extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        logger.info(f"Feature extractor saved to {extractor_path}")
        
        return model_path, tokenizer_path, scaler_path, extractor_path
    
    def predict_match_score(self, resume_text, job_description):
        """Predict match score for a resume-job pair"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare text data
        combined_text = f"{resume_text} [SEP] {job_description}"
        sequence = self.tokenizer.texts_to_sequences([combined_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Extract numerical features
        resume_skills = self.feature_extractor.extract_skills(resume_text)
        job_skills = self.feature_extractor.extract_skills(job_description)
        resume_exp = self.feature_extractor.extract_experience_years(resume_text)
        job_exp = self.feature_extractor.extract_experience_years(job_description)
        resume_projects = self.feature_extractor.extract_project_count(resume_text)
        resume_education = self.feature_extractor.extract_education_level(resume_text)
        
        numerical_features = np.array([[
            len(resume_skills), len(job_skills),
            len(set(resume_skills) & set(job_skills)),
            len(set(resume_skills) | set(job_skills)),
            resume_exp, job_exp, resume_projects, resume_education,
            len(resume_text.split()), len(job_description.split())
        ]])
        
        numerical_features_scaled = self.scaler.transform(numerical_features)
        
        # Predict
        prediction = self.model.predict([padded_sequence, numerical_features_scaled])
        match_score = float(prediction[0][0]) * 100  # Convert to percentage
        
        return {
            'match_score': round(match_score, 2),
            'extracted_skills': resume_skills,
            'required_skills': job_skills,
            'skill_overlap': len(set(resume_skills) & set(job_skills)),
            'experience_years': resume_exp,
            'required_experience': job_exp,
            'projects_count': resume_projects,
            'education_level': resume_education
        }

def main():
    """Main training function"""
    logger.info("Starting LSTM Resume Matcher Training")
    
    # Initialize trainer
    trainer = LSTMResumeMatcherTrainer(max_features=15000, max_len=600)
    
    try:
        # Load and preprocess data
        ai_screening, updated_resumes, all_jobs = trainer.load_and_preprocess_data()
        
        # Create training pairs
        training_df = trainer.create_training_pairs(ai_screening, updated_resumes, all_jobs)
        
        # Prepare features
        X_text = trainer.prepare_text_data(training_df)
        X_numerical = trainer.extract_numerical_features(training_df)
        y = training_df['match_score'].values
        
        logger.info(f"Text features shape: {X_text.shape}")
        logger.info(f"Numerical features shape: {X_numerical.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target range: {y.min():.3f} - {y.max():.3f}")
        
        # Train model
        history = trainer.train_model(X_text, X_numerical, y, epochs=30, batch_size=32)
        
        # Save everything
        paths = trainer.save_model_and_components()
        logger.info("Training completed successfully!")
        logger.info(f"Saved files: {paths}")
        
        # Test prediction
        sample_resume = """
        Skills: Python, Machine Learning, TensorFlow, Scikit-learn, Pandas, NumPy
        Experience: 3 years of data science experience
        Education: M.Sc in Computer Science
        Projects: 5 machine learning projects completed
        """
        
        sample_job = """
        We are looking for a Data Scientist with 2+ years of experience in Python,
        Machine Learning, and statistical analysis. Must have experience with
        TensorFlow or PyTorch and data manipulation libraries like Pandas.
        """
        
        result = trainer.predict_match_score(sample_resume, sample_job)
        logger.info(f"Sample prediction: {result}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()