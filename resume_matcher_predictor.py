import pandas as pd
import numpy as np
import re
import pickle
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class ResumeMatcherPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.feature_extractor = None
        self.load_models()
    
    def load_models(self):
        """Load the trained LSTM model and supporting components"""
        try:
            # Try to load the best model
            model_paths = [
                "lstm_resume_matcher_best.h5",
                os.path.join(os.path.dirname(__file__), "lstm_resume_matcher_best.h5"),
                os.path.join(os.path.dirname(__file__), "..", "lstm_resume_matcher_best.h5")
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.model = keras.models.load_model(model_path)
                        print(f"LSTM model loaded from: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load model from {model_path}: {e}")
                        continue
            
            if not model_loaded:
                print("LSTM model not found")
                return
            
            # Load supporting components (use only the new retrained components)
            component_paths = [
                "lstm_resume_matcher_extractor_20250822_150311.pkl",
                "lstm_resume_matcher_scaler_20250822_150311.pkl", 
                "lstm_resume_matcher_tokenizer_20250822_150311.pkl"
            ]
            
            for component_path in component_paths:
                full_paths = [
                    component_path,
                    os.path.join(os.path.dirname(__file__), component_path),
                    os.path.join(os.path.dirname(__file__), "..", component_path)
                ]
                
                component_loaded = False
                for full_path in full_paths:
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, 'rb') as f:
                                component = pickle.load(f)
                            
                            if 'extractor' in component_path:
                                self.feature_extractor = component
                            elif 'scaler' in component_path:
                                self.scaler = component
                            elif 'tokenizer' in component_path:
                                self.tokenizer = component
                            
                            print(f"Component loaded from: {full_path}")
                            component_loaded = True
                            break
                        except Exception as e:
                            print(f"Failed to load component from {full_path}: {e}")
                            continue
                
                if not component_loaded:
                    print(f"Warning: Could not load {component_path}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def extract_features(self, resume_text: str, job_description: str) -> Dict:
        """Extract features from resume and job description"""
        features = {}
        
        # Basic text features
        features['resume_length'] = len(resume_text)
        features['jd_length'] = len(job_description)
        
        # Experience extraction
        experience_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s*)?(?:the\s*)?(?:field|industry|role)'
        ]
        
        candidate_experience = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text.lower())
            if matches:
                candidate_experience = max([int(match) for match in matches])
                break
        
        required_experience = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, job_description.lower())
            if matches:
                required_experience = max([int(match) for match in matches])
                break
        
        features['candidate_experience'] = candidate_experience
        features['required_experience'] = required_experience
        features['experience_gap'] = required_experience - candidate_experience
        
        # Project count
        project_patterns = [
            r'project',
            r'developed',
            r'built',
            r'created',
            r'implemented'
        ]
        
        project_count = 0
        for pattern in project_patterns:
            project_count += len(re.findall(pattern, resume_text.lower()))
        
        features['projects_count'] = project_count
        
        # Education level
        education_levels = {
            'phd': 4,
            'doctorate': 4,
            'master': 3,
            'bachelor': 2,
            'associate': 1,
            'diploma': 1,
            'high school': 0
        }
        
        education_level = 0
        for level, score in education_levels.items():
            if level in resume_text.lower():
                education_level = max(education_level, score)
        
        features['education_level'] = education_level
        
        # Skills matching
        skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'nodejs',
            'sql', 'mongodb', 'aws', 'docker', 'kubernetes', 'git', 'agile',
            'machine learning', 'tensorflow', 'pytorch', 'scikit-learn'
        ]
        
        resume_skills = []
        jd_skills = []
        
        for skill in skills:
            if skill in resume_text.lower():
                resume_skills.append(skill)
            if skill in job_description.lower():
                jd_skills.append(skill)
        
        matching_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        extra_skills = list(set(resume_skills) - set(jd_skills))
        
        features['matching_skills'] = matching_skills
        features['missing_skills'] = missing_skills
        features['extra_skills'] = extra_skills
        features['skill_match_ratio'] = len(matching_skills) / max(len(jd_skills), 1)
        
        return features
    
    def predict_single_match(self, resume_text: str, job_description: str) -> Dict:
        """Predict match score for a single resume-job pair"""
        try:
            # Extract features
            features = self.extract_features(resume_text, job_description)
            
            # If model is not loaded, return basic analysis
            if self.model is None:
                return self._basic_analysis(features)
            
            # Prepare input for model
            if self.tokenizer and self.scaler:
                # Tokenize and pad sequences
                resume_seq = self.tokenizer.texts_to_sequences([resume_text])
                jd_seq = self.tokenizer.texts_to_sequences([job_description])
                
                # Pad sequences
                max_len = 1000  # Adjust based on your model
                resume_padded = tf.keras.preprocessing.sequence.pad_sequences(resume_seq, maxlen=max_len)
                jd_padded = tf.keras.preprocessing.sequence.pad_sequences(jd_seq, maxlen=max_len)
                
                # Combine features
                numerical_features = np.array([
                    features['resume_length'],
                    features['jd_length'],
                    features['candidate_experience'],
                    features['required_experience'],
                    features['experience_gap'],
                    features['projects_count'],
                    features['education_level'],
                    features['skill_match_ratio']
                ]).reshape(1, -1)
                
                # Scale features
                if self.scaler:
                    numerical_features = self.scaler.transform(numerical_features)
                
                # Make prediction
                prediction = self.model.predict([resume_padded, jd_padded, numerical_features])
                overall_score = float(prediction[0][0] * 100)
                
            else:
                # Fallback to basic analysis
                return self._basic_analysis(features)
            
            # Calculate component scores
            skill_score = features['skill_match_ratio'] * 100
            experience_score = max(0, 100 - abs(features['experience_gap']) * 20)
            
            # Generate recommendation
            if overall_score >= 80:
                recommendation = "Strong match! This candidate appears to be an excellent fit for the position."
            elif overall_score >= 60:
                recommendation = "Good match. Consider this candidate with some training or skill development."
            elif overall_score >= 40:
                recommendation = "Moderate match. The candidate has some relevant skills but may need significant development."
            else:
                recommendation = "Weak match. Consider looking for candidates with more relevant experience and skills."
            
            return {
                'overall_match_score': round(overall_score, 1),
                'skill_match_score': round(skill_score, 1),
                'experience_match_score': round(experience_score, 1),
                'candidate_experience': features['candidate_experience'],
                'required_experience': features['required_experience'],
                'projects_count': features['projects_count'],
                'education_level': features['education_level'],
                'matching_skills': features['matching_skills'],
                'missing_skills': features['missing_skills'],
                'extra_skills': features['extra_skills'],
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._basic_analysis(self.extract_features(resume_text, job_description))
    
    def _basic_analysis(self, features: Dict) -> Dict:
        """Basic analysis when model is not available"""
        skill_score = features['skill_match_ratio'] * 100
        experience_score = max(0, 100 - abs(features['experience_gap']) * 20)
        overall_score = (skill_score + experience_score) / 2
        
        if overall_score >= 80:
            recommendation = "Strong match based on skills and experience analysis."
        elif overall_score >= 60:
            recommendation = "Good match with some areas for improvement."
        elif overall_score >= 40:
            recommendation = "Moderate match, consider skill development."
        else:
            recommendation = "Weak match, significant skill gaps identified."
        
        return {
            'overall_match_score': round(overall_score, 1),
            'skill_match_score': round(skill_score, 1),
            'experience_match_score': round(experience_score, 1),
            'candidate_experience': features['candidate_experience'],
            'required_experience': features['required_experience'],
            'projects_count': features['projects_count'],
            'education_level': features['education_level'],
            'matching_skills': features['matching_skills'],
            'missing_skills': features['missing_skills'],
            'extra_skills': features['extra_skills'],
            'recommendation': recommendation
        }
