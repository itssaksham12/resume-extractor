#!/usr/bin/env python3
"""
Resume Matcher Predictor - Use trained LSTM model for predictions
Loads the trained model and provides easy-to-use prediction functions.
"""

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import json
from datetime import datetime
import os
import glob
import logging
import sys

# Add the current directory to Python path so pickle can find the classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the feature extractor class to make it available for pickle
try:
    from lstm_resume_matcher import AdvancedFeatureExtractor
except ImportError:
    # If import fails, define a minimal version for compatibility
    class AdvancedFeatureExtractor:
        def __init__(self):
            self.technical_skills = []
            self.soft_skills = []
            self.experience_patterns = []
            self.project_patterns = []
        
        def extract_skills(self, text):
            return []
        
        def extract_experience_years(self, text):
            return 0
        
        def extract_project_count(self, text):
            return 0
        
        def extract_education_level(self, text):
            return 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeMatcherPredictor:
    """Predictor class for trained LSTM resume matcher"""
    
    def __init__(self, model_path=None, tokenizer_path=None, scaler_path=None, extractor_path=None):
        """
        Initialize predictor with model components
        If paths are None, will try to find the latest saved model
        """
        if model_path is None:
            model_path = self.find_latest_model()
        if tokenizer_path is None:
            tokenizer_path = self.find_latest_component('tokenizer')
        if scaler_path is None:
            scaler_path = self.find_latest_component('scaler')
        if extractor_path is None:
            extractor_path = self.find_latest_component('extractor')
        
        logger.info(f"Loading model from: {model_path}")
        try:
            self.model = load_model(model_path)
        except ValueError as e:
            if "Could not deserialize" in str(e):
                logger.warning("Model loading failed due to version compatibility. Loading with custom_objects...")
                # Load model without compile to avoid metric deserialization issues
                self.model = load_model(model_path, compile=False)
                # Recompile with compatible metrics
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                self.model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae', 'mse']
                )
                logger.info("Model loaded and recompiled successfully!")
            else:
                raise e
        
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        logger.info(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Loading feature extractor from: {extractor_path}")
        try:
            with open(extractor_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)
            logger.info("Feature extractor loaded successfully!")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Could not load feature extractor: {e}")
            logger.info("Creating new feature extractor...")
            self.feature_extractor = AdvancedFeatureExtractor()
            logger.info("New feature extractor created!")
        
        # Get max_len from model input shape
        self.max_len = self.model.input[0].shape[1]
        
        logger.info("Model loaded successfully!")
    
    def find_latest_model(self):
        """Find the latest saved model file"""
        model_files = glob.glob("lstm_resume_matcher_*.h5")
        if not model_files:
            raise FileNotFoundError("No model files found. Please train the model first.")
        return max(model_files, key=os.path.getctime)
    
    def find_latest_component(self, component_type):
        """Find the latest saved component (tokenizer, scaler, extractor)"""
        component_files = glob.glob(f"lstm_resume_matcher_{component_type}_*.pkl")
        if not component_files:
            raise FileNotFoundError(f"No {component_type} files found. Please train the model first.")
        return max(component_files, key=os.path.getctime)
    
    def predict_single_match(self, resume_text, job_description):
        """Predict match score for a single resume-job pair"""
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
        
        skill_intersection = set(resume_skills) & set(job_skills)
        skill_union = set(resume_skills) | set(job_skills)
        
        numerical_features = np.array([[
            len(resume_skills),           # Number of skills in resume
            len(job_skills),             # Number of skills required
            len(skill_intersection),     # Skill overlap
            len(skill_union),            # Total unique skills
            resume_exp,                  # Resume experience years
            job_exp,                     # Required experience years
            resume_projects,             # Number of projects
            resume_education,            # Education level
            len(resume_text.split()),    # Resume length
            len(job_description.split()) # Job description length
        ]])
        
        numerical_features_scaled = self.scaler.transform(numerical_features)
        
        # Predict
        prediction = self.model.predict([padded_sequence, numerical_features_scaled], verbose=0)
        match_score = float(prediction[0][0]) * 100  # Convert to percentage
        
        # Calculate additional metrics
        skill_match_percentage = (len(skill_intersection) / len(skill_union) * 100) if skill_union else 0
        experience_match = (resume_exp / job_exp) if job_exp > 0 else 1.0
        experience_match_percentage = min(experience_match * 100, 100)
        
        return {
            'overall_match_score': round(match_score, 2),
            'skill_match_score': round(skill_match_percentage, 2),
            'experience_match_score': round(experience_match_percentage, 2),
            'extracted_skills': resume_skills,
            'required_skills': job_skills,
            'matching_skills': list(skill_intersection),
            'missing_skills': list(set(job_skills) - set(resume_skills)),
            'extra_skills': list(set(resume_skills) - set(job_skills)),
            'candidate_experience': resume_exp,
            'required_experience': job_exp,
            'projects_count': resume_projects,
            'education_level': resume_education,
            'recommendation': self.get_recommendation(match_score)
        }
    
    def get_recommendation(self, match_score):
        """Get hiring recommendation based on match score"""
        if match_score >= 80:
            return "Strong Match - Highly Recommended"
        elif match_score >= 65:
            return "Good Match - Recommended"
        elif match_score >= 50:
            return "Moderate Match - Consider for Interview"
        elif match_score >= 35:
            return "Weak Match - May Consider with Additional Training"
        else:
            return "Poor Match - Not Recommended"
    
    def predict_batch(self, resume_job_pairs):
        """Predict match scores for multiple resume-job pairs"""
        results = []
        
        for i, (resume_text, job_description, metadata) in enumerate(resume_job_pairs):
            try:
                result = self.predict_single_match(resume_text, job_description)
                result.update(metadata)  # Add any additional metadata
                result['pair_id'] = i
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} resume-job pairs")
                    
            except Exception as e:
                logger.error(f"Error processing pair {i}: {str(e)}")
                results.append({
                    'pair_id': i,
                    'error': str(e),
                    'overall_match_score': 0,
                    **metadata
                })
        
        return results
    
    def rank_candidates(self, candidates, job_description, top_k=10):
        """Rank candidates for a specific job"""
        results = []
        
        for candidate in candidates:
            resume_text = candidate.get('resume_text', '')
            candidate_name = candidate.get('name', f'Candidate_{len(results)}')
            
            match_result = self.predict_single_match(resume_text, job_description)
            match_result['candidate_name'] = candidate_name
            match_result.update({k: v for k, v in candidate.items() if k != 'resume_text'})
            
            results.append(match_result)
        
        # Sort by overall match score
        results.sort(key=lambda x: x['overall_match_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results[:top_k]
    
    def analyze_skill_gaps(self, resume_text, job_description):
        """Analyze skill gaps and provide improvement suggestions"""
        result = self.predict_single_match(resume_text, job_description)
        
        # Skill gap analysis
        missing_critical_skills = []
        missing_nice_to_have = []
        
        # Categorize missing skills (this is a simplified approach)
        critical_keywords = ['python', 'java', 'sql', 'machine learning', 'deep learning']
        
        for skill in result['missing_skills']:
            if any(keyword in skill.lower() for keyword in critical_keywords):
                missing_critical_skills.append(skill)
            else:
                missing_nice_to_have.append(skill)
        
        # Experience gap
        exp_gap = max(0, result['required_experience'] - result['candidate_experience'])
        
        # Suggestions
        suggestions = []
        
        if missing_critical_skills:
            suggestions.append(f"Critical skills to develop: {', '.join(missing_critical_skills)}")
        
        if missing_nice_to_have:
            suggestions.append(f"Additional skills to consider: {', '.join(missing_nice_to_have[:3])}")
        
        if exp_gap > 0:
            suggestions.append(f"Gain {exp_gap} more years of relevant experience")
        
        if result['projects_count'] < 3:
            suggestions.append("Build more projects to demonstrate practical skills")
        
        return {
            'current_match_score': result['overall_match_score'],
            'missing_critical_skills': missing_critical_skills,
            'missing_nice_to_have': missing_nice_to_have,
            'experience_gap_years': exp_gap,
            'improvement_suggestions': suggestions,
            'potential_score_improvement': min(100, result['overall_match_score'] + len(missing_critical_skills) * 10)
        }

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Resume Matcher Predictor')
    parser.add_argument('--mode', choices=['single', 'batch', 'rank'], default='single',
                       help='Prediction mode')
    parser.add_argument('--resume', type=str, help='Resume text or file path')
    parser.add_argument('--job', type=str, help='Job description text or file path')
    parser.add_argument('--input_file', type=str, help='Input CSV file for batch processing')
    parser.add_argument('--output_file', type=str, help='Output file for results')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ResumeMatcherPredictor(model_path=args.model_path)
    
    if args.mode == 'single':
        # Single prediction
        if not args.resume or not args.job:
            print("Please provide both --resume and --job for single prediction")
            return
        
        # Read resume and job description
        resume_text = args.resume
        if os.path.isfile(args.resume):
            with open(args.resume, 'r', encoding='utf-8') as f:
                resume_text = f.read()
        
        job_text = args.job
        if os.path.isfile(args.job):
            with open(args.job, 'r', encoding='utf-8') as f:
                job_text = f.read()
        
        # Predict
        result = predictor.predict_single_match(resume_text, job_text)
        
        # Display results
        print("\n" + "="*50)
        print("RESUME MATCH ANALYSIS")
        print("="*50)
        print(f"Overall Match Score: {result['overall_match_score']}%")
        print(f"Skill Match Score: {result['skill_match_score']}%")
        print(f"Experience Match Score: {result['experience_match_score']}%")
        print(f"Recommendation: {result['recommendation']}")
        print(f"\nMatching Skills ({len(result['matching_skills'])}): {', '.join(result['matching_skills'])}")
        print(f"Missing Skills ({len(result['missing_skills'])}): {', '.join(result['missing_skills'])}")
        print(f"Extra Skills ({len(result['extra_skills'])}): {', '.join(result['extra_skills'])}")
        print(f"\nCandidate Experience: {result['candidate_experience']} years")
        print(f"Required Experience: {result['required_experience']} years")
        print(f"Projects: {result['projects_count']}")
        print(f"Education Level: {result['education_level']}")
        
        # Skill gap analysis
        gap_analysis = predictor.analyze_skill_gaps(resume_text, job_text)
        print(f"\n" + "-"*30)
        print("IMPROVEMENT SUGGESTIONS")
        print("-"*30)
        for suggestion in gap_analysis['improvement_suggestions']:
            print(f"• {suggestion}")
        print(f"\nPotential Score with Improvements: {gap_analysis['potential_score_improvement']}%")
        
    elif args.mode == 'batch':
        # Batch processing
        if not args.input_file:
            print("Please provide --input_file for batch processing")
            return
        
        # Read input file
        df = pd.read_csv(args.input_file)
        
        # Prepare pairs
        pairs = []
        for _, row in df.iterrows():
            pairs.append((
                row['resume_text'],
                row['job_description'],
                {'candidate_id': row.get('candidate_id', ''), 'job_id': row.get('job_id', '')}
            ))
        
        # Predict
        results = predictor.predict_batch(pairs)
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = args.output_file or f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    else:
        print("Invalid mode. Use --mode single, batch, or rank")

if __name__ == "__main__":
    main()