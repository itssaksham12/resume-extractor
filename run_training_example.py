#!/usr/bin/env python3
"""
Example script to run the LSTM training and demonstrate usage
"""

import sys
import os
from lstm_resume_matcher import LSTMResumeMatcherTrainer
from resume_matcher_predictor import ResumeMatcherPredictor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training_example():
    """Run a quick training example with reduced data size"""
    logger.info("Starting LSTM Resume Matcher Training Example")
    
    # Initialize trainer with smaller parameters for quick testing
    trainer = LSTMResumeMatcherTrainer(max_features=5000, max_len=300)
    
    try:
        # Load and preprocess data
        ai_screening, updated_resumes, all_jobs = trainer.load_and_preprocess_data()
        
        # Create smaller training set for quick testing
        logger.info("Creating smaller training set for example...")
        ai_screening_small = ai_screening.head(200)  # Use only first 200 records
        updated_resumes_small = updated_resumes.head(500)  # Use only first 500 records
        
        training_df = trainer.create_training_pairs(
            ai_screening_small, 
            updated_resumes_small, 
            all_jobs.head(1000),  # Use only first 1000 jobs
            num_pairs=2000  # Create only 2000 pairs for quick training
        )
        
        logger.info(f"Created {len(training_df)} training pairs")
        
        # Prepare features
        X_text = trainer.prepare_text_data(training_df)
        X_numerical = trainer.extract_numerical_features(training_df)
        y = training_df['match_score'].values
        
        logger.info(f"Text features shape: {X_text.shape}")
        logger.info(f"Numerical features shape: {X_numerical.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Train model with fewer epochs for quick example
        history = trainer.train_model(
            X_text, X_numerical, y, 
            epochs=10,  # Reduced epochs for quick training
            batch_size=32
        )
        
        # Save model
        paths = trainer.save_model_and_components()
        logger.info("Training completed successfully!")
        logger.info(f"Saved files: {paths}")
        
        return trainer, paths
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def test_prediction(model_paths=None):
    """Test prediction with sample data"""
    logger.info("Testing prediction...")
    
    # Initialize predictor
    if model_paths:
        predictor = ResumeMatcherPredictor(
            model_path=model_paths[0],
            tokenizer_path=model_paths[1],
            scaler_path=model_paths[2],
            extractor_path=model_paths[3]
        )
    else:
        # Use latest saved model
        predictor = ResumeMatcherPredictor()
    
    # Test cases
    test_cases = [
        {
            'name': 'High Match Data Scientist',
            'resume': """
            Skills: Python, Machine Learning, TensorFlow, Scikit-learn, Pandas, NumPy, SQL, AWS
            Experience: 4 years of data science experience at tech companies
            Education: M.Sc in Computer Science from Stanford University
            Projects: Led 8 machine learning projects including recommendation systems and NLP models
            Certifications: AWS Certified Machine Learning Specialist, Google ML Engineer
            """,
            'job': """
            We are seeking a Data Scientist with 3+ years of experience in Python,
            Machine Learning, and statistical analysis. Must have experience with
            TensorFlow or PyTorch, data manipulation libraries like Pandas, and
            cloud platforms. Experience with NLP and recommendation systems preferred.
            Strong SQL skills required.
            """
        },
        {
            'name': 'Moderate Match Software Engineer',
            'resume': """
            Skills: Java, Spring Boot, React, JavaScript, HTML, CSS, MySQL
            Experience: 2 years of full-stack development experience
            Education: B.Tech in Computer Science
            Projects: Built 3 web applications using Java and React
            """,
            'job': """
            Looking for a Senior Software Engineer with 5+ years of experience in
            Python, Django, PostgreSQL, and AWS. Must have experience with
            microservices architecture and containerization using Docker.
            Machine learning experience is a plus.
            """
        },
        {
            'name': 'Entry Level Match',
            'resume': """
            Skills: Python, HTML, CSS, JavaScript, Git
            Experience: Fresh graduate with internship experience
            Education: B.Sc in Computer Science
            Projects: 2 academic projects in web development
            """,
            'job': """
            Entry-level Python Developer position. Looking for candidates with
            basic Python knowledge, willingness to learn, and some project experience.
            Fresh graduates welcome. Training will be provided.
            """
        }
    ]
    
    print("\n" + "="*80)
    print("RESUME MATCHING TEST RESULTS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-'*60}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'-'*60}")
        
        try:
            result = predictor.predict_single_match(test_case['resume'], test_case['job'])
            
            print(f"Overall Match Score: {result['overall_match_score']}%")
            print(f"Skill Match Score: {result['skill_match_score']}%")
            print(f"Experience Match Score: {result['experience_match_score']}%")
            print(f"Recommendation: {result['recommendation']}")
            print(f"Matching Skills: {', '.join(result['matching_skills'])}")
            print(f"Missing Skills: {', '.join(result['missing_skills'][:5])}")  # Show first 5
            print(f"Experience: {result['candidate_experience']} years (Required: {result['required_experience']})")
            print(f"Projects: {result['projects_count']}")
            
            # Get improvement suggestions
            gap_analysis = predictor.analyze_skill_gaps(test_case['resume'], test_case['job'])
            print(f"\nTop Improvement Suggestions:")
            for suggestion in gap_analysis['improvement_suggestions'][:3]:
                print(f"  • {suggestion}")
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
    
    print(f"\n{'='*80}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Resume Matcher Example')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                       help='Mode: train model, test prediction, or both')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with reduced dataset size')
    
    args = parser.parse_args()
    
    model_paths = None
    
    if args.mode in ['train', 'both']:
        logger.info("Starting training...")
        if args.quick:
            logger.info("Using quick training mode with reduced dataset")
        trainer, model_paths = run_training_example()
    
    if args.mode in ['test', 'both']:
        logger.info("Starting prediction test...")
        test_prediction(model_paths)
    
    logger.info("Example completed!")

if __name__ == "__main__":
    main()