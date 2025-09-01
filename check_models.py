#!/usr/bin/env python3
"""
Script to check for model files and their locations
"""

import os
from pathlib import Path

def check_model_files():
    """Check for model files in various locations"""
    
    print("🔍 Checking for model files...")
    print("="*50)
    
    # Define model files to look for
    model_files = [
        "bert_skills_model.pth",
        "bert_summarizer_model.pth", 
        "lstm_resume_matcher_best.h5",
        "lstm_resume_matcher_20250828_113623.h5"
    ]
    
    # Define search directories
    search_dirs = [
        Path("."),  # Current directory
        Path(".."),  # Parent directory
        Path("models"),  # Models directory
        Path("app"),  # App directory
        Path("backend"),  # Backend directory
    ]
    
    found_files = {}
    
    for model_file in model_files:
        print(f"\n📄 Looking for: {model_file}")
        found_locations = []
        
        for search_dir in search_dirs:
            model_path = search_dir / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                found_locations.append(f"{model_path} ({size_mb:.1f} MB)")
        
        if found_locations:
            print(f"  ✅ Found in:")
            for location in found_locations:
                print(f"    - {location}")
            found_files[model_file] = found_locations
        else:
            print(f"  ❌ Not found")
    
    # Check for supporting LSTM files
    print(f"\n📚 Looking for LSTM supporting files...")
    lstm_files = [
        "lstm_resume_matcher_tokenizer_20250828_113623.pkl",
        "lstm_resume_matcher_scaler_20250828_113623.pkl", 
        "lstm_resume_matcher_extractor_20250828_113623.pkl"
    ]
    
    for lstm_file in lstm_files:
        found = False
        for search_dir in search_dirs:
            lstm_path = search_dir / lstm_file
            if lstm_path.exists():
                size_kb = lstm_path.stat().st_size / 1024
                print(f"  ✅ {lstm_file}: {lstm_path} ({size_kb:.1f} KB)")
                found = True
                break
        if not found:
            print(f"  ❌ {lstm_file}: Not found")
    
    print("\n" + "="*50)
    print(f"📊 Summary: Found {len(found_files)} out of {len(model_files)} model files")
    
    if found_files:
        print(f"\n✅ Available models:")
        for model, locations in found_files.items():
            print(f"  - {model}")
    
    print(f"\n💡 Tip: Move model files to the root directory for easier access")
    
    return found_files

if __name__ == "__main__":
    check_model_files()
