#!/usr/bin/env python3
"""
Script to locate model files and move them to correct locations
"""

import os
import shutil
from pathlib import Path

def locate_and_organize_models():
    """Locate model files and organize them properly"""
    
    print("🔍 Locating model files...")
    
    # Define model files to look for
    model_files = {
        "bert_skills_model.pth": "BERT Skills Model",
        "bert_summarizer_model.pth": "BERT Summarizer Model", 
        "lstm_resume_matcher_best.h5": "LSTM Resume Matcher",
        "lstm_resume_matcher_20250828_113623.h5": "LSTM Resume Matcher (Timestamped)"
    }
    
    # Supporting files
    support_files = [
        "lstm_resume_matcher_tokenizer_20250828_113623.pkl",
        "lstm_resume_matcher_scaler_20250828_113623.pkl",
        "lstm_resume_matcher_extractor_20250828_113623.pkl"
    ]
    
    all_files = list(model_files.keys()) + support_files
    
    # Search entire directory tree
    found_files = {}
    
    print("📂 Searching directory tree...")
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != '__pycache__']
        
        for file in files:
            if file in all_files:
                file_path = Path(root) / file
                size_mb = file_path.stat().st_size / (1024 * 1024)
                found_files[file] = {
                    'path': file_path,
                    'size_mb': size_mb
                }
                print(f"  ✅ Found: {file} ({size_mb:.1f} MB) at {file_path}")
    
    print(f"\n📊 Found {len(found_files)} model files")
    
    # Move/copy files to root directory for easier access
    target_dir = Path(".")
    moved_files = []
    
    for filename, info in found_files.items():
        source_path = info['path']
        target_path = target_dir / filename
        
        if source_path != target_path:
            try:
                # Copy file to root (don't move to preserve original)
                shutil.copy2(source_path, target_path)
                print(f"📋 Copied {filename} to root directory")
                moved_files.append(filename)
            except Exception as e:
                print(f"❌ Failed to copy {filename}: {e}")
    
    print(f"\n✅ Organized {len(moved_files)} model files in root directory")
    
    # Verify organization
    print("\n🔍 Verification - Files in root directory:")
    for filename in model_files.keys():
        file_path = Path(".") / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {filename} - NOT FOUND")
    
    return found_files

if __name__ == "__main__":
    locate_and_organize_models()
