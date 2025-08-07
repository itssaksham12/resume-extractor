#!/usr/bin/env python3
"""
Setup script to ensure RAKE-NLTK is available for the resume matcher application.
This script should be run before building the PyInstaller executable.
"""

import subprocess
import sys
import os

def install_rake_nltk():
    """Install RAKE-NLTK if not already installed."""
    try:
        from rake_nltk import Rake
        print("✓ RAKE-NLTK is already installed and working.")
        return True
    except ImportError:
        print("✗ RAKE-NLTK not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "rake-nltk"], 
                         check=True, capture_output=True, text=True)
            print("✓ RAKE-NLTK installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install RAKE-NLTK: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error installing RAKE-NLTK: {e}")
            return False

def install_nltk_stopwords():
    """Install NLTK stopwords if not already installed."""
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
        print("✓ NLTK stopwords are already installed.")
        return True
    except LookupError:
        print("✗ NLTK stopwords not found. Installing...")
        try:
            nltk.download('stopwords')
            print("✓ NLTK stopwords installed successfully!")
            return True
        except Exception as e:
            print(f"✗ Failed to install NLTK stopwords: {e}")
            return False

def verify_installation():
    """Verify that RAKE-NLTK is properly installed and working."""
    try:
        from rake_nltk import Rake
        import nltk
        print(f"✓ NLTK version: {nltk.__version__}")
        
        # Test RAKE functionality
        r = Rake()
        r.extract_keywords_from_text("This is a test sentence for RAKE keyword extraction.")
        keywords = r.get_ranked_phrases()
        print(f"✓ RAKE-NLTK working successfully. Test extracted {len(keywords)} keywords")
        return True
    except Exception as e:
        print(f"✗ RAKE-NLTK verification failed: {e}")
        return False

def main():
    print("Setting up RAKE-NLTK for Resume Matcher Application")
    print("=" * 50)
    
    # Install RAKE-NLTK
    if not install_rake_nltk():
        print("\n❌ Setup failed. Please install RAKE-NLTK manually:")
        print("   pip install rake-nltk")
        sys.exit(1)
    
    # Install NLTK stopwords
    if not install_nltk_stopwords():
        print("\n❌ NLTK stopwords setup failed.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Verification failed. Please check your RAKE-NLTK installation.")
        sys.exit(1)
    
    print("\n✅ RAKE-NLTK setup completed successfully!")
    print("You can now build the PyInstaller executable.")

if __name__ == "__main__":
    main() 