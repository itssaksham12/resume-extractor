#!/usr/bin/env python3
"""
Build script for the Resume Matcher Application.
This script handles spaCy setup and PyInstaller building.
"""

import subprocess
import sys
import os
import shutil

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def setup_dependencies():
    """Set up RAKE-NLTK dependencies."""
    print("Setting up RAKE-NLTK dependencies...")
    
    # Install RAKE-NLTK if not already installed
    try:
        import rake_nltk
    except ImportError:
        print("RAKE-NLTK not found. Installing...")
        if not run_command(f"{sys.executable} -m pip install rake-nltk", "Installing RAKE-NLTK"):
            return False
    
    # Install NLTK if not already installed
    try:
        import nltk
    except ImportError:
        print("NLTK not found. Installing...")
        if not run_command(f"{sys.executable} -m pip install nltk", "Installing NLTK"):
            return False
    
    # Download NLTK stopwords
    if not run_command(f"{sys.executable} -c \"import nltk; nltk.download('stopwords')\"", "Downloading NLTK stopwords"):
        return False
    
    # Verify the installation
    try:
        from rake_nltk import Rake
        import nltk
        r = Rake()
        r.extract_keywords_from_text("This is a test sentence for RAKE.")
        print("✅ RAKE-NLTK verified successfully")
        return True
    except Exception as e:
        print(f"❌ RAKE-NLTK verification failed: {e}")
        return False

def clean_build():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))
                print(f"Removed {os.path.join(root, file)}")

def build_executable():
    """Build the PyInstaller executable."""
    print("\n🔨 Building PyInstaller executable...")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        if not run_command(f"{sys.executable} -m pip install pyinstaller", "Installing PyInstaller"):
            return False
    
    # Build the executable
    if not run_command("pyinstaller app.spec", "Building executable with PyInstaller"):
        return False
    
    return True

def verify_build():
    """Verify the built executable."""
    print("\n🔍 Verifying built executable...")
    
    executable_path = "dist/app"
    if not os.path.exists(executable_path):
        print(f"❌ Executable not found at {executable_path}")
        return False
    
    print(f"✅ Executable found at {executable_path}")
    print(f"📁 Executable size: {os.path.getsize(executable_path) / (1024*1024):.2f} MB")
    
    return True

def main():
    print("🚀 Resume Matcher Application Builder")
    print("=" * 50)
    
    # Step 1: Set up RAKE-NLTK dependencies
    if not setup_dependencies():
        print("\n❌ RAKE-NLTK setup failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Clean previous builds
    clean_build()
    
    # Step 3: Build executable
    if not build_executable():
        print("\n❌ Build failed. Exiting.")
        sys.exit(1)
    
    # Step 4: Verify build
    if not verify_build():
        print("\n❌ Build verification failed. Exiting.")
        sys.exit(1)
    
    print("\n🎉 Build completed successfully!")
    print(f"📦 Your executable is ready at: {os.path.abspath('dist/app')}")
    print("\nTo run the application:")
    print("   ./dist/app")
    print("\n💡 Note: RAKE-NLTK provides lightweight keyword extraction without complex model dependencies!")

if __name__ == "__main__":
    main() 