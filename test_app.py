#!/usr/bin/env python3
"""
Simple test script to verify the Streamlit app works
"""

import os
import sys

# Set environment variables to prevent mutex issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported")
    except Exception as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported")
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    return True

def test_streamlit_app():
    """Test if the streamlit app can be imported"""
    print("\n🔍 Testing Streamlit app import...")
    
    try:
        import streamlit_app
        print("✅ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Resume Reviewer - Compatibility Test")
    print("=" * 50)
    
    # Test basic imports
    if not test_imports():
        print("\n❌ Basic imports failed")
        return False
    
    # Test streamlit app
    if not test_streamlit_app():
        print("\n❌ Streamlit app import failed")
        return False
    
    print("\n✅ All tests passed! The app should work.")
    print("\n🎯 To run the app:")
    print("   streamlit run streamlit_app.py")
    print("   OR")
    print("   python run_streamlit.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

