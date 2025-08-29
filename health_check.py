#!/usr/bin/env python3
"""
Health check script for the Resume Reviewer Streamlit app
"""

import sys
import os
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        logger.error(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.8+")
        return False

def check_core_dependencies():
    """Check if core dependencies can be imported"""
    dependencies = {
        'streamlit': 'Streamlit web framework',
        'torch': 'PyTorch machine learning',
        'tensorflow': 'TensorFlow machine learning', 
        'sklearn': 'Scikit-learn machine learning',
        'nltk': 'Natural Language Toolkit',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'PyPDF2': 'PDF processing'
    }
    
    results = {}
    for dep, description in dependencies.items():
        try:
            importlib.import_module(dep)
            logger.info(f"✅ {dep}: {description}")
            results[dep] = True
        except ImportError:
            logger.error(f"❌ {dep}: {description} - NOT FOUND")
            results[dep] = False
    
    return results

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'bert_skills_extractor.py',
        'bert_summarizer.py', 
        'lstm_resume_matcher.py'
    ]
    
    results = {}
    for file in model_files:
        if os.path.exists(file):
            logger.info(f"✅ Model file: {file}")
            results[file] = True
        else:
            logger.warning(f"⚠️ Model file: {file} - NOT FOUND (optional)")
            results[file] = False
    
    return results

def check_streamlit_app():
    """Check if the main Streamlit app can be imported"""
    try:
        # Try to compile the streamlit app
        import py_compile
        py_compile.compile('streamlit_app.py', doraise=True)
        logger.info("✅ Streamlit app syntax is valid")
        return True
    except py_compile.PyCompileError as e:
        logger.error(f"❌ Streamlit app has syntax errors: {e}")
        return False
    except FileNotFoundError:
        logger.error("❌ streamlit_app.py not found")
        return False

def check_configuration():
    """Check Streamlit configuration"""
    config_file = '.streamlit/config.toml'
    if os.path.exists(config_file):
        logger.info("✅ Streamlit configuration file found")
        return True
    else:
        logger.warning("⚠️ Streamlit configuration file not found (using defaults)")
        return False

def main():
    """Run all health checks"""
    logger.info("🔍 Starting Resume Reviewer Health Check...")
    logger.info("=" * 50)
    
    checks = []
    
    # Check Python version
    checks.append(("Python Version", check_python_version()))
    
    # Check core dependencies
    logger.info("\n📦 Checking Core Dependencies...")
    dep_results = check_core_dependencies()
    checks.append(("Core Dependencies", all(dep_results.values())))
    
    # Check model files
    logger.info("\n🤖 Checking Model Files...")
    model_results = check_model_files()
    checks.append(("Model Files", any(model_results.values())))
    
    # Check Streamlit app
    logger.info("\n🚀 Checking Streamlit App...")
    checks.append(("Streamlit App", check_streamlit_app()))
    
    # Check configuration
    logger.info("\n⚙️ Checking Configuration...")
    checks.append(("Configuration", check_configuration()))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 HEALTH CHECK SUMMARY")
    logger.info("=" * 50)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("🎉 ALL CHECKS PASSED - Ready for deployment!")
        return 0
    else:
        logger.error("⚠️ Some checks failed - Review issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
