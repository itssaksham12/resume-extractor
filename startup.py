#!/usr/bin/env python3
"""
Startup script for Resume Extractor on Render
Handles NLTK downloads and starts the app
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        logger.info("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ NLTK download failed: {e}")

def main():
    """Main startup function"""
    logger.info("🚀 Starting Resume Extractor API...")
    
    # Download NLTK data
    download_nltk_data()
    
    # Import and run the app
    try:
        from app import app
        import uvicorn
        
        # Get port from environment
        port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"🌐 Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        logger.error(f"❌ Failed to start app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
