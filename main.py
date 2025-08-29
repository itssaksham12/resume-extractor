#!/usr/bin/env python3
"""
Main entry point for Resume Extractor API on Render
Redirects to the backend FastAPI application
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change working directory to backend
os.chdir(backend_dir)

# Import and run the FastAPI app
from main import app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=port)
