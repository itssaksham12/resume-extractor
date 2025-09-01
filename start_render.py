#!/usr/bin/env python3
"""
Production start script for Render deployment
Optimized for Render's environment with proper port binding
"""

import os
import sys
from pathlib import Path

# Set critical environment variables BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# Disable multithreading for compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Change to backend directory for model loading
backend_dir = Path(__file__).parent / "backend"
os.chdir(backend_dir)

if __name__ == "__main__":
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"  # Required for Render
    
    print(f"🚀 Starting Resume Extractor API for Render")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path}")
    
    # Check if model files exist
    import glob
    model_files = glob.glob("*.pth") + glob.glob("*.h5") + glob.glob("../*.pth") + glob.glob("../*.h5")
    print(f"🎯 Model files found: {model_files}")
    
    # Import and run with uvicorn
    import uvicorn
    from main import app
    
    # Production configuration for Render - optimized for immediate port binding
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        loop="asyncio",
        workers=1,  # Single worker for ML models
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
