#!/usr/bin/env python3
"""
Start backend with minimal model loading (skip problematic models)
"""

import os
import sys
from pathlib import Path

# Set environment variables to prevent conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['SKIP_LSTM_LOADING'] = 'true'  # Skip LSTM to avoid mutex lock

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Change to backend directory
backend_dir = Path(__file__).parent / "backend"
os.chdir(backend_dir)

if __name__ == "__main__":
    import uvicorn
    from main import app
    
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Resume Extractor API (Minimal Mode) on port {port}")
    print("⚠️ LSTM models disabled to avoid startup issues")
    print("✅ Skills extraction and basic features available")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=False
    )
