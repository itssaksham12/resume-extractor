#!/usr/bin/env python3
"""
Alternative Gunicorn-based start script for Render deployment
Use this if uvicorn direct start has issues
"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Change to backend directory
backend_dir = Path(__file__).parent / "backend"
os.chdir(backend_dir)

if __name__ == "__main__":
    # Get port from environment
    port = int(os.environ.get("PORT", 10000))
    
    print(f"🚀 Starting Resume Extractor API with Gunicorn on port {port}")
    
    # Run with gunicorn
    import subprocess
    import signal
    
    gunicorn_cmd = [
        "gunicorn",
        "main:app",
        "--bind", f"0.0.0.0:{port}",
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--workers", "1",
        "--timeout", "120",
        "--max-requests", "1000",
        "--max-requests-jitter", "100",
        "--preload",
        "--log-level", "info"
    ]
    
    try:
        process = subprocess.run(gunicorn_cmd)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Gracefully shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting with Gunicorn: {e}")
        sys.exit(1)
