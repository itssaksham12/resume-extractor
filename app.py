#!/usr/bin/env python3
"""
Fallback entry point for Render deployment
Redirects to the proper start script for compatibility
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set critical environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

if __name__ == "__main__":
    print("🔄 app.py: Redirecting to start_render.py for proper initialization...")
    
    # Import and run the proper start script
    from start_render import *
