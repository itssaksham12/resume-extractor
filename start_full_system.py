#!/usr/bin/env python3
"""
Start the full system with all heavy models - aggressive TensorFlow fixes
"""

import os
import sys
from pathlib import Path

# CRITICAL: Set ALL environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# Aggressive TensorFlow fixes for macOS mutex lock
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=false'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

# PyTorch settings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Change to backend directory
backend_dir = Path(__file__).parent / "backend"
os.chdir(backend_dir)

if __name__ == "__main__":
    print("🚀 Starting Resume Extractor with FULL AI MODELS")
    print("🔧 All TensorFlow mutex lock fixes applied")
    print("⚡ Heavy models will load (BERT + LSTM)")
    print("⏱️ This may take 1-2 minutes...")
    
    import uvicorn
    from main import app
    
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=30
    )
