#!/usr/bin/env python3
"""
Start backend with timeout and better error handling
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Set environment variables
env_vars = {
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TOKENIZERS_PARALLELISM': 'false',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'KMP_INIT_AT_FORK': 'FALSE',
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    'TF_ENABLE_ONEDNN_OPTS': '0'
}

def start_backend():
    """Start backend with timeout monitoring"""
    
    print("🚀 Starting Resume Extractor API with timeout monitoring...")
    
    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    # Start the backend process
    try:
        process = subprocess.Popen(
            [sys.executable, "start.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("⏱️ Monitoring startup (60 second timeout)...")
        
        startup_timeout = 60
        start_time = time.time()
        
        while True:
            # Check if process finished
            if process.poll() is not None:
                print("❌ Backend process exited unexpectedly")
                return False
            
            # Check timeout
            if time.time() - start_time > startup_timeout:
                print("⏰ Startup timeout reached")
                print("🔄 Backend may be hung on model loading")
                print("💡 Try running: python check_models.py")
                
                # Kill the process
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                return False
            
            # Read output
            try:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    
                    # Check for successful startup
                    if "Application startup complete" in line or "Uvicorn running on" in line:
                        print("✅ Backend started successfully!")
                        print("🌐 API available at: http://localhost:8000")
                        print("📚 API docs at: http://localhost:8000/docs")
                        return True
                        
            except Exception as e:
                print(f"Error reading output: {e}")
                break
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Startup cancelled by user")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

if __name__ == "__main__":
    success = start_backend()
    if not success:
        print("\n🔧 Troubleshooting tips:")
        print("1. Check model files: python check_models.py")
        print("2. Check dependencies: pip list | grep -E 'torch|tensorflow'")
        print("3. Try with reduced models (edit backend/main.py)")
        sys.exit(1)
