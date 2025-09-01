#!/usr/bin/env python3
"""
Complete fix script - locate models, test TensorFlow, and run system
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up all environment variables"""
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
        'TF_DISABLE_MKL': '1',
        'TF_DISABLE_POOL_ALLOCATOR': '1',
        'TF_XLA_FLAGS': '--tf_xla_cpu_global_jit=false',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Environment variables set")

def locate_models():
    """Quick model location check"""
    print("🔍 Checking for model files...")
    
    model_files = [
        "bert_skills_model.pth",
        "bert_summarizer_model.pth", 
        "lstm_resume_matcher_best.h5"
    ]
    
    found = 0
    for model_file in model_files:
        if Path(model_file).exists():
            size_mb = Path(model_file).stat().st_size / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
            found += 1
        else:
            print(f"  ❌ {model_file} - Not found")
    
    if found == 0:
        print("⚠️ No model files found in root directory")
        print("🔧 Running model locator...")
        try:
            subprocess.run([sys.executable, "locate_models.py"], check=True)
        except:
            print("❌ Model locator failed")
    
    return found

def test_tensorflow():
    """Test TensorFlow in subprocess"""
    print("🧪 Testing TensorFlow...")
    
    try:
        result = subprocess.run([sys.executable, "test_tensorflow.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if "ALL TESTS PASSED" in result.stdout:
            print("✅ TensorFlow tests passed")
            return True
        else:
            print("❌ TensorFlow tests failed")
            print("Output:", result.stdout)
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ TensorFlow test timed out (mutex lock likely)")
        return False
    except Exception as e:
        print(f"❌ TensorFlow test error: {e}")
        return False

def start_backend():
    """Start the backend with all fixes"""
    print("🚀 Starting backend...")
    
    try:
        # Start backend process
        process = subprocess.Popen(
            [sys.executable, "start_full_system.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("⏱️ Monitoring startup (90 second timeout)...")
        
        import time
        start_time = time.time()
        
        while True:
            if process.poll() is not None:
                print("❌ Backend process exited")
                return False
            
            if time.time() - start_time > 90:
                print("⏰ Startup timeout - killing process")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                return False
            
            try:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    
                    if "Uvicorn running on" in line or "Application startup complete" in line:
                        print("✅ Backend started successfully!")
                        print("🌐 API: http://localhost:8000")
                        print("📚 Docs: http://localhost:8000/docs")
                        print("🎨 Frontend: http://localhost:3000")
                        return True
                        
            except Exception as e:
                print(f"Error: {e}")
                break
            
            time.sleep(0.1)
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def main():
    """Main execution"""
    print("🔧 Complete Resume Extractor Fix & Run")
    print("="*50)
    
    # 1. Setup environment
    setup_environment()
    
    # 2. Locate models
    model_count = locate_models()
    
    # 3. Test TensorFlow
    tf_ok = test_tensorflow()
    
    if not tf_ok:
        print("\n❌ TensorFlow mutex lock issue detected")
        print("💡 Suggested fixes:")
        print("   1. pip uninstall tensorflow && pip install tensorflow==2.13.0")
        print("   2. Restart terminal and try again")
        print("   3. Use Docker for isolation")
        return False
    
    # 4. Start backend
    success = start_backend()
    
    if success:
        print("\n🎉 System running successfully!")
        print("✅ Backend: http://localhost:8000")
        print("✅ Frontend: http://localhost:3000")
        print(f"✅ Models found: {model_count}/3")
    else:
        print("\n❌ Failed to start system")
        print("🔍 Check logs above for errors")
    
    return success

if __name__ == "__main__":
    main()
