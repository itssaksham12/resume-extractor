#!/usr/bin/env python3
"""
Test TensorFlow import and configuration to diagnose mutex lock issues
"""

import os
import sys

# Set all environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

def test_tensorflow():
    """Test TensorFlow import and basic operations"""
    
    print("🧪 Testing TensorFlow import...")
    
    try:
        print("1️⃣ Importing TensorFlow...")
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully: {tf.__version__}")
        
        print("2️⃣ Configuring TensorFlow...")
        # Disable GPU
        tf.config.set_visible_devices([], 'GPU')
        print("✅ GPU disabled")
        
        # Set threading
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        print("✅ Single-threaded mode set")
        
        # Test basic operation
        print("3️⃣ Testing basic TensorFlow operation...")
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✅ Basic operation successful: {c.numpy()}")
        
        print("4️⃣ Testing model loading capability...")
        # Create a simple model to test model operations
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model creation successful")
        
        # Test model saving/loading
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model.save(tmp.name)
            loaded_model = tf.keras.models.load_model(tmp.name)
            print("✅ Model save/load successful")
            os.unlink(tmp.name)
        
        print("\n🎉 ALL TENSORFLOW TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lstm_imports():
    """Test importing LSTM-related components"""
    
    print("\n🧪 Testing LSTM component imports...")
    
    try:
        print("1️⃣ Importing lstm_resume_matcher...")
        from lstm_resume_matcher import LSTMResumeMatcherTrainer, AdvancedFeatureExtractor
        print("✅ LSTM components imported successfully")
        
        print("2️⃣ Creating instances...")
        trainer = LSTMResumeMatcherTrainer()
        extractor = AdvancedFeatureExtractor()
        print("✅ LSTM instances created successfully")
        
        print("\n🎉 ALL LSTM IMPORT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ LSTM import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 TensorFlow and LSTM Diagnostic Test")
    print("="*50)
    
    # Test TensorFlow
    tf_success = test_tensorflow()
    
    if tf_success:
        # Test LSTM imports
        lstm_success = test_lstm_imports()
        
        if lstm_success:
            print("\n✅ ALL TESTS PASSED - Your system should work!")
            print("🚀 Try running: python start_full_system.py")
        else:
            print("\n⚠️ LSTM imports failed - check lstm_resume_matcher.py")
    else:
        print("\n❌ TensorFlow tests failed - mutex lock or configuration issue")
        print("💡 Try installing a different TensorFlow version:")
        print("   pip uninstall tensorflow")
        print("   pip install tensorflow==2.13.0")
    
    print("\n" + "="*50)
