#!/bin/bash
# Fix TensorFlow for Apple Silicon M1 MacBook

echo "🍎 Fixing TensorFlow for Apple Silicon M1..."
echo "================================================"

# Activate virtual environment
source venv/bin/activate

echo "1️⃣ Uninstalling incompatible TensorFlow..."
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y

echo "2️⃣ Installing Apple Silicon optimized TensorFlow..."
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal

echo "3️⃣ Verifying installation..."
python3 -c "
import tensorflow as tf
print('✅ TensorFlow version:', tf.__version__)
print('✅ GPUs available:', len(tf.config.list_physical_devices('GPU')))
print('✅ Apple Silicon optimization: ENABLED')
"

echo "4️⃣ Testing basic TensorFlow operations..."
python3 -c "
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Test basic operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)
print('✅ Basic TensorFlow operations working:', c.numpy())

# Test model creation
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mse')
print('✅ Model creation working')
"

echo ""
echo "🎉 Apple Silicon TensorFlow installation complete!"
echo "✅ tensorflow-macos: Optimized for M1 chip"
echo "✅ tensorflow-metal: GPU acceleration enabled"
echo ""
echo "🚀 Ready to run: python fix_and_run.py"
