#!/bin/bash

echo "🔧 Fixing ML dependencies for Python 3.13 on macOS..."

# Activate virtual environment
source venv/bin/activate

echo "📦 Uninstalling problematic ML libraries..."
pip uninstall torch tensorflow transformers scikit-learn -y

echo "📥 Installing compatible versions..."

# Install latest PyTorch for CPU (compatible with Python 3.13)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install latest TensorFlow (compatible with Python 3.13)
pip install tensorflow==2.20.0

# Install other ML libraries (use latest compatible versions)
pip install transformers==4.36.0
pip install scikit-learn==1.4.0
pip install numpy>=1.26.0

# Install other requirements
pip install -r backend/requirements.txt

echo "✅ Dependencies fixed! Now try running:"
echo "python start.py"
