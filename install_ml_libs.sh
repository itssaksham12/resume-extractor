#!/bin/bash

echo "🔧 Installing ML libraries for Python 3.13..."

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing PyTorch (CPU version)..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "📦 Installing TensorFlow..."
pip install tensorflow==2.20.0

echo "📦 Installing other ML libraries..."
pip install transformers==4.36.0
pip install scikit-learn
pip install --upgrade numpy

echo "📦 Installing remaining backend requirements..."
pip install -r backend/requirements.txt

echo "✅ All dependencies installed! Now try:"
echo "python start.py"
