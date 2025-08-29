#!/bin/bash

# Resume Extractor - Deployment Script for Render
# This script helps deploy the FastAPI + React application to Render

set -e  # Exit on any error

echo "🚀 Resume Extractor - Render Deployment Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "render.yaml" ]; then
    print_error "render.yaml not found. Make sure you're in the project root directory."
    exit 1
fi

print_status "Starting deployment preparation..."

# 1. Check model files
print_status "Checking model files..."

MODELS_MISSING=0

if [ ! -f "lstm_resume_matcher_best.h5" ]; then
    print_warning "LSTM model file not found: lstm_resume_matcher_best.h5"
    MODELS_MISSING=1
fi

if [ ! -f "app/bert_summarizer_model.pth" ]; then
    print_warning "BERT summarizer model not found: app/bert_summarizer_model.pth"
    MODELS_MISSING=1
fi

# Check support files
if [ ! -f "lstm_resume_matcher_tokenizer_20250828_113623.pkl" ]; then
    print_warning "LSTM tokenizer file not found"
    MODELS_MISSING=1
fi

if [ $MODELS_MISSING -eq 1 ]; then
    print_warning "Some model files are missing. The application will use fallback methods."
    print_warning "For full functionality, ensure all model files are present."
fi

# 2. Validate backend requirements
print_status "Validating backend requirements..."

if [ ! -f "backend/requirements.txt" ]; then
    print_error "Backend requirements.txt not found"
    exit 1
fi

if [ ! -f "backend/main.py" ]; then
    print_error "Backend main.py not found"
    exit 1
fi

# 3. Validate frontend files
print_status "Validating frontend files..."

if [ ! -f "frontend/package.json" ]; then
    print_error "Frontend package.json not found"
    exit 1
fi

if [ ! -f "frontend/src/App.js" ]; then
    print_error "Frontend App.js not found"
    exit 1
fi

# 4. Create models directory structure
print_status "Creating models directory structure..."

mkdir -p backend/models
mkdir -p frontend/build

# Copy model files to backend directory if they exist
if [ -f "lstm_resume_matcher_best.h5" ]; then
    cp lstm_resume_matcher_best.h5 backend/
    print_status "Copied LSTM model to backend directory"
fi

if [ -f "app/bert_summarizer_model.pth" ]; then
    cp app/bert_summarizer_model.pth backend/
    print_status "Copied BERT summarizer to backend directory"
fi

# Copy support files
for file in lstm_resume_matcher_*.pkl; do
    if [ -f "$file" ]; then
        cp "$file" backend/
        print_status "Copied support file: $file"
    fi
done

# 5. Update backend imports to handle model locations
print_status "Backend is configured to handle multiple model paths..."

# 6. Create .renderignore files
print_status "Creating .renderignore files..."

cat > .renderignore << EOF
# Development files
*.log
*.pyc
__pycache__/
.DS_Store
.env
.venv/
venv/

# Documentation
*.md
docs/

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/

# Old streamlit files
streamlit_app*.py
run_streamlit.py
main_streamlit_app.py

# Large datasets (keep models)
*.csv
resume_samples.txt
normlized_classes.txt
EOF

# 7. Display deployment information
print_status "Deployment preparation complete!"
echo ""
echo "📋 Deployment Summary:"
echo "======================="
echo "• Backend: FastAPI application with full ML model support"
echo "• Frontend: React application with Material-UI"
echo "• Models: $(ls -1 backend/*.h5 backend/*.pth 2>/dev/null | wc -l) model files copied"
echo "• Architecture: Microservices (separate backend/frontend)"
echo ""
echo "🔧 Next Steps for Render Deployment:"
echo "===================================="
echo "1. Commit all changes to your Git repository"
echo "2. Push to GitHub/GitLab"
echo "3. Connect your repository to Render"
echo "4. The render.yaml file will automatically configure both services"
echo ""
echo "📊 Service Configuration:"
echo "• Backend API: resume-extractor-api.onrender.com"
echo "• Frontend App: resume-extractor-frontend.onrender.com"
echo "• CORS configured for cross-origin requests"
echo ""
echo "⚠️  Important Notes:"
echo "• First deployment may take 10-15 minutes due to model loading"
echo "• Backend requires at least 512MB RAM (starter plan recommended)"
echo "• Models will be loaded on startup - check logs for status"
echo ""

if [ $MODELS_MISSING -eq 1 ]; then
    print_warning "Some AI models are missing. App will work with reduced functionality."
    echo "• Skills extraction: Rule-based fallback"
    echo "• Resume analysis: Basic matching"
    echo "• Text summarization: Rule-based"
fi

print_status "Deployment script completed successfully!"
echo "🚀 Ready for Render deployment!"
