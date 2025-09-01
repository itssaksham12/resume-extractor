# Resume Extractor

AI-powered resume analysis system with React frontend and FastAPI backend.

## Quick Start

### Backend
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Start backend server
python start.py
```
Backend runs on http://localhost:8000

### Frontend
```bash
# Install Node dependencies and start React app
chmod +x start_frontend.sh
./start_frontend.sh
```
Frontend runs on http://localhost:3000

## Features

- **Skills Extraction**: BERT-powered skill identification
- **Resume Analysis**: LSTM-based matching algorithm  
- **Text Summarization**: AI-powered resume summaries
- **ChromaDB Integration**: Vector database for resumes
- **React Frontend**: Modern web interface
- **FastAPI Backend**: Production-ready REST API

## Tech Stack

- **Frontend**: React, Material-UI
- **Backend**: FastAPI, Python
- **AI Models**: BERT, LSTM, PyTorch, TensorFlow
- **Database**: ChromaDB
- **Deployment**: AWS-ready

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.