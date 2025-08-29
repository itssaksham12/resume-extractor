# 🚀 Resume Reviewer - Streamlit Deployment Guide

## Overview

This guide covers the deployment of the Resume Reviewer application, which has been successfully migrated from Gradio to Streamlit for optimal deployment performance.

## ✅ Migration Completed

### What's Changed
- **Migrated from Gradio to Streamlit** for better deployment capabilities
- **Optimized imports** with graceful fallbacks for missing components
- **Enhanced UI** with modern Streamlit components
- **Improved caching** using `@st.cache_resource` for better performance
- **Clean architecture** with modular functions and error handling

### Features Included
1. **🎯 Skills Extraction** - Extract skills from job descriptions using BERT or rule-based methods
2. **📊 Resume Analysis** - Match resumes against job descriptions with LSTM scoring
3. **📝 Text Summarization** - Generate summaries using BERT or rule-based algorithms

## 📁 Project Structure

```
resume-extractor-clean/
├── streamlit_app.py              # Main Streamlit application
├── run_streamlit.py              # Production deployment script
├── health_check.py               # Deployment health check
├── requirements.txt              # Optimized dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── bert_skills_extractor.py     # AI skills extraction
├── bert_summarizer.py           # AI text summarization  
├── lstm_resume_matcher.py       # AI resume matching
└── enhanced_models/             # Model files directory
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Health Check

```bash
python health_check.py
```

### 3. Start Application

#### Option A: Simple Start
```bash
streamlit run streamlit_app.py
```

#### Option B: Production Start
```bash
python run_streamlit.py
```

#### Option C: Custom Configuration
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Production Server
```bash
python run_streamlit.py
```
This script includes:
- Environment configuration
- Graceful shutdown handling
- Error logging
- Production optimizations

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "run_streamlit.py"]
```

Build and run:
```bash
docker build -t resume-reviewer .
docker run -p 8501:8501 resume-reviewer
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

#### Heroku
Create `Procfile`:
```
web: python run_streamlit.py
```

#### AWS/GCP/Azure
Use the provided `run_streamlit.py` with your preferred container service.

## ⚙️ Configuration

### Environment Variables
```bash
export PORT=8501                    # Server port
export HOST=0.0.0.0                # Server host
export STREAMLIT_SERVER_HEADLESS=true
export TF_CPP_MIN_LOG_LEVEL=2       # Reduce TensorFlow logs
export TOKENIZERS_PARALLELISM=false # Disable tokenizer warnings
```

### Streamlit Config (`.streamlit/config.toml`)
The application includes optimized Streamlit configuration for production deployment.

## 🤖 Model Requirements

### Required Models
- `bert_skills_extractor.py` - Core skills extraction (✅ included)
- `bert_summarizer.py` - Text summarization (✅ included)
- `lstm_resume_matcher.py` - Resume matching (✅ included)

### Optional Model Files
- `bert_skills_model.pth` - Trained BERT skills model
- `bert_summarizer_model.pth` - Trained summarization model
- `lstm_resume_matcher_best.h5` - Trained LSTM matching model

**Note**: The application gracefully falls back to rule-based methods if AI models are not available.

## 🔧 Performance Optimizations

### Implemented Optimizations
1. **Model Caching** - `@st.cache_resource` for model loading
2. **Lazy Loading** - Models loaded only when needed
3. **Error Handling** - Graceful fallbacks for missing components
4. **Memory Management** - Optimized imports and dependencies
5. **UI Optimization** - Efficient Streamlit components

### Recommended Server Specs
- **Minimum**: 2 CPU cores, 4GB RAM
- **Recommended**: 4 CPU cores, 8GB RAM
- **With All AI Models**: 8GB+ RAM recommended

## 📊 Monitoring

### Health Check Endpoint
Run health checks:
```bash
python health_check.py
```

### Application Logs
The application provides detailed logging for:
- Model loading status
- Feature availability
- Error handling
- Performance metrics

## 🔒 Security Considerations

### Production Security
- CORS disabled for production
- XSRF protection configured
- Error details hidden in production
- Usage stats disabled

### File Upload Security
- PDF files are processed in memory
- No persistent file storage
- Input validation on all text fields

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: Missing dependencies
**Solution**: 
```bash
pip install -r requirements.txt
python health_check.py
```

#### 2. Model Loading Errors
**Problem**: AI models not found
**Solution**: Application falls back to rule-based methods automatically

#### 3. Memory Issues
**Problem**: High memory usage
**Solution**: 
- Restart application periodically
- Monitor memory usage
- Consider reducing model complexity

#### 4. Port Already in Use
**Problem**: Port 8501 is occupied
**Solution**:
```bash
export PORT=8502
python run_streamlit.py
```

### Debug Mode
Enable debug logging:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run streamlit_app.py
```

## 📈 Scaling

### Horizontal Scaling
- Use load balancer with multiple instances
- Each instance is stateless
- Models are cached per instance

### Vertical Scaling
- Increase RAM for better model performance
- More CPU cores for concurrent users
- SSD storage for faster model loading

## 🎯 Best Practices

### Deployment Best Practices
1. **Use health checks** before deployment
2. **Monitor resource usage** regularly
3. **Implement graceful shutdowns**
4. **Use environment-specific configurations**
5. **Regular model updates** for better accuracy

### Performance Best Practices
1. **Warm up models** on startup
2. **Use caching** for repeated requests
3. **Monitor memory usage**
4. **Implement request timeouts**
5. **Use CDN** for static assets

## 🆘 Support

### Getting Help
1. Check application logs
2. Run health check script
3. Review troubleshooting section
4. Check model file availability

### Performance Monitoring
- Monitor CPU and memory usage
- Track response times
- Monitor error rates
- Check model accuracy metrics

---

## 🎉 Success!

Your Resume Reviewer application is now fully migrated to Streamlit and ready for deployment. The application includes:

- ✅ **Complete feature migration** from Gradio
- ✅ **Production-ready optimizations**
- ✅ **Deployment scripts and configuration**
- ✅ **Health monitoring tools**
- ✅ **Comprehensive documentation**

Start the application with `python run_streamlit.py` and enjoy your upgraded Resume Reviewer! 🚀
