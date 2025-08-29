import axios from 'axios';

// API base URL - automatically handles development vs production
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for model processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response?.status === 422) {
      // Validation errors
      const details = error.response.data?.detail;
      if (Array.isArray(details)) {
        const messages = details.map(d => d.msg).join(', ');
        error.userMessage = `Validation Error: ${messages}`;
      } else {
        error.userMessage = 'Please check your input and try again.';
      }
    } else if (error.response?.status === 500) {
      error.userMessage = 'Server error. Please try again later.';
    } else if (error.code === 'ECONNABORTED') {
      error.userMessage = 'Request timeout. Please try again.';
    } else {
      error.userMessage = error.response?.data?.detail || 'Something went wrong.';
    }
    
    return Promise.reject(error);
  }
);

// Health check API
export const healthAPI = {
  check: async () => {
    const response = await api.get('/');
    return response.data;
  },
};

// Skills extraction API
export const skillsAPI = {
  extractSkills: async (text, useAIModel = true, threshold = 0.3) => {
    const response = await api.post('/api/extract-skills', {
      text,
      use_ai_model: useAIModel,
      threshold,
    });
    return response.data;
  },
};

// Resume analysis API
export const resumeAPI = {
  analyzeResume: async (resumeText, jobDescription, useAIModel = true) => {
    const response = await api.post('/api/analyze-resume', {
      resume_text: resumeText,
      job_description: jobDescription,
      use_ai_model: useAIModel,
    });
    return response.data;
  },
  
  uploadPDF: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/upload-pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 1 minute for file upload
    });
    return response.data;
  },
};

// Text summarization API
export const summarizerAPI = {
  summarizeText: async (text, maxSentences = 3, textType = 'general') => {
    const response = await api.post('/api/summarize', {
      text,
      max_sentences: maxSentences,
      text_type: textType,
    });
    return response.data;
  },
};

// Utility function to handle API errors consistently
export const handleAPIError = (error, defaultMessage = 'An error occurred') => {
  if (error.userMessage) {
    return error.userMessage;
  }
  
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }
  
  return defaultMessage;
};

// Export the base API instance for custom requests
export default api;
