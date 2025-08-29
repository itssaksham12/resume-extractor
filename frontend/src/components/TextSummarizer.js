import React, { useState } from 'react';
import {
  Container,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Box,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Divider,
  Chip,
} from '@mui/material';
import {
  Summarize as SummaryIcon,
  AutoFixHigh as AutoIcon,
  Speed as SpeedIcon,
  CompareArrows as CompareIcon,
} from '@mui/icons-material';
import { summarizerAPI, handleAPIError } from '../services/api';

const TextSummarizer = () => {
  const [inputText, setInputText] = useState('');
  const [maxSentences, setMaxSentences] = useState(3);
  const [textType, setTextType] = useState('general');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter text to summarize');
      return;
    }

    if (inputText.trim().length < 50) {
      setError('Please enter at least 50 characters of text');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await summarizerAPI.summarizeText(inputText, maxSentences, textType);
      setResult(data);
    } catch (err) {
      setError(handleAPIError(err, 'Failed to summarize text'));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setResult(null);
    setError(null);
  };

  const sampleTexts = {
    job_description: `We are seeking a highly skilled Senior Software Engineer to join our dynamic development team. The ideal candidate will have extensive experience in full-stack development, with proficiency in modern JavaScript frameworks such as React and Node.js. You will be responsible for designing and implementing scalable web applications, collaborating with cross-functional teams, and mentoring junior developers. The role requires strong problem-solving skills, experience with cloud technologies like AWS or Azure, and familiarity with agile development methodologies. We offer competitive compensation, comprehensive benefits, flexible work arrangements, and opportunities for professional growth in a fast-paced, innovative environment.`,
    
    resume: `John Smith is a seasoned software engineer with over 8 years of experience in developing enterprise-level applications. He has extensive expertise in JavaScript, Python, and Java, with particular strength in React, Node.js, and Django frameworks. Throughout his career, John has led multiple successful projects, including the development of a microservices architecture that improved system performance by 40%. He holds a Master's degree in Computer Science from MIT and has experience working in agile environments. John is passionate about clean code, test-driven development, and has contributed to several open-source projects. He is seeking new challenges in a senior engineering role where he can leverage his technical skills and leadership experience.`,
    
    general: `Artificial Intelligence has rapidly transformed various industries over the past decade, revolutionizing how businesses operate and deliver services. Machine learning algorithms have become increasingly sophisticated, enabling computers to perform tasks that were once thought to be exclusively human. From healthcare diagnostics to autonomous vehicles, AI applications are now ubiquitous in our daily lives. Natural language processing has made significant strides, allowing for more intuitive human-computer interactions through voice assistants and chatbots. The ethical implications of AI development have also gained prominence, with researchers and policymakers working together to establish guidelines for responsible AI implementation. As we move forward, the integration of AI with other emerging technologies like blockchain and quantum computing promises to unlock even more innovative solutions to complex global challenges.`
  };

  const getCompressionColor = (ratio) => {
    if (ratio >= 70) return 'success';
    if (ratio >= 50) return 'warning';
    return 'info';
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <SummaryIcon fontSize="large" color="primary" />
        Text Summarization
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Generate concise summaries of job descriptions, resumes, or any text using AI-powered extractive summarization.
      </Typography>

      {/* Input Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Text Input */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📝 Input Text
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={12}
                label="Enter text to summarize"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your text here (job description, resume, article, etc.)..."
                variant="outlined"
                sx={{ mb: 2 }}
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Characters: {inputText.length} | Words: {inputText.split(/\s+/).filter(word => word.length > 0).length}
                </Typography>
                
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleClear}
                  disabled={loading}
                >
                  Clear
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Configuration */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ⚙️ Configuration
              </Typography>
              
              {/* Text Type */}
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>Text Type</InputLabel>
                <Select
                  value={textType}
                  label="Text Type"
                  onChange={(e) => setTextType(e.target.value)}
                >
                  <MenuItem value="general">General Text</MenuItem>
                  <MenuItem value="job_description">Job Description</MenuItem>
                  <MenuItem value="resume">Resume</MenuItem>
                </Select>
              </FormControl>
              
              {/* Max Sentences */}
              <Typography variant="body2" gutterBottom>
                Summary Length: {maxSentences} sentence{maxSentences !== 1 ? 's' : ''}
              </Typography>
              <Slider
                value={maxSentences}
                onChange={(e, newValue) => setMaxSentences(newValue)}
                min={1}
                max={10}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' }
                ]}
                valueLabelDisplay="auto"
                sx={{ mb: 3 }}
              />
              
              {/* Sample Texts */}
              <Typography variant="body2" gutterBottom sx={{ fontWeight: 'bold' }}>
                📋 Try Sample Texts:
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {Object.entries(sampleTexts).map(([type, text]) => (
                  <Button
                    key={type}
                    variant="outlined"
                    size="small"
                    onClick={() => {
                      setInputText(text);
                      setTextType(type === 'general' ? 'general' : type);
                    }}
                    sx={{ textTransform: 'none' }}
                  >
                    {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Button>
                ))}
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              {/* Summarize Button */}
              <Button
                variant="contained"
                fullWidth
                size="large"
                onClick={handleSummarize}
                disabled={loading || !inputText.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <AutoIcon />}
              >
                {loading ? 'Summarizing...' : 'Generate Summary'}
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {result && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📄 Generated Summary
            </Typography>
            
            {/* Summary Statistics */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" color="primary.main">
                    {result.original_length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Original Words
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" color="success.main">
                    {result.summary_length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Summary Words
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" color={`${getCompressionColor(result.compression_ratio)}.main`}>
                    {result.compression_ratio.toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Compression
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <SpeedIcon color="action" />
                  <Box>
                    <Typography variant="h6">
                      {result.processing_time?.toFixed(2)}s
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Processing
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </Grid>

            {/* Model Information */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <Chip
                label={result.model_used}
                color="primary"
                variant="outlined"
                size="small"
              />
              <Chip
                label={`${maxSentences} sentence${maxSentences !== 1 ? 's' : ''}`}
                color="secondary"
                variant="outlined"
                size="small"
              />
              <Chip
                label={textType.replace('_', ' ')}
                color="success"
                variant="outlined"
                size="small"
              />
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Summary Text */}
            <Paper sx={{ p: 3, bgcolor: 'grey.50' }}>
              <Typography variant="body1" sx={{ lineHeight: 1.8, fontSize: '1.1rem' }}>
                {result.summary}
              </Typography>
            </Paper>

            {/* Comparison */}
            <Box sx={{ mt: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
              <CompareIcon color="primary" />
              <Typography variant="body2" color="text.secondary">
                Reduced from {result.original_length} to {result.summary_length} words 
                ({result.compression_ratio.toFixed(1)}% compression)
              </Typography>
            </Box>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default TextSummarizer;
