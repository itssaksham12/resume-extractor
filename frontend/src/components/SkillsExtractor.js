import React, { useState } from 'react';
import {
  Container,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Chip,
  Grid,
  FormControlLabel,
  Switch,
  CircularProgress,
  Alert,
  Box,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Paper,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Psychology as PsychologyIcon,
  Speed as SpeedIcon,
  Category as CategoryIcon,
} from '@mui/icons-material';
import { skillsAPI, handleAPIError } from '../services/api';

const SkillsExtractor = () => {
  const [jobDescription, setJobDescription] = useState('');
  const [useAIModel, setUseAIModel] = useState(true);
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleExtractSkills = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await skillsAPI.extractSkills(jobDescription, useAIModel, threshold);
      setResult(data);
    } catch (err) {
      setError(handleAPIError(err, 'Failed to extract skills'));
    } finally {
      setLoading(false);
    }
  };

  const handleClearInput = () => {
    setJobDescription('');
    setResult(null);
    setError(null);
  };

  const getCategoryDisplayName = (category) => {
    return category
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getCategoryColor = (category) => {
    const colors = {
      programming_languages: 'primary',
      web_technologies: 'secondary',
      databases: 'success',
      data_science_ml: 'warning',
      cloud_devops: 'info',
      mobile_development: 'error',
      soft_skills: 'default',
      tools_frameworks: 'primary'
    };
    return colors[category] || 'default';
  };

  const sampleJobDescriptions = [
    "We are looking for a Python developer with experience in Django, React, and PostgreSQL for our web application team.",
    "Senior Data Scientist needed with expertise in Machine Learning, TensorFlow, Python, and AWS for our AI initiatives.",
    "Frontend Developer position requires React, JavaScript, HTML, CSS, and Node.js experience for building modern web applications."
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <PsychologyIcon fontSize="large" color="primary" />
        Skills Extraction
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Extract required skills from job descriptions using AI-powered analysis or rule-based methods.
      </Typography>

      {/* Input Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Job Description Input
          </Typography>
          
          <TextField
            fullWidth
            multiline
            rows={8}
            label="Enter Job Description"
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the job description here to extract required skills..."
            sx={{ mb: 3 }}
            variant="outlined"
          />
          
          {/* Sample Job Descriptions */}
          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" color="text.secondary">
                📝 Try Sample Job Descriptions
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {sampleJobDescriptions.map((sample, index) => (
                  <Grid item xs={12} key={index}>
                    <Paper 
                      sx={{ p: 2, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
                      onClick={() => setJobDescription(sample)}
                    >
                      <Typography variant="body2">{sample}</Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Configuration Options */}
          <Box sx={{ mb: 3 }}>
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useAIModel}
                      onChange={(e) => setUseAIModel(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Use AI Model"
                />
              </Grid>
              
              {useAIModel && (
                <Grid item xs={12} md={8}>
                  <Typography variant="body2" gutterBottom>
                    Confidence Threshold: {threshold}
                  </Typography>
                  <Slider
                    value={threshold}
                    onChange={(e, newValue) => setThreshold(newValue)}
                    min={0.1}
                    max={0.9}
                    step={0.1}
                    marks={[
                      { value: 0.1, label: '0.1' },
                      { value: 0.5, label: '0.5' },
                      { value: 0.9, label: '0.9' }
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Grid>
              )}
            </Grid>
          </Box>

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              onClick={handleExtractSkills}
              disabled={loading || !jobDescription.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
              size="large"
            >
              {loading ? 'Extracting...' : 'Extract Skills'}
            </Button>
            
            <Button
              variant="outlined"
              onClick={handleClearInput}
              disabled={loading}
            >
              Clear
            </Button>
          </Box>
        </CardContent>
      </Card>

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
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CategoryIcon color="primary" />
              Extraction Results
            </Typography>
            
            {/* Summary Stats */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4" color="primary.main">
                    {result.total_skills}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Skills
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4" color="success.main">
                    {Object.keys(result.skill_categories).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Categories
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4" color="warning.main">
                    {(result.confidence * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence
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
                      Processing Time
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            </Grid>

            <Typography variant="body2" color="text.secondary" gutterBottom>
              Model Used: <strong>{result.model_used}</strong>
            </Typography>
            
            <Divider sx={{ my: 2 }} />

            {/* All Skills */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                All Extracted Skills ({result.skills.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {result.skills.map((skill, index) => (
                  <Chip 
                    key={index} 
                    label={skill} 
                    color="primary" 
                    variant="outlined"
                    size="medium"
                  />
                ))}
              </Box>
            </Box>

            {/* Skills by Category */}
            <Typography variant="h6" gutterBottom>
              Skills by Category
            </Typography>
            
            {Object.keys(result.skill_categories).length > 0 ? (
              <Grid container spacing={2}>
                {Object.entries(result.skill_categories).map(([category, skills]) => (
                  <Grid item xs={12} md={6} key={category}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography 
                          variant="subtitle1" 
                          gutterBottom 
                          color={`${getCategoryColor(category)}.main`}
                          sx={{ fontWeight: 'bold' }}
                        >
                          {getCategoryDisplayName(category)} ({skills.length})
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {skills.map((skill, index) => (
                            <Chip 
                              key={index} 
                              label={skill} 
                              size="small" 
                              color={getCategoryColor(category)}
                              variant="filled"
                            />
                          ))}
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Alert severity="info">
                No categorized skills found. All skills are listed above.
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default SkillsExtractor;
