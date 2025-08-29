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
  LinearProgress,
  Paper,
  Divider,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  CloudUpload as UploadIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Add as AddIcon,
  Person as PersonIcon,
  Work as WorkIcon,
  School as SchoolIcon,
  Code as CodeIcon,
} from '@mui/icons-material';
import { resumeAPI, handleAPIError } from '../services/api';

const ResumeAnalyzer = () => {
  const [tabValue, setTabValue] = useState(0);
  const [jobDescription, setJobDescription] = useState('');
  const [resumeText, setResumeText] = useState('');
  const [useAIModel, setUseAIModel] = useState(true);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleAnalyzeResume = async () => {
    if (!jobDescription.trim()) {
      setError('Please enter a job description');
      return;
    }

    if (!resumeText.trim()) {
      setError('Please enter resume text or upload a PDF file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await resumeAPI.analyzeResume(resumeText, jobDescription, useAIModel);
      setResult(data);
    } catch (err) {
      setError(handleAPIError(err, 'Failed to analyze resume'));
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please upload a PDF file');
      return;
    }

    setUploadLoading(true);
    setError(null);

    try {
      const data = await resumeAPI.uploadPDF(file);
      if (data.extraction_success) {
        setResumeText(data.text);
        setUploadedFile({
          name: data.filename,
          pages: data.pages,
          wordCount: data.word_count
        });
      } else {
        setError('Failed to extract text from PDF');
      }
    } catch (err) {
      setError(handleAPIError(err, 'Failed to process PDF file'));
    } finally {
      setUploadLoading(false);
    }

    // Clear the input
    event.target.value = '';
  };

  const handleClear = () => {
    setJobDescription('');
    setResumeText('');
    setResult(null);
    setError(null);
    setUploadedFile(null);
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getRecommendationSeverity = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'info';
    if (score >= 40) return 'warning';
    return 'error';
  };

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <AnalyticsIcon fontSize="large" color="primary" />
        Resume Analysis
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Analyze how well a resume matches a job description using AI-powered algorithms.
      </Typography>

      {/* Input Section */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Job Description */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <WorkIcon color="primary" />
                Job Description
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={12}
                label="Enter Job Description"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the job description here..."
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Resume Input */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <PersonIcon color="primary" />
                Resume Input
              </Typography>
              
              {/* File Upload */}
              <Box sx={{ mb: 2 }}>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={uploadLoading ? <CircularProgress size={20} /> : <UploadIcon />}
                  disabled={uploadLoading}
                  fullWidth
                  sx={{ mb: 1 }}
                >
                  {uploadLoading ? 'Processing PDF...' : 'Upload PDF Resume'}
                  <input
                    type="file"
                    hidden
                    accept=".pdf"
                    onChange={handleFileUpload}
                  />
                </Button>
                
                {uploadedFile && (
                  <Alert severity="success" sx={{ mt: 1 }}>
                    📄 {uploadedFile.name} ({uploadedFile.pages} pages, {uploadedFile.wordCount} words)
                  </Alert>
                )}
              </Box>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Or paste resume text:
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={8}
                label="Resume Text"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste resume text here or upload a PDF file above..."
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Configuration and Actions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={useAIModel}
                    onChange={(e) => setUseAIModel(e.target.checked)}
                    color="primary"
                  />
                }
                label="Use AI Model (LSTM)"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  onClick={handleClear}
                  disabled={loading}
                >
                  Clear All
                </Button>
                
                <Button
                  variant="contained"
                  onClick={handleAnalyzeResume}
                  disabled={loading || !jobDescription.trim() || !resumeText.trim()}
                  startIcon={loading ? <CircularProgress size={20} /> : <AnalyticsIcon />}
                  size="large"
                >
                  {loading ? 'Analyzing...' : 'Analyze Match'}
                </Button>
              </Box>
            </Grid>
          </Grid>
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
            <Typography variant="h6" gutterBottom>
              📊 Analysis Results
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Model: {result.model_used} | Processing Time: {result.processing_time?.toFixed(2)}s
            </Typography>

            {/* Match Scores */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h3" color={`${getScoreColor(result.overall_match_score)}.main`}>
                    {result.overall_match_score.toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                    Overall Match
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={result.overall_match_score}
                    color={getScoreColor(result.overall_match_score)}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h3" color={`${getScoreColor(result.skill_match_score)}.main`}>
                    {result.skill_match_score.toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                    Skills Match
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={result.skill_match_score}
                    color={getScoreColor(result.skill_match_score)}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="h3" color={`${getScoreColor(result.experience_match_score)}.main`}>
                    {result.experience_match_score.toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                    Experience Match
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={result.experience_match_score}
                    color={getScoreColor(result.experience_match_score)}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </Paper>
              </Grid>
            </Grid>

            {/* Recommendation */}
            <Alert severity={getRecommendationSeverity(result.overall_match_score)} sx={{ mb: 3 }}>
              <Typography variant="h6">Recommendation</Typography>
              <Typography>{result.recommendation}</Typography>
            </Alert>

            {/* Detailed Analysis Tabs */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
                <Tab label="Skills Analysis" />
                <Tab label="Candidate Profile" />
              </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
              <Grid container spacing={3}>
                {/* Matching Skills */}
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CheckIcon color="success" />
                    Matching Skills ({result.matching_skills?.length || 0})
                  </Typography>
                  {result.matching_skills?.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.matching_skills.map((skill, index) => (
                        <Chip key={index} label={skill} color="success" size="small" />
                      ))}
                    </Box>
                  ) : (
                    <Typography color="text.secondary">No matching skills found</Typography>
                  )}
                </Grid>

                {/* Missing Skills */}
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CancelIcon color="error" />
                    Missing Skills ({result.missing_skills?.length || 0})
                  </Typography>
                  {result.missing_skills?.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.missing_skills.map((skill, index) => (
                        <Chip key={index} label={skill} color="error" size="small" variant="outlined" />
                      ))}
                    </Box>
                  ) : (
                    <Typography color="text.secondary">No missing skills</Typography>
                  )}
                </Grid>

                {/* Extra Skills */}
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AddIcon color="primary" />
                    Additional Skills ({result.extra_skills?.length || 0})
                  </Typography>
                  {result.extra_skills?.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.extra_skills.map((skill, index) => (
                        <Chip key={index} label={skill} color="primary" size="small" variant="outlined" />
                      ))}
                    </Box>
                  ) : (
                    <Typography color="text.secondary">No additional skills found</Typography>
                  )}
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {result.candidate_profile && Object.keys(result.candidate_profile).length > 0 ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <WorkIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Experience"
                          secondary={`${result.candidate_profile.experience || 0} years`}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <CodeIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Projects"
                          secondary={`${result.candidate_profile.projects || 0} projects`}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <List>
                      <ListItem>
                        <ListItemIcon>
                          <SchoolIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Education Level"
                          secondary={`Level ${result.candidate_profile.education || 0}`}
                        />
                      </ListItem>
                      
                      <ListItem>
                        <ListItemIcon>
                          <CheckIcon color="success" />
                        </ListItemIcon>
                        <ListItemText
                          primary="Skill Overlap"
                          secondary={`${result.candidate_profile.skill_overlap || 0} matching skills`}
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
              ) : (
                <Alert severity="info">
                  Candidate profile information is not available. This may occur when using rule-based analysis.
                </Alert>
              )}
            </TabPanel>
          </CardContent>
        </Card>
      )}
    </Container>
  );
};

export default ResumeAnalyzer;
