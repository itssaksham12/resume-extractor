import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  Analytics as AnalyticsIcon,
  Summarize as SummaryIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  CloudDone as CloudIcon,
} from '@mui/icons-material';
import { healthAPI } from '../services/api';

const Dashboard = () => {
  const navigate = useNavigate();
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await healthAPI.check();
        setHealthStatus(health);
      } catch (error) {
        console.error('Health check failed:', error);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  const features = [
    {
      title: 'Skills Extraction',
      description: 'Extract required skills from job descriptions using AI-powered BERT models or rule-based analysis.',
      icon: <PsychologyIcon fontSize="large" />,
      path: '/skills',
      color: 'primary',
      capabilities: ['BERT AI Model', 'Rule-based Fallback', 'Skill Categorization']
    },
    {
      title: 'Resume Analysis',
      description: 'Analyze resume-job compatibility using advanced LSTM models with comprehensive matching algorithms.',
      icon: <AnalyticsIcon fontSize="large" />,
      path: '/analyze',
      color: 'secondary',
      capabilities: ['LSTM Neural Networks', 'Match Scoring', 'PDF Processing']
    },
    {
      title: 'Text Summarization',
      description: 'Generate concise summaries of job descriptions and resumes using BERT-based extractive summarization.',
      icon: <SummaryIcon fontSize="large" />,
      path: '/summarize',
      color: 'success',
      capabilities: ['BERT Summarizer', 'Extractive Methods', 'Configurable Length']
    }
  ];

  const getModelStatusIcon = (isLoaded) => {
    return isLoaded ? <CheckIcon color="success" /> : <WarningIcon color="warning" />;
  };

  const getModelStatusText = (isLoaded) => {
    return isLoaded ? 'Loaded' : 'Using Fallback';
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          🚀 Resume Extractor AI
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          AI-powered resume and job description analysis platform
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Extract skills, analyze compatibility, and generate summaries using state-of-the-art machine learning models
        </Typography>
      </Box>

      {/* System Status */}
      {healthStatus && (
        <Alert 
          severity={Object.values(healthStatus.models_loaded).every(loaded => loaded) ? "success" : "warning"}
          sx={{ mb: 4 }}
        >
          <Typography variant="h6">System Status: {healthStatus.status}</Typography>
          <Typography variant="body2">
            {Object.values(healthStatus.models_loaded).filter(loaded => loaded).length} of {Object.keys(healthStatus.models_loaded).length} AI models loaded
          </Typography>
        </Alert>
      )}

      {/* Feature Cards */}
      <Grid container spacing={4} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4
                }
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box sx={{ color: `${feature.color}.main`, mb: 2 }}>
                  {feature.icon}
                </Box>
                
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                  {feature.title}
                </Typography>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  {feature.description}
                </Typography>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, justifyContent: 'center' }}>
                  {feature.capabilities.map((capability, capIndex) => (
                    <Chip 
                      key={capIndex}
                      label={capability}
                      size="small"
                      color={feature.color}
                      variant="outlined"
                    />
                  ))}
                </Box>
              </CardContent>
              
              <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                <Button
                  variant="contained"
                  color={feature.color}
                  onClick={() => navigate(feature.path)}
                  sx={{ borderRadius: 2 }}
                >
                  Get Started
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Model Status and Quick Stats */}
      <Grid container spacing={4}>
        {/* Model Status */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <MemoryIcon color="primary" />
                AI Models Status
              </Typography>
              
              {healthStatus ? (
                <List>
                  {Object.entries(healthStatus.models_loaded).map(([model, isLoaded], index) => (
                    <React.Fragment key={model}>
                      <ListItem>
                        <ListItemIcon>
                          {getModelStatusIcon(isLoaded)}
                        </ListItemIcon>
                        <ListItemText
                          primary={model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          secondary={getModelStatusText(isLoaded)}
                        />
                        <Chip
                          label={isLoaded ? 'Active' : 'Fallback'}
                          color={isLoaded ? 'success' : 'warning'}
                          size="small"
                        />
                      </ListItem>
                      {index < Object.entries(healthStatus.models_loaded).length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">Loading status...</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Stats */}
        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <SpeedIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="primary.main">
                  Fast
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  AI-powered analysis
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12}>
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <CloudIcon color="success" sx={{ fontSize: 40, mb: 1 }} />
                <Typography variant="h4" color="success.main">
                  Cloud
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Always available
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Quick Start Guide */}
      <Card sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🚀 Quick Start Guide
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary.main" gutterBottom>
                  1. Extract Skills
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Start by extracting required skills from job descriptions
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="secondary.main" gutterBottom>
                  2. Analyze Resume
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload resumes and analyze compatibility with job requirements
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="success.main" gutterBottom>
                  3. Get Insights
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Review detailed analysis and recommendations
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Dashboard;
