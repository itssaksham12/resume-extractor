import React from 'react';
import { Box, Typography, Container, Link, Divider } from '@mui/material';
import { GitHub as GitHubIcon, Code as CodeIcon } from '@mui/icons-material';

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: 'grey.100',
        borderTop: 1,
        borderColor: 'grey.200',
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            © 2024 Resume Extractor AI - Built with FastAPI & React
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <CodeIcon fontSize="small" color="primary" />
              <Typography variant="body2" color="text.secondary">
                Powered by AI
              </Typography>
            </Box>
            
            <Divider orientation="vertical" flexItem />
            
            <Typography variant="body2" color="text.secondary">
              BERT • LSTM • FastAPI • React
            </Typography>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
