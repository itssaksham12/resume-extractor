import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  Alert,
  Collapse,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  WorkOutline as WorkIcon,
  Analytics as AnalyticsIcon,
  Summarize as SummaryIcon,
  Dashboard as DashboardIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { healthAPI } from '../services/api';

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [anchorEl, setAnchorEl] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [showHealthAlert, setShowHealthAlert] = useState(false);

  // Navigation items
  const navItems = [
    { path: '/', label: 'Dashboard', icon: <DashboardIcon /> },
    { path: '/skills', label: 'Skills Extraction', icon: <WorkIcon /> },
    { path: '/analyze', label: 'Resume Analysis', icon: <AnalyticsIcon /> },
    { path: '/summarize', label: 'Text Summary', icon: <SummaryIcon /> },
  ];

  // Check health status on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await healthAPI.check();
        setHealthStatus(health);
        
        // Show alert if some models are not loaded
        const modelsLoaded = Object.values(health.models_loaded);
        const allLoaded = modelsLoaded.every(loaded => loaded);
        setShowHealthAlert(!allLoaded);
      } catch (error) {
        console.error('Health check failed:', error);
        setShowHealthAlert(true);
      }
    };

    checkHealth();
  }, []);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNavigation = (path) => {
    navigate(path);
    handleMenuClose();
  };

  const getModelStatusColor = (isLoaded) => {
    return isLoaded ? 'success' : 'warning';
  };

  const getCurrentPageTitle = () => {
    const currentItem = navItems.find(item => item.path === location.pathname);
    return currentItem ? currentItem.label : 'Resume Extractor';
  };

  return (
    <>
      <AppBar position="static" elevation={2}>
        <Toolbar>
          {/* Logo and Title */}
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <WorkIcon sx={{ mr: 2, fontSize: 32 }} />
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontWeight: 'bold',
                display: { xs: 'none', sm: 'block' }
              }}
            >
              Resume Extractor AI
            </Typography>
            {isMobile && (
              <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
                {getCurrentPageTitle()}
              </Typography>
            )}
          </Box>

          {/* Navigation - Desktop */}
          {!isMobile && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  color="inherit"
                  startIcon={item.icon}
                  onClick={() => handleNavigation(item.path)}
                  variant={location.pathname === item.path ? 'outlined' : 'text'}
                  sx={{
                    mx: 0.5,
                    color: location.pathname === item.path ? 'primary.main' : 'inherit',
                    backgroundColor: location.pathname === item.path ? 'rgba(255,255,255,0.1)' : 'transparent',
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </Box>
          )}

          {/* Mobile Menu */}
          {isMobile && (
            <>
              <IconButton
                color="inherit"
                onClick={handleMenuOpen}
                sx={{ ml: 2 }}
              >
                <MenuIcon />
              </IconButton>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                PaperProps={{
                  sx: { mt: 1.5, minWidth: 200 }
                }}
              >
                {navItems.map((item) => (
                  <MenuItem
                    key={item.path}
                    onClick={() => handleNavigation(item.path)}
                    selected={location.pathname === item.path}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {item.icon}
                      {item.label}
                    </Box>
                  </MenuItem>
                ))}
              </Menu>
            </>
          )}

          {/* Model Status Indicator */}
          {healthStatus && (
            <Box sx={{ ml: 2, display: 'flex', gap: 1 }}>
              {Object.entries(healthStatus.models_loaded).map(([model, isLoaded]) => (
                <Chip
                  key={model}
                  label={model.replace('_', ' ')}
                  size="small"
                  color={getModelStatusColor(isLoaded)}
                  sx={{ display: { xs: 'none', md: 'flex' } }}
                />
              ))}
            </Box>
          )}
        </Toolbar>
      </AppBar>

      {/* Health Status Alert */}
      <Collapse in={showHealthAlert}>
        <Alert 
          severity="warning" 
          action={
            <IconButton
              color="inherit"
              size="small"
              onClick={() => setShowHealthAlert(false)}
            >
              <CloseIcon fontSize="inherit" />
            </IconButton>
          }
          sx={{ borderRadius: 0 }}
        >
          Some AI models are not loaded. The application will use rule-based fallbacks for missing features.
        </Alert>
      </Collapse>
    </>
  );
};

export default Header;
