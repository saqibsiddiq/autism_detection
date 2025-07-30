# ASD Behavioral Analysis Platform

## Overview

This application is an educational and screening support platform for Autism Spectrum Disorder (ASD) behavioral analysis. It combines traditional questionnaire-based assessments with computer vision-based gaze pattern analysis to provide comprehensive behavioral insights. The platform uses Streamlit for the web interface, MediaPipe for facial landmark detection, and machine learning models for pattern analysis.

**Important Note**: This is an educational tool only and is not intended for medical diagnosis. All results must be interpreted by qualified healthcare professionals.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page navigation
- **UI Components**: Interactive forms, real-time video processing, data visualizations
- **Responsive Design**: Wide layout configuration with expandable sidebar
- **Real-time Processing**: WebRTC integration for live video analysis

### Backend Architecture
- **Core Framework**: Python-based Streamlit application
- **Processing Pipeline**: Modular design with separate components for questionnaire processing, gaze analysis, and machine learning predictions
- **Session Management**: Streamlit session state for maintaining user data across pages
- **Data Flow**: Sequential assessment workflow (questionnaire → gaze assessment → results)

### Computer Vision Pipeline
- **Face Detection**: MediaPipe Face Mesh for facial landmark detection
- **Gaze Estimation**: Custom gaze analyzer using eye landmarks and iris tracking
- **Real-time Processing**: Frame-by-frame analysis with performance optimization (every 3rd frame)
- **Feature Extraction**: Gaze patterns, fixation analysis, and eye contact metrics

## Key Components

### 1. Assessment Modules
- **Questionnaire System** (`pages/questionnaire.py`): M-CHAT-R and AQ-10 based screening questions
- **Gaze Assessment** (`pages/gaze_assessment.py`): Computer vision-based eye tracking and gaze pattern analysis
- **Results Dashboard** (`pages/results.py`): Comprehensive analysis with ML insights and visualizations

### 2. Machine Learning Pipeline
- **Behavioral Model** (`models/behavioral_model.py`): Multi-algorithm ensemble (Random Forest, Gradient Boosting, Logistic Regression)
- **Feature Engineering**: Combines questionnaire scores with gaze metrics
- **Preprocessing**: StandardScaler for feature normalization

### 3. Computer Vision Engine
- **Gaze Analyzer** (`models/gaze_analyzer.py`): MediaPipe-based facial landmark detection and gaze estimation
- **Video Processor** (`utils/camera_utils.py`): Real-time video processing with WebRTC integration
- **Performance Optimization**: Frame skipping and efficient landmark processing

### 4. Data Processing
- **Data Processor** (`utils/data_processor.py`): Weighted scoring system for questionnaire responses
- **Visualization Engine**: Plotly-based charts and graphs for result presentation
- **Feature Extraction**: Behavioral metrics and gaze pattern analysis

## Data Flow

1. **Initial Setup**: User consents to terms and privacy policy
2. **Questionnaire Phase**: 
   - Load questions from JSON configuration
   - Collect weighted responses
   - Calculate domain-specific scores
3. **Gaze Assessment Phase**:
   - Initialize WebRTC video stream
   - Process frames through MediaPipe pipeline
   - Extract gaze features and eye contact metrics
4. **Analysis Phase**:
   - Combine questionnaire and gaze data
   - Apply machine learning models
   - Generate risk assessment scores
5. **Results Presentation**:
   - Create comprehensive visualizations
   - Provide educational context
   - Generate downloadable reports

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **StreamLit-WebRTC**: Real-time video processing
- **MediaPipe**: Facial landmark detection and pose estimation
- **OpenCV**: Computer vision operations
- **Plotly**: Interactive data visualizations

### Machine Learning Stack
- **Scikit-learn**: ML algorithms and preprocessing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Joblib**: Model serialization

### Data Storage
- **JSON**: Configuration files for questions and settings
- **Session State**: In-memory storage for user data during assessment
- **Local Processing**: All video data processed locally for privacy

## Deployment Strategy

### Current Architecture
- **Single-Page Application**: Streamlit-based web interface
- **Local Processing**: All video analysis happens client-side
- **No Database**: Uses JSON files and session state for data management
- **Privacy-First**: No external data transmission for video processing

### Scalability Considerations
- **Stateless Design**: Each session is independent
- **Modular Components**: Easy to extend with additional assessment types
- **Performance Optimization**: Frame processing optimization for real-time analysis
- **Cross-Platform**: Works on devices with webcam capability

### Security and Privacy
- **Local Video Processing**: No video data transmitted to servers
- **Medical Disclaimer**: Clear warnings about diagnostic limitations
- **Consent Management**: Explicit user consent for camera access
- **Data Retention**: No persistent storage of personal data

The application follows a privacy-first approach with all sensitive processing happening locally on the user's device, making it suitable for educational and preliminary screening purposes while maintaining strict privacy standards.