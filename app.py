import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ASD Behavioral Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'assessment_data' not in st.session_state:
    st.session_state.assessment_data = {}
if 'gaze_data' not in st.session_state:
    st.session_state.gaze_data = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

def main():
    st.title("🧠 ASD Behavioral Analysis Platform")
    
    # Medical disclaimer
    with st.expander("⚠️ Important Medical Disclaimer - Please Read", expanded=False):
        st.warning("""
        **MEDICAL DISCLAIMER:**
        
        This application is designed for educational and screening support purposes only. It is NOT a diagnostic tool and should never be used as a substitute for professional medical advice, diagnosis, or treatment.
        
        - Results from this application do not constitute a medical diagnosis
        - Always consult with qualified healthcare professionals for proper evaluation
        - This tool may produce false positives or false negatives
        - Early intervention and professional assessment are crucial for autism spectrum disorders
        
        By using this application, you acknowledge that you understand these limitations and will seek appropriate professional medical advice.
        """)
    
    # Privacy notice
    with st.expander("🔒 Privacy Notice", expanded=False):
        st.info("""
        **Your Privacy is Our Priority:**
        
        - All video processing happens locally on your device
        - No video data is transmitted or stored on external servers
        - Assessment data is only stored temporarily in your browser session
        - You can clear all data at any time using the sidebar options
        """)
    
    # Navigation
    st.sidebar.title("Navigation")
    
    # Assessment progress
    steps = ["Overview", "Questionnaire", "Gaze Assessment", "Results", "Educational Resources"]
    current_step = st.sidebar.selectbox("Assessment Steps", steps, index=st.session_state.current_step)
    
    # Update current step
    st.session_state.current_step = steps.index(current_step)
    
    # Progress bar
    progress = (st.session_state.current_step + 1) / len(steps)
    st.sidebar.progress(progress)
    st.sidebar.write(f"Progress: {int(progress * 100)}%")
    
    # Clear data option
    if st.sidebar.button("🗑️ Clear All Data"):
        for key in list(st.session_state.keys()):
            if key not in ['current_step']:
                del st.session_state[key]
        st.session_state.assessment_data = {}
        st.session_state.gaze_data = []
        st.rerun()
    
    # Main content based on current step
    if current_step == "Overview":
        show_overview()
    elif current_step == "Questionnaire":
        show_questionnaire()
    elif current_step == "Gaze Assessment":
        show_gaze_assessment()
    elif current_step == "Results":
        show_results()
    elif current_step == "Educational Resources":
        show_education()

def show_overview():
    st.header("Welcome to the ASD Behavioral Analysis Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Assessment
        
        This comprehensive assessment combines traditional behavioral questionnaires with advanced gaze pattern analysis to provide insights into autism spectrum disorder (ASD) characteristics.
        
        **The assessment includes:**
        
        1. **Behavioral Questionnaire** - Based on validated screening tools including M-CHAT-R and AQ-10
        2. **Gaze Pattern Analysis** - Real-time eye tracking during social attention tasks
        3. **Comprehensive Results** - Combined analysis of behavioral and gaze data
        4. **Educational Resources** - Information about ASD and early intervention
        
        ### What You'll Need
        
        - A device with a camera (webcam or mobile camera)
        - Good lighting for optimal face detection
        - Approximately 15-20 minutes to complete
        - A quiet environment for the assessment
        """)
        
        if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        st.markdown("""
        ### Key Features
        
        ✅ **Evidence-Based**
        - Validated questionnaires
        - Research-backed gaze metrics
        
        ✅ **Privacy-First**
        - Local video processing
        - No data transmission
        
        ✅ **Comprehensive**
        - Multiple assessment modalities
        - Detailed visualizations
        
        ✅ **Accessible**
        - Mobile-friendly design
        - Clear instructions
        """)
        
        st.info("""
        **Target Age Range:**
        This assessment is designed for individuals aged 18 months and above. For younger children, parental assistance may be needed.
        """)

def show_questionnaire():
    from pages.questionnaire import show_questionnaire_page
    show_questionnaire_page()

def show_gaze_assessment():
    from pages.gaze_assessment import show_gaze_assessment_page
    show_gaze_assessment_page()

def show_results():
    from pages.results import show_results_page
    show_results_page()

def show_education():
    from pages.education import show_education_page
    show_education_page()

if __name__ == "__main__":
    main()
