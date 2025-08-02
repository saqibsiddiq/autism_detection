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
import uuid
import time
from datetime import datetime
from database.models import db_manager

# Page configuration
st.set_page_config(
    page_title="ASD Behavioral Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
try:
    db_manager.create_tables()
except Exception as e:
    st.error(f"Database initialization error: {e}")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'user_id' not in st.session_state:
    user = db_manager.get_user_by_session(st.session_state.session_id)
    if not user:
        user = db_manager.create_user(st.session_state.session_id)
    st.session_state.user_id = user.id
if 'assessment_id' not in st.session_state:
    st.session_state.assessment_id = None
if 'current_test' not in st.session_state:
    st.session_state.current_test = 0
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

def main():
    # Title and description
    st.title("🧠 ASD Behavioral Analysis Platform")
    st.markdown("""
    ### Advanced Gaze Pattern Analysis for Autism Spectrum Disorder Detection
    
    This platform uses computer vision and behavioral analysis to assess concentration levels 
    and detect autism-related behavioral patterns through visual stimuli and gaze tracking.
    
    **⚠️ Important Medical Disclaimer:**
    This is an educational tool only and not intended for medical diagnosis. 
    All results must be interpreted by qualified healthcare professionals.
    """)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Assessment Tests")
    
    test_options = [
        "🏠 Overview",
        "👁️ Face Recognition Test", 
        "🎭 Social Attention Test",
        "🎨 Visual Pattern Test",
        "🎬 Motion Tracking Test",
        "📊 Results & Analysis",
        "📈 Admin Dashboard"
    ]
    
    selected_test = st.sidebar.selectbox("Select Test", test_options, index=st.session_state.current_test)
    st.session_state.current_test = test_options.index(selected_test)
    
    # Display selected page
    if selected_test == "🏠 Overview":
        show_overview()
    elif selected_test == "👁️ Face Recognition Test":
        show_face_recognition_test()
    elif selected_test == "🎭 Social Attention Test":
        show_social_attention_test()
    elif selected_test == "🎨 Visual Pattern Test":
        show_visual_pattern_test()
    elif selected_test == "🎬 Motion Tracking Test":
        show_motion_tracking_test()
    elif selected_test == "📊 Results & Analysis":
        show_results_analysis()
    elif selected_test == "📈 Admin Dashboard":
        from pages.admin_dashboard import show_admin_dashboard
        show_admin_dashboard()

def show_overview():
    st.header("🎯 Assessment Overview")
    
    # Test descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Tests")
        
        tests_info = [
            {
                "name": "👁️ Face Recognition Test",
                "description": "Measures gaze patterns when viewing human faces vs objects",
                "duration": "3-5 minutes",
                "key_metrics": "Eye contact duration, face preference ratio"
            },
            {
                "name": "🎭 Social Attention Test", 
                "description": "Analyzes attention to social vs non-social stimuli",
                "duration": "4-6 minutes",
                "key_metrics": "Social attention score, gaze distribution"
            },
            {
                "name": "🎨 Visual Pattern Test",
                "description": "Evaluates preference for patterns and repetitive visuals",
                "duration": "3-4 minutes", 
                "key_metrics": "Pattern fixation, visual scanning behavior"
            },
            {
                "name": "🎬 Motion Tracking Test",
                "description": "Tracks eye movements following moving objects",
                "duration": "2-3 minutes",
                "key_metrics": "Smooth pursuit, saccadic movements"
            }
        ]
        
        for test in tests_info:
            with st.expander(test["name"], expanded=False):
                st.write(f"**Description:** {test['description']}")
                st.write(f"**Duration:** {test['duration']}")
                st.write(f"**Key Metrics:** {test['key_metrics']}")
    
    with col2:
        st.subheader("Getting Started")
        
        st.markdown("""
        ### 📋 Before You Begin:
        
        1. **Camera Setup**: Ensure your camera is working and positioned at eye level
        2. **Lighting**: Good lighting on your face for accurate tracking
        3. **Distance**: Sit 18-24 inches from the screen
        4. **Environment**: Quiet space with minimal distractions
        
        ### 🔬 How It Works:
        
        - **Face Detection**: Uses MediaPipe for precise facial landmark detection
        - **Gaze Tracking**: Analyzes eye movements and fixation patterns
        - **Behavioral Analysis**: AI models assess concentration and attention patterns
        - **Real-time Processing**: All analysis happens locally on your device
        
        ### 📊 What You'll Get:
        
        - Detailed gaze pattern analysis
        - Concentration level assessment
        - Behavioral pattern insights
        - Comprehensive report with visualizations
        """)
        
        # Start assessment button
        st.markdown("---")
        if st.button("🚀 Start Behavioral Assessment", type="primary", use_container_width=True):
            # Create new assessment
            if not st.session_state.assessment_id:
                assessment = db_manager.create_assessment(st.session_state.user_id, "behavioral_analysis")
                st.session_state.assessment_id = assessment.id
            
            st.session_state.current_test = 1  # Go to first test
            st.rerun()

def show_face_recognition_test():
    from pages.face_recognition_test import show_face_recognition_test_page
    show_face_recognition_test_page()

def show_social_attention_test():
    from pages.social_attention_test import show_social_attention_test_page
    show_social_attention_test_page()

def show_visual_pattern_test():
    from pages.visual_pattern_test import show_visual_pattern_test_page
    show_visual_pattern_test_page()

def show_motion_tracking_test():
    from pages.motion_tracking_test import show_motion_tracking_test_page
    show_motion_tracking_test_page()

def show_results_analysis():
    from pages.results_analysis import show_results_analysis_page
    show_results_analysis_page()

if __name__ == "__main__":
    main()