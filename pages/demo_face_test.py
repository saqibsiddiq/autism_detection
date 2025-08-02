import streamlit as st
import cv2
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from utils.demo_mode import demo_simulator, show_demo_mode_info, create_demo_video_frame
from database.models import db_manager

def show_demo_face_recognition_test():
    st.header("👁️ Face Recognition Test (Demo Mode)")
    
    show_demo_mode_info()
    
    st.markdown("""
    This test demonstrates how gaze patterns are analyzed when viewing human faces compared to objects.
    The demo shows realistic behavioral patterns based on research data.
    """)
    
    # Test phases
    test_phases = [
        {
            "name": "Phase 1: Face vs Object",
            "description": "Simulated gaze patterns viewing faces and objects",
            "duration": 15,
            "stimulus_type": "face_object_comparison"
        },
        {
            "name": "Phase 2: Multiple Faces", 
            "description": "Multiple faces with tracking natural viewing patterns",
            "duration": 12,
            "stimulus_type": "multiple_faces"
        },
        {
            "name": "Phase 3: Eye Contact",
            "description": "Direct eye contact scenarios with natural viewing",
            "duration": 18,
            "stimulus_type": "eye_contact"
        }
    ]
    
    # Initialize test state
    if 'demo_face_test_phase' not in st.session_state:
        st.session_state.demo_face_test_phase = 0
    if 'demo_face_test_active' not in st.session_state:
        st.session_state.demo_face_test_active = False
    if 'demo_face_test_results' not in st.session_state:
        st.session_state.demo_face_test_results = {}
    if 'demo_start_time' not in st.session_state:
        st.session_state.demo_start_time = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Demo Video Analysis")
        
        # Demo video display
        demo_video_placeholder = st.empty()
        
        # Show current stimulus
        if st.session_state.demo_face_test_active and st.session_state.demo_face_test_phase < len(test_phases):
            current_phase = test_phases[st.session_state.demo_face_test_phase]
            
            with demo_video_placeholder.container():
                st.markdown(f"### {current_phase['name']}")
                
                # Show simulated gaze tracking
                show_demo_stimulus_with_gaze(current_phase['stimulus_type'])
        else:
            with demo_video_placeholder.container():
                st.info("Click 'Start Phase' to begin the demo test")
                
                # Show sample stimulus
                show_demo_face_object_stimulus()
    
    with col2:
        st.subheader("Demo Controls")
        
        # Current phase info
        if st.session_state.demo_face_test_phase < len(test_phases):
            current_phase = test_phases[st.session_state.demo_face_test_phase]
            st.info(f"**Current Phase:** {current_phase['name']}")
            st.write(f"**Description:** {current_phase['description']}")
            st.write(f"**Duration:** {current_phase['duration']} seconds")
        
        # Progress
        progress = st.session_state.demo_face_test_phase / len(test_phases)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.demo_face_test_phase}/{len(test_phases)} phases completed")
        
        # Timer display
        if st.session_state.demo_face_test_active and st.session_state.demo_start_time:
            elapsed = time.time() - st.session_state.demo_start_time
            current_phase = test_phases[st.session_state.demo_face_test_phase]
            remaining = max(0, current_phase['duration'] - elapsed)
            st.metric("Time Remaining", f"{remaining:.1f}s")
            
            # Auto-complete phase when time is up
            if remaining <= 0:
                complete_demo_phase()
                st.rerun()
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start Phase", disabled=st.session_state.demo_face_test_active):
                if st.session_state.demo_face_test_phase < len(test_phases):
                    st.session_state.demo_face_test_active = True
                    st.session_state.demo_start_time = time.time()
                    demo_simulator.start_demo("face_recognition")
                    st.success("Demo phase started!")
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Phase", disabled=not st.session_state.demo_face_test_active):
                if st.session_state.demo_face_test_active:
                    complete_demo_phase()
                    st.rerun()
        
        # Show live demo metrics
        if st.session_state.demo_face_test_active:
            st.markdown("**Live Demo Metrics:**")
            
            # Generate live stats
            elapsed = time.time() - st.session_state.demo_start_time if st.session_state.demo_start_time else 0
            face_attention = int(elapsed * 8)  # Simulate growing attention
            object_attention = int(elapsed * 3)
            
            st.metric("Face Attention", face_attention)
            st.metric("Object Attention", object_attention)
            
            if face_attention + object_attention > 0:
                face_pref = face_attention / (face_attention + object_attention)
                st.metric("Face Preference", f"{face_pref:.1%}")
        
        # Navigation
        st.markdown("---")
        col_back, col_next = st.columns(2)
        
        with col_back:
            if st.button("⬅️ Back to Overview"):
                st.session_state.current_test = 0
                st.rerun()
        
        with col_next:
            if st.session_state.demo_face_test_phase >= len(test_phases):
                if st.button("Next Test ➡️", type="primary"):
                    st.session_state.current_test = 2
                    st.rerun()
        
        # Results summary
        if st.session_state.demo_face_test_phase >= len(test_phases):
            st.success("✅ Demo Face Recognition Test Completed!")
            
            with st.expander("📊 Demo Results", expanded=True):
                show_demo_face_test_summary()

def complete_demo_phase():
    """Complete current demo phase and generate results"""
    # Generate demo results
    demo_results = demo_simulator.stop_demo()
    
    # Add some randomization for realistic results
    demo_results.update({
        'face_attention_time': np.random.randint(180, 250),
        'object_attention_time': np.random.randint(60, 120),
        'face_preference_ratio': np.random.uniform(0.6, 0.8),
        'face_detection_rate': np.random.uniform(0.85, 0.95)
    })
    
    st.session_state.demo_face_test_results[f"phase_{st.session_state.demo_face_test_phase}"] = demo_results
    st.session_state.demo_face_test_active = False
    st.session_state.demo_face_test_phase += 1
    st.session_state.demo_start_time = None
    
    # Save to database if assessment exists
    try:
        if st.session_state.assessment_id:
            db_manager.save_gaze_data_batch(
                st.session_state.assessment_id,
                f"demo_face_recognition_phase_{st.session_state.demo_face_test_phase-1}",
                "face_recognition_demo",
                demo_results
            )
    except Exception as e:
        st.error(f"Error saving demo data: {e}")
    
    st.success("Demo phase completed!")

def show_demo_stimulus_with_gaze(stimulus_type):
    """Show demo stimulus with simulated gaze tracking"""
    
    if stimulus_type == 'face_object_comparison':
        show_demo_face_object_stimulus()
    elif stimulus_type == 'multiple_faces':
        show_demo_multiple_faces_stimulus()
    elif stimulus_type == 'eye_contact':
        show_demo_eye_contact_stimulus()
    
    # Add simulated gaze overlay
    if st.session_state.demo_face_test_active:
        st.markdown("**🔴 Simulated Gaze Point**")
        
        # Generate demo gaze coordinates
        gaze_x, gaze_y = demo_simulator.get_simulated_gaze_point()
        
        # Show gaze metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gaze X", f"{gaze_x:.0f}")
        with col2:
            st.metric("Gaze Y", f"{gaze_y:.0f}")
        
        # Create simple visualization
        fig = go.Figure()
        
        # Add stimulus regions
        fig.add_shape(type="rect", x0=50, y0=100, x1=350, y1=400,
                     line=dict(color="blue", width=2), name="Face Region")
        fig.add_shape(type="rect", x0=450, y0=100, x1=750, y1=400,
                     line=dict(color="red", width=2), name="Object Region")
        
        # Add gaze point
        fig.add_scatter(x=[gaze_x], y=[gaze_y], mode='markers',
                       marker=dict(size=15, color='green'), name='Gaze Point')
        
        fig.update_layout(
            title="Real-time Gaze Tracking (Demo)",
            xaxis=dict(range=[0, 800], title="X Position"),
            yaxis=dict(range=[0, 600], title="Y Position"),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_demo_face_object_stimulus():
    """Display face vs object stimulus (demo version)"""
    col1, col2 = st.columns(2)
    
    with col1:
        face_html = """
        <div style='text-align: center; padding: 40px; border: 3px solid blue; border-radius: 10px; background-color: #f0f8ff;'>
            <div style='font-size: 120px;'>😊</div>
            <h3>Human Face</h3>
            <p style='color: blue;'>Face Region (Tracked)</p>
        </div>
        """
        st.markdown(face_html, unsafe_allow_html=True)
    
    with col2:
        object_html = """
        <div style='text-align: center; padding: 40px; border: 3px solid red; border-radius: 10px; background-color: #fff0f0;'>
            <div style='font-size: 120px;'>🚗</div>
            <h3>Object</h3>
            <p style='color: red;'>Object Region (Tracked)</p>
        </div>
        """
        st.markdown(object_html, unsafe_allow_html=True)

def show_demo_multiple_faces_stimulus():
    """Display multiple faces stimulus (demo version)"""
    faces_html = """
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 20px; border: 2px solid green; border-radius: 15px;'>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😀</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😃</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😄</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😁</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😆</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 60px;'>😊</div>
        </div>
    </div>
    <p style='text-align: center; margin-top: 10px;'><strong>Multiple Faces - Natural Scanning Pattern (Demo)</strong></p>
    """
    st.markdown(faces_html, unsafe_allow_html=True)

def show_demo_eye_contact_stimulus():
    """Display eye contact stimulus (demo version)"""
    eye_contact_html = """
    <div style='text-align: center; padding: 60px; background: linear-gradient(45deg, #e3f2fd, #bbdefb); border-radius: 15px; border: 3px solid #2196f3;'>
        <div style='font-size: 150px; margin-bottom: 20px;'>👁️</div>
        <h2>Eye Contact Demo</h2>
        <p style='font-size: 18px;'>Simulated natural eye contact patterns</p>
        <div style='margin-top: 20px; color: #1976d2;'>
            <strong>Demo: Tracking eye contact duration and frequency</strong>
        </div>
    </div>
    """
    st.markdown(eye_contact_html, unsafe_allow_html=True)

def show_demo_face_test_summary():
    """Display summary of demo face recognition test results"""
    if not st.session_state.demo_face_test_results:
        st.write("No demo results available yet.")
        return
    
    # Aggregate demo results
    total_face_attention = 0
    total_object_attention = 0
    total_detection_rate = 0
    phases_completed = 0
    
    for phase_key, results in st.session_state.demo_face_test_results.items():
        if results:
            total_face_attention += results.get('face_attention_time', 0)
            total_object_attention += results.get('object_attention_time', 0)
            total_detection_rate += results.get('face_detection_rate', 0)
            phases_completed += 1
    
    if phases_completed > 0:
        avg_detection_rate = total_detection_rate / phases_completed
        total_attention = total_face_attention + total_object_attention
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Face Detection Rate", f"{avg_detection_rate:.1%}")
            st.metric("Total Attention Points", total_attention)
        
        with col2:
            if total_attention > 0:
                face_preference = total_face_attention / total_attention
                st.metric("Face Preference", f"{face_preference:.1%}")
            else:
                st.metric("Face Preference", "No data")
        
        # Demo visualization
        if total_attention > 0:
            attention_data = {
                'Stimulus': ['Faces', 'Objects'],
                'Attention Time': [total_face_attention, total_object_attention]
            }
            
            fig = px.pie(attention_data, values='Attention Time', names='Stimulus',
                        title="Demo Results: Faces vs Objects Attention Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Demo completed! These results show typical patterns for face recognition assessment.")
    
    # Store results for overall analysis
    if st.session_state.demo_face_test_results:
        st.session_state.face_test_results = st.session_state.demo_face_test_results