import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import threading
from utils.camera_utils import VideoProcessor, get_rtc_configuration, create_assessment_tasks, analyze_task_performance
from database.models import db_manager

def show_gaze_assessment_page():
    st.header("👁️ Gaze Pattern Assessment")
    
    st.markdown("""
    This assessment uses your device's camera to analyze gaze patterns and eye movements during specific tasks.
    Your privacy is protected - all processing happens locally on your device.
    """)
    
    # Privacy and consent
    with st.expander("🔒 Privacy and Consent", expanded=False):
        st.info("""
        **Camera Usage Consent:**
        - Your camera will be used to detect face landmarks and estimate gaze direction
        - No video data is transmitted or stored on external servers
        - All processing happens locally in your browser
        - You can stop the assessment at any time
        - Camera access will end when you close this page
        """)
        
        consent = st.checkbox("I consent to camera usage for gaze analysis")
        if not consent:
            st.warning("Camera consent is required to proceed with gaze assessment.")
            return
    
    # Assessment configuration
    st.subheader("Assessment Configuration")
    
    # Select assessment tasks
    available_tasks = create_assessment_tasks()
    selected_tasks = st.multiselect(
        "Select assessment tasks to perform:",
        options=[task['name'] for task in available_tasks],
        default=[task['name'] for task in available_tasks[:2]],
        help="Choose which gaze assessment tasks to include"
    )
    
    if not selected_tasks:
        st.warning("Please select at least one assessment task.")
        return
    
    # Camera configuration
    st.subheader("Camera Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        **Setup Tips:**
        - Ensure good lighting
        - Position camera at eye level
        - Maintain 18-24 inches distance
        - Minimize background distractions
        """)
    
    with col1:
        # Initialize session state
        if 'current_task_index' not in st.session_state:
            st.session_state.current_task_index = 0
            st.session_state.task_data = {}
            st.session_state.assessment_active = False
            st.session_state.task_start_time = None
        
        # Get current task
        if st.session_state.current_task_index < len(selected_tasks):
            current_task_name = selected_tasks[st.session_state.current_task_index]
            current_task = next(task for task in available_tasks if task['name'] == current_task_name)
            
            st.info(f"**Current Task:** {current_task['name']}")
            st.write(f"**Instructions:** {current_task['instructions']}")
            st.write(f"**Duration:** {current_task['duration']} seconds")
            
            # WebRTC configuration
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Video processor
            class GazeAssessmentProcessor(VideoProcessorBase):
                def __init__(self):
                    self.video_processor = VideoProcessor()
                    self.task_active = False
                    self.task_data = []
                    self.lock = threading.Lock()
                
                def recv(self, frame):
                    # Process frame through gaze analyzer
                    processed_frame = self.video_processor.recv(frame)
                    
                    # Collect data if task is active
                    if self.task_active:
                        with self.lock:
                            gaze_data = self.video_processor.get_analysis_data()
                            if gaze_data:
                                self.task_data.extend(gaze_data[-1:])  # Get latest data point
                    
                    return processed_frame
                
                def start_task(self):
                    self.task_active = True
                    self.task_data = []
                
                def stop_task(self):
                    self.task_active = False
                    return self.task_data.copy()
            
            # Create webrtc streamer
            webrtc_ctx = webrtc_streamer(
                key="gaze-assessment",
                video_processor_factory=GazeAssessmentProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False
                },
                async_processing=True,
            )
            
            # Task controls
            if webrtc_ctx.video_processor:
                col_start, col_stop, col_next = st.columns(3)
                
                with col_start:
                    if st.button("▶️ Start Task", disabled=st.session_state.assessment_active):
                        st.session_state.assessment_active = True
                        st.session_state.task_start_time = time.time()
                        webrtc_ctx.video_processor.start_task()
                        st.rerun()
                
                with col_stop:
                    if st.button("⏹️ Stop Task", disabled=not st.session_state.assessment_active):
                        if st.session_state.assessment_active:
                            task_data = webrtc_ctx.video_processor.stop_task()
                            st.session_state.task_data[current_task_name] = task_data
                            st.session_state.assessment_active = False
                            
                            # Save gaze data to database
                            try:
                                db_manager.save_gaze_data_batch(
                                    st.session_state.assessment_id,
                                    current_task_name,
                                    current_task['type'],
                                    task_data
                                )
                            except Exception as e:
                                st.error(f"Error saving gaze data: {e}")
                            
                            st.success(f"Task '{current_task_name}' completed!")
                
                with col_next:
                    task_completed = current_task_name in st.session_state.task_data
                    if st.button("Next Task ➡️", disabled=not task_completed):
                        st.session_state.current_task_index += 1
                        st.rerun()
                
                # Task progress
                if st.session_state.assessment_active and st.session_state.task_start_time:
                    elapsed_time = time.time() - st.session_state.task_start_time
                    remaining_time = max(0, current_task['duration'] - elapsed_time)
                    
                    progress = min(elapsed_time / current_task['duration'], 1.0)
                    st.progress(progress)
                    
                    if remaining_time > 0:
                        st.write(f"⏱️ Time remaining: {remaining_time:.1f} seconds")
                    else:
                        st.success("⏰ Task time completed! You can stop the task now.")
                
                # Show task completion status
                st.subheader("Task Completion Status")
                for i, task_name in enumerate(selected_tasks):
                    status = "✅ Completed" if task_name in st.session_state.task_data else "⏳ Pending"
                    current_indicator = "👉 " if i == st.session_state.current_task_index else ""
                    st.write(f"{current_indicator}{task_name}: {status}")
        
        else:
            # All tasks completed
            st.success("🎉 All assessment tasks completed!")
            
            # Process and store results
            if st.session_state.task_data:
                processed_gaze_data = process_all_task_data(st.session_state.task_data, available_tasks)
                st.session_state.gaze_assessment_results = processed_gaze_data
                
                # Show summary
                show_gaze_assessment_summary(processed_gaze_data)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Back to Questionnaire"):
                    st.session_state.current_step = 1
                    st.rerun()
            
            with col2:
                if st.button("🔄 Restart Assessment"):
                    # Reset assessment state
                    st.session_state.current_task_index = 0
                    st.session_state.task_data = {}
                    st.session_state.assessment_active = False
                    st.session_state.task_start_time = None
                    st.rerun()
            
            with col3:
                if st.button("Next: View Results ➡️", type="primary"):
                    st.session_state.current_step = 3
                    st.rerun()

def process_all_task_data(task_data_dict, available_tasks):
    """Process gaze data from all completed tasks"""
    processed_results = {}
    
    for task_name, task_data in task_data_dict.items():
        # Find task configuration
        task_config = next((task for task in available_tasks if task['name'] == task_name), None)
        
        if task_config and task_data:
            # Analyze task performance
            task_analysis = analyze_task_performance(task_data, task_config['type'])
            
            processed_results[task_name] = {
                'task_type': task_config['type'],
                'duration': task_config['duration'],
                'raw_data': task_data,
                'analysis': task_analysis
            }
    
    # Calculate overall metrics
    if processed_results:
        processed_results['overall_metrics'] = calculate_overall_gaze_metrics(processed_results)
    
    return processed_results

def calculate_overall_gaze_metrics(task_results):
    """Calculate overall gaze metrics across all tasks"""
    all_data = []
    
    # Combine data from all tasks
    for task_name, task_result in task_results.items():
        if task_name != 'overall_metrics' and 'raw_data' in task_result:
            all_data.extend(task_result['raw_data'])
    
    if not all_data:
        return {}
    
    # Calculate combined metrics
    from utils.data_processor import DataProcessor
    processor = DataProcessor()
    
    overall_metrics = processor.process_gaze_data(all_data)
    
    # Add task-specific aggregations
    task_performances = {}
    for task_name, task_result in task_results.items():
        if task_name != 'overall_metrics' and 'analysis' in task_result:
            analysis = task_result['analysis']
            task_performances[task_name] = {
                'eye_contact_score': analysis.get('avg_eye_contact_score', 0),
                'social_attention_score': analysis.get('avg_social_attention_score', 0),
                'face_detection_rate': analysis.get('face_detection_rate', 0),
                'gaze_stability': analysis.get('gaze_stability', 0)
            }
    
    overall_metrics['task_performances'] = task_performances
    
    return overall_metrics

def show_gaze_assessment_summary(gaze_results):
    """Display summary of gaze assessment results"""
    st.subheader("📊 Gaze Assessment Summary")
    
    if 'overall_metrics' not in gaze_results:
        st.write("No data available for summary.")
        return
    
    overall = gaze_results['overall_metrics']
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Face Detection",
            f"{overall.get('face_detection_rate', 0):.1%}",
            help="Percentage of frames where face was detected"
        )
    
    with col2:
        st.metric(
            "Avg Eye Contact",
            f"{overall.get('avg_eye_contact_score', 0):.2f}",
            help="Average eye contact score (0-1 scale)"
        )
    
    with col3:
        st.metric(
            "Social Attention",
            f"{overall.get('avg_social_attention_score', 0):.2f}",
            help="Average social attention score (0-1 scale)"
        )
    
    with col4:
        total_time = overall.get('total_samples', 0) * 33  # Convert to milliseconds
        st.metric(
            "Assessment Duration",
            f"{total_time/1000:.1f}s",
            help="Total duration of gaze assessment"
        )
    
    # Task-specific results
    if 'task_performances' in overall:
        st.subheader("Task-Specific Performance")
        
        task_perf = overall['task_performances']
        
        # Create performance comparison chart
        import plotly.graph_objects as go
        
        tasks = list(task_perf.keys())
        eye_contact_scores = [task_perf[task]['eye_contact_score'] for task in tasks]
        social_attention_scores = [task_perf[task]['social_attention_score'] for task in tasks]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Eye Contact',
            x=tasks,
            y=eye_contact_scores,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Social Attention',
            x=tasks,
            y=social_attention_scores,
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Task Performance Comparison',
            xaxis_title='Tasks',
            yaxis_title='Score (0-1)',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
