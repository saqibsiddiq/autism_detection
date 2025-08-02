import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import threading
from database.models import db_manager
import plotly.express as px
import plotly.graph_objects as go

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Test variables
        self.test_active = False
        self.current_stimulus = None
        self.stimulus_start_time = None
        self.test_results = []
        self.frame_count = 0
        
        # Eye tracking data
        self.gaze_data = []
        self.face_detected_frames = 0
        self.total_frames = 0
        
        # Stimulus regions (will be set based on stimulus type)
        self.face_region = None
        self.object_region = None
        
    def set_stimulus(self, stimulus_type, regions):
        """Set current stimulus and regions of interest"""
        self.current_stimulus = stimulus_type
        self.face_region = regions.get('face_region')
        self.object_region = regions.get('object_region')
        self.stimulus_start_time = time.time()
        
    def start_test(self):
        self.test_active = True
        self.frame_count = 0
        self.gaze_data = []
        self.face_detected_frames = 0
        self.total_frames = 0
        
    def stop_test(self):
        self.test_active = False
        return self.get_results()
        
    def get_results(self):
        if not self.gaze_data:
            return {}
            
        results = {
            'total_frames': self.total_frames,
            'face_detected_frames': self.face_detected_frames,
            'face_detection_rate': self.face_detected_frames / max(self.total_frames, 1),
            'gaze_points': len(self.gaze_data),
            'avg_gaze_x': np.mean([g['gaze_x'] for g in self.gaze_data]) if self.gaze_data else 0,
            'avg_gaze_y': np.mean([g['gaze_y'] for g in self.gaze_data]) if self.gaze_data else 0,
            'gaze_dispersion_x': np.std([g['gaze_x'] for g in self.gaze_data]) if self.gaze_data else 0,
            'gaze_dispersion_y': np.std([g['gaze_y'] for g in self.gaze_data]) if self.gaze_data else 0,
        }
        
        # Calculate attention to face vs object regions
        if self.face_region and self.object_region:
            face_attention = 0
            object_attention = 0
            
            for gaze in self.gaze_data:
                gx, gy = gaze['gaze_x'], gaze['gaze_y']
                
                # Check if gaze is in face region
                if (self.face_region[0] <= gx <= self.face_region[2] and 
                    self.face_region[1] <= gy <= self.face_region[3]):
                    face_attention += 1
                    
                # Check if gaze is in object region  
                if (self.object_region[0] <= gx <= self.object_region[2] and
                    self.object_region[1] <= gy <= self.object_region[3]):
                    object_attention += 1
            
            total_attention = face_attention + object_attention
            if total_attention > 0:
                results['face_preference_ratio'] = face_attention / total_attention
                results['object_preference_ratio'] = object_attention / total_attention
            else:
                results['face_preference_ratio'] = 0
                results['object_preference_ratio'] = 0
                
            results['face_attention_time'] = face_attention
            results['object_attention_time'] = object_attention
        
        return results
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.total_frames += 1
        
        if not self.test_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process every 3rd frame for performance
        if self.frame_count % 3 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                self.face_detected_frames += 1
                
                for face_landmarks in results.multi_face_landmarks:
                    # Get eye landmarks for gaze estimation
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    h, w = img.shape[:2]
                    
                    # Calculate gaze point (simplified estimation)
                    left_eye_center = np.mean([[face_landmarks.landmark[i].x * w, 
                                              face_landmarks.landmark[i].y * h] for i in left_eye_indices], axis=0)
                    right_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                               face_landmarks.landmark[i].y * h] for i in right_eye_indices], axis=0)
                    
                    gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    
                    # Store gaze data
                    self.gaze_data.append({
                        'timestamp': time.time(),
                        'gaze_x': gaze_x,
                        'gaze_y': gaze_y,
                        'stimulus': self.current_stimulus,
                        'frame_count': self.frame_count
                    })
                    
                    # Draw eye tracking visualization
                    cv2.circle(img, (int(gaze_x), int(gaze_y)), 5, (0, 255, 0), -1)
                    
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        
        # Draw stimulus regions if active
        if self.test_active and self.current_stimulus:
            if self.face_region:
                cv2.rectangle(img, (self.face_region[0], self.face_region[1]), 
                            (self.face_region[2], self.face_region[3]), (255, 0, 0), 2)
                cv2.putText(img, "FACE", (self.face_region[0], self.face_region[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                          
            if self.object_region:
                cv2.rectangle(img, (self.object_region[0], self.object_region[1]),
                            (self.object_region[2], self.object_region[3]), (0, 0, 255), 2)
                cv2.putText(img, "OBJECT", (self.object_region[0], self.object_region[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display test info
        if self.test_active:
            cv2.putText(img, f"Test: {self.current_stimulus}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Frames: {self.total_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def show_face_recognition_test_page():
    st.header("👁️ Face Recognition Test")
    
    st.markdown("""
    This test measures your gaze patterns when viewing human faces compared to objects.
    Research shows that individuals with ASD often show different attention patterns to faces.
    """)
    
    # Test phases
    test_phases = [
        {
            "name": "Phase 1: Face vs Object",
            "description": "Look at the screen while images of faces and objects are displayed",
            "duration": 30,
            "stimulus_type": "face_object_comparison"
        },
        {
            "name": "Phase 2: Multiple Faces", 
            "description": "Multiple faces will be shown - track your natural gaze patterns",
            "duration": 20,
            "stimulus_type": "multiple_faces"
        },
        {
            "name": "Phase 3: Eye Contact",
            "description": "Direct eye contact scenarios - maintain natural viewing",
            "duration": 25,
            "stimulus_type": "eye_contact"
        }
    ]
    
    # Initialize test state
    if 'face_test_phase' not in st.session_state:
        st.session_state.face_test_phase = 0
    if 'face_test_active' not in st.session_state:
        st.session_state.face_test_active = False
    if 'face_test_results' not in st.session_state:
        st.session_state.face_test_results = {}
    
    # Create two columns - one for video, one for controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Video Analysis")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Create video processor
        webrtc_ctx = webrtc_streamer(
            key="face_recognition_test",
            video_processor_factory=FaceRecognitionProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display current stimulus
        if st.session_state.face_test_active and st.session_state.face_test_phase < len(test_phases):
            current_phase = test_phases[st.session_state.face_test_phase]
            
            # Create stimulus display area
            stimulus_placeholder = st.empty()
            
            with stimulus_placeholder.container():
                st.markdown(f"### {current_phase['name']}")
                
                # Generate stimulus based on phase
                if current_phase['stimulus_type'] == 'face_object_comparison':
                    show_face_object_stimulus(webrtc_ctx)
                elif current_phase['stimulus_type'] == 'multiple_faces':
                    show_multiple_faces_stimulus(webrtc_ctx)
                elif current_phase['stimulus_type'] == 'eye_contact':
                    show_eye_contact_stimulus(webrtc_ctx)
    
    with col2:
        st.subheader("Test Controls")
        
        # Show current phase info
        if st.session_state.face_test_phase < len(test_phases):
            current_phase = test_phases[st.session_state.face_test_phase]
            st.info(f"**Current Phase:** {current_phase['name']}")
            st.write(f"**Description:** {current_phase['description']}")
            st.write(f"**Duration:** {current_phase['duration']} seconds")
        
        # Test progress
        progress = st.session_state.face_test_phase / len(test_phases)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.face_test_phase}/{len(test_phases)} phases completed")
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start Phase", disabled=st.session_state.face_test_active):
                if webrtc_ctx.video_processor and st.session_state.face_test_phase < len(test_phases):
                    st.session_state.face_test_active = True
                    webrtc_ctx.video_processor.start_test()
                    st.success("Phase started!")
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Phase", disabled=not st.session_state.face_test_active):
                if webrtc_ctx.video_processor and st.session_state.face_test_active:
                    results = webrtc_ctx.video_processor.stop_test()
                    st.session_state.face_test_results[f"phase_{st.session_state.face_test_phase}"] = results
                    st.session_state.face_test_active = False
                    st.session_state.face_test_phase += 1
                    
                    # Save to database
                    try:
                        if st.session_state.assessment_id:
                            db_manager.save_gaze_data_batch(
                                st.session_state.assessment_id,
                                f"face_recognition_phase_{st.session_state.face_test_phase-1}",
                                "face_recognition",
                                results
                            )
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                    
                    st.success("Phase completed!")
                    st.rerun()
        
        # Navigation buttons
        st.markdown("---")
        
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("⬅️ Back to Overview"):
                st.session_state.current_test = 0
                st.rerun()
        
        with col_next:
            if st.session_state.face_test_phase >= len(test_phases):
                if st.button("Next Test ➡️", type="primary"):
                    st.session_state.current_test = 2  # Go to social attention test
                    st.rerun()
        
        # Show results if test completed
        if st.session_state.face_test_phase >= len(test_phases):
            st.success("✅ Face Recognition Test Completed!")
            
            with st.expander("📊 Quick Results", expanded=True):
                show_face_test_summary()

def show_face_object_stimulus(webrtc_ctx):
    """Display face vs object stimulus"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Face stimulus (using emoji as placeholder)
        face_html = """
        <div style='text-align: center; padding: 40px; border: 3px solid blue; border-radius: 10px; background-color: #f0f8ff;'>
            <div style='font-size: 120px;'>😊</div>
            <h3>Human Face</h3>
        </div>
        """
        st.markdown(face_html, unsafe_allow_html=True)
        
        # Set face region for tracking
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_stimulus("face_object", {
                'face_region': [50, 100, 350, 400],  # x1, y1, x2, y2
                'object_region': [450, 100, 750, 400]
            })
    
    with col2:
        # Object stimulus
        object_html = """
        <div style='text-align: center; padding: 40px; border: 3px solid red; border-radius: 10px; background-color: #fff0f0;'>
            <div style='font-size: 120px;'>🚗</div>
            <h3>Object</h3>
        </div>
        """
        st.markdown(object_html, unsafe_allow_html=True)

def show_multiple_faces_stimulus(webrtc_ctx):
    """Display multiple faces stimulus"""
    faces_html = """
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; padding: 20px;'>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😀</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😃</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😄</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😁</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😆</div>
        </div>
        <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 20px;'>
            <div style='font-size: 80px;'>😊</div>
        </div>
    </div>
    """
    st.markdown(faces_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_stimulus("multiple_faces", {
            'face_region': [0, 0, 800, 600]  # Full screen tracking
        })

def show_eye_contact_stimulus(webrtc_ctx):
    """Display eye contact stimulus"""
    eye_contact_html = """
    <div style='text-align: center; padding: 60px; background: linear-gradient(45deg, #e3f2fd, #bbdefb);'>
        <div style='font-size: 150px; margin-bottom: 20px;'>👁️</div>
        <h2>Maintain Natural Eye Contact</h2>
        <p style='font-size: 18px;'>Look at the eyes naturally - don't force it</p>
    </div>
    """
    st.markdown(eye_contact_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_stimulus("eye_contact", {
            'face_region': [300, 200, 500, 300]  # Eye region
        })

def show_face_test_summary():
    """Display summary of face recognition test results"""
    if not st.session_state.face_test_results:
        st.write("No results available yet.")
        return
    
    # Aggregate results across all phases
    total_face_attention = 0
    total_object_attention = 0
    total_detection_rate = 0
    phases_completed = 0
    
    for phase_key, results in st.session_state.face_test_results.items():
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
        
        # Visualization
        if total_attention > 0:
            attention_data = {
                'Stimulus': ['Faces', 'Objects'],
                'Attention Time': [total_face_attention, total_object_attention]
            }
            
            fig = px.pie(attention_data, values='Attention Time', names='Stimulus',
                        title="Attention Distribution: Faces vs Objects")
            st.plotly_chart(fig, use_container_width=True)