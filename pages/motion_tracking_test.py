import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import math
from database.models import db_manager
import plotly.express as px
import plotly.graph_objects as go

class MotionTrackingProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.test_active = False
        self.current_motion = None
        self.motion_start_time = None
        self.gaze_data = []
        self.frame_count = 0
        
        # Motion tracking variables
        self.target_position = [400, 300]  # Center of screen
        self.target_speed = 2
        self.target_direction = 0
        self.motion_type = "circular"
        
        # Tracking accuracy metrics
        self.tracking_accuracy_scores = []
        self.smooth_pursuit_quality = []
        self.saccadic_movements = []
        
    def set_motion_type(self, motion_type):
        """Set the type of motion for tracking"""
        self.motion_type = motion_type
        self.motion_start_time = time.time()
        self.target_direction = 0
        
    def start_test(self):
        self.test_active = True
        self.frame_count = 0
        self.gaze_data = []
        self.tracking_accuracy_scores = []
        self.smooth_pursuit_quality = []
        self.saccadic_movements = []
        
    def stop_test(self):
        self.test_active = False
        return self.get_results()
        
    def get_results(self):
        if not self.gaze_data:
            return {}
            
        # Calculate tracking metrics
        avg_tracking_accuracy = np.mean(self.tracking_accuracy_scores) if self.tracking_accuracy_scores else 0
        pursuit_quality = np.mean(self.smooth_pursuit_quality) if self.smooth_pursuit_quality else 0
        saccadic_count = len(self.saccadic_movements)
        
        # Calculate gaze velocity and smoothness
        velocities = []
        if len(self.gaze_data) > 1:
            for i in range(1, len(self.gaze_data)):
                dt = self.gaze_data[i]['timestamp'] - self.gaze_data[i-1]['timestamp']
                if dt > 0:
                    dx = self.gaze_data[i]['gaze_x'] - self.gaze_data[i-1]['gaze_x']
                    dy = self.gaze_data[i]['gaze_y'] - self.gaze_data[i-1]['gaze_y']
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
                    velocities.append(velocity)
        
        results = {
            'total_gaze_points': len(self.gaze_data),
            'tracking_accuracy': avg_tracking_accuracy,
            'smooth_pursuit_quality': pursuit_quality,
            'saccadic_movements_count': saccadic_count,
            'avg_gaze_velocity': np.mean(velocities) if velocities else 0,
            'gaze_velocity_std': np.std(velocities) if velocities else 0,
            'motion_type': self.motion_type,
            'test_duration': time.time() - self.motion_start_time if self.motion_start_time else 0
        }
        
        return results
    
    def update_target_position(self):
        """Update target position based on motion type"""
        if not self.motion_start_time:
            return
            
        elapsed_time = time.time() - self.motion_start_time
        
        if self.motion_type == "circular":
            # Circular motion
            radius = 150
            center_x, center_y = 400, 300
            angle = elapsed_time * self.target_speed
            self.target_position[0] = center_x + radius * math.cos(angle)
            self.target_position[1] = center_y + radius * math.sin(angle)
            
        elif self.motion_type == "horizontal":
            # Horizontal motion
            amplitude = 300
            center_x, center_y = 400, 300
            self.target_position[0] = center_x + amplitude * math.sin(elapsed_time * self.target_speed)
            self.target_position[1] = center_y
            
        elif self.motion_type == "vertical":
            # Vertical motion
            amplitude = 200
            center_x, center_y = 400, 300
            self.target_position[0] = center_x
            self.target_position[1] = center_y + amplitude * math.sin(elapsed_time * self.target_speed)
            
        elif self.motion_type == "figure8":
            # Figure-8 motion
            amplitude_x, amplitude_y = 200, 150
            center_x, center_y = 400, 300
            t = elapsed_time * self.target_speed
            self.target_position[0] = center_x + amplitude_x * math.sin(t)
            self.target_position[1] = center_y + amplitude_y * math.sin(2 * t)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if not self.test_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Update target position
        self.update_target_position()
        
        # Process every 2nd frame
        if self.frame_count % 2 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = img.shape[:2]
                    
                    # Calculate gaze point
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    left_eye_center = np.mean([[face_landmarks.landmark[i].x * w, 
                                              face_landmarks.landmark[i].y * h] for i in left_eye_indices], axis=0)
                    right_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                               face_landmarks.landmark[i].y * h] for i in right_eye_indices], axis=0)
                    
                    gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    
                    # Calculate tracking accuracy
                    distance_to_target = math.sqrt(
                        (gaze_x - self.target_position[0])**2 + 
                        (gaze_y - self.target_position[1])**2
                    )
                    
                    # Accuracy score (closer = better, max distance = 100 pixels for full score)
                    accuracy_score = max(0, 1 - distance_to_target / 100)
                    self.tracking_accuracy_scores.append(accuracy_score)
                    
                    # Detect saccadic movements (rapid gaze changes)
                    if len(self.gaze_data) > 0:
                        last_gaze = self.gaze_data[-1]
                        gaze_movement = math.sqrt(
                            (gaze_x - last_gaze['gaze_x'])**2 + 
                            (gaze_y - last_gaze['gaze_y'])**2
                        )
                        
                        # If movement is large and fast, it's likely a saccade
                        time_diff = time.time() - last_gaze['timestamp']
                        if time_diff > 0 and gaze_movement / time_diff > 500:  # Threshold for saccadic movement
                            self.saccadic_movements.append({
                                'timestamp': time.time(),
                                'magnitude': gaze_movement,
                                'velocity': gaze_movement / time_diff
                            })
                    
                    # Calculate smooth pursuit quality
                    if len(self.gaze_data) > 5:
                        # Look at recent gaze positions to assess smoothness
                        recent_positions = [(g['gaze_x'], g['gaze_y']) for g in self.gaze_data[-5:]]
                        recent_positions.append((gaze_x, gaze_y))
                        
                        # Calculate smoothness as inverse of position variance
                        x_positions = [pos[0] for pos in recent_positions]
                        y_positions = [pos[1] for pos in recent_positions]
                        smoothness = 1 / (1 + np.var(x_positions) + np.var(y_positions))
                        self.smooth_pursuit_quality.append(smoothness)
                    
                    # Store gaze data
                    self.gaze_data.append({
                        'timestamp': time.time(),
                        'gaze_x': gaze_x,
                        'gaze_y': gaze_y,
                        'target_x': self.target_position[0],
                        'target_y': self.target_position[1],
                        'tracking_accuracy': accuracy_score,
                        'distance_to_target': distance_to_target,
                        'motion_type': self.motion_type,
                        'frame_count': self.frame_count
                    })
                    
                    # Draw gaze point
                    cv2.circle(img, (int(gaze_x), int(gaze_y)), 8, (0, 255, 0), -1)
                    
                    # Draw tracking line
                    cv2.line(img, (int(gaze_x), int(gaze_y)), 
                            (int(self.target_position[0]), int(self.target_position[1])), 
                            (255, 255, 0), 2)
                    
                    # Draw face mesh (simplified)
                    self.mp_drawing.draw_landmarks(
                        img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        
        # Draw moving target
        cv2.circle(img, (int(self.target_position[0]), int(self.target_position[1])), 15, (0, 0, 255), -1)
        cv2.circle(img, (int(self.target_position[0]), int(self.target_position[1])), 20, (0, 0, 255), 2)
        
        # Display test info
        if self.test_active:
            cv2.putText(img, f"Motion: {self.motion_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if self.tracking_accuracy_scores:
                avg_accuracy = np.mean(self.tracking_accuracy_scores[-10:])  # Last 10 measurements
                cv2.putText(img, f"Accuracy: {avg_accuracy:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Saccades: {len(self.saccadic_movements)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def show_motion_tracking_test_page():
    st.header("🎬 Motion Tracking Test")
    
    st.markdown("""
    This test evaluates your ability to track moving objects with smooth eye movements.
    It measures smooth pursuit and saccadic eye movement patterns associated with ASD.
    """)
    
    # Motion types
    motion_tests = [
        {
            "name": "Test 1: Circular Motion",
            "description": "Follow a target moving in a circular pattern",
            "duration": 20,
            "motion_type": "circular"
        },
        {
            "name": "Test 2: Horizontal Motion",
            "description": "Track horizontal back-and-forth movement",
            "duration": 15,
            "motion_type": "horizontal"
        },
        {
            "name": "Test 3: Vertical Motion",
            "description": "Follow vertical up-and-down movement",
            "duration": 15,
            "motion_type": "vertical"
        },
        {
            "name": "Test 4: Figure-8 Motion",
            "description": "Track complex figure-8 movement pattern",
            "duration": 25,
            "motion_type": "figure8"
        }
    ]
    
    # Initialize test state
    if 'motion_test_phase' not in st.session_state:
        st.session_state.motion_test_phase = 0
    if 'motion_test_active' not in st.session_state:
        st.session_state.motion_test_active = False
    if 'motion_test_results' not in st.session_state:
        st.session_state.motion_test_results = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Motion Tracking Analysis")
        
        # Instructions
        st.info("👁️ Follow the red target with your eyes as it moves around the screen. Try to keep your gaze on the target as smoothly as possible.")
        
        # WebRTC setup
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="motion_tracking_test",
            video_processor_factory=MotionTrackingProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Set motion type if test is active
        if (st.session_state.motion_test_active and 
            st.session_state.motion_test_phase < len(motion_tests) and 
            webrtc_ctx.video_processor):
            
            current_test = motion_tests[st.session_state.motion_test_phase]
            webrtc_ctx.video_processor.set_motion_type(current_test['motion_type'])
    
    with col2:
        st.subheader("Test Controls")
        
        # Current test info
        if st.session_state.motion_test_phase < len(motion_tests):
            current_test = motion_tests[st.session_state.motion_test_phase]
            st.info(f"**Current:** {current_test['name']}")
            st.write(f"**Task:** {current_test['description']}")
            st.write(f"**Duration:** {current_test['duration']} seconds")
        
        # Progress
        progress = st.session_state.motion_test_phase / len(motion_tests)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.motion_test_phase}/{len(motion_tests)} tests")
        
        # Real-time metrics
        if (st.session_state.motion_test_active and webrtc_ctx.video_processor and 
            hasattr(webrtc_ctx.video_processor, 'tracking_accuracy_scores')):
            
            with st.container():
                st.markdown("**Live Metrics:**")
                
                if webrtc_ctx.video_processor.tracking_accuracy_scores:
                    recent_accuracy = np.mean(webrtc_ctx.video_processor.tracking_accuracy_scores[-10:])
                    st.metric("Current Accuracy", f"{recent_accuracy:.1%}")
                
                st.metric("Saccadic Movements", len(webrtc_ctx.video_processor.saccadic_movements))
                
                if webrtc_ctx.video_processor.smooth_pursuit_quality:
                    recent_smoothness = np.mean(webrtc_ctx.video_processor.smooth_pursuit_quality[-10:])
                    st.metric("Pursuit Smoothness", f"{recent_smoothness:.3f}")
        
        # Controls
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start", disabled=st.session_state.motion_test_active):
                if webrtc_ctx.video_processor and st.session_state.motion_test_phase < len(motion_tests):
                    st.session_state.motion_test_active = True
                    webrtc_ctx.video_processor.start_test()
                    st.success("Motion test started!")
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop", disabled=not st.session_state.motion_test_active):
                if webrtc_ctx.video_processor and st.session_state.motion_test_active:
                    results = webrtc_ctx.video_processor.stop_test()
                    st.session_state.motion_test_results[f"test_{st.session_state.motion_test_phase}"] = results
                    st.session_state.motion_test_active = False
                    st.session_state.motion_test_phase += 1
                    
                    # Save to database
                    try:
                        if st.session_state.assessment_id:
                            db_manager.save_gaze_data_batch(
                                st.session_state.assessment_id,
                                f"motion_tracking_test_{st.session_state.motion_test_phase-1}",
                                "motion_tracking",
                                results
                            )
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                    
                    st.success("Test completed!")
                    st.rerun()
        
        # Navigation
        st.markdown("---")
        col_back, col_next = st.columns(2)
        
        with col_back:
            if st.button("⬅️ Previous Test"):
                st.session_state.current_test = 3  # Back to visual pattern
                st.rerun()
        
        with col_next:
            if st.session_state.motion_test_phase >= len(motion_tests):
                if st.button("View Results ➡️", type="primary"):
                    st.session_state.current_test = 5  # Go to results
                    st.rerun()
        
        # Results summary
        if st.session_state.motion_test_phase >= len(motion_tests):
            st.success("✅ Motion Tracking Test Completed!")
            
            with st.expander("📊 Quick Results", expanded=True):
                show_motion_test_summary()

def show_motion_test_summary():
    """Display summary of motion tracking test results"""
    if not st.session_state.motion_test_results:
        st.write("No results available yet.")
        return
    
    # Aggregate results
    tracking_accuracies = []
    pursuit_qualities = []
    saccadic_counts = []
    gaze_velocities = []
    tests_completed = 0
    
    for test_key, results in st.session_state.motion_test_results.items():
        if results:
            tracking_accuracies.append(results.get('tracking_accuracy', 0))
            pursuit_qualities.append(results.get('smooth_pursuit_quality', 0))
            saccadic_counts.append(results.get('saccadic_movements_count', 0))
            gaze_velocities.append(results.get('avg_gaze_velocity', 0))
            tests_completed += 1
    
    if tests_completed > 0:
        avg_tracking_accuracy = np.mean(tracking_accuracies)
        avg_pursuit_quality = np.mean(pursuit_qualities)
        total_saccades = sum(saccadic_counts)
        avg_gaze_velocity = np.mean(gaze_velocities)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avg Tracking Accuracy", f"{avg_tracking_accuracy:.1%}")
            st.metric("Smooth Pursuit Quality", f"{avg_pursuit_quality:.3f}")
        
        with col2:
            st.metric("Total Saccadic Movements", total_saccades)
            st.metric("Avg Gaze Velocity", f"{avg_gaze_velocity:.1f} px/s")
        
        # Visualizations
        motion_types = ['Circular', 'Horizontal', 'Vertical', 'Figure-8']
        
        # Tracking accuracy by motion type
        fig1 = px.bar(x=motion_types[:len(tracking_accuracies)], y=tracking_accuracies,
                     title="Tracking Accuracy by Motion Type",
                     labels={'x': 'Motion Type', 'y': 'Tracking Accuracy'})
        fig1.update_traces(marker_color='#28a745')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Saccadic movements by motion type
        fig2 = px.bar(x=motion_types[:len(saccadic_counts)], y=saccadic_counts,
                     title="Saccadic Movements by Motion Type",
                     labels={'x': 'Motion Type', 'y': 'Saccadic Count'})
        fig2.update_traces(marker_color='#dc3545')
        st.plotly_chart(fig2, use_container_width=True)