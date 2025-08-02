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

class VisualPatternProcessor(VideoProcessorBase):
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
        self.current_pattern = None
        self.pattern_start_time = None
        self.gaze_data = []
        self.frame_count = 0
        self.pattern_fixations = 0
        self.random_fixations = 0
        
        # Pattern regions
        self.pattern_regions = []
        self.random_regions = []
        
    def set_pattern(self, pattern_type, pattern_regions, random_regions):
        """Set current pattern and regions"""
        self.current_pattern = pattern_type
        self.pattern_regions = pattern_regions
        self.random_regions = random_regions
        self.pattern_start_time = time.time()
        
    def start_test(self):
        self.test_active = True
        self.frame_count = 0
        self.gaze_data = []
        self.pattern_fixations = 0
        self.random_fixations = 0
        
    def stop_test(self):
        self.test_active = False
        return self.get_results()
        
    def get_results(self):
        if not self.gaze_data:
            return {}
            
        total_fixations = self.pattern_fixations + self.random_fixations
        
        # Calculate fixation duration and scanning patterns
        fixation_durations = []
        if len(self.gaze_data) > 1:
            for i in range(1, len(self.gaze_data)):
                duration = self.gaze_data[i]['timestamp'] - self.gaze_data[i-1]['timestamp']
                fixation_durations.append(duration)
        
        # Calculate gaze dispersion
        gaze_x_coords = [g['gaze_x'] for g in self.gaze_data]
        gaze_y_coords = [g['gaze_y'] for g in self.gaze_data]
        
        results = {
            'total_gaze_points': len(self.gaze_data),
            'pattern_fixations': self.pattern_fixations,
            'random_fixations': self.random_fixations,
            'pattern_preference_ratio': self.pattern_fixations / max(total_fixations, 1),
            'avg_fixation_duration': np.mean(fixation_durations) if fixation_durations else 0,
            'fixation_variability': np.std(fixation_durations) if fixation_durations else 0,
            'gaze_dispersion_x': np.std(gaze_x_coords) if gaze_x_coords else 0,
            'gaze_dispersion_y': np.std(gaze_y_coords) if gaze_y_coords else 0,
            'scanning_efficiency': len(set([(int(g['gaze_x']/50), int(g['gaze_y']/50)) for g in self.gaze_data])),
            'pattern_type': self.current_pattern
        }
        
        return results
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if not self.test_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process every 3rd frame
        if self.frame_count % 3 == 0:
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
                    
                    # Check pattern vs random regions
                    is_pattern = False
                    is_random = False
                    
                    for region in self.pattern_regions:
                        if (region[0] <= gaze_x <= region[2] and region[1] <= gaze_y <= region[3]):
                            self.pattern_fixations += 1
                            is_pattern = True
                            break
                    
                    for region in self.random_regions:
                        if (region[0] <= gaze_x <= region[2] and region[1] <= gaze_y <= region[3]):
                            self.random_fixations += 1
                            is_random = True
                            break
                    
                    # Store gaze data
                    self.gaze_data.append({
                        'timestamp': time.time(),
                        'gaze_x': gaze_x,
                        'gaze_y': gaze_y,
                        'is_pattern': is_pattern,
                        'is_random': is_random,
                        'pattern_type': self.current_pattern,
                        'frame_count': self.frame_count
                    })
                    
                    # Draw gaze point
                    color = (255, 0, 0) if is_pattern else (0, 0, 255) if is_random else (255, 255, 255)
                    cv2.circle(img, (int(gaze_x), int(gaze_y)), 6, color, -1)
                    
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        
        # Draw regions
        for region in self.pattern_regions:
            cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 2)
            cv2.putText(img, "PATTERN", (region[0], region[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for region in self.random_regions:
            cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 2)
            cv2.putText(img, "RANDOM", (region[0], region[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display info
        if self.test_active:
            cv2.putText(img, f"Pattern: {self.current_pattern}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, f"Pattern Fixations: {self.pattern_fixations}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img, f"Random Fixations: {self.random_fixations}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def show_visual_pattern_test_page():
    st.header("🎨 Visual Pattern Test")
    
    st.markdown("""
    This test evaluates your preference for patterns and repetitive visual elements.
    People with ASD often show enhanced pattern recognition and preference for structured visuals.
    """)
    
    # Pattern test types
    pattern_tests = [
        {
            "name": "Test 1: Geometric Patterns",
            "description": "Observe geometric patterns vs random shapes",
            "duration": 20,
            "pattern_type": "geometric"
        },
        {
            "name": "Test 2: Repetitive Sequences",
            "description": "View repetitive sequences vs irregular arrangements",
            "duration": 25,
            "pattern_type": "sequences"
        },
        {
            "name": "Test 3: Symmetrical Designs",
            "description": "Compare symmetrical vs asymmetrical designs",
            "duration": 20,
            "pattern_type": "symmetry"
        }
    ]
    
    # Initialize test state
    if 'pattern_test_phase' not in st.session_state:
        st.session_state.pattern_test_phase = 0
    if 'pattern_test_active' not in st.session_state:
        st.session_state.pattern_test_active = False
    if 'pattern_test_results' not in st.session_state:
        st.session_state.pattern_test_results = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pattern Recognition Analysis")
        
        # WebRTC setup
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="visual_pattern_test",
            video_processor_factory=VisualPatternProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display current pattern test
        if st.session_state.pattern_test_active and st.session_state.pattern_test_phase < len(pattern_tests):
            current_test = pattern_tests[st.session_state.pattern_test_phase]
            
            stimulus_placeholder = st.empty()
            
            with stimulus_placeholder.container():
                st.markdown(f"### {current_test['name']}")
                
                if current_test['pattern_type'] == 'geometric':
                    show_geometric_patterns(webrtc_ctx)
                elif current_test['pattern_type'] == 'sequences':
                    show_sequence_patterns(webrtc_ctx)
                elif current_test['pattern_type'] == 'symmetry':
                    show_symmetry_patterns(webrtc_ctx)
    
    with col2:
        st.subheader("Test Controls")
        
        # Current test info
        if st.session_state.pattern_test_phase < len(pattern_tests):
            current_test = pattern_tests[st.session_state.pattern_test_phase]
            st.info(f"**Current:** {current_test['name']}")
            st.write(f"**Task:** {current_test['description']}")
            st.write(f"**Duration:** {current_test['duration']} seconds")
        
        # Progress
        progress = st.session_state.pattern_test_phase / len(pattern_tests)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.pattern_test_phase}/{len(pattern_tests)} tests")
        
        # Controls
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start", disabled=st.session_state.pattern_test_active):
                if webrtc_ctx.video_processor and st.session_state.pattern_test_phase < len(pattern_tests):
                    st.session_state.pattern_test_active = True
                    webrtc_ctx.video_processor.start_test()
                    st.success("Test started!")
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop", disabled=not st.session_state.pattern_test_active):
                if webrtc_ctx.video_processor and st.session_state.pattern_test_active:
                    results = webrtc_ctx.video_processor.stop_test()
                    st.session_state.pattern_test_results[f"test_{st.session_state.pattern_test_phase}"] = results
                    st.session_state.pattern_test_active = False
                    st.session_state.pattern_test_phase += 1
                    
                    # Save to database
                    try:
                        if st.session_state.assessment_id:
                            db_manager.save_gaze_data_batch(
                                st.session_state.assessment_id,
                                f"visual_pattern_test_{st.session_state.pattern_test_phase-1}",
                                "visual_pattern",
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
                st.session_state.current_test = 2  # Back to social attention
                st.rerun()
        
        with col_next:
            if st.session_state.pattern_test_phase >= len(pattern_tests):
                if st.button("Next Test ➡️", type="primary"):
                    st.session_state.current_test = 4  # Go to motion tracking
                    st.rerun()
        
        # Results summary
        if st.session_state.pattern_test_phase >= len(pattern_tests):
            st.success("✅ Visual Pattern Test Completed!")
            
            with st.expander("📊 Quick Results", expanded=True):
                show_pattern_test_summary()

def show_geometric_patterns(webrtc_ctx):
    """Display geometric patterns"""
    patterns_html = """
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 30px; padding: 20px;'>
        <div style='text-align: center; border: 3px solid red; border-radius: 15px; padding: 20px; background: #fff5f5;'>
            <h3>Geometric Patterns</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0;'>
                <div style='width: 40px; height: 40px; background: #ff6b6b; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #4ecdc4; border-radius: 50%; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #45b7d1; transform: rotate(45deg); margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #ff6b6b; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #4ecdc4; border-radius: 50%; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #45b7d1; transform: rotate(45deg); margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #ff6b6b; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #4ecdc4; border-radius: 50%; margin: auto;'></div>
                <div style='width: 40px; height: 40px; background: #45b7d1; transform: rotate(45deg); margin: auto;'></div>
            </div>
        </div>
        <div style='text-align: center; border: 3px solid blue; border-radius: 15px; padding: 20px; background: #f0f8ff;'>
            <h3>Random Shapes</h3>
            <div style='position: relative; height: 200px; margin: 20px 0;'>
                <div style='position: absolute; top: 20px; left: 30px; width: 25px; height: 35px; background: #feca57; border-radius: 10px;'></div>
                <div style='position: absolute; top: 80px; left: 100px; width: 45px; height: 20px; background: #ff9ff3; transform: rotate(30deg);'></div>
                <div style='position: absolute; top: 50px; left: 180px; width: 30px; height: 30px; background: #48dbfb; border-radius: 50%;'></div>
                <div style='position: absolute; top: 120px; left: 60px; width: 40px; height: 15px; background: #0abde3;'></div>
                <div style='position: absolute; top: 40px; left: 250px; width: 20px; height: 40px; background: #fd79a8; transform: rotate(-20deg);'></div>
            </div>
        </div>
    </div>
    """
    st.markdown(patterns_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        pattern_regions = [(50, 100, 350, 350)]  # Left side - patterns
        random_regions = [(450, 100, 750, 350)]  # Right side - random
        webrtc_ctx.video_processor.set_pattern("geometric", pattern_regions, random_regions)

def show_sequence_patterns(webrtc_ctx):
    """Display sequence patterns"""
    sequences_html = """
    <div style='padding: 30px; background: #f8f9fa; border-radius: 15px;'>
        <h2 style='text-align: center; margin-bottom: 30px;'>Pattern Sequences</h2>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 40px;'>
            <div style='border: 3px solid red; border-radius: 10px; padding: 20px; background: white;'>
                <h3 style='text-align: center; color: red;'>Repetitive Sequence</h3>
                <div style='display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px; margin: 20px 0;'>
                    <span style='font-size: 30px;'>🔴</span>
                    <span style='font-size: 30px;'>🔵</span>
                    <span style='font-size: 30px;'>🟢</span>
                    <span style='font-size: 30px;'>🔴</span>
                    <span style='font-size: 30px;'>🔵</span>
                    <span style='font-size: 30px;'>🟢</span>
                    <span style='font-size: 30px;'>🔴</span>
                    <span style='font-size: 30px;'>🔵</span>
                    <span style='font-size: 30px;'>🟢</span>
                </div>
                <p style='text-align: center; font-size: 14px;'>ABC-ABC-ABC Pattern</p>
            </div>
            
            <div style='border: 3px solid blue; border-radius: 10px; padding: 20px; background: white;'>
                <h3 style='text-align: center; color: blue;'>Random Arrangement</h3>
                <div style='display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px; margin: 20px 0;'>
                    <span style='font-size: 30px;'>🟡</span>
                    <span style='font-size: 30px;'>🔵</span>
                    <span style='font-size: 30px;'>🔴</span>
                    <span style='font-size: 30px;'>🟢</span>
                    <span style='font-size: 30px;'>🟡</span>
                    <span style='font-size: 30px;'>🔴</span>
                    <span style='font-size: 30px;'>🔵</span>
                    <span style='font-size: 30px;'>🟢</span>
                    <span style='font-size: 30px;'>🔴</span>
                </div>
                <p style='text-align: center; font-size: 14px;'>No Pattern</p>
            </div>
        </div>
    </div>
    """
    st.markdown(sequences_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        pattern_regions = [(50, 150, 350, 350)]  # Left side - sequence
        random_regions = [(450, 150, 750, 350)]  # Right side - random
        webrtc_ctx.video_processor.set_pattern("sequences", pattern_regions, random_regions)

def show_symmetry_patterns(webrtc_ctx):
    """Display symmetry patterns"""
    symmetry_html = """
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 30px; padding: 20px;'>
        <div style='text-align: center; border: 3px solid red; border-radius: 15px; padding: 30px; background: #fff5f5;'>
            <h3>Symmetrical Design</h3>
            <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 20px auto; width: 200px;'>
                <div></div>
                <div></div>
                <div style='width: 20px; height: 20px; background: #e74c3c; margin: auto; border-radius: 50%;'></div>
                <div></div>
                <div></div>
                
                <div></div>
                <div style='width: 20px; height: 20px; background: #3498db; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #f39c12; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #3498db; margin: auto;'></div>
                <div></div>
                
                <div style='width: 20px; height: 20px; background: #2ecc71; margin: auto; border-radius: 50%;'></div>
                <div style='width: 20px; height: 20px; background: #9b59b6; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #e67e22; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #9b59b6; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #2ecc71; margin: auto; border-radius: 50%;'></div>
                
                <div></div>
                <div style='width: 20px; height: 20px; background: #3498db; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #f39c12; margin: auto;'></div>
                <div style='width: 20px; height: 20px; background: #3498db; margin: auto;'></div>
                <div></div>
                
                <div></div>
                <div></div>
                <div style='width: 20px; height: 20px; background: #e74c3c; margin: auto; border-radius: 50%;'></div>
                <div></div>
                <div></div>
            </div>
        </div>
        
        <div style='text-align: center; border: 3px solid blue; border-radius: 15px; padding: 30px; background: #f0f8ff;'>
            <h3>Asymmetrical Design</h3>
            <div style='position: relative; height: 180px; margin: 20px auto; width: 200px;'>
                <div style='position: absolute; top: 10px; left: 20px; width: 25px; height: 25px; background: #e74c3c; border-radius: 50%;'></div>
                <div style='position: absolute; top: 50px; left: 80px; width: 30px; height: 15px; background: #3498db;'></div>
                <div style='position: absolute; top: 90px; left: 40px; width: 20px; height: 30px; background: #2ecc71;'></div>
                <div style='position: absolute; top: 30px; left: 150px; width: 35px; height: 20px; background: #f39c12; transform: rotate(45deg);'></div>
                <div style='position: absolute; top: 120px; left: 120px; width: 15px; height: 15px; background: #9b59b6; border-radius: 50%;'></div>
                <div style='position: absolute; top: 70px; left: 10px; width: 20px; height: 25px; background: #e67e22; transform: rotate(-30deg);'></div>
                <div style='position: absolute; top: 140px; left: 170px; width: 25px; height: 10px; background: #1abc9c;'></div>
            </div>
        </div>
    </div>
    """
    st.markdown(symmetry_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        pattern_regions = [(50, 100, 350, 350)]  # Left side - symmetrical
        random_regions = [(450, 100, 750, 350)]  # Right side - asymmetrical
        webrtc_ctx.video_processor.set_pattern("symmetry", pattern_regions, random_regions)

def show_pattern_test_summary():
    """Display summary of pattern test results"""
    if not st.session_state.pattern_test_results:
        st.write("No results available yet.")
        return
    
    # Aggregate results
    total_pattern_fixations = 0
    total_random_fixations = 0
    pattern_preferences = []
    avg_fixation_durations = []
    tests_completed = 0
    
    for test_key, results in st.session_state.pattern_test_results.items():
        if results:
            total_pattern_fixations += results.get('pattern_fixations', 0)
            total_random_fixations += results.get('random_fixations', 0)
            pattern_preferences.append(results.get('pattern_preference_ratio', 0))
            avg_fixation_durations.append(results.get('avg_fixation_duration', 0))
            tests_completed += 1
    
    if tests_completed > 0:
        overall_pattern_preference = np.mean(pattern_preferences)
        overall_fixation_duration = np.mean(avg_fixation_durations)
        total_fixations = total_pattern_fixations + total_random_fixations
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pattern Preference", f"{overall_pattern_preference:.1%}")
            st.metric("Avg Fixation Duration", f"{overall_fixation_duration:.3f}s")
        
        with col2:
            st.metric("Tests Completed", tests_completed)
            st.metric("Total Fixations", total_fixations)
        
        # Visualization
        if total_fixations > 0:
            fixation_data = {
                'Visual Type': ['Patterns', 'Random'],
                'Fixation Count': [total_pattern_fixations, total_random_fixations]
            }
            
            fig = px.pie(fixation_data, values='Fixation Count', names='Visual Type',
                        title="Pattern vs Random Visual Attention",
                        color_discrete_map={'Patterns': '#dc3545', 'Random': '#007bff'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern preference by test
            test_names = [f"Test {i+1}" for i in range(len(pattern_preferences))]
            fig2 = px.bar(x=test_names, y=pattern_preferences,
                         title="Pattern Preference by Test Type",
                         labels={'x': 'Test', 'y': 'Pattern Preference Ratio'})
            fig2.update_traces(marker_color='#fd7e14')
            st.plotly_chart(fig2, use_container_width=True)