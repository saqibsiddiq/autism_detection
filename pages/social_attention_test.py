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

class SocialAttentionProcessor(VideoProcessorBase):
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
        self.current_scenario = None
        self.scenario_start_time = None
        self.gaze_data = []
        self.frame_count = 0
        self.social_attention_score = 0
        self.non_social_attention_score = 0
        
        # Define regions for social vs non-social stimuli
        self.social_regions = []
        self.non_social_regions = []
        
    def set_scenario(self, scenario_type, social_regions, non_social_regions):
        """Set current scenario and regions of interest"""
        self.current_scenario = scenario_type
        self.social_regions = social_regions
        self.non_social_regions = non_social_regions
        self.scenario_start_time = time.time()
        
    def start_test(self):
        self.test_active = True
        self.frame_count = 0
        self.gaze_data = []
        self.social_attention_score = 0
        self.non_social_attention_score = 0
        
    def stop_test(self):
        self.test_active = False
        return self.get_results()
        
    def get_results(self):
        if not self.gaze_data:
            return {}
            
        total_attention = self.social_attention_score + self.non_social_attention_score
        
        results = {
            'total_gaze_points': len(self.gaze_data),
            'social_attention_score': self.social_attention_score,
            'non_social_attention_score': self.non_social_attention_score,
            'social_attention_ratio': self.social_attention_score / max(total_attention, 1),
            'scenario': self.current_scenario,
            'avg_gaze_x': np.mean([g['gaze_x'] for g in self.gaze_data]) if self.gaze_data else 0,
            'avg_gaze_y': np.mean([g['gaze_y'] for g in self.gaze_data]) if self.gaze_data else 0,
            'gaze_variability': np.std([g['gaze_x'] for g in self.gaze_data]) if self.gaze_data else 0,
        }
        
        return results
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if not self.test_active:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process every 2nd frame for better performance
        if self.frame_count % 2 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate gaze point
                    h, w = img.shape[:2]
                    
                    # Get eye center points
                    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    left_eye_center = np.mean([[face_landmarks.landmark[i].x * w, 
                                              face_landmarks.landmark[i].y * h] for i in left_eye_indices], axis=0)
                    right_eye_center = np.mean([[face_landmarks.landmark[i].x * w,
                                               face_landmarks.landmark[i].y * h] for i in right_eye_indices], axis=0)
                    
                    gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
                    gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
                    
                    # Check which regions the gaze falls into
                    is_social = False
                    is_non_social = False
                    
                    for region in self.social_regions:
                        if (region[0] <= gaze_x <= region[2] and region[1] <= gaze_y <= region[3]):
                            self.social_attention_score += 1
                            is_social = True
                            break
                    
                    for region in self.non_social_regions:
                        if (region[0] <= gaze_x <= region[2] and region[1] <= gaze_y <= region[3]):
                            self.non_social_attention_score += 1
                            is_non_social = True
                            break
                    
                    # Store gaze data
                    self.gaze_data.append({
                        'timestamp': time.time(),
                        'gaze_x': gaze_x,
                        'gaze_y': gaze_y,
                        'is_social': is_social,
                        'is_non_social': is_non_social,
                        'scenario': self.current_scenario,
                        'frame_count': self.frame_count
                    })
                    
                    # Draw gaze point
                    color = (0, 255, 0) if is_social else (0, 0, 255) if is_non_social else (255, 255, 255)
                    cv2.circle(img, (int(gaze_x), int(gaze_y)), 8, color, -1)
                    
                    # Draw face mesh
                    self.mp_drawing.draw_landmarks(
                        img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        
        # Draw regions
        for region in self.social_regions:
            cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)
            cv2.putText(img, "SOCIAL", (region[0], region[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for region in self.non_social_regions:
            cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 2)
            cv2.putText(img, "NON-SOCIAL", (region[0], region[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display test info
        if self.test_active:
            cv2.putText(img, f"Scenario: {self.current_scenario}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(img, f"Social: {self.social_attention_score}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Non-Social: {self.non_social_attention_score}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def show_social_attention_test_page():
    st.header("🎭 Social Attention Test")
    
    st.markdown("""
    This test analyzes your attention patterns to social versus non-social stimuli.
    People with ASD often show different preferences for social information.
    """)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Scenario 1: People vs Objects",
            "description": "View scenes with both people and objects",
            "duration": 25,
            "scenario_type": "people_objects"
        },
        {
            "name": "Scenario 2: Social Interactions",
            "description": "Observe social interaction scenes",
            "duration": 30,
            "scenario_type": "social_interaction"
        },
        {
            "name": "Scenario 3: Group Activities",
            "description": "Watch group activity scenarios",
            "duration": 20,
            "scenario_type": "group_activity"
        }
    ]
    
    # Initialize test state
    if 'social_test_scenario' not in st.session_state:
        st.session_state.social_test_scenario = 0
    if 'social_test_active' not in st.session_state:
        st.session_state.social_test_active = False
    if 'social_test_results' not in st.session_state:
        st.session_state.social_test_results = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Social Attention Analysis")
        
        # WebRTC setup with multiple STUN servers
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.cloudflare.com:3478"]},
                {"urls": ["stun:openrelay.metered.ca:80"]}
            ],
            "iceTransportPolicy": "all",
            "iceCandidatePoolSize": 10
        })
        
        webrtc_ctx = webrtc_streamer(
            key="social_attention_test",
            video_processor_factory=SocialAttentionProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display current scenario
        if st.session_state.social_test_active and st.session_state.social_test_scenario < len(test_scenarios):
            current_scenario = test_scenarios[st.session_state.social_test_scenario]
            
            stimulus_placeholder = st.empty()
            
            with stimulus_placeholder.container():
                st.markdown(f"### {current_scenario['name']}")
                
                if current_scenario['scenario_type'] == 'people_objects':
                    show_people_objects_scenario(webrtc_ctx)
                elif current_scenario['scenario_type'] == 'social_interaction':
                    show_social_interaction_scenario(webrtc_ctx)
                elif current_scenario['scenario_type'] == 'group_activity':
                    show_group_activity_scenario(webrtc_ctx)
    
    with col2:
        st.subheader("Test Controls")
        
        # Current scenario info
        if st.session_state.social_test_scenario < len(test_scenarios):
            current_scenario = test_scenarios[st.session_state.social_test_scenario]
            st.info(f"**Current:** {current_scenario['name']}")
            st.write(f"**Task:** {current_scenario['description']}")
            st.write(f"**Duration:** {current_scenario['duration']} seconds")
        
        # Progress
        progress = st.session_state.social_test_scenario / len(test_scenarios)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.social_test_scenario}/{len(test_scenarios)} scenarios")
        
        # Controls
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start", disabled=st.session_state.social_test_active):
                if webrtc_ctx.video_processor and st.session_state.social_test_scenario < len(test_scenarios):
                    st.session_state.social_test_active = True
                    webrtc_ctx.video_processor.start_test()
                    st.success("Scenario started!")
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop", disabled=not st.session_state.social_test_active):
                if webrtc_ctx.video_processor and st.session_state.social_test_active:
                    results = webrtc_ctx.video_processor.stop_test()
                    st.session_state.social_test_results[f"scenario_{st.session_state.social_test_scenario}"] = results
                    st.session_state.social_test_active = False
                    st.session_state.social_test_scenario += 1
                    
                    # Save to database
                    try:
                        if st.session_state.assessment_id:
                            db_manager.save_gaze_data_batch(
                                st.session_state.assessment_id,
                                f"social_attention_scenario_{st.session_state.social_test_scenario-1}",
                                "social_attention",
                                results
                            )
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                    
                    st.success("Scenario completed!")
                    st.rerun()
        
        # Navigation
        st.markdown("---")
        col_back, col_next = st.columns(2)
        
        with col_back:
            if st.button("⬅️ Previous Test"):
                st.session_state.current_test = 1  # Go back to face recognition
                st.rerun()
        
        with col_next:
            if st.session_state.social_test_scenario >= len(test_scenarios):
                if st.button("Next Test ➡️", type="primary"):
                    st.session_state.current_test = 3  # Go to visual pattern test
                    st.rerun()
        
        # Results summary
        if st.session_state.social_test_scenario >= len(test_scenarios):
            st.success("✅ Social Attention Test Completed!")
            
            with st.expander("📊 Quick Results", expanded=True):
                show_social_test_summary()

def show_people_objects_scenario(webrtc_ctx):
    """Display people vs objects scenario"""
    scenario_html = """
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 30px; padding: 20px;'>
        <div style='text-align: center; border: 3px solid green; border-radius: 15px; padding: 30px; background: #f0fff0;'>
            <div style='font-size: 100px; margin-bottom: 10px;'>👨‍👩‍👧‍👦</div>
            <h3>Family Group</h3>
            <p>Social Stimulus</p>
        </div>
        <div style='text-align: center; border: 3px solid red; border-radius: 15px; padding: 30px; background: #fff0f0;'>
            <div style='font-size: 100px; margin-bottom: 10px;'>🏠🚗📱</div>
            <h3>Objects</h3>
            <p>Non-Social Stimulus</p>
        </div>
    </div>
    """
    st.markdown(scenario_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        social_regions = [(50, 100, 350, 400)]  # Left side - people
        non_social_regions = [(450, 100, 750, 400)]  # Right side - objects
        webrtc_ctx.video_processor.set_scenario("people_objects", social_regions, non_social_regions)

def show_social_interaction_scenario(webrtc_ctx):
    """Display social interaction scenario"""
    interaction_html = """
    <div style='text-align: center; padding: 40px; background: linear-gradient(45deg, #e8f5e8, #f0fff0); border-radius: 20px;'>
        <div style='display: flex; justify-content: center; align-items: center; gap: 30px; margin-bottom: 20px;'>
            <div style='font-size: 80px;'>🗣️</div>
            <div style='font-size: 80px;'>↔️</div>
            <div style='font-size: 80px;'>👂</div>
        </div>
        <h2>Social Interaction Scene</h2>
        <p style='font-size: 18px;'>Two people having a conversation</p>
        <div style='margin-top: 30px; display: flex; justify-content: space-around;'>
            <div style='border: 2px solid green; padding: 20px; border-radius: 10px;'>
                <div style='font-size: 60px;'>👩</div>
                <p>Speaker</p>
            </div>
            <div style='border: 2px solid green; padding: 20px; border-radius: 10px;'>
                <div style='font-size: 60px;'>👨</div>
                <p>Listener</p>
            </div>
        </div>
    </div>
    """
    st.markdown(interaction_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        social_regions = [(200, 300, 350, 450), (450, 300, 600, 450)]  # Two people
        non_social_regions = [(50, 50, 150, 150), (650, 50, 750, 150)]  # Background elements
        webrtc_ctx.video_processor.set_scenario("social_interaction", social_regions, non_social_regions)

def show_group_activity_scenario(webrtc_ctx):
    """Display group activity scenario"""
    group_html = """
    <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 15px;'>
        <h2>Group Activity</h2>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 30px 0;'>
            <div style='border: 2px solid green; padding: 15px; border-radius: 8px;'>
                <div style='font-size: 50px;'>👩‍🏫</div>
                <p>Teacher</p>
            </div>
            <div style='border: 2px solid green; padding: 15px; border-radius: 8px;'>
                <div style='font-size: 50px;'>👨‍🎓</div>
                <p>Student 1</p>
            </div>
            <div style='border: 2px solid green; padding: 15px; border-radius: 8px;'>
                <div style='font-size: 50px;'>👩‍🎓</div>
                <p>Student 2</p>
            </div>
            <div style='border: 2px solid green; padding: 15px; border-radius: 8px;'>
                <div style='font-size: 50px;'>👨‍🎓</div>
                <p>Student 3</p>
            </div>
        </div>
        <div style='border: 2px solid red; padding: 20px; border-radius: 10px; background: #fff5f5;'>
            <div style='font-size: 60px;'>📚📝✏️</div>
            <p>Learning Materials (Non-Social)</p>
        </div>
    </div>
    """
    st.markdown(group_html, unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        social_regions = [
            (100, 150, 200, 250),  # Teacher
            (250, 150, 350, 250),  # Student 1
            (400, 150, 500, 250),  # Student 2
            (550, 150, 650, 250)   # Student 3
        ]
        non_social_regions = [(300, 350, 500, 450)]  # Materials
        webrtc_ctx.video_processor.set_scenario("group_activity", social_regions, non_social_regions)

def show_social_test_summary():
    """Display summary of social attention test results"""
    if not st.session_state.social_test_results:
        st.write("No results available yet.")
        return
    
    # Aggregate results
    total_social = 0
    total_non_social = 0
    scenarios_completed = 0
    social_ratios = []
    
    for scenario_key, results in st.session_state.social_test_results.items():
        if results:
            total_social += results.get('social_attention_score', 0)
            total_non_social += results.get('non_social_attention_score', 0)
            social_ratios.append(results.get('social_attention_ratio', 0))
            scenarios_completed += 1
    
    if scenarios_completed > 0:
        avg_social_ratio = np.mean(social_ratios)
        total_attention = total_social + total_non_social
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Social Attention Ratio", f"{avg_social_ratio:.1%}")
            st.metric("Total Attention Points", total_attention)
        
        with col2:
            st.metric("Scenarios Completed", scenarios_completed)
            if total_attention > 0:
                social_preference = total_social / total_attention
                st.metric("Overall Social Preference", f"{social_preference:.1%}")
        
        # Visualization
        if total_attention > 0:
            attention_data = {
                'Stimulus Type': ['Social', 'Non-Social'],
                'Attention Score': [total_social, total_non_social]
            }
            
            fig = px.pie(attention_data, values='Attention Score', names='Stimulus Type',
                        title="Social vs Non-Social Attention Distribution",
                        color_discrete_map={'Social': '#28a745', 'Non-Social': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Social ratio across scenarios
            scenario_names = [f"Scenario {i+1}" for i in range(len(social_ratios))]
            fig2 = px.bar(x=scenario_names, y=social_ratios,
                         title="Social Attention Ratio by Scenario",
                         labels={'x': 'Scenario', 'y': 'Social Attention Ratio'})
            fig2.update_traces(marker_color='#17a2b8')
            st.plotly_chart(fig2, use_container_width=True)