import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
import threading
import time
from models.gaze_analyzer import GazeAnalyzer

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.gaze_analyzer = GazeAnalyzer()
        self.frame_count = 0
        self.last_analysis_time = time.time()
        self.analysis_data = []
        self.lock = threading.Lock()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 3rd frame for performance
        if self.frame_count % 3 == 0:
            gaze_data = self.gaze_analyzer.process_frame(img)
            
            with self.lock:
                self.analysis_data.append(gaze_data)
                
            # Draw gaze visualization on frame
            img = self._draw_gaze_visualization(img, gaze_data)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_gaze_visualization(self, frame, gaze_data):
        """Draw gaze tracking visualization on frame"""
        h, w = frame.shape[:2]
        
        if gaze_data['face_detected']:
            # Draw gaze point
            gaze_x = int(gaze_data['gaze_x'])
            gaze_y = int(gaze_data['gaze_y'])
            
            # Ensure coordinates are within frame bounds
            gaze_x = max(0, min(gaze_x, w-1))
            gaze_y = max(0, min(gaze_y, h-1))
            
            # Draw gaze point
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 0), 2)
            
            # Draw eye contact region
            center_x, center_y = w // 2, h // 2
            region_width = int(w * 0.2)
            region_height = int(h * 0.2)
            
            cv2.rectangle(
                frame,
                (center_x - region_width//2, center_y - region_height//2),
                (center_x + region_width//2, center_y + region_height//2),
                (255, 255, 0), 2
            )
            
            # Draw metrics
            metrics_text = [
                f"Eye Contact: {gaze_data['eye_contact_score']:.2f}",
                f"Social Attention: {gaze_data['social_attention_score']:.2f}",
                f"Fixation: {gaze_data['fixation_duration']:.0f}ms"
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(
                    frame, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        else:
            # No face detected
            cv2.putText(
                frame, "No face detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        return frame
    
    def get_analysis_data(self):
        """Get collected analysis data"""
        with self.lock:
            return self.analysis_data.copy()
    
    def reset_data(self):
        """Reset analysis data"""
        with self.lock:
            self.analysis_data = []
        self.gaze_analyzer.reset_data()

def get_rtc_configuration():
    """Get RTC configuration for WebRTC"""
    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }

def create_assessment_tasks():
    """Create assessment tasks for gaze analysis"""
    tasks = [
        {
            "name": "Social Attention Task",
            "description": "Look at the person in the center of the screen",
            "duration": 10,
            "instructions": "Please look at the face in the center of the screen when it appears.",
            "type": "social_attention"
        },
        {
            "name": "Joint Attention Task", 
            "description": "Follow the gaze direction of the person on screen",
            "duration": 15,
            "instructions": "Look where the person on screen is looking.",
            "type": "joint_attention"
        },
        {
            "name": "Free Viewing Task",
            "description": "Look naturally at the screen",
            "duration": 10,
            "instructions": "Look at the screen naturally, as you normally would.",
            "type": "free_viewing"
        }
    ]
    
    return tasks

def analyze_task_performance(gaze_data, task_type):
    """Analyze performance on specific gaze tasks"""
    if not gaze_data:
        return {}
    
    analysis = {
        'total_samples': len(gaze_data),
        'face_detection_rate': sum([1 for d in gaze_data if d['face_detected']]) / len(gaze_data),
        'avg_eye_contact_score': np.mean([d['eye_contact_score'] for d in gaze_data]),
        'avg_social_attention_score': np.mean([d['social_attention_score'] for d in gaze_data]),
        'avg_fixation_duration': np.mean([d['fixation_duration'] for d in gaze_data if d['fixation_duration'] > 0]),
        'total_fixation_time': sum([d['fixation_duration'] for d in gaze_data]),
        'gaze_stability': 1.0 - np.std([d['saccade_amplitude'] for d in gaze_data]) / 100.0  # Normalized stability measure
    }
    
    # Task-specific analysis
    if task_type == "social_attention":
        analysis['social_attention_performance'] = analysis['avg_eye_contact_score']
    elif task_type == "joint_attention":
        # For joint attention, we'd need more complex analysis of gaze following
        analysis['joint_attention_performance'] = analysis['gaze_stability']
    elif task_type == "free_viewing":
        analysis['natural_gaze_pattern'] = analysis['gaze_stability']
    
    return analysis
