import streamlit as st
import numpy as np
import time
import random
from typing import Dict, List, Tuple

class DemoGazeSimulator:
    """Simulates realistic gaze patterns for demo purposes"""
    
    def __init__(self):
        self.demo_active = False
        self.start_time = None
        self.gaze_history = []
        self.current_stimulus = None
        
    def start_demo(self, stimulus_type: str):
        """Start demo mode with specified stimulus"""
        self.demo_active = True
        self.start_time = time.time()
        self.current_stimulus = stimulus_type
        self.gaze_history = []
        
    def stop_demo(self):
        """Stop demo mode and return results"""
        self.demo_active = False
        return self.generate_demo_results()
    
    def get_simulated_gaze_point(self, frame_width: int = 800, frame_height: int = 600) -> Tuple[float, float]:
        """Generate realistic gaze point based on current stimulus"""
        if not self.demo_active:
            return frame_width // 2, frame_height // 2
            
        elapsed_time = time.time() - self.start_time
        
        # Simulate different gaze patterns based on stimulus type
        if self.current_stimulus == "face_recognition":
            return self._simulate_face_gaze(frame_width, frame_height, elapsed_time)
        elif self.current_stimulus == "social_attention":
            return self._simulate_social_gaze(frame_width, frame_height, elapsed_time)
        elif self.current_stimulus == "visual_pattern":
            return self._simulate_pattern_gaze(frame_width, frame_height, elapsed_time)
        elif self.current_stimulus == "motion_tracking":
            return self._simulate_motion_gaze(frame_width, frame_height, elapsed_time)
        else:
            return self._simulate_random_gaze(frame_width, frame_height)
    
    def _simulate_face_gaze(self, width: int, height: int, elapsed_time: float) -> Tuple[float, float]:
        """Simulate gaze patterns typical for face recognition tasks"""
        # Focus more on left side (face region) with some attention to right side (objects)
        if elapsed_time % 4 < 2.5:  # 62.5% time on faces
            # Face region (left side)
            x = random.gauss(width * 0.25, width * 0.08)
            y = random.gauss(height * 0.4, height * 0.1)
        else:
            # Object region (right side)
            x = random.gauss(width * 0.75, width * 0.08)
            y = random.gauss(height * 0.4, height * 0.1)
            
        # Add natural eye movement noise
        x += random.gauss(0, 15)
        y += random.gauss(0, 15)
        
        return max(0, min(width, x)), max(0, min(height, y))
    
    def _simulate_social_gaze(self, width: int, height: int, elapsed_time: float) -> Tuple[float, float]:
        """Simulate social attention patterns"""
        # Varying preference for social vs non-social stimuli
        social_preference = 0.7  # 70% preference for social stimuli
        
        if random.random() < social_preference:
            # Focus on social regions
            social_regions = [
                (width * 0.2, height * 0.3, width * 0.4, height * 0.5),  # Person 1
                (width * 0.6, height * 0.3, width * 0.8, height * 0.5),  # Person 2
            ]
            region = random.choice(social_regions)
            x = random.uniform(region[0], region[2])
            y = random.uniform(region[1], region[3])
        else:
            # Focus on non-social elements
            x = random.uniform(width * 0.1, width * 0.9)
            y = random.uniform(height * 0.1, height * 0.9)
            
        return x, y
    
    def _simulate_pattern_gaze(self, width: int, height: int, elapsed_time: float) -> Tuple[float, float]:
        """Simulate pattern recognition gaze behavior"""
        # Higher preference for structured patterns
        pattern_preference = 0.8  # 80% preference for patterns
        
        if random.random() < pattern_preference:
            # Pattern region (left side)
            x = random.gauss(width * 0.25, width * 0.1)
            y = random.gauss(height * 0.5, height * 0.15)
        else:
            # Random region (right side)
            x = random.gauss(width * 0.75, width * 0.1)
            y = random.gauss(height * 0.5, height * 0.15)
            
        return max(0, min(width, x)), max(0, min(height, y))
    
    def _simulate_motion_gaze(self, width: int, height: int, elapsed_time: float) -> Tuple[float, float]:
        """Simulate motion tracking behavior"""
        # Follow a moving target with some tracking error
        center_x, center_y = width // 2, height // 2
        
        # Circular motion
        radius = 100
        angle = elapsed_time * 2  # Speed of motion
        
        target_x = center_x + radius * np.cos(angle)
        target_y = center_y + radius * np.sin(angle)
        
        # Add tracking error (less accurate tracking)
        tracking_accuracy = 0.8
        error_x = random.gauss(0, 30 * (1 - tracking_accuracy))
        error_y = random.gauss(0, 30 * (1 - tracking_accuracy))
        
        gaze_x = target_x + error_x
        gaze_y = target_y + error_y
        
        return max(0, min(width, gaze_x)), max(0, min(height, gaze_y))
    
    def _simulate_random_gaze(self, width: int, height: int) -> Tuple[float, float]:
        """Simulate random gaze movement"""
        x = random.uniform(width * 0.1, width * 0.9)
        y = random.uniform(height * 0.1, height * 0.9)
        return x, y
    
    def generate_demo_results(self) -> Dict:
        """Generate realistic demo results"""
        if not self.gaze_history:
            return {}
            
        # Generate results based on stimulus type
        if self.current_stimulus == "face_recognition":
            return {
                'face_attention_time': random.randint(150, 250),
                'object_attention_time': random.randint(80, 120),
                'face_preference_ratio': random.uniform(0.55, 0.75),
                'face_detection_rate': random.uniform(0.85, 0.95),
                'total_frames': random.randint(800, 1200)
            }
        elif self.current_stimulus == "social_attention":
            return {
                'social_attention_score': random.randint(180, 280),
                'non_social_attention_score': random.randint(60, 120),
                'social_attention_ratio': random.uniform(0.65, 0.82),
                'total_gaze_points': random.randint(300, 500)
            }
        elif self.current_stimulus == "visual_pattern":
            return {
                'pattern_fixations': random.randint(120, 200),
                'random_fixations': random.randint(40, 80),
                'pattern_preference_ratio': random.uniform(0.7, 0.9),
                'avg_fixation_duration': random.uniform(0.2, 0.4),
                'total_gaze_points': random.randint(200, 350)
            }
        elif self.current_stimulus == "motion_tracking":
            return {
                'tracking_accuracy': random.uniform(0.6, 0.85),
                'smooth_pursuit_quality': random.uniform(0.5, 0.8),
                'saccadic_movements_count': random.randint(15, 35),
                'avg_gaze_velocity': random.uniform(200, 400),
                'total_gaze_points': random.randint(250, 400)
            }
        
        return {}

def show_demo_mode_info():
    """Show information about demo mode"""
    st.info("""
    **🎮 Demo Mode Active**
    
    Since no camera is available, the system is running in demo mode with simulated gaze tracking.
    This demonstrates how the behavioral analysis works with realistic patterns.
    
    **What you'll see:**
    - Simulated eye movements and gaze points
    - Realistic behavioral patterns for each test
    - Sample analysis results based on research data
    - Complete assessment workflow demonstration
    """)

def create_demo_video_frame(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a demo video frame with simulated face detection"""
    # Create a simple demo frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a background gradient
    for y in range(height):
        frame[y, :, 0] = int(50 + (y / height) * 30)  # Blue gradient
        frame[y, :, 1] = int(60 + (y / height) * 40)  # Green gradient
        frame[y, :, 2] = int(70 + (y / height) * 50)  # Red gradient
    
    # Add face outline (simplified)
    center_x, center_y = width // 2, height // 2
    
    # Face oval
    face_color = (200, 180, 160)
    for angle in np.linspace(0, 2*np.pi, 100):
        x = int(center_x + 80 * np.cos(angle))
        y = int(center_y + 100 * np.sin(angle))
        if 0 <= x < width and 0 <= y < height:
            frame[y-2:y+3, x-2:x+3] = face_color
    
    # Eyes
    eye_color = (100, 100, 100)
    left_eye_x, left_eye_y = center_x - 30, center_y - 20
    right_eye_x, right_eye_y = center_x + 30, center_y - 20
    
    frame[left_eye_y-5:left_eye_y+5, left_eye_x-8:left_eye_x+8] = eye_color
    frame[right_eye_y-5:right_eye_y+5, right_eye_x-8:right_eye_x+8] = eye_color
    
    # Add demo text
    demo_text_color = (255, 255, 0)
    demo_text = "DEMO MODE - Simulated Face Detection"
    
    # Simple text rendering (pixel by pixel would be complex, so just add a colored bar)
    frame[20:40, 20:400] = demo_text_color
    
    return frame

# Global demo simulator instance
demo_simulator = DemoGazeSimulator()