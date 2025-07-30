import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

class GazeAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        # Data storage
        self.gaze_history = deque(maxlen=100)
        self.fixation_history = deque(maxlen=50)
        self.eye_contact_history = deque(maxlen=100)
        
        # Calibration data
        self.is_calibrated = False
        self.calibration_points = []
        self.gaze_offset = [0, 0]
        
        # Analysis parameters
        self.fixation_threshold = 50  # pixels
        self.fixation_duration_threshold = 100  # milliseconds
        self.eye_contact_threshold = 0.3  # normalized distance threshold
        
    def process_frame(self, frame):
        """Process a single frame and extract gaze data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        gaze_data = {
            'timestamp': time.time(),
            'face_detected': False,
            'gaze_x': 0,
            'gaze_y': 0,
            'eye_contact_score': 0,
            'fixation_duration': 0,
            'saccade_amplitude': 0,
            'social_attention_score': 0
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            gaze_data['face_detected'] = True
            
            # Extract eye regions
            left_eye_landmarks = self._get_eye_landmarks(face_landmarks, self.LEFT_EYE_INDICES)
            right_eye_landmarks = self._get_eye_landmarks(face_landmarks, self.RIGHT_EYE_INDICES)
            
            # Extract iris positions
            left_iris = self._get_iris_landmarks(face_landmarks, self.LEFT_IRIS_INDICES)
            right_iris = self._get_iris_landmarks(face_landmarks, self.RIGHT_IRIS_INDICES)
            
            # Calculate gaze direction
            gaze_point = self._calculate_gaze_direction(
                left_eye_landmarks, right_eye_landmarks,
                left_iris, right_iris,
                frame.shape
            )
            
            gaze_data['gaze_x'] = gaze_point[0]
            gaze_data['gaze_y'] = gaze_point[1]
            
            # Calculate eye contact score
            gaze_data['eye_contact_score'] = self._calculate_eye_contact_score(gaze_point, frame.shape)
            
            # Update gaze history
            self.gaze_history.append(gaze_point)
            
            # Calculate fixation duration
            gaze_data['fixation_duration'] = self._calculate_fixation_duration()
            
            # Calculate saccade amplitude
            gaze_data['saccade_amplitude'] = self._calculate_saccade_amplitude()
            
            # Calculate social attention score
            gaze_data['social_attention_score'] = self._calculate_social_attention_score(gaze_point, frame.shape)
            
            # Store eye contact history
            self.eye_contact_history.append(gaze_data['eye_contact_score'])
        
        return gaze_data
    
    def _get_eye_landmarks(self, face_landmarks, eye_indices):
        """Extract eye landmark coordinates"""
        eye_points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            eye_points.append([landmark.x, landmark.y])
        return np.array(eye_points)
    
    def _get_iris_landmarks(self, face_landmarks, iris_indices):
        """Extract iris landmark coordinates"""
        iris_points = []
        for idx in iris_indices:
            landmark = face_landmarks.landmark[idx]
            iris_points.append([landmark.x, landmark.y])
        return np.array(iris_points)
    
    def _calculate_gaze_direction(self, left_eye, right_eye, left_iris, right_iris, frame_shape):
        """Calculate gaze direction from eye and iris positions"""
        h, w = frame_shape[:2]
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Calculate iris centers
        left_iris_center = np.mean(left_iris, axis=0)
        right_iris_center = np.mean(right_iris, axis=0)
        
        # Calculate gaze vectors (iris relative to eye center)
        left_gaze_vector = left_iris_center - left_eye_center
        right_gaze_vector = right_iris_center - right_eye_center
        
        # Average the gaze vectors
        avg_gaze_vector = (left_gaze_vector + right_gaze_vector) / 2
        
        # Convert to screen coordinates
        gaze_x = avg_gaze_vector[0] * w
        gaze_y = avg_gaze_vector[1] * h
        
        # Apply calibration offset if available
        gaze_x += self.gaze_offset[0]
        gaze_y += self.gaze_offset[1]
        
        return [gaze_x, gaze_y]
    
    def _calculate_eye_contact_score(self, gaze_point, frame_shape):
        """Calculate eye contact score based on gaze direction"""
        h, w = frame_shape[:2]
        
        # Define eye contact region (center of screen)
        center_x, center_y = w // 2, h // 2
        eye_contact_region_width = w * 0.2  # 20% of screen width
        eye_contact_region_height = h * 0.2  # 20% of screen height
        
        # Calculate distance from gaze point to center
        distance = math.sqrt(
            (gaze_point[0] - center_x) ** 2 + 
            (gaze_point[1] - center_y) ** 2
        )
        
        # Normalize distance
        max_distance = math.sqrt(eye_contact_region_width ** 2 + eye_contact_region_height ** 2)
        normalized_distance = min(distance / max_distance, 1.0)
        
        # Convert to eye contact score (closer to center = higher score)
        eye_contact_score = max(0, 1.0 - normalized_distance)
        
        return eye_contact_score
    
    def _calculate_fixation_duration(self):
        """Calculate current fixation duration"""
        if len(self.gaze_history) < 2:
            return 0
        
        current_point = self.gaze_history[-1]
        fixation_start_time = time.time()
        fixation_duration = 0
        
        # Look backwards through gaze history to find fixation start
        for i in range(len(self.gaze_history) - 2, -1, -1):
            prev_point = self.gaze_history[i]
            distance = math.sqrt(
                (current_point[0] - prev_point[0]) ** 2 + 
                (current_point[1] - prev_point[1]) ** 2
            )
            
            if distance > self.fixation_threshold:
                break
            
            fixation_duration += 33  # Assuming ~30 FPS (33ms per frame)
        
        return fixation_duration
    
    def _calculate_saccade_amplitude(self):
        """Calculate amplitude of the last saccade"""
        if len(self.gaze_history) < 2:
            return 0
        
        current_point = self.gaze_history[-1]
        prev_point = self.gaze_history[-2]
        
        amplitude = math.sqrt(
            (current_point[0] - prev_point[0]) ** 2 + 
            (current_point[1] - prev_point[1]) ** 2
        )
        
        return amplitude
    
    def _calculate_social_attention_score(self, gaze_point, frame_shape):
        """Calculate social attention score based on gaze patterns"""
        h, w = frame_shape[:2]
        
        # Define social regions (upper half of screen where faces typically appear)
        social_region_top = h * 0.1
        social_region_bottom = h * 0.6
        
        social_attention_score = 0
        
        # Check if gaze is in social region
        if social_region_top <= gaze_point[1] <= social_region_bottom:
            social_attention_score += 0.5
        
        # Check for eye contact
        eye_contact_score = self._calculate_eye_contact_score(gaze_point, frame_shape)
        social_attention_score += eye_contact_score * 0.5
        
        return min(social_attention_score, 1.0)
    
    def get_gaze_summary(self):
        """Get summary statistics of gaze data"""
        if len(self.gaze_history) == 0:
            return {}
        
        gaze_points = list(self.gaze_history)
        eye_contact_scores = list(self.eye_contact_history)
        
        summary = {
            'total_samples': len(gaze_points),
            'avg_gaze_x': np.mean([p[0] for p in gaze_points]),
            'avg_gaze_y': np.mean([p[1] for p in gaze_points]),
            'gaze_dispersion_x': np.std([p[0] for p in gaze_points]),
            'gaze_dispersion_y': np.std([p[1] for p in gaze_points]),
        }
        
        if eye_contact_scores:
            summary.update({
                'avg_eye_contact_score': np.mean(eye_contact_scores),
                'total_eye_contact_time': sum([1 for score in eye_contact_scores if score > self.eye_contact_threshold]) * 33,  # ms
                'eye_contact_frequency': len([score for score in eye_contact_scores if score > self.eye_contact_threshold]) / len(eye_contact_scores)
            })
        
        return summary
    
    def reset_data(self):
        """Reset all gaze tracking data"""
        self.gaze_history.clear()
        self.fixation_history.clear()
        self.eye_contact_history.clear()
        self.is_calibrated = False
        self.calibration_points = []
        self.gaze_offset = [0, 0]
    
    def calibrate(self, calibration_data):
        """Calibrate gaze tracking using calibration points"""
        if len(calibration_data) >= 4:  # Need at least 4 points for calibration
            # Simple offset calibration
            actual_points = [point['actual'] for point in calibration_data]
            measured_points = [point['measured'] for point in calibration_data]
            
            avg_offset_x = np.mean([a[0] - m[0] for a, m in zip(actual_points, measured_points)])
            avg_offset_y = np.mean([a[1] - m[1] for a, m in zip(actual_points, measured_points)])
            
            self.gaze_offset = [avg_offset_x, avg_offset_y]
            self.is_calibrated = True
            
            return True
        
        return False
