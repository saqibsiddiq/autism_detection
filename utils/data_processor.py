import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataProcessor:
    def __init__(self):
        self.questionnaire_weights = {
            # M-CHAT-R questions (higher weight for core symptoms)
            'enjoys_being_swung': 0.8,
            'interest_in_other_children': 1.0,
            'enjoys_climbing': 0.6,
            'enjoys_peek_a_boo': 0.9,
            'pretend_play': 1.0,
            'uses_index_finger': 0.8,
            'brings_objects_to_show': 1.0,
            'eye_contact': 1.0,
            'unusual_finger_movements': 0.9,
            'tries_to_attract_attention': 0.9,
            
            # AQ-10 questions
            'notices_small_sounds': 0.7,
            'concentrates_on_whole_picture': 0.6,
            'easy_to_do_several_things': 0.7,
            'enjoys_social_chit_chat': 0.9,
            'finds_easy_to_read_between_lines': 0.8,
            'knows_how_to_tell_stories': 0.7,
            'drawn_to_people': 0.9,
            'enjoys_social_activities': 0.9,
            'finds_easy_to_work_out_intentions': 0.8,
            'good_at_social_chit_chat': 0.9
        }
    
    def process_questionnaire_data(self, raw_answers):
        """Process questionnaire responses into structured data"""
        processed_data = {}
        
        # Calculate weighted scores
        total_weighted_score = 0
        max_possible_score = 0
        
        for question, answer in raw_answers.items():
            weight = self.questionnaire_weights.get(question, 1.0)
            weighted_score = answer * weight
            
            processed_data[question] = answer
            total_weighted_score += weighted_score
            max_possible_score += weight
        
        # Calculate normalized scores
        processed_data['total_score'] = total_weighted_score
        processed_data['normalized_score'] = total_weighted_score / max_possible_score if max_possible_score > 0 else 0
        
        # Calculate domain-specific scores
        processed_data.update(self._calculate_domain_scores(raw_answers))
        
        return processed_data
    
    def _calculate_domain_scores(self, answers):
        """Calculate scores for specific behavioral domains"""
        domains = {
            'social_communication': [
                'interest_in_other_children', 'enjoys_peek_a_boo', 'brings_objects_to_show',
                'eye_contact', 'tries_to_attract_attention', 'enjoys_social_chit_chat',
                'drawn_to_people', 'enjoys_social_activities'
            ],
            'repetitive_behaviors': [
                'unusual_finger_movements', 'notices_small_sounds', 'concentrates_on_whole_picture'
            ],
            'social_cognition': [
                'pretend_play', 'finds_easy_to_read_between_lines', 'knows_how_to_tell_stories',
                'finds_easy_to_work_out_intentions', 'good_at_social_chit_chat'
            ],
            'adaptive_functioning': [
                'enjoys_being_swung', 'enjoys_climbing', 'uses_index_finger',
                'easy_to_do_several_things'
            ]
        }
        
        domain_scores = {}
        for domain, questions in domains.items():
            domain_score = 0
            domain_count = 0
            
            for question in questions:
                if question in answers:
                    domain_score += answers[question]
                    domain_count += 1
            
            domain_scores[f'{domain}_score'] = domain_score / domain_count if domain_count > 0 else 0
        
        return domain_scores
    
    def process_gaze_data(self, gaze_data_list):
        """Process gaze tracking data into analysis metrics"""
        if not gaze_data_list:
            return {}
        
        df = pd.DataFrame(gaze_data_list)
        
        # Basic statistics
        gaze_metrics = {
            'total_samples': len(df),
            'face_detection_rate': df['face_detected'].mean(),
            'avg_eye_contact_score': df['eye_contact_score'].mean(),
            'std_eye_contact_score': df['eye_contact_score'].std(),
            'avg_social_attention_score': df['social_attention_score'].mean(),
            'std_social_attention_score': df['social_attention_score'].std(),
        }
        
        # Only calculate these metrics for frames where face was detected
        face_detected_df = df[df['face_detected'] == True]
        
        if len(face_detected_df) > 0:
            gaze_metrics.update({
                'avg_fixation_duration': face_detected_df['fixation_duration'].mean(),
                'std_fixation_duration': face_detected_df['fixation_duration'].std(),
                'avg_saccade_amplitude': face_detected_df['saccade_amplitude'].mean(),
                'std_saccade_amplitude': face_detected_df['saccade_amplitude'].std(),
                'gaze_dispersion_x': face_detected_df['gaze_x'].std(),
                'gaze_dispersion_y': face_detected_df['gaze_y'].std(),
            })
            
            # Calculate eye contact metrics
            eye_contact_threshold = 0.5
            eye_contact_frames = face_detected_df[face_detected_df['eye_contact_score'] > eye_contact_threshold]
            
            gaze_metrics.update({
                'eye_contact_frequency': len(eye_contact_frames) / len(face_detected_df),
                'total_eye_contact_time': len(eye_contact_frames) * 33,  # Assuming 30fps
                'avg_eye_contact_duration': self._calculate_avg_eye_contact_duration(face_detected_df),
            })
        
        return gaze_metrics
    
    def _calculate_avg_eye_contact_duration(self, df, threshold=0.5):
        """Calculate average duration of eye contact episodes"""
        eye_contact_binary = (df['eye_contact_score'] > threshold).astype(int)
        
        # Find continuous segments of eye contact
        diff = np.diff(np.concatenate(([0], eye_contact_binary, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        durations = ends - starts
        return np.mean(durations) * 33  # Convert to milliseconds (assuming 30fps)
    
    def create_comprehensive_report(self, questionnaire_data, gaze_data, prediction_results):
        """Create a comprehensive assessment report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'assessment_type': 'comprehensive',
            'questionnaire_summary': self._create_questionnaire_summary(questionnaire_data),
            'gaze_analysis_summary': self._create_gaze_summary(gaze_data),
            'ml_prediction': prediction_results,
            'risk_assessment': self._calculate_risk_assessment(questionnaire_data, gaze_data, prediction_results),
            'recommendations': self._generate_recommendations(questionnaire_data, gaze_data, prediction_results)
        }
        
        return report
    
    def _create_questionnaire_summary(self, questionnaire_data):
        """Create summary of questionnaire responses"""
        if not questionnaire_data:
            return {}
        
        summary = {
            'total_questions_answered': len([k for k, v in questionnaire_data.items() if not k.endswith('_score')]),
            'social_communication_score': questionnaire_data.get('social_communication_score', 0),
            'repetitive_behaviors_score': questionnaire_data.get('repetitive_behaviors_score', 0),
            'social_cognition_score': questionnaire_data.get('social_cognition_score', 0),
            'adaptive_functioning_score': questionnaire_data.get('adaptive_functioning_score', 0),
            'overall_score': questionnaire_data.get('normalized_score', 0)
        }
        
        return summary
    
    def _create_gaze_summary(self, gaze_data):
        """Create summary of gaze analysis"""
        if not gaze_data:
            return {}
        
        return {
            'assessment_duration': len(gaze_data) * 33,  # milliseconds
            'face_detection_quality': gaze_data.get('face_detection_rate', 0),
            'eye_contact_performance': gaze_data.get('avg_eye_contact_score', 0),
            'social_attention_performance': gaze_data.get('avg_social_attention_score', 0),
            'gaze_stability': 1.0 - min(gaze_data.get('std_saccade_amplitude', 0) / 100.0, 1.0),
            'fixation_patterns': {
                'avg_duration': gaze_data.get('avg_fixation_duration', 0),
                'consistency': 1.0 - min(gaze_data.get('std_fixation_duration', 0) / 1000.0, 1.0)
            }
        }
    
    def _calculate_risk_assessment(self, questionnaire_data, gaze_data, prediction_results):
        """Calculate overall risk assessment"""
        risk_factors = []
        risk_score = 0.0
        
        # Questionnaire-based risk factors
        if questionnaire_data:
            social_comm_score = questionnaire_data.get('social_communication_score', 0)
            if social_comm_score < 0.3:  # Low social communication
                risk_factors.append("Low social communication scores")
                risk_score += 0.3
            
            repetitive_score = questionnaire_data.get('repetitive_behaviors_score', 0)
            if repetitive_score > 0.7:  # High repetitive behaviors
                risk_factors.append("Elevated repetitive behavior indicators")
                risk_score += 0.2
        
        # Gaze-based risk factors
        if gaze_data:
            eye_contact_score = gaze_data.get('avg_eye_contact_score', 0)
            if eye_contact_score < 0.3:  # Low eye contact
                risk_factors.append("Reduced eye contact patterns")
                risk_score += 0.2
            
            social_attention_score = gaze_data.get('avg_social_attention_score', 0)
            if social_attention_score < 0.4:  # Low social attention
                risk_factors.append("Atypical social attention patterns")
                risk_score += 0.2
        
        # ML model prediction
        if prediction_results:
            ml_risk = prediction_results.get('probability_asd_indicators', 0)
            if ml_risk > 0.5:
                risk_factors.append("Machine learning model indicates elevated risk")
                risk_score += ml_risk * 0.3
        
        risk_level = "Low"
        if risk_score > 0.3:
            risk_level = "Moderate"
        if risk_score > 0.6:
            risk_level = "Elevated"
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'confidence': prediction_results.get('confidence', 0) if prediction_results else 0.5
        }
    
    def _generate_recommendations(self, questionnaire_data, gaze_data, prediction_results):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Always include professional consultation recommendation
        recommendations.append({
            'category': 'Professional Consultation',
            'priority': 'High',
            'recommendation': 'Consult with a qualified healthcare professional or developmental pediatrician for comprehensive evaluation.',
            'rationale': 'Professional assessment is essential for accurate diagnosis and appropriate intervention planning.'
        })
        
        # Questionnaire-based recommendations
        if questionnaire_data:
            social_score = questionnaire_data.get('social_communication_score', 0)
            if social_score < 0.4:
                recommendations.append({
                    'category': 'Social Communication',
                    'priority': 'High',
                    'recommendation': 'Consider social skills training or speech-language therapy evaluation.',
                    'rationale': 'Assessment indicates potential challenges in social communication.'
                })
        
        # Gaze-based recommendations
        if gaze_data:
            eye_contact_score = gaze_data.get('avg_eye_contact_score', 0)
            if eye_contact_score < 0.3:
                recommendations.append({
                    'category': 'Social Attention',
                    'priority': 'Moderate',
                    'recommendation': 'Practice eye contact and social attention activities in natural settings.',
                    'rationale': 'Gaze analysis suggests potential difficulties with eye contact and social attention.'
                })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Early Intervention',
                'priority': 'High',
                'recommendation': 'Explore early intervention services if concerns are confirmed.',
                'rationale': 'Early intervention can significantly improve outcomes for individuals with ASD.'
            },
            {
                'category': 'Support Resources',
                'priority': 'Moderate',
                'recommendation': 'Connect with autism support organizations and family resources.',
                'rationale': 'Support networks provide valuable information and assistance for families.'
            }
        ])
        
        return recommendations
    
    def create_visualization_data(self, questionnaire_data, gaze_data):
        """Prepare data for visualizations"""
        viz_data = {}
        
        # Questionnaire visualization data
        if questionnaire_data:
            domain_scores = {
                'Social Communication': questionnaire_data.get('social_communication_score', 0),
                'Repetitive Behaviors': questionnaire_data.get('repetitive_behaviors_score', 0),
                'Social Cognition': questionnaire_data.get('social_cognition_score', 0),
                'Adaptive Functioning': questionnaire_data.get('adaptive_functioning_score', 0)
            }
            viz_data['domain_scores'] = domain_scores
        
        # Gaze visualization data
        if gaze_data:
            gaze_metrics = {
                'Eye Contact': gaze_data.get('avg_eye_contact_score', 0),
                'Social Attention': gaze_data.get('avg_social_attention_score', 0),
                'Gaze Stability': 1.0 - min(gaze_data.get('std_saccade_amplitude', 0) / 100.0, 1.0),
                'Face Detection': gaze_data.get('face_detection_rate', 0)
            }
            viz_data['gaze_metrics'] = gaze_metrics
        
        return viz_data
