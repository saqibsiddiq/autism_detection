import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

class BehavioralModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def prepare_features(self, questionnaire_data, gaze_data=None):
        """Prepare features from questionnaire and gaze data"""
        features = {}
        
        # Behavioral features from questionnaire
        behavioral_scores = self._calculate_behavioral_scores(questionnaire_data)
        features.update(behavioral_scores)
        
        # Gaze features if available
        if gaze_data and len(gaze_data) > 0:
            gaze_features = self._calculate_gaze_features(gaze_data)
            features.update(gaze_features)
        
        return features
    
    def _calculate_behavioral_scores(self, questionnaire_data):
        """Calculate behavioral domain scores"""
        scores = {}
        
        # M-CHAT-R style questions (social communication)
        social_questions = [
            'enjoys_being_swung', 'interest_in_other_children', 'enjoys_climbing',
            'enjoys_peek_a_boo', 'pretend_play', 'uses_index_finger',
            'brings_objects_to_show', 'eye_contact', 'unusual_finger_movements',
            'tries_to_attract_attention'
        ]
        
        # AQ-10 style questions (autism traits)
        autism_trait_questions = [
            'notices_small_sounds', 'concentrates_on_whole_picture', 'easy_to_do_several_things',
            'enjoys_social_chit_chat', 'finds_easy_to_read_between_lines', 'knows_how_to_tell_stories',
            'drawn_to_people', 'enjoys_social_activities', 'finds_easy_to_work_out_intentions',
            'good_at_social_chit_chat'
        ]
        
        # Calculate domain scores
        social_score = sum([questionnaire_data.get(q, 0) for q in social_questions if q in questionnaire_data])
        autism_traits_score = sum([questionnaire_data.get(q, 0) for q in autism_trait_questions if q in questionnaire_data])
        
        scores['social_communication_score'] = social_score
        scores['autism_traits_score'] = autism_traits_score
        scores['total_behavioral_score'] = social_score + autism_traits_score
        
        # Individual question scores
        for question, answer in questionnaire_data.items():
            scores[f'q_{question}'] = answer
            
        return scores
    
    def _calculate_gaze_features(self, gaze_data):
        """Calculate gaze pattern features"""
        if not gaze_data:
            return {}
            
        gaze_df = pd.DataFrame(gaze_data)
        features = {}
        
        # Basic gaze metrics
        if 'fixation_duration' in gaze_df.columns:
            features['avg_fixation_duration'] = gaze_df['fixation_duration'].mean()
            features['std_fixation_duration'] = gaze_df['fixation_duration'].std()
            features['total_fixation_time'] = gaze_df['fixation_duration'].sum()
        
        if 'saccade_amplitude' in gaze_df.columns:
            features['avg_saccade_amplitude'] = gaze_df['saccade_amplitude'].mean()
            features['std_saccade_amplitude'] = gaze_df['saccade_amplitude'].std()
        
        # Eye contact metrics
        if 'eye_contact_duration' in gaze_df.columns:
            features['total_eye_contact'] = gaze_df['eye_contact_duration'].sum()
            features['avg_eye_contact'] = gaze_df['eye_contact_duration'].mean()
            features['eye_contact_frequency'] = len(gaze_df[gaze_df['eye_contact_duration'] > 0])
        
        # Social attention metrics
        if 'social_attention_score' in gaze_df.columns:
            features['avg_social_attention'] = gaze_df['social_attention_score'].mean()
            features['social_attention_variability'] = gaze_df['social_attention_score'].std()
        
        # Gaze pattern regularity
        if 'gaze_x' in gaze_df.columns and 'gaze_y' in gaze_df.columns:
            features['gaze_dispersion_x'] = gaze_df['gaze_x'].std()
            features['gaze_dispersion_y'] = gaze_df['gaze_y'].std()
            features['gaze_center_tendency'] = np.sqrt(gaze_df['gaze_x'].var() + gaze_df['gaze_y'].var())
        
        return features
    
    def create_synthetic_training_data(self, n_samples=1000):
        """Create synthetic training data based on ASD research patterns"""
        np.random.seed(42)
        
        # Generate synthetic behavioral data
        data = []
        labels = []
        
        for i in range(n_samples):
            # Randomly assign ASD vs typical development
            is_asd = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% ASD prevalence in synthetic data
            
            sample = {}
            
            # Behavioral features based on research findings
            if is_asd:
                # ASD patterns
                sample['social_communication_score'] = np.random.normal(15, 5)  # Lower social communication
                sample['autism_traits_score'] = np.random.normal(25, 6)  # Higher autism traits
                sample['avg_fixation_duration'] = np.random.normal(800, 200)  # Longer fixations
                sample['total_eye_contact'] = np.random.normal(20, 10)  # Less eye contact
                sample['avg_social_attention'] = np.random.normal(0.3, 0.1)  # Lower social attention
                sample['gaze_dispersion_x'] = np.random.normal(100, 30)  # More dispersed gaze
                sample['gaze_dispersion_y'] = np.random.normal(100, 30)
            else:
                # Typical development patterns
                sample['social_communication_score'] = np.random.normal(25, 4)  # Higher social communication
                sample['autism_traits_score'] = np.random.normal(12, 4)  # Lower autism traits
                sample['avg_fixation_duration'] = np.random.normal(600, 150)  # Shorter fixations
                sample['total_eye_contact'] = np.random.normal(60, 15)  # More eye contact
                sample['avg_social_attention'] = np.random.normal(0.7, 0.1)  # Higher social attention
                sample['gaze_dispersion_x'] = np.random.normal(60, 20)  # More focused gaze
                sample['gaze_dispersion_y'] = np.random.normal(60, 20)
            
            # Add some noise and individual variation
            for key in sample:
                if sample[key] > 0:
                    sample[key] += np.random.normal(0, abs(sample[key]) * 0.1)  # 10% noise
                
            data.append(sample)
            labels.append(is_asd)
        
        return pd.DataFrame(data), np.array(labels)
    
    def train_models(self):
        """Train the ensemble of models"""
        # Create synthetic training data
        X_df, y = self.create_synthetic_training_data()
        
        # Store feature names
        self.feature_names = X_df.columns.tolist()
        
        # Convert to numpy array and scale
        X = self.scaler.fit_transform(X_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train each model
        self.model_performance = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.model_performance[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        self.is_trained = True
        return self.model_performance
    
    def predict(self, features_dict):
        """Make prediction using ensemble of models"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features_dict.get(feature_name, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = self.scaler.transform(feature_vector)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(feature_vector)[0]
            prob = model.predict_proba(feature_vector)[0]
            
            predictions[name] = pred
            probabilities[name] = {
                'typical': prob[0],
                'asd_indicators': prob[1]
            }
        
        # Ensemble prediction (majority vote)
        ensemble_pred = max(set(predictions.values()), key=list(predictions.values()).count)
        
        # Average probabilities
        avg_prob_typical = np.mean([prob['typical'] for prob in probabilities.values()])
        avg_prob_asd = np.mean([prob['asd_indicators'] for prob in probabilities.values()])
        
        return {
            'prediction': ensemble_pred,
            'confidence': max(avg_prob_typical, avg_prob_asd),
            'probability_typical': avg_prob_typical,
            'probability_asd_indicators': avg_prob_asd,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities
        }
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if not self.is_trained:
            self.train_models()
        
        rf_model = self.models['random_forest']
        importance_scores = rf_model.feature_importances_
        
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = importance_scores[i]
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features
