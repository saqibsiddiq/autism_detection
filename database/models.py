from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
import json

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    age_group = Column(String)  # e.g., "18-24 months", "adult"
    consent_given = Column(Boolean, default=False)

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    assessment_type = Column(String)  # "questionnaire", "gaze", "combined"
    status = Column(String, default="in_progress")  # "in_progress", "completed", "abandoned"
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_duration = Column(Integer)  # seconds

class QuestionnaireResponse(Base):
    __tablename__ = "questionnaire_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, index=True)
    question_id = Column(String)
    question_text = Column(Text)
    response_value = Column(Float)
    response_text = Column(String)
    domain = Column(String)  # "social_communication", "autism_traits", etc.
    weight = Column(Float)
    is_critical_item = Column(Boolean, default=False)
    answered_at = Column(DateTime, default=datetime.utcnow)

class GazeData(Base):
    __tablename__ = "gaze_data"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, index=True)
    task_name = Column(String)
    task_type = Column(String)
    frame_number = Column(Integer)
    timestamp = Column(Float)
    face_detected = Column(Boolean)
    gaze_x = Column(Float)
    gaze_y = Column(Float)
    eye_contact_score = Column(Float)
    fixation_duration = Column(Float)
    saccade_amplitude = Column(Float)
    social_attention_score = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)

class AssessmentResult(Base):
    __tablename__ = "assessment_results"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, index=True)
    questionnaire_scores = Column(JSON)  # Domain scores from questionnaire
    gaze_metrics = Column(JSON)  # Aggregated gaze analysis metrics
    ml_prediction = Column(JSON)  # Machine learning model results
    risk_assessment = Column(JSON)  # Risk level and factors
    recommendations = Column(JSON)  # Generated recommendations
    overall_score = Column(Float)
    risk_level = Column(String)  # "low", "moderate", "elevated"
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def create_user(self, session_id: str, age_group: str = None, consent_given: bool = False) -> User:
        """Create a new user session"""
        db = self.get_session()
        try:
            user = User(
                session_id=session_id,
                age_group=age_group,
                consent_given=consent_given
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        finally:
            db.close()
    
    def get_user_by_session(self, session_id: str) -> User:
        """Get user by session ID"""
        db = self.get_session()
        try:
            return db.query(User).filter(User.session_id == session_id).first()
        finally:
            db.close()
    
    def create_assessment(self, user_id: int, assessment_type: str) -> Assessment:
        """Create a new assessment"""
        db = self.get_session()
        try:
            assessment = Assessment(
                user_id=user_id,
                assessment_type=assessment_type
            )
            db.add(assessment)
            db.commit()
            db.refresh(assessment)
            return assessment
        finally:
            db.close()
    
    def save_questionnaire_response(self, assessment_id: int, question_id: str, 
                                  question_text: str, response_value: float,
                                  response_text: str = None, domain: str = None,
                                  weight: float = 1.0, is_critical_item: bool = False):
        """Save a questionnaire response"""
        db = self.get_session()
        try:
            response = QuestionnaireResponse(
                assessment_id=assessment_id,
                question_id=question_id,
                question_text=question_text,
                response_value=response_value,
                response_text=response_text,
                domain=domain,
                weight=weight,
                is_critical_item=is_critical_item
            )
            db.add(response)
            db.commit()
            return response
        finally:
            db.close()
    
    def save_gaze_data_batch(self, assessment_id: int, task_name: str, task_type: str, 
                           gaze_data_list: list):
        """Save a batch of gaze data points"""
        db = self.get_session()
        try:
            gaze_records = []
            for i, data_point in enumerate(gaze_data_list):
                gaze_record = GazeData(
                    assessment_id=assessment_id,
                    task_name=task_name,
                    task_type=task_type,
                    frame_number=i,
                    timestamp=data_point.get('timestamp', 0),
                    face_detected=data_point.get('face_detected', False),
                    gaze_x=data_point.get('gaze_x', 0),
                    gaze_y=data_point.get('gaze_y', 0),
                    eye_contact_score=data_point.get('eye_contact_score', 0),
                    fixation_duration=data_point.get('fixation_duration', 0),
                    saccade_amplitude=data_point.get('saccade_amplitude', 0),
                    social_attention_score=data_point.get('social_attention_score', 0)
                )
                gaze_records.append(gaze_record)
            
            db.add_all(gaze_records)
            db.commit()
            return len(gaze_records)
        finally:
            db.close()
    
    def save_assessment_results(self, assessment_id: int, questionnaire_scores: dict,
                              gaze_metrics: dict, ml_prediction: dict,
                              risk_assessment: dict, recommendations: list):
        """Save final assessment results"""
        db = self.get_session()
        try:
            result = AssessmentResult(
                assessment_id=assessment_id,
                questionnaire_scores=questionnaire_scores,
                gaze_metrics=gaze_metrics,
                ml_prediction=ml_prediction,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                overall_score=ml_prediction.get('probability_asd_indicators', 0),
                risk_level=risk_assessment.get('risk_level', 'unknown'),
                confidence_score=ml_prediction.get('confidence', 0)
            )
            db.add(result)
            db.commit()
            db.refresh(result)
            return result
        finally:
            db.close()
    
    def complete_assessment(self, assessment_id: int):
        """Mark assessment as completed"""
        db = self.get_session()
        try:
            assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
            if assessment:
                assessment.status = "completed"
                assessment.completed_at = datetime.utcnow()
                if assessment.started_at:
                    duration = (datetime.utcnow() - assessment.started_at).total_seconds()
                    assessment.total_duration = int(duration)
                db.commit()
                return assessment
        finally:
            db.close()
    
    def get_assessment_results(self, assessment_id: int) -> AssessmentResult:
        """Get assessment results"""
        db = self.get_session()
        try:
            return db.query(AssessmentResult).filter(
                AssessmentResult.assessment_id == assessment_id
            ).first()
        finally:
            db.close()
    
    def get_user_assessments(self, user_id: int) -> list:
        """Get all assessments for a user"""
        db = self.get_session()
        try:
            return db.query(Assessment).filter(Assessment.user_id == user_id).all()
        finally:
            db.close()
    
    def get_assessment_statistics(self) -> dict:
        """Get general statistics about assessments"""
        db = self.get_session()
        try:
            total_users = db.query(User).count()
            total_assessments = db.query(Assessment).count()
            completed_assessments = db.query(Assessment).filter(
                Assessment.status == "completed"
            ).count()
            
            # Risk level distribution
            risk_levels = db.query(AssessmentResult.risk_level).all()
            risk_distribution = {}
            for level in risk_levels:
                if level[0]:
                    risk_distribution[level[0]] = risk_distribution.get(level[0], 0) + 1
            
            return {
                "total_users": total_users,
                "total_assessments": total_assessments,
                "completed_assessments": completed_assessments,
                "completion_rate": completed_assessments / max(total_assessments, 1),
                "risk_distribution": risk_distribution
            }
        finally:
            db.close()

# Global database manager instance
db_manager = DatabaseManager()