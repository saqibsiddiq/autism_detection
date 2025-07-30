import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def show_results_page():
    st.header("📊 Assessment Results")
    
    # Check if we have both questionnaire and gaze data
    has_questionnaire = 'processed_questionnaire_data' in st.session_state
    has_gaze = 'gaze_assessment_results' in st.session_state
    
    if not has_questionnaire and not has_gaze:
        st.warning("No assessment data found. Please complete the questionnaire and/or gaze assessment first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Go to Questionnaire"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("👁️ Go to Gaze Assessment"):
                st.session_state.current_step = 2
                st.rerun()
        return
    
    # Medical disclaimer at top of results
    st.error("""
    ⚠️ **IMPORTANT DISCLAIMER**: These results are for educational and screening support purposes only. 
    They do NOT constitute a medical diagnosis. Always consult with qualified healthcare professionals 
    for proper evaluation and diagnosis.
    """)
    
    # Get data
    questionnaire_data = st.session_state.get('processed_questionnaire_data', {})
    gaze_data = st.session_state.get('gaze_assessment_results', {})
    
    # Process data and generate ML prediction
    ml_results = generate_ml_prediction(questionnaire_data, gaze_data)
    
    # Store results for comprehensive report
    if 'comprehensive_results' not in st.session_state:
        st.session_state.comprehensive_results = create_comprehensive_report(
            questionnaire_data, gaze_data, ml_results
        )
    
    # Create tabs for different result views
    tabs = st.tabs(["🎯 Summary", "📋 Questionnaire", "👁️ Gaze Analysis", "🤖 ML Insights", "📄 Report"])
    
    with tabs[0]:
        show_summary_results(questionnaire_data, gaze_data, ml_results)
    
    with tabs[1]:
        show_questionnaire_results(questionnaire_data)
    
    with tabs[2]:
        show_gaze_results(gaze_data)
    
    with tabs[3]:
        show_ml_results(ml_results)
    
    with tabs[4]:
        show_comprehensive_report()
    
    # Navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Assessment"):
            st.session_state.current_step = 2
            st.rerun()
    
    with col2:
        if st.button("📧 Download Report", type="primary"):
            download_report()
    
    with col3:
        if st.button("📚 Educational Resources ➡️"):
            st.session_state.current_step = 4
            st.rerun()

def generate_ml_prediction(questionnaire_data, gaze_data):
    """Generate ML prediction from assessment data"""
    from models.behavioral_model import BehavioralModel
    from utils.data_processor import DataProcessor
    
    model = BehavioralModel()
    processor = DataProcessor()
    
    # Prepare features
    gaze_metrics = {}
    if gaze_data and 'overall_metrics' in gaze_data:
        gaze_metrics = gaze_data['overall_metrics']
    
    # Convert gaze data to the format expected by the model
    gaze_list = []
    if gaze_metrics:
        # Create a synthetic gaze data list from metrics
        for i in range(10):  # Create some sample points
            gaze_point = {
                'fixation_duration': gaze_metrics.get('avg_fixation_duration', 0),
                'saccade_amplitude': gaze_metrics.get('avg_saccade_amplitude', 0),
                'eye_contact_duration': gaze_metrics.get('avg_eye_contact_score', 0) * 100,
                'social_attention_score': gaze_metrics.get('avg_social_attention_score', 0),
                'gaze_x': gaze_metrics.get('avg_gaze_x', 320),
                'gaze_y': gaze_metrics.get('avg_gaze_y', 240)
            }
            gaze_list.append(gaze_point)
    
    # Prepare features for ML model
    features = model.prepare_features(questionnaire_data, gaze_list)
    
    # Generate prediction
    prediction = model.predict(features)
    
    # Add feature importance
    feature_importance = model.get_feature_importance()
    prediction['feature_importance'] = feature_importance
    
    return prediction

def show_summary_results(questionnaire_data, gaze_data, ml_results):
    """Show summary of all results"""
    st.subheader("🎯 Assessment Summary")
    
    # Overall risk assessment
    risk_score = ml_results.get('probability_asd_indicators', 0)
    confidence = ml_results.get('confidence', 0)
    
    # Risk level categorization
    if risk_score < 0.3:
        risk_level = "Low"
        risk_color = "green"
    elif risk_score < 0.6:
        risk_level = "Moderate"
        risk_color = "orange"
    else:
        risk_level = "Elevated"
        risk_color = "red"
    
    # Display risk assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Level",
            risk_level,
            help="Overall risk assessment based on combined data"
        )
    
    with col2:
        st.metric(
            "ASD Indicators",
            f"{risk_score:.1%}",
            help="Probability of ASD-related characteristics"
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            help="Model confidence in prediction"
        )
    
    # Risk level indicator
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {risk_color}; color: white; text-align: center; margin: 10px 0;">
        <strong>Risk Level: {risk_level}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    findings = []
    
    # Questionnaire findings
    if questionnaire_data:
        social_score = questionnaire_data.get('social_communication_score', 0)
        if social_score < 0.4:
            findings.append("⚠️ Lower scores in social communication domain")
        elif social_score > 0.7:
            findings.append("✅ Strong social communication patterns")
        
        repetitive_score = questionnaire_data.get('repetitive_behaviors_score', 0)
        if repetitive_score > 0.6:
            findings.append("⚠️ Elevated repetitive behavior indicators")
    
    # Gaze findings
    if gaze_data and 'overall_metrics' in gaze_data:
        gaze_metrics = gaze_data['overall_metrics']
        eye_contact_score = gaze_metrics.get('avg_eye_contact_score', 0)
        
        if eye_contact_score < 0.3:
            findings.append("⚠️ Reduced eye contact patterns observed")
        elif eye_contact_score > 0.7:
            findings.append("✅ Strong eye contact patterns observed")
        
        face_detection = gaze_metrics.get('face_detection_rate', 0)
        if face_detection < 0.8:
            findings.append("ℹ️ Limited face detection during assessment (may affect gaze analysis)")
    
    # Display findings
    if findings:
        for finding in findings:
            st.write(finding)
    else:
        st.info("Assessment results are within typical ranges.")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = [
        "🔸 **Consult a Healthcare Professional**: Discuss these results with a qualified healthcare provider or developmental specialist",
        "🔸 **Consider Comprehensive Evaluation**: If concerns persist, seek a comprehensive developmental assessment",
        "🔸 **Monitor Development**: Continue observing social communication and behavioral patterns",
        "🔸 **Support Resources**: Connect with local autism support organizations and family resources"
    ]
    
    if risk_level in ["Moderate", "Elevated"]:
        recommendations.insert(1, "🔸 **Early Intervention**: Consider early intervention services if recommended by professionals")
    
    for rec in recommendations:
        st.markdown(rec)

def show_questionnaire_results(questionnaire_data):
    """Show detailed questionnaire results"""
    st.subheader("📋 Questionnaire Analysis")
    
    if not questionnaire_data:
        st.info("No questionnaire data available.")
        return
    
    # Domain scores
    st.subheader("Domain Scores")
    
    domain_scores = {
        'Social Communication': questionnaire_data.get('social_communication_score', 0),
        'Repetitive Behaviors': questionnaire_data.get('repetitive_behaviors_score', 0),
        'Social Cognition': questionnaire_data.get('social_cognition_score', 0),
        'Adaptive Functioning': questionnaire_data.get('adaptive_functioning_score', 0)
    }
    
    # Create radar chart
    fig = go.Figure()
    
    categories = list(domain_scores.keys())
    values = list(domain_scores.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Behavioral Domain Scores"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed domain analysis
    for domain, score in domain_scores.items():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric(domain, f"{score:.2f}")
        
        with col2:
            # Interpretation
            if score < 0.3:
                st.write("🔴 Below typical range - may indicate areas of concern")
            elif score < 0.6:
                st.write("🟡 Within lower typical range")
            else:
                st.write("🟢 Within typical range")
    
    # Response pattern analysis
    st.subheader("Response Patterns")
    
    # Get individual question responses
    question_responses = {k: v for k, v in questionnaire_data.items() if not k.endswith('_score')}
    
    if question_responses:
        # Create histogram of responses
        response_values = list(question_responses.values())
        
        fig = px.histogram(
            x=response_values,
            nbins=10,
            title="Distribution of Question Responses",
            labels={'x': 'Response Score', 'y': 'Frequency'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_gaze_results(gaze_data):
    """Show detailed gaze analysis results"""
    st.subheader("👁️ Gaze Pattern Analysis")
    
    if not gaze_data or 'overall_metrics' not in gaze_data:
        st.info("No gaze assessment data available.")
        return
    
    overall_metrics = gaze_data['overall_metrics']
    
    # Key gaze metrics
    st.subheader("Gaze Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Face Detection Rate",
            f"{overall_metrics.get('face_detection_rate', 0):.1%}",
            help="Percentage of frames where face was successfully detected"
        )
    
    with col2:
        st.metric(
            "Average Eye Contact",
            f"{overall_metrics.get('avg_eye_contact_score', 0):.2f}",
            help="Average eye contact score during assessment"
        )
    
    with col3:
        st.metric(
            "Social Attention",
            f"{overall_metrics.get('avg_social_attention_score', 0):.2f}",
            help="Average social attention score"
        )
    
    with col4:
        st.metric(
            "Gaze Stability",
            f"{1.0 - min(overall_metrics.get('std_saccade_amplitude', 0) / 100.0, 1.0):.2f}",
            help="Measure of gaze pattern stability"
        )
    
    # Task-specific performance
    if 'task_performances' in overall_metrics:
        st.subheader("Task Performance Comparison")
        
        task_perf = overall_metrics['task_performances']
        
        # Create performance comparison
        tasks = list(task_perf.keys())
        metrics = ['eye_contact_score', 'social_attention_score', 'face_detection_rate', 'gaze_stability']
        metric_names = ['Eye Contact', 'Social Attention', 'Face Detection', 'Gaze Stability']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names,
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [task_perf[task].get(metric, 0) for task in tasks]
            
            fig.add_trace(
                go.Bar(x=tasks, y=values, name=name, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Performance Across Tasks")
        st.plotly_chart(fig, use_container_width=True)
    
    # Gaze pattern visualization
    st.subheader("Gaze Pattern Analysis")
    
    # Time series of key metrics (if available)
    time_series_data = []
    for task_name, task_result in gaze_data.items():
        if task_name != 'overall_metrics' and 'raw_data' in task_result:
            for i, data_point in enumerate(task_result['raw_data']):
                time_series_data.append({
                    'time': i * 33,  # milliseconds
                    'eye_contact_score': data_point.get('eye_contact_score', 0),
                    'social_attention_score': data_point.get('social_attention_score', 0),
                    'task': task_name
                })
    
    if time_series_data:
        df = pd.DataFrame(time_series_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Eye Contact Over Time', 'Social Attention Over Time'],
            shared_xaxes=True
        )
        
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            
            fig.add_trace(
                go.Scatter(x=task_data['time'], y=task_data['eye_contact_score'], 
                          name=f'{task} - Eye Contact', mode='lines'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=task_data['time'], y=task_data['social_attention_score'], 
                          name=f'{task} - Social Attention', mode='lines'),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_layout(height=600, title_text="Gaze Metrics Over Time")
        
        st.plotly_chart(fig, use_container_width=True)

def show_ml_results(ml_results):
    """Show ML model results and insights"""
    st.subheader("🤖 Machine Learning Analysis")
    
    if not ml_results:
        st.info("No ML analysis available.")
        return
    
    # Model predictions
    st.subheader("Model Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction probabilities
        prob_typical = ml_results.get('probability_typical', 0)
        prob_asd = ml_results.get('probability_asd_indicators', 0)
        
        fig = go.Figure(data=[
            go.Bar(x=['Typical Development', 'ASD Indicators'], 
                   y=[prob_typical, prob_asd],
                   marker_color=['lightblue', 'lightcoral'])
        ])
        
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Individual model predictions
        individual_preds = ml_results.get('individual_predictions', {})
        individual_probs = ml_results.get('individual_probabilities', {})
        
        if individual_preds:
            st.write("**Individual Model Predictions:**")
            for model_name, prediction in individual_preds.items():
                prob = individual_probs.get(model_name, {}).get('asd_indicators', 0)
                st.write(f"- {model_name.title()}: {prediction} ({prob:.2%})")
    
    # Feature importance
    if 'feature_importance' in ml_results:
        st.subheader("Feature Importance")
        
        feature_importance = ml_results['feature_importance'][:10]  # Top 10 features
        
        features = [item[0] for item in feature_importance]
        importance = [item[1] for item in feature_importance]
        
        fig = go.Figure(data=[
            go.Bar(x=importance, y=features, orientation='h',
                   marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature interpretation
        st.subheader("Feature Interpretation")
        st.write("""
        The features shown above are the most influential factors in the ML model's prediction:
        
        - **Higher importance** = more influence on the final prediction
        - **Behavioral features** (starting with 'q_') come from questionnaire responses
        - **Gaze features** include eye contact, fixation patterns, and social attention metrics
        - **Combined features** integrate multiple data sources for comprehensive analysis
        """)
    
    # Model performance info
    st.subheader("Model Information")
    
    st.info("""
    **About the ML Models:**
    
    This analysis uses an ensemble of machine learning models trained on synthetic data that 
    reflects patterns found in autism research literature:
    
    - **Random Forest**: Combines multiple decision trees for robust predictions
    - **Gradient Boosting**: Sequential learning for complex pattern recognition  
    - **Logistic Regression**: Linear model for interpretable baseline predictions
    
    The ensemble approach helps improve prediction reliability by combining different model strengths.
    """)

def show_comprehensive_report():
    """Show comprehensive assessment report"""
    st.subheader("📄 Comprehensive Assessment Report")
    
    if 'comprehensive_results' not in st.session_state:
        st.warning("No comprehensive report available.")
        return
    
    report = st.session_state.comprehensive_results
    
    # Report header
    st.markdown(f"""
    **Assessment Date:** {report.get('timestamp', 'Unknown')}  
    **Assessment Type:** {report.get('assessment_type', 'Unknown').title()}
    """)
    
    # Executive summary
    st.subheader("Executive Summary")
    
    risk_assessment = report.get('risk_assessment', {})
    
    st.markdown(f"""
    **Risk Level:** {risk_assessment.get('risk_level', 'Unknown')}  
    **Overall Risk Score:** {risk_assessment.get('risk_score', 0):.2f}  
    **Model Confidence:** {risk_assessment.get('confidence', 0):.1%}
    """)
    
    # Risk factors
    risk_factors = risk_assessment.get('risk_factors', [])
    if risk_factors:
        st.write("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.write(f"- {factor}")
    
    # Detailed findings
    st.subheader("Detailed Findings")
    
    # Questionnaire summary
    q_summary = report.get('questionnaire_summary', {})
    if q_summary:
        st.write("**Behavioral Assessment:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Social Communication", f"{q_summary.get('social_communication_score', 0):.2f}")
        with col2:
            st.metric("Repetitive Behaviors", f"{q_summary.get('repetitive_behaviors_score', 0):.2f}")
        with col3:
            st.metric("Social Cognition", f"{q_summary.get('social_cognition_score', 0):.2f}")
        with col4:
            st.metric("Adaptive Functioning", f"{q_summary.get('adaptive_functioning_score', 0):.2f}")
    
    # Gaze analysis summary
    gaze_summary = report.get('gaze_analysis_summary', {})
    if gaze_summary:
        st.write("**Gaze Pattern Analysis:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Face Detection", f"{gaze_summary.get('face_detection_quality', 0):.1%}")
        with col2:
            st.metric("Eye Contact", f"{gaze_summary.get('eye_contact_performance', 0):.2f}")
        with col3:
            st.metric("Social Attention", f"{gaze_summary.get('social_attention_performance', 0):.2f}")
        with col4:
            st.metric("Gaze Stability", f"{gaze_summary.get('gaze_stability', 0):.2f}")
    
    # Recommendations
    st.subheader("Recommendations")
    
    recommendations = report.get('recommendations', [])
    for rec in recommendations:
        priority_emoji = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Moderate' else "🟢"
        
        with st.expander(f"{priority_emoji} {rec['category']} ({rec['priority']} Priority)"):
            st.write(f"**Recommendation:** {rec['recommendation']}")
            st.write(f"**Rationale:** {rec['rationale']}")
    
    # Important disclaimers
    st.subheader("Important Disclaimers")
    
    st.error("""
    **MEDICAL DISCLAIMER:**
    
    This assessment is for educational and screening support purposes only. It is NOT a diagnostic tool:
    
    - Results do not constitute a medical diagnosis
    - Professional evaluation is required for proper diagnosis
    - Early intervention and professional assessment are crucial
    - This tool may produce false positives or false negatives
    
    Always consult with qualified healthcare professionals for comprehensive evaluation and diagnosis.
    """)

def create_comprehensive_report(questionnaire_data, gaze_data, ml_results):
    """Create comprehensive assessment report"""
    from utils.data_processor import DataProcessor
    
    processor = DataProcessor()
    
    # Process gaze data for report
    gaze_metrics = {}
    if gaze_data and 'overall_metrics' in gaze_data:
        gaze_metrics = gaze_data['overall_metrics']
    
    # Create comprehensive report
    report = processor.create_comprehensive_report(
        questionnaire_data, gaze_metrics, ml_results
    )
    
    return report

def download_report():
    """Generate downloadable report"""
    if 'comprehensive_results' not in st.session_state:
        st.error("No report data available.")
        return
    
    report = st.session_state.comprehensive_results
    
    # Create formatted report text
    report_text = f"""
ASD BEHAVIORAL ANALYSIS REPORT
Generated: {report.get('timestamp', 'Unknown')}

EXECUTIVE SUMMARY
================
Risk Level: {report.get('risk_assessment', {}).get('risk_level', 'Unknown')}
Risk Score: {report.get('risk_assessment', {}).get('risk_score', 0):.2f}
Model Confidence: {report.get('risk_assessment', {}).get('confidence', 0):.1%}

DETAILED FINDINGS
=================

Behavioral Assessment:
{report.get('questionnaire_summary', {})}

Gaze Pattern Analysis:
{report.get('gaze_analysis_summary', {})}

RECOMMENDATIONS
===============
"""
    
    recommendations = report.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        report_text += f"""
{i}. {rec['category']} ({rec['priority']} Priority)
   Recommendation: {rec['recommendation']}
   Rationale: {rec['rationale']}
"""
    
    report_text += """

IMPORTANT DISCLAIMERS
====================
This assessment is for educational and screening support purposes only. 
It is NOT a diagnostic tool and should never be used as a substitute for 
professional medical advice, diagnosis, or treatment.

- Results do not constitute a medical diagnosis
- Professional evaluation is required for proper diagnosis  
- Early intervention and professional assessment are crucial
- This tool may produce false positives or false negatives

Always consult with qualified healthcare professionals for comprehensive 
evaluation and diagnosis.
"""
    
    # Offer download
    st.download_button(
        label="📄 Download Report as Text",
        data=report_text,
        file_name=f"asd_assessment_report_{report.get('timestamp', 'unknown').replace(':', '-')}.txt",
        mime="text/plain"
    )
