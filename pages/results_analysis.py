import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from database.models import db_manager
import time
from datetime import datetime

def show_results_analysis_page():
    st.header("📊 Comprehensive Results & Analysis")
    
    # Check if any tests have been completed
    test_results = {
        'face_recognition': st.session_state.get('face_test_results', {}),
        'social_attention': st.session_state.get('social_test_results', {}),
        'visual_pattern': st.session_state.get('pattern_test_results', {}),
        'motion_tracking': st.session_state.get('motion_test_results', {})
    }
    
    completed_tests = [test for test, results in test_results.items() if results]
    
    if not completed_tests:
        st.warning("⚠️ No test results available. Please complete at least one behavioral test to see analysis.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Overview", use_container_width=True):
                st.session_state.current_test = 0
                st.rerun()
        with col2:
            if st.button("👁️ Start Tests", use_container_width=True):
                st.session_state.current_test = 1
                st.rerun()
        return
    
    # Generate comprehensive analysis
    analysis_results = generate_comprehensive_analysis(test_results)
    
    # Save results to database
    save_analysis_to_database(analysis_results)
    
    # Display analysis tabs
    tabs = st.tabs([
        "🎯 Overall Assessment",
        "📈 Detailed Metrics", 
        "🧠 Behavioral Insights",
        "📋 Clinical Interpretation",
        "📄 Generate Report"
    ])
    
    with tabs[0]:
        show_overall_assessment(analysis_results)
    
    with tabs[1]:
        show_detailed_metrics(test_results, analysis_results)
    
    with tabs[2]:
        show_behavioral_insights(analysis_results)
    
    with tabs[3]:
        show_clinical_interpretation(analysis_results)
    
    with tabs[4]:
        show_report_generation(analysis_results)

def generate_comprehensive_analysis(test_results):
    """Generate comprehensive behavioral analysis from all test results"""
    
    analysis = {
        'timestamp': datetime.now(),
        'tests_completed': [],
        'overall_scores': {},
        'behavioral_patterns': {},
        'risk_indicators': {},
        'recommendations': [],
        'confidence_level': 0
    }
    
    # Analyze each completed test
    total_tests = len([r for r in test_results.values() if r])
    
    # Face Recognition Analysis
    if test_results['face_recognition']:
        face_analysis = analyze_face_recognition_results(test_results['face_recognition'])
        analysis['tests_completed'].append('Face Recognition')
        analysis['overall_scores']['face_preference'] = face_analysis['face_preference_score']
        analysis['behavioral_patterns']['social_orientation'] = face_analysis['social_orientation']
        analysis['risk_indicators']['reduced_face_attention'] = face_analysis['risk_level']
    
    # Social Attention Analysis
    if test_results['social_attention']:
        social_analysis = analyze_social_attention_results(test_results['social_attention'])
        analysis['tests_completed'].append('Social Attention')
        analysis['overall_scores']['social_attention'] = social_analysis['social_attention_score']
        analysis['behavioral_patterns']['social_engagement'] = social_analysis['engagement_pattern']
        analysis['risk_indicators']['social_attention_deficit'] = social_analysis['risk_level']
    
    # Visual Pattern Analysis
    if test_results['visual_pattern']:
        pattern_analysis = analyze_visual_pattern_results(test_results['visual_pattern'])
        analysis['tests_completed'].append('Visual Pattern')
        analysis['overall_scores']['pattern_preference'] = pattern_analysis['pattern_preference_score']
        analysis['behavioral_patterns']['visual_processing'] = pattern_analysis['processing_style']
        analysis['risk_indicators']['pattern_fixation'] = pattern_analysis['risk_level']
    
    # Motion Tracking Analysis
    if test_results['motion_tracking']:
        motion_analysis = analyze_motion_tracking_results(test_results['motion_tracking'])
        analysis['tests_completed'].append('Motion Tracking')
        analysis['overall_scores']['tracking_ability'] = motion_analysis['tracking_score']
        analysis['behavioral_patterns']['eye_movement'] = motion_analysis['movement_pattern']
        analysis['risk_indicators']['atypical_eye_movements'] = motion_analysis['risk_level']
    
    # Calculate overall risk assessment
    analysis['overall_risk_level'] = calculate_overall_risk(analysis['risk_indicators'])
    analysis['confidence_level'] = min(total_tests * 0.25, 1.0)  # More tests = higher confidence
    
    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(analysis)
    
    return analysis

def analyze_face_recognition_results(face_results):
    """Analyze face recognition test results"""
    total_face_attention = 0
    total_object_attention = 0
    phases_completed = 0
    
    for phase_key, results in face_results.items():
        if results:
            total_face_attention += results.get('face_attention_time', 0)
            total_object_attention += results.get('object_attention_time', 0)
            phases_completed += 1
    
    total_attention = total_face_attention + total_object_attention
    face_preference_score = total_face_attention / max(total_attention, 1)
    
    # Determine social orientation and risk level
    if face_preference_score >= 0.6:
        social_orientation = "Strong social orientation"
        risk_level = "low"
    elif face_preference_score >= 0.4:
        social_orientation = "Moderate social orientation"
        risk_level = "moderate"
    else:
        social_orientation = "Reduced social orientation"
        risk_level = "high"
    
    return {
        'face_preference_score': face_preference_score,
        'social_orientation': social_orientation,
        'risk_level': risk_level,
        'total_attention_points': total_attention
    }

def analyze_social_attention_results(social_results):
    """Analyze social attention test results"""
    total_social = 0
    total_non_social = 0
    social_ratios = []
    
    for scenario_key, results in social_results.items():
        if results:
            total_social += results.get('social_attention_score', 0)
            total_non_social += results.get('non_social_attention_score', 0)
            social_ratios.append(results.get('social_attention_ratio', 0))
    
    avg_social_ratio = np.mean(social_ratios) if social_ratios else 0
    
    if avg_social_ratio >= 0.65:
        engagement_pattern = "High social engagement"
        risk_level = "low"
    elif avg_social_ratio >= 0.35:
        engagement_pattern = "Moderate social engagement"
        risk_level = "moderate"
    else:
        engagement_pattern = "Low social engagement"
        risk_level = "high"
    
    return {
        'social_attention_score': avg_social_ratio,
        'engagement_pattern': engagement_pattern,
        'risk_level': risk_level,
        'total_social_attention': total_social
    }

def analyze_visual_pattern_results(pattern_results):
    """Analyze visual pattern test results"""
    pattern_preferences = []
    fixation_durations = []
    
    for test_key, results in pattern_results.items():
        if results:
            pattern_preferences.append(results.get('pattern_preference_ratio', 0))
            fixation_durations.append(results.get('avg_fixation_duration', 0))
    
    avg_pattern_preference = np.mean(pattern_preferences) if pattern_preferences else 0
    avg_fixation_duration = np.mean(fixation_durations) if fixation_durations else 0
    
    # High pattern preference may indicate ASD-related visual processing differences
    if avg_pattern_preference >= 0.7:
        processing_style = "High pattern preference"
        risk_level = "moderate"  # Not necessarily negative, but characteristic
    elif avg_pattern_preference >= 0.5:
        processing_style = "Moderate pattern preference"
        risk_level = "low"
    else:
        processing_style = "Low pattern preference"
        risk_level = "low"
    
    return {
        'pattern_preference_score': avg_pattern_preference,
        'processing_style': processing_style,
        'risk_level': risk_level,
        'avg_fixation_duration': avg_fixation_duration
    }

def analyze_motion_tracking_results(motion_results):
    """Analyze motion tracking test results"""
    tracking_accuracies = []
    pursuit_qualities = []
    saccadic_counts = []
    
    for test_key, results in motion_results.items():
        if results:
            tracking_accuracies.append(results.get('tracking_accuracy', 0))
            pursuit_qualities.append(results.get('smooth_pursuit_quality', 0))
            saccadic_counts.append(results.get('saccadic_movements_count', 0))
    
    avg_tracking_accuracy = np.mean(tracking_accuracies) if tracking_accuracies else 0
    avg_pursuit_quality = np.mean(pursuit_qualities) if pursuit_qualities else 0
    total_saccades = sum(saccadic_counts)
    
    if avg_tracking_accuracy >= 0.7 and avg_pursuit_quality >= 0.5:
        movement_pattern = "Smooth eye movement tracking"
        risk_level = "low"
    elif avg_tracking_accuracy >= 0.5 or avg_pursuit_quality >= 0.3:
        movement_pattern = "Moderate tracking ability"
        risk_level = "moderate"
    else:
        movement_pattern = "Atypical eye movement patterns"
        risk_level = "high"
    
    return {
        'tracking_score': avg_tracking_accuracy,
        'movement_pattern': movement_pattern,
        'risk_level': risk_level,
        'total_saccades': total_saccades
    }

def calculate_overall_risk(risk_indicators):
    """Calculate overall risk level from individual test risks"""
    if not risk_indicators:
        return "insufficient_data"
    
    risk_values = {'low': 0, 'moderate': 1, 'high': 2}
    risk_scores = [risk_values.get(risk, 0) for risk in risk_indicators.values()]
    avg_risk_score = np.mean(risk_scores)
    
    if avg_risk_score <= 0.5:
        return "low"
    elif avg_risk_score <= 1.2:
        return "moderate"
    else:
        return "high"

def generate_recommendations(analysis):
    """Generate personalized recommendations based on analysis"""
    recommendations = []
    
    risk_level = analysis['overall_risk_level']
    patterns = analysis['behavioral_patterns']
    
    # General recommendations
    recommendations.append("Continue monitoring behavioral patterns with qualified healthcare professionals")
    
    # Specific recommendations based on patterns
    if 'social_orientation' in patterns and 'reduced' in patterns['social_orientation'].lower():
        recommendations.append("Consider social skills training and structured interaction opportunities")
    
    if 'social_engagement' in patterns and 'low' in patterns['social_engagement'].lower():
        recommendations.append("Explore social communication interventions and peer interaction programs")
    
    if 'visual_processing' in patterns and 'high pattern' in patterns['visual_processing'].lower():
        recommendations.append("Leverage pattern recognition strengths in learning and development activities")
    
    if 'eye_movement' in patterns and 'atypical' in patterns['eye_movement'].lower():
        recommendations.append("Consider occupational therapy focusing on visual-motor integration")
    
    if risk_level == "high":
        recommendations.append("Recommend comprehensive developmental assessment by qualified specialists")
    elif risk_level == "moderate":
        recommendations.append("Consider follow-up assessment and monitoring of developmental progress")
    
    return recommendations

def show_overall_assessment(analysis):
    """Display overall assessment summary"""
    st.subheader("🎯 Overall Assessment Summary")
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tests Completed", len(analysis['tests_completed']))
    
    with col2:
        risk_level = analysis['overall_risk_level']
        risk_color = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(risk_level, "⚪")
        st.metric("Risk Level", f"{risk_color} {risk_level.title()}")
    
    with col3:
        confidence = analysis['confidence_level']
        st.metric("Analysis Confidence", f"{confidence:.1%}")
    
    with col4:
        avg_score = np.mean(list(analysis['overall_scores'].values())) if analysis['overall_scores'] else 0
        st.metric("Average Score", f"{avg_score:.2f}")
    
    # Risk level explanation
    st.markdown("---")
    
    risk_level = analysis['overall_risk_level']
    
    if risk_level == "low":
        st.success("""
        **Low Risk Indication**
        
        The behavioral patterns observed are generally within typical ranges. This suggests:
        - Normal social attention and engagement patterns
        - Typical visual processing and eye movement behaviors
        - Age-appropriate developmental indicators
        """)
    elif risk_level == "moderate":
        st.warning("""
        **Moderate Risk Indication**
        
        Some behavioral patterns show mild differences that may warrant attention:
        - Mixed results across different assessment areas
        - Some atypical patterns that could benefit from monitoring
        - Consider follow-up assessment for comprehensive evaluation
        """)
    else:  # high risk
        st.error("""
        **Higher Risk Indication**
        
        Several behavioral patterns suggest potential developmental differences:
        - Multiple areas showing atypical patterns
        - Patterns consistent with autism spectrum characteristics
        - Strongly recommend professional assessment by qualified specialists
        """)
    
    # Behavioral patterns summary
    st.markdown("### 🧠 Key Behavioral Patterns")
    
    for pattern_type, description in analysis['behavioral_patterns'].items():
        st.write(f"**{pattern_type.replace('_', ' ').title()}:** {description}")

def show_detailed_metrics(test_results, analysis):
    """Display detailed metrics from all tests"""
    st.subheader("📈 Detailed Test Metrics")
    
    # Create metrics comparison chart
    metrics_data = []
    
    for test_name, scores in analysis['overall_scores'].items():
        metrics_data.append({
            'Test': test_name.replace('_', ' ').title(),
            'Score': scores,
            'Category': 'Behavioral Assessment'
        })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        
        fig = px.bar(df, x='Test', y='Score', 
                    title="Test Scores Overview",
                    color='Score',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual test details
    for test_name, results in test_results.items():
        if results:
            with st.expander(f"📊 {test_name.replace('_', ' ').title()} Details", expanded=False):
                show_individual_test_details(test_name, results)

def show_individual_test_details(test_name, results):
    """Show detailed results for individual test"""
    
    if test_name == 'face_recognition':
        total_face = sum(r.get('face_attention_time', 0) for r in results.values() if r)
        total_object = sum(r.get('object_attention_time', 0) for r in results.values() if r)
        
        if total_face + total_object > 0:
            attention_data = pd.DataFrame({
                'Stimulus': ['Faces', 'Objects'],
                'Attention Time': [total_face, total_object]
            })
            
            fig = px.pie(attention_data, values='Attention Time', names='Stimulus',
                        title="Face vs Object Attention Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    elif test_name == 'social_attention':
        social_scores = [r.get('social_attention_ratio', 0) for r in results.values() if r]
        if social_scores:
            scenario_names = [f"Scenario {i+1}" for i in range(len(social_scores))]
            
            fig = px.bar(x=scenario_names, y=social_scores,
                        title="Social Attention Ratio by Scenario")
            st.plotly_chart(fig, use_container_width=True)
    
    elif test_name == 'visual_pattern':
        pattern_prefs = [r.get('pattern_preference_ratio', 0) for r in results.values() if r]
        if pattern_prefs:
            test_names = [f"Test {i+1}" for i in range(len(pattern_prefs))]
            
            fig = px.bar(x=test_names, y=pattern_prefs,
                        title="Pattern Preference by Test Type")
            st.plotly_chart(fig, use_container_width=True)
    
    elif test_name == 'motion_tracking':
        tracking_acc = [r.get('tracking_accuracy', 0) for r in results.values() if r]
        if tracking_acc:
            motion_types = ['Circular', 'Horizontal', 'Vertical', 'Figure-8'][:len(tracking_acc)]
            
            fig = px.bar(x=motion_types, y=tracking_acc,
                        title="Tracking Accuracy by Motion Type")
            st.plotly_chart(fig, use_container_width=True)

def show_behavioral_insights(analysis):
    """Display behavioral insights and interpretations"""
    st.subheader("🧠 Behavioral Insights")
    
    st.markdown("""
    ### Understanding the Results
    
    This analysis examines behavioral patterns associated with autism spectrum characteristics:
    """)
    
    insights = {
        "Social Attention": {
            "description": "Measures preference for social versus non-social stimuli",
            "typical": "Strong preference for social information and faces",
            "atypical": "Reduced attention to social cues, preference for objects or patterns"
        },
        "Face Recognition": {
            "description": "Evaluates gaze patterns when viewing human faces",
            "typical": "Natural eye contact and face preference",
            "atypical": "Reduced face attention, avoidance of eye contact"
        },
        "Visual Processing": {
            "description": "Assesses preference for patterns and visual structure",
            "typical": "Balanced attention to patterns and random stimuli",
            "atypical": "Strong preference for patterns, repetitive visual elements"
        },
        "Eye Movement": {
            "description": "Analyzes smooth pursuit and tracking abilities",
            "typical": "Smooth tracking, appropriate saccadic movements",
            "atypical": "Jerky movements, difficulty with smooth pursuit"
        }
    }
    
    for domain, info in insights.items():
        with st.expander(f"📖 {domain}", expanded=False):
            st.write(f"**What it measures:** {info['description']}")
            st.write(f"**Typical pattern:** {info['typical']}")
            st.write(f"**Atypical pattern:** {info['atypical']}")
    
    # Research context
    st.markdown("---")
    st.markdown("""
    ### 📚 Research Context
    
    These assessments are based on established research findings about behavioral differences 
    in autism spectrum disorders:
    
    - **Social Attention**: Individuals with ASD often show reduced attention to social stimuli (Klin et al., 2002)
    - **Face Processing**: Atypical face processing patterns are commonly observed (Dawson et al., 2005)
    - **Visual Patterns**: Enhanced pattern detection and preference for structured visuals (Mottron et al., 2006)
    - **Eye Movements**: Differences in smooth pursuit and saccadic movements (Takarae et al., 2007)
    
    **Important:** These patterns exist on a spectrum and individual differences are significant.
    """)

def show_clinical_interpretation(analysis):
    """Display clinical interpretation and recommendations"""
    st.subheader("📋 Clinical Interpretation")
    
    st.warning("""
    **⚠️ Important Medical Disclaimer**
    
    This analysis is for educational and screening purposes only. It is NOT a diagnostic tool 
    and should NOT be used for medical diagnosis. All results must be interpreted by 
    qualified healthcare professionals.
    """)
    
    # Risk level interpretation
    risk_level = analysis['overall_risk_level']
    
    st.markdown("### 🎯 Risk Assessment Interpretation")
    
    if risk_level == "low":
        st.success("""
        **Low Risk Level:**
        - Behavioral patterns are generally within typical developmental ranges
        - No immediate concerns indicated by current assessment
        - Continue routine developmental monitoring
        """)
    elif risk_level == "moderate":
        st.warning("""
        **Moderate Risk Level:**
        - Some behavioral patterns show mild atypicalities
        - May benefit from closer monitoring and follow-up assessment
        - Consider consultation with developmental specialists if concerns persist
        """)
    else:
        st.error("""
        **Higher Risk Level:**
        - Multiple behavioral patterns suggest potential developmental differences
        - Patterns consistent with autism spectrum characteristics
        - Strongly recommend comprehensive evaluation by qualified professionals
        """)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    for i, recommendation in enumerate(analysis['recommendations'], 1):
        st.write(f"{i}. {recommendation}")
    
    # Next steps
    st.markdown("### 🚀 Suggested Next Steps")
    
    if risk_level == "high":
        st.markdown("""
        1. **Immediate:** Schedule consultation with developmental pediatrician or child psychologist
        2. **Assessment:** Request comprehensive autism spectrum evaluation
        3. **Documentation:** Bring these results to share with healthcare providers
        4. **Support:** Explore early intervention services if appropriate
        """)
    elif risk_level == "moderate":
        st.markdown("""
        1. **Monitoring:** Continue observing behavioral patterns in daily life
        2. **Follow-up:** Consider repeat assessment in 3-6 months
        3. **Consultation:** Discuss results with pediatrician at next visit
        4. **Documentation:** Keep record of developmental milestones
        """)
    else:
        st.markdown("""
        1. **Routine:** Continue regular developmental check-ups
        2. **Awareness:** Stay informed about developmental milestones
        3. **Support:** Maintain supportive environment for healthy development
        4. **Monitoring:** Be aware of any emerging concerns
        """)

def show_report_generation(analysis):
    """Generate downloadable report"""
    st.subheader("📄 Generate Assessment Report")
    
    st.markdown("Create a comprehensive report of the behavioral analysis results.")
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        include_detailed_metrics = st.checkbox("Include Detailed Test Metrics", value=True)
        include_visualizations = st.checkbox("Include Data Visualizations", value=True)
    
    with col2:
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_research_context = st.checkbox("Include Research Background", value=False)
    
    if st.button("📋 Generate Report", type="primary"):
        report_content = generate_report_content(
            analysis, 
            include_detailed_metrics,
            include_visualizations,
            include_recommendations,
            include_research_context
        )
        
        # Create download button
        st.download_button(
            label="📥 Download Report (PDF-ready)",
            data=report_content,
            file_name=f"asd_behavioral_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        # Preview
        with st.expander("📖 Report Preview", expanded=True):
            st.text_area("Report Content:", report_content, height=400)

def generate_report_content(analysis, include_metrics, include_viz, include_rec, include_research):
    """Generate formatted report content"""
    
    content = f"""
ASD BEHAVIORAL ANALYSIS REPORT
=============================

Generated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
----------------
Tests Completed: {', '.join(analysis['tests_completed'])}
Overall Risk Level: {analysis['overall_risk_level'].upper()}
Analysis Confidence: {analysis['confidence_level']:.1%}

ASSESSMENT RESULTS
-----------------
"""
    
    # Add overall scores
    for test, score in analysis['overall_scores'].items():
        content += f"{test.replace('_', ' ').title()}: {score:.3f}\n"
    
    content += "\nBEHAVIORAL PATTERNS\n"
    content += "------------------\n"
    
    for pattern, description in analysis['behavioral_patterns'].items():
        content += f"{pattern.replace('_', ' ').title()}: {description}\n"
    
    if include_rec:
        content += "\nRECOMMENDATIONS\n"
        content += "--------------\n"
        for i, rec in enumerate(analysis['recommendations'], 1):
            content += f"{i}. {rec}\n"
    
    content += "\nIMPORTANT DISCLAIMER\n"
    content += "-------------------\n"
    content += """This analysis is for educational and screening purposes only. 
It is NOT a diagnostic tool and should NOT be used for medical diagnosis. 
All results must be interpreted by qualified healthcare professionals.

For questions or concerns, please consult with:
- Developmental pediatrician
- Child psychologist
- Licensed clinical social worker
- Board-certified psychiatrist
"""
    
    if include_research:
        content += "\nRESEARCH BACKGROUND\n"
        content += "------------------\n"
        content += """This assessment is based on established research in autism spectrum disorders:

- Social attention differences (Klin et al., 2002)
- Face processing atypicalities (Dawson et al., 2005)  
- Enhanced pattern detection (Mottron et al., 2006)
- Eye movement differences (Takarae et al., 2007)
"""
    
    return content

def save_analysis_to_database(analysis):
    """Save comprehensive analysis to database"""
    try:
        if st.session_state.assessment_id:
            # Save final results
            db_manager.save_assessment_results(
                st.session_state.assessment_id,
                analysis['overall_scores'],
                analysis['behavioral_patterns'],
                {
                    'overall_risk_level': analysis['overall_risk_level'],
                    'confidence_level': analysis['confidence_level'],
                    'tests_completed': analysis['tests_completed']
                },
                analysis['risk_indicators'],
                analysis['recommendations']
            )
            
            # Mark assessment as completed
            db_manager.complete_assessment(st.session_state.assessment_id)
            
    except Exception as e:
        st.error(f"Error saving analysis to database: {e}")