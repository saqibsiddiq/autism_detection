import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from database.models import db_manager

def show_admin_dashboard():
    st.header("📊 Admin Dashboard")
    
    # Password protection for admin access
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("Admin access required. Please enter the admin password.")
        password = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":  # Simple password for demo
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
        return
    
    # Admin dashboard content
    st.success("✅ Admin access granted")
    
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # Tabs for different admin views
    tabs = st.tabs([
        "📈 Overview", 
        "👥 Users", 
        "📋 Assessments", 
        "📊 Analytics",
        "🗄️ Database"
    ])
    
    with tabs[0]:
        show_overview_stats()
    
    with tabs[1]:
        show_user_management()
    
    with tabs[2]:
        show_assessment_management()
    
    with tabs[3]:
        show_analytics()
    
    with tabs[4]:
        show_database_management()

def show_overview_stats():
    st.subheader("System Overview")
    
    try:
        stats = db_manager.get_assessment_statistics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", stats.get("total_users", 0))
        
        with col2:
            st.metric("Total Assessments", stats.get("total_assessments", 0))
        
        with col3:
            st.metric("Completed Assessments", stats.get("completed_assessments", 0))
        
        with col4:
            completion_rate = stats.get("completion_rate", 0)
            st.metric("Completion Rate", f"{completion_rate:.1%}")
        
        # Risk distribution chart
        if stats.get("risk_distribution"):
            st.subheader("Risk Level Distribution")
            
            risk_data = stats["risk_distribution"]
            fig = px.pie(
                values=list(risk_data.values()),
                names=list(risk_data.keys()),
                title="Assessment Risk Levels"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def show_user_management():
    st.subheader("User Management")
    
    try:
        db = db_manager.get_session()
        
        # Get recent users
        from database.models import User, Assessment
        users = db.query(User).order_by(User.created_at.desc()).limit(50).all()
        
        if users:
            user_data = []
            for user in users:
                assessments = db.query(Assessment).filter(Assessment.user_id == user.id).all()
                user_data.append({
                    "ID": user.id,
                    "Session ID": user.session_id[:8] + "...",
                    "Created": user.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Age Group": user.age_group or "Not specified",
                    "Consent": "✅" if user.consent_given else "❌",
                    "Assessments": len(assessments),
                    "Completed": len([a for a in assessments if a.status == "completed"])
                })
            
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
            
            # User activity over time
            st.subheader("User Registration Timeline")
            daily_users = {}
            for user in users:
                date = user.created_at.date()
                daily_users[date] = daily_users.get(date, 0) + 1
            
            if daily_users:
                dates = list(daily_users.keys())
                counts = list(daily_users.values())
                
                fig = px.line(x=dates, y=counts, title="Daily User Registrations")
                fig.update_xaxes(title="Date")
                fig.update_yaxes(title="New Users")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No users found in database.")
        
        db.close()
        
    except Exception as e:
        st.error(f"Error loading user data: {e}")

def show_assessment_management():
    st.subheader("Assessment Management")
    
    try:
        db = db_manager.get_session()
        
        from database.models import Assessment, AssessmentResult, QuestionnaireResponse
        
        # Recent assessments
        assessments = db.query(Assessment).order_by(Assessment.started_at.desc()).limit(20).all()
        
        if assessments:
            assessment_data = []
            for assessment in assessments:
                result = db.query(AssessmentResult).filter(
                    AssessmentResult.assessment_id == assessment.id
                ).first()
                
                assessment_data.append({
                    "ID": assessment.id,
                    "User ID": assessment.user_id,
                    "Type": assessment.assessment_type,
                    "Status": assessment.status,
                    "Started": assessment.started_at.strftime("%Y-%m-%d %H:%M"),
                    "Duration (min)": f"{assessment.total_duration // 60}:{assessment.total_duration % 60:02d}" if assessment.total_duration else "N/A",
                    "Risk Level": result.risk_level.title() if result else "N/A",
                    "Overall Score": f"{result.overall_score:.2f}" if result and result.overall_score else "N/A"
                })
            
            df = pd.DataFrame(assessment_data)
            st.dataframe(df, use_container_width=True)
            
            # Assessment completion funnel
            st.subheader("Assessment Completion Funnel")
            
            status_counts = {}
            for assessment in assessments:
                status_counts[assessment.status] = status_counts.get(assessment.status, 0) + 1
            
            if status_counts:
                fig = px.funnel(
                    y=list(status_counts.keys()),
                    x=list(status_counts.values()),
                    title="Assessment Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No assessments found in database.")
        
        db.close()
        
    except Exception as e:
        st.error(f"Error loading assessment data: {e}")

def show_analytics():
    st.subheader("Advanced Analytics")
    
    try:
        db = db_manager.get_session()
        
        from database.models import AssessmentResult, QuestionnaireResponse, GazeData
        
        # Risk level trends over time
        results = db.query(AssessmentResult).order_by(AssessmentResult.created_at).all()
        
        if results:
            st.subheader("Risk Level Trends")
            
            # Create timeline data
            timeline_data = []
            for result in results:
                timeline_data.append({
                    "Date": result.created_at.date(),
                    "Risk Level": result.risk_level or "unknown",
                    "Overall Score": result.overall_score or 0,
                    "Confidence": result.confidence_score or 0
                })
            
            df = pd.DataFrame(timeline_data)
            
            if not df.empty:
                # Risk level distribution over time
                fig = px.histogram(
                    df, x="Date", color="Risk Level",
                    title="Risk Level Distribution Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Overall score distribution
                fig2 = px.histogram(
                    df, x="Overall Score", nbins=20,
                    title="Overall Score Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Question response analytics
        st.subheader("Question Response Patterns")
        
        responses = db.query(QuestionnaireResponse).all()
        
        if responses:
            response_data = []
            for response in responses:
                response_data.append({
                    "Question ID": response.question_id,
                    "Domain": response.domain or "unknown",
                    "Response Value": response.response_value,
                    "Weight": response.weight,
                    "Critical Item": response.is_critical_item
                })
            
            response_df = pd.DataFrame(response_data)
            
            if not response_df.empty:
                # Average scores by domain
                domain_avg = response_df.groupby('Domain')['Response Value'].mean().reset_index()
                
                fig3 = px.bar(
                    domain_avg, x="Domain", y="Response Value",
                    title="Average Response Scores by Domain"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Critical item analysis
                critical_analysis = response_df.groupby(['Question ID', 'Critical Item'])['Response Value'].mean().reset_index()
                
                fig4 = px.box(
                    response_df, x="Critical Item", y="Response Value",
                    title="Response Distribution: Critical vs Non-Critical Items"
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        # Gaze data analytics
        st.subheader("Gaze Analysis Patterns")
        
        gaze_data = db.query(GazeData).limit(1000).all()  # Limit for performance
        
        if gaze_data:
            gaze_df_data = []
            for gaze in gaze_data:
                gaze_df_data.append({
                    "Task Type": gaze.task_type or "unknown",
                    "Eye Contact Score": gaze.eye_contact_score or 0,
                    "Social Attention Score": gaze.social_attention_score or 0,
                    "Fixation Duration": gaze.fixation_duration or 0,
                    "Face Detected": gaze.face_detected
                })
            
            gaze_df = pd.DataFrame(gaze_df_data)
            
            if not gaze_df.empty:
                # Average gaze metrics by task type
                task_avg = gaze_df.groupby('Task Type').agg({
                    'Eye Contact Score': 'mean',
                    'Social Attention Score': 'mean',
                    'Fixation Duration': 'mean'
                }).reset_index()
                
                fig5 = px.bar(
                    task_avg.melt(id_vars=['Task Type'], var_name='Metric', value_name='Score'),
                    x='Task Type', y='Score', color='Metric',
                    title="Average Gaze Metrics by Task Type"
                )
                st.plotly_chart(fig5, use_container_width=True)
        
        db.close()
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def show_database_management():
    st.subheader("Database Management")
    
    try:
        db = db_manager.get_session()
        
        # Table statistics
        from database.models import User, Assessment, QuestionnaireResponse, GazeData, AssessmentResult
        
        st.subheader("Table Statistics")
        
        table_stats = [
            ("Users", db.query(User).count()),
            ("Assessments", db.query(Assessment).count()),
            ("Questionnaire Responses", db.query(QuestionnaireResponse).count()),
            ("Gaze Data Points", db.query(GazeData).count()),
            ("Assessment Results", db.query(AssessmentResult).count())
        ]
        
        for table_name, count in table_stats:
            st.metric(table_name, count)
        
        # Database cleanup options
        st.subheader("Database Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Old Sessions", help="Remove sessions older than 30 days"):
                try:
                    cutoff_date = datetime.utcnow() - timedelta(days=30)
                    old_users = db.query(User).filter(User.created_at < cutoff_date).all()
                    
                    for user in old_users:
                        # Delete associated data
                        assessments = db.query(Assessment).filter(Assessment.user_id == user.id).all()
                        for assessment in assessments:
                            db.query(QuestionnaireResponse).filter(QuestionnaireResponse.assessment_id == assessment.id).delete()
                            db.query(GazeData).filter(GazeData.assessment_id == assessment.id).delete()
                            db.query(AssessmentResult).filter(AssessmentResult.assessment_id == assessment.id).delete()
                        
                        db.query(Assessment).filter(Assessment.user_id == user.id).delete()
                        db.delete(user)
                    
                    db.commit()
                    st.success(f"Cleaned {len(old_users)} old sessions")
                    
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
                    db.rollback()
        
        with col2:
            if st.button("📊 Export Data", help="Export assessment data as CSV"):
                try:
                    # Create export data
                    export_data = []
                    
                    assessments = db.query(Assessment).filter(Assessment.status == "completed").all()
                    
                    for assessment in assessments:
                        result = db.query(AssessmentResult).filter(AssessmentResult.assessment_id == assessment.id).first()
                        if result:
                            export_data.append({
                                "assessment_id": assessment.id,
                                "user_id": assessment.user_id,
                                "completed_at": assessment.completed_at,
                                "duration_minutes": assessment.total_duration // 60 if assessment.total_duration else 0,
                                "risk_level": result.risk_level,
                                "overall_score": result.overall_score,
                                "confidence_score": result.confidence_score
                            })
                    
                    if export_data:
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"asd_assessment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No completed assessments to export")
                        
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        # Raw SQL query interface
        st.subheader("SQL Query Interface")
        st.warning("⚠️ Advanced users only. Be careful with DELETE/UPDATE queries.")
        
        sql_query = st.text_area(
            "Enter SQL Query:",
            placeholder="SELECT * FROM users LIMIT 10;",
            height=100
        )
        
        if st.button("Execute Query"):
            if sql_query.strip():
                try:
                    result = db.execute(sql_query)
                    
                    if sql_query.strip().upper().startswith("SELECT"):
                        rows = result.fetchall()
                        if rows:
                            columns = result.keys()
                            query_df = pd.DataFrame(rows, columns=columns)
                            st.dataframe(query_df, use_container_width=True)
                        else:
                            st.info("Query returned no results")
                    else:
                        db.commit()
                        st.success("Query executed successfully")
                        
                except Exception as e:
                    st.error(f"Query error: {e}")
                    db.rollback()
        
        db.close()
        
    except Exception as e:
        st.error(f"Error in database management: {e}")