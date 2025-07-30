import streamlit as st
import json
import os

def load_questions():
    """Load questionnaire questions from JSON file"""
    questions_file = "data/asd_questions.json"
    
    if os.path.exists(questions_file):
        with open(questions_file, 'r') as f:
            return json.load(f)
    else:
        # Fallback questions if file doesn't exist
        return get_default_questions()

def get_default_questions():
    """Default questions based on M-CHAT-R and AQ-10"""
    return {
        "sections": [
            {
                "name": "Early Development and Social Communication",
                "description": "Questions about early development and social communication patterns",
                "questions": [
                    {
                        "id": "enjoys_being_swung",
                        "text": "Does your child enjoy being swung, bounced on your knee, etc.?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "interest_in_other_children",
                        "text": "Does your child take an interest in other children?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "enjoys_climbing",
                        "text": "Does your child like climbing on things, such as up stairs?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "enjoys_peek_a_boo",
                        "text": "Does your child enjoy playing peek-a-boo/hide-and-seek?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "pretend_play",
                        "text": "Does your child ever pretend, for example, to talk on the phone or take care of dolls, or pretend other things?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "uses_index_finger",
                        "text": "Does your child ever use his/her index finger to point, to ask for something?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "brings_objects_to_show",
                        "text": "Does your child ever bring objects over to you (parent) to show you something?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "eye_contact",
                        "text": "Does your child look you in the eye for more than a second or two?",
                        "type": "yes_no",
                        "reverse_scored": False
                    },
                    {
                        "id": "unusual_finger_movements",
                        "text": "Does your child make unusual finger movements near his/her face?",
                        "type": "yes_no",
                        "reverse_scored": True
                    },
                    {
                        "id": "tries_to_attract_attention",
                        "text": "Does your child ever try to attract your attention to his/her own activity?",
                        "type": "yes_no",
                        "reverse_scored": False
                    }
                ]
            },
            {
                "name": "Social Interaction and Communication",
                "description": "Questions about social interaction and communication preferences",
                "questions": [
                    {
                        "id": "notices_small_sounds",
                        "text": "I often notice small sounds when others do not",
                        "type": "likert",
                        "reverse_scored": True
                    },
                    {
                        "id": "concentrates_on_whole_picture",
                        "text": "I usually concentrate more on the whole picture, rather than the small details",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "easy_to_do_several_things",
                        "text": "I find it easy to do more than one thing at once",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "enjoys_social_chit_chat",
                        "text": "I enjoy social chit-chat",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "finds_easy_to_read_between_lines",
                        "text": "I find it easy to 'read between the lines' when someone is talking to me",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "knows_how_to_tell_stories",
                        "text": "I know how to tell if someone listening to me is getting bored",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "drawn_to_people",
                        "text": "When I'm reading a story I find it difficult to work out the characters' intentions",
                        "type": "likert",
                        "reverse_scored": True
                    },
                    {
                        "id": "enjoys_social_activities",
                        "text": "I like to collect information about categories of things",
                        "type": "likert",
                        "reverse_scored": True
                    },
                    {
                        "id": "finds_easy_to_work_out_intentions",
                        "text": "I find it easy to work out what someone is thinking or feeling just by looking at their face",
                        "type": "likert",
                        "reverse_scored": False
                    },
                    {
                        "id": "good_at_social_chit_chat",
                        "text": "I find it difficult to work out people's intentions",
                        "type": "likert",
                        "reverse_scored": True
                    }
                ]
            }
        ]
    }

def show_questionnaire_page():
    st.header("📋 Behavioral Assessment Questionnaire")
    
    st.markdown("""
    This questionnaire combines elements from validated screening tools including the Modified Checklist 
    for Autism in Toddlers, Revised (M-CHAT-R) and the Autism Spectrum Quotient (AQ-10).
    
    Please answer all questions honestly based on typical behavior patterns.
    """)
    
    # Load questions
    questions_data = load_questions()
    
    # Initialize responses in session state
    if 'questionnaire_responses' not in st.session_state:
        st.session_state.questionnaire_responses = {}
    
    # Progress tracking
    total_questions = sum(len(section['questions']) for section in questions_data['sections'])
    answered_questions = len(st.session_state.questionnaire_responses)
    
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        st.progress(answered_questions / total_questions)
    with progress_col2:
        st.write(f"{answered_questions}/{total_questions} answered")
    
    # Display sections
    for section in questions_data['sections']:
        with st.expander(f"📖 {section['name']}", expanded=True):
            st.write(section['description'])
            
            for question in section['questions']:
                question_id = question['id']
                question_text = question['text']
                question_type = question['type']
                
                # Display question
                st.markdown(f"**{question_text}**")
                
                if question_type == "yes_no":
                    response = st.radio(
                        f"Response for: {question_text}",
                        options=["Yes", "No"],
                        key=question_id,
                        label_visibility="collapsed",
                        horizontal=True
                    )
                    
                    # Convert to numeric score
                    if response == "Yes":
                        score = 0 if question.get('reverse_scored', False) else 1
                    else:
                        score = 1 if question.get('reverse_scored', False) else 0
                    
                    st.session_state.questionnaire_responses[question_id] = score
                
                elif question_type == "likert":
                    response = st.select_slider(
                        f"Response for: {question_text}",
                        options=[
                            "Definitely agree",
                            "Slightly agree", 
                            "Slightly disagree",
                            "Definitely disagree"
                        ],
                        key=question_id,
                        label_visibility="collapsed"
                    )
                    
                    # Convert to numeric score (0-3 scale)
                    score_map = {
                        "Definitely agree": 3,
                        "Slightly agree": 2,
                        "Slightly disagree": 1,
                        "Definitely disagree": 0
                    }
                    
                    score = score_map[response]
                    if question.get('reverse_scored', False):
                        score = 3 - score  # Reverse the score
                    
                    # Normalize to 0-1 scale
                    st.session_state.questionnaire_responses[question_id] = score / 3.0
                
                st.divider()
    
    # Show summary and navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Overview"):
            st.session_state.current_step = 0
            st.rerun()
    
    with col2:
        if answered_questions == total_questions:
            st.success("✅ All questions completed!")
            
            # Show quick summary
            with st.expander("📊 Quick Summary", expanded=False):
                show_questionnaire_summary()
        else:
            st.warning(f"Please complete all {total_questions} questions to proceed.")
    
    with col3:
        if answered_questions == total_questions:
            if st.button("Next: Gaze Assessment ➡️", type="primary"):
                st.session_state.current_step = 2
                st.rerun()

def show_questionnaire_summary():
    """Display a quick summary of questionnaire responses"""
    if not st.session_state.questionnaire_responses:
        st.write("No responses recorded yet.")
        return
    
    from utils.data_processor import DataProcessor
    processor = DataProcessor()
    
    processed_data = processor.process_questionnaire_data(st.session_state.questionnaire_responses)
    
    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Social Communication",
            f"{processed_data.get('social_communication_score', 0):.2f}",
            help="Score for social communication domain"
        )
    
    with col2:
        st.metric(
            "Repetitive Behaviors", 
            f"{processed_data.get('repetitive_behaviors_score', 0):.2f}",
            help="Score for repetitive behaviors domain"
        )
    
    with col3:
        st.metric(
            "Social Cognition",
            f"{processed_data.get('social_cognition_score', 0):.2f}",
            help="Score for social cognition domain"
        )
    
    with col4:
        st.metric(
            "Overall Score",
            f"{processed_data.get('normalized_score', 0):.2f}",
            help="Normalized overall score"
        )
    
    # Store processed data for later use
    st.session_state.processed_questionnaire_data = processed_data
