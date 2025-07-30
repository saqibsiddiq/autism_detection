import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def show_education_page():
    st.header("📚 Educational Resources")
    
    st.markdown("""
    This section provides evidence-based information about Autism Spectrum Disorders (ASD), 
    early intervention, and support resources for individuals and families.
    """)
    
    # Create tabs for different educational topics
    tabs = st.tabs([
        "🧠 About ASD", 
        "🔍 Early Signs", 
        "🏥 Getting Help", 
        "🎯 Interventions", 
        "👨‍👩‍👧‍👦 Family Support",
        "📖 Resources"
    ])
    
    with tabs[0]:
        show_about_asd()
    
    with tabs[1]:
        show_early_signs()
    
    with tabs[2]:
        show_getting_help()
    
    with tabs[3]:
        show_interventions()
    
    with tabs[4]:
        show_family_support()
    
    with tabs[5]:
        show_resources()

def show_about_asd():
    st.subheader("Understanding Autism Spectrum Disorders")
    
    st.markdown("""
    ### What is Autism Spectrum Disorder (ASD)?
    
    Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by:
    
    - **Social communication and interaction challenges**
    - **Restricted and repetitive patterns of behavior, interests, or activities**
    - **Symptoms present in early developmental period**
    - **Symptoms that cause significant impairment in daily functioning**
    """)
    
    # ASD Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Key Statistics:**
        - Affects approximately 1 in 36 children in the US
        - More common in boys than girls (4:1 ratio)
        - Can be diagnosed as early as 18 months
        - Lifelong condition with varying support needs
        """)
    
    with col2:
        st.success("""
        **Important Facts:**
        - ASD occurs across all racial, ethnic, and socioeconomic groups
        - Early intervention significantly improves outcomes
        - Many individuals with ASD live independent, fulfilling lives
        - Autism is not caused by vaccines or parenting styles
        """)
    
    # Spectrum visualization
    st.subheader("The Autism Spectrum")
    
    st.markdown("""
    ASD is called a "spectrum" because it affects individuals differently and to varying degrees:
    """)
    
    # Create spectrum visualization
    spectrum_data = {
        'Support Level': ['Level 1', 'Level 2', 'Level 3'],
        'Description': [
            'Requiring Support',
            'Requiring Substantial Support', 
            'Requiring Very Substantial Support'
        ],
        'Characteristics': [
            'May struggle with social situations, organization, and transitions',
            'Significant challenges with verbal/nonverbal communication',
            'Severe challenges with communication and daily functioning'
        ]
    }
    
    spectrum_df = pd.DataFrame(spectrum_data)
    
    for i, row in spectrum_df.iterrows():
        with st.expander(f"**{row['Support Level']}: {row['Description']}**"):
            st.write(row['Characteristics'])
    
    st.markdown("""
    ### Core Features of ASD
    
    #### 1. Social Communication and Interaction
    - Difficulty with social-emotional reciprocity
    - Challenges with nonverbal communication
    - Problems developing and maintaining relationships
    
    #### 2. Restricted and Repetitive Behaviors
    - Repetitive motor movements or speech
    - Strong need for routines and rituals
    - Highly focused special interests
    - Sensory sensitivities or seeking behaviors
    """)

def show_early_signs():
    st.subheader("Early Signs and Red Flags")
    
    st.warning("""
    **Important:** These are potential warning signs, not diagnostic criteria. 
    Only qualified professionals can diagnose ASD. If you have concerns, 
    consult with your healthcare provider.
    """)
    
    # Age-based signs
    age_groups = {
        "6-12 Months": [
            "Limited eye contact",
            "Doesn't smile or show facial expressions",
            "Doesn't respond to their name",
            "Limited gesturing (pointing, waving)"
        ],
        "12-18 Months": [
            "No single words by 16 months",
            "Doesn't point to show interest",
            "Unusual attachment to objects",
            "Loss of previously acquired skills"
        ],
        "18-24 Months": [
            "No two-word phrases by 24 months",
            "Limited pretend play",
            "Repetitive behaviors increase",
            "Difficulty with changes in routine"
        ],
        "2-3 Years": [
            "Limited social interaction with peers",
            "Intense focus on specific topics",
            "Sensory sensitivities become apparent",
            "Communication remains limited or regresses"
        ]
    }
    
    for age, signs in age_groups.items():
        with st.expander(f"🕐 **{age}**"):
            st.markdown("**Potential signs to watch for:**")
            for sign in signs:
                st.write(f"• {sign}")
    
    # M-CHAT-R screening
    st.subheader("M-CHAT-R Screening Tool")
    
    st.info("""
    The Modified Checklist for Autism in Toddlers, Revised (M-CHAT-R) is a validated 
    screening tool for toddlers between 16-30 months. Key screening questions include:
    """)
    
    mchat_questions = [
        "Does your child enjoy being swung or bounced on your knee?",
        "Does your child take an interest in other children?",
        "Does your child enjoy playing peek-a-boo/hide-and-seek?",
        "Does your child ever pretend (e.g., talk on phone, care for dolls)?",
        "Does your child ever point to ask for something?",
        "Does your child look you in the eye for more than a second or two?",
        "Does your child try to attract your attention to their activity?"
    ]
    
    for i, question in enumerate(mchat_questions, 1):
        st.write(f"{i}. {question}")
    
    st.markdown("""
    **Note:** This assessment tool includes questions based on M-CHAT-R principles.
    """)

def show_getting_help():
    st.subheader("Getting Professional Help")
    
    st.markdown("""
    ### When to Seek Evaluation
    
    Consider seeking professional evaluation if:
    - You notice multiple early warning signs
    - Your child loses previously acquired skills
    - You have concerns about social communication
    - Your child shows repetitive behaviors that interfere with daily life
    - Daycare or school providers express concerns
    """)
    
    # Professional types
    st.subheader("Types of Professionals")
    
    professionals = {
        "Developmental Pediatrician": {
            "Role": "Medical doctor specializing in child development",
            "Services": "Comprehensive developmental evaluations, medical management",
            "When to See": "For initial evaluation and ongoing medical care"
        },
        "Child Psychologist": {
            "Role": "Mental health professional specializing in children",
            "Services": "Psychological testing, behavioral assessments, therapy",
            "When to See": "For psychological evaluation and behavioral support"
        },
        "Speech-Language Pathologist": {
            "Role": "Communication disorders specialist",
            "Services": "Communication assessment and therapy",
            "When to See": "For speech and language concerns"
        },
        "Occupational Therapist": {
            "Role": "Specialist in daily living skills and sensory processing",
            "Services": "Sensory integration, fine motor skills, daily living skills",
            "When to See": "For sensory sensitivities and motor skill challenges"
        }
    }
    
    for prof, info in professionals.items():
        with st.expander(f"👩‍⚕️ **{prof}**"):
            st.write(f"**Role:** {info['Role']}")
            st.write(f"**Services:** {info['Services']}")
            st.write(f"**When to See:** {info['When to See']}")
    
    # Evaluation process
    st.subheader("The Evaluation Process")
    
    eval_steps = [
        "**Initial Screening:** Brief questionnaires and developmental checklists",
        "**Comprehensive Evaluation:** Detailed assessment of communication, behavior, and development",
        "**Medical Evaluation:** Rule out other medical conditions",
        "**Multidisciplinary Team:** Input from various specialists",
        "**Results and Recommendations:** Diagnosis (if applicable) and treatment plan"
    ]
    
    for i, step in enumerate(eval_steps, 1):
        st.write(f"{i}. {step}")
    
    st.info("""
    **Remember:** Early identification and intervention can significantly improve outcomes. 
    Don't wait - if you have concerns, seek professional guidance.
    """)

def show_interventions():
    st.subheader("Evidence-Based Interventions")
    
    st.markdown("""
    ### Early Intervention (Birth to 3 years)
    
    Early intervention services are crucial for optimal outcomes:
    """)
    
    ei_services = {
        "Applied Behavior Analysis (ABA)": {
            "Description": "Systematic approach to understanding and changing behavior",
            "Benefits": "Improves communication, social skills, and reduces challenging behaviors",
            "Evidence": "Most researched intervention with strong evidence base"
        },
        "Speech-Language Therapy": {
            "Description": "Targets communication skills development",
            "Benefits": "Improves verbal and nonverbal communication",
            "Evidence": "Essential component of comprehensive intervention"
        },
        "Occupational Therapy": {
            "Description": "Focuses on daily living skills and sensory processing",
            "Benefits": "Improves fine motor skills and sensory regulation",
            "Evidence": "Effective for addressing sensory sensitivities"
        },
        "Developmental/Relationship-Based Approaches": {
            "Description": "Focus on building relationships and emotional connections",
            "Benefits": "Improves social engagement and emotional regulation",
            "Evidence": "Promising approach, especially for young children"
        }
    }
    
    for intervention, details in ei_services.items():
        with st.expander(f"🎯 **{intervention}**"):
            st.write(f"**Description:** {details['Description']}")
            st.write(f"**Benefits:** {details['Benefits']}")
            st.write(f"**Evidence:** {details['Evidence']}")
    
    # School-age interventions
    st.subheader("School-Age Interventions (3+ years)")
    
    school_services = [
        "**Special Education Services:** Individualized Education Programs (IEPs)",
        "**Inclusion Programs:** Participation in general education with supports",
        "**Social Skills Training:** Structured programs to develop peer relationships",
        "**Assistive Technology:** Communication devices and learning supports",
        "**Transition Planning:** Preparation for adult life and independence"
    ]
    
    for service in school_services:
        st.write(f"• {service}")
    
    # Treatment principles
    st.subheader("Key Treatment Principles")
    
    principles = [
        "**Individualized:** Tailored to each person's unique needs and strengths",
        "**Evidence-Based:** Using interventions with scientific support",
        "**Intensive:** Sufficient hours and frequency for meaningful progress",
        "**Family-Centered:** Involving families as partners in treatment",
        "**Comprehensive:** Addressing all areas of need",
        "**Lifelong:** Ongoing support and services as needed"
    ]
    
    for principle in principles:
        st.write(f"• {principle}")

def show_family_support():
    st.subheader("Supporting Families")
    
    st.markdown("""
    ### For Parents and Caregivers
    
    Receiving an autism diagnosis or having concerns about your child can be overwhelming. 
    Remember that you are not alone, and there are many resources and strategies to help.
    """)
    
    # Coping strategies
    st.subheader("Coping Strategies")
    
    coping_strategies = [
        "**Educate Yourself:** Learn about autism from reputable sources",
        "**Connect with Others:** Join support groups and connect with other families",
        "**Advocate for Your Child:** Learn about rights and available services",
        "**Take Care of Yourself:** Maintain your own physical and mental health",
        "**Celebrate Strengths:** Focus on your child's unique abilities and progress",
        "**Be Patient:** Progress may be slow but is often meaningful"
    ]
    
    for strategy in coping_strategies:
        st.write(f"• {strategy}")
    
    # Sibling support
    st.subheader("Supporting Siblings")
    
    st.info("""
    **Siblings of children with autism may need special support:**
    
    • Age-appropriate explanations about autism
    • Individual attention and quality time
    • Opportunities to express feelings and concerns
    • Connection with other siblings of children with disabilities
    • Recognition of their own needs and interests
    """)
    
    # Family strategies
    st.subheader("Daily Life Strategies")
    
    daily_strategies = {
        "Structure and Routine": [
            "Create predictable daily schedules",
            "Use visual schedules and calendars",
            "Prepare for changes in advance",
            "Establish consistent bedtime routines"
        ],
        "Communication": [
            "Use clear, simple language",
            "Give time to process information",
            "Use visual supports when helpful",
            "Practice patience with communication attempts"
        ],
        "Sensory Considerations": [
            "Identify sensory preferences and sensitivities",
            "Create calm, sensory-friendly spaces",
            "Gradually introduce new sensory experiences",
            "Use sensory breaks when needed"
        ],
        "Behavior Support": [
            "Identify triggers for challenging behaviors",
            "Use positive reinforcement strategies",
            "Teach alternative communication methods",
            "Seek professional help for persistent challenges"
        ]
    }
    
    for category, strategies in daily_strategies.items():
        with st.expander(f"🏠 **{category}**"):
            for strategy in strategies:
                st.write(f"• {strategy}")

def show_resources():
    st.subheader("Additional Resources")
    
    # National organizations
    st.subheader("🏛️ National Organizations")
    
    organizations = {
        "Autism Society of America": {
            "Website": "autism-society.org",
            "Services": "Information, advocacy, local chapter referrals"
        },
        "Autism Speaks": {
            "Website": "autismspeaks.org", 
            "Services": "Research funding, awareness, resource database"
        },
        "Association for Behavior Analysis International": {
            "Website": "abainternational.org",
            "Services": "Professional standards, provider directories"
        },
        "National Autistic Society (UK)": {
            "Website": "autism.org.uk",
            "Services": "Information, services, advocacy"
        }
    }
    
    for org, info in organizations.items():
        st.write(f"**{org}**")
        st.write(f"Website: {info['Website']}")
        st.write(f"Services: {info['Services']}")
        st.write("")
    
    # Government resources
    st.subheader("🏛️ Government Resources")
    
    gov_resources = [
        "**CDC Autism Information:** cdc.gov/autism",
        "**NIH/NIMH Autism Research:** nimh.nih.gov/autism",
        "**Early Intervention Program Directory:** cdc.gov/ncbddd/childdevelopment/early-intervention.html",
        "**Individuals with Disabilities Education Act (IDEA):** sites.ed.gov/idea"
    ]
    
    for resource in gov_resources:
        st.write(f"• {resource}")
    
    # Books and publications
    st.subheader("📚 Recommended Reading")
    
    books = [
        "**'More Than Words' by Fern Sussman** - Communication strategies for parents",
        "**'The Reason I Jump' by Naoki Higashida** - Perspective from someone with autism",
        "**'Uniquely Human' by Barry Prizant** - Strengths-based approach to autism",
        "**'Ten Things Every Child with Autism Wishes You Knew' by Ellen Notbohm** - Practical insights"
    ]
    
    for book in books:
        st.write(f"• {book}")
    
    # Apps and tools
    st.subheader("📱 Helpful Apps and Tools")
    
    apps = [
        "**Visual Schedule Apps:** First-Then Visual Schedule, Choiceworks",
        "**Communication Apps:** Proloquo2Go, TouchChat, LAMP Words for Life",
        "**Social Stories Apps:** Social Stories Creator & Library, Stories2Learn",
        "**Sensory Tools:** Autism iHelp, Sensory Apps"
    ]
    
    for app in apps:
        st.write(f"• {app}")
    
    # Crisis resources
    st.subheader("🆘 Crisis and Support Resources")
    
    st.error("""
    **If you or someone you know is in crisis:**
    
    • National Suicide Prevention Lifeline: 988
    • Crisis Text Line: Text HOME to 741741
    • National Domestic Violence Hotline: 1-800-799-7233
    • Autism Crisis Helpline (varies by location - contact local autism organizations)
    """)
    
    # Disclaimer
    st.subheader("Important Disclaimer")
    
    st.warning("""
    **Please Note:** This information is for educational purposes only and is not intended 
    to replace professional medical advice, diagnosis, or treatment. Always seek the advice 
    of qualified healthcare providers with questions about autism or any medical condition.
    
    The resources listed here are provided for informational purposes. Inclusion does not 
    constitute endorsement, and families should evaluate services and providers carefully.
    """)
    
    # Navigation
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("⬅️ Back to Results"):
            st.session_state.current_step = 3
            st.rerun()
    
    with col2:
        if st.button("🏠 Return to Overview"):
            st.session_state.current_step = 0
            st.rerun()
