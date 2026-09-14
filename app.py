import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ElimuPulse",
    page_icon="🎓",
    layout="wide"
)



# =====================================================
# LOAD SAVED FILES
# =====================================================

model = joblib.load(
    "course_completion_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)

training_columns = joblib.load(
    "training_columns.pkl"
)

fees_paid = joblib.load(
    "fees_paid_mapping.pkl"
)

discount_used = joblib.load(
    "discount_used_mapping.pkl"
)

education_levels = joblib.load(
    "education_mapping.pkl"
)

internet_quality = joblib.load(
    "internet_mapping.pkl"
)

course_levels = joblib.load(
    "course_level_mapping.pkl"
)



# =====================================================
# HEADER
# =====================================================

st.title("🎓 ElimuPulse")

st.subheader(
    "A Machine Learning Early Warning System for Student Course Completion"
)


st.write(
    """
    ElimuPulse predicts whether a student is likely to complete an online
    course and helps identify learners who may require early intervention.
    """
)



# =====================================================
# ABOUT THE SYSTEM
# =====================================================

st.header("📌 About the System")


st.write(
    """
    Online learning platforms collect large amounts of student activity data,
    but identifying students at risk of non-completion early remains a challenge.

    ElimuPulse uses machine learning to analyse:

    • Student characteristics
    • Learning behaviour
    • Course engagement
    • Assessment performance
    • Payment information

    The goal is to support educators and institutions in providing timely
    support to students who may be at risk.
    """
)



# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("📊 Dataset Information")


st.write(
    """
    The machine learning model was trained using:

    **100,000 student records** obtained from Kaggle.

    The dataset contains information from several categories:
    """
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        ### 👤 Student Information

        - Age
        - Gender
        - Education Level
        - Employment Status
        """
    )


with col2:

    st.markdown(
        """
        ### 📈 Learning Behaviour

        - Login Frequency
        - Video Completion Rate
        - Progress Percentage
        - Time Spent Hours
        - Days Since Last Login
        """
    )


with col3:

    st.markdown(
        """
        ### 📚 Course & Performance

        - Course Category
        - Course Level
        - Quiz Scores
        - Assignments
        - Payment Information
        """
    )



# =====================================================
# HOW TO USE
# =====================================================

st.header("🚀 How to Use the System")


st.markdown(
    """
    Follow these steps:

    **Step 1:** Enter student information.

    **Step 2:** Adjust learning behaviour indicators using the sliders.

    **Step 3:** Click **Predict Course Completion**.

    **Step 4:** Review:
    
    - Completion prediction
    - Probability score
    - Recommended intervention actions

    """
)



# =====================================================
# MODEL INFORMATION
# =====================================================

st.header("🤖 Model Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Selected Model",
        "Logistic Regression"
    )


with col2:

    st.metric(
        "Accuracy",
        "≈60%"
    )


with col3:

    st.metric(
        "Explainability",
        "SHAP"
    )


st.info(
    """
    The final model was selected after comparing:

    • Logistic Regression
    • Decision Tree
    • Random Forest
    • XGBoost
    • KNN

    Logistic Regression was selected because it provided the strongest overall
    performance while remaining interpretable.

    SHAP analysis was used to identify the factors influencing predictions.
    """
)



# =====================================================
# INPUT SECTION
# =====================================================

st.divider()

st.header("👤 Student Information")


col1, col2, col3 = st.columns(3)


with col1:

    age = st.slider(
        "Age",
        15,
        70,
        22
    )


    education = st.selectbox(
        "Education Level",
        list(education_levels.keys())
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )


    employment = st.selectbox(
        "Employment Status",
        [
            "Employed",
            "Self-Employed",
            "Student",
            "Unemployed"
        ]
    )


with col3:

    device = st.selectbox(
        "Device Type",
        [
            "Laptop",
            "Mobile",
            "Tablet"
        ]
    )


    internet = st.selectbox(
        "Internet Connection Quality",
        list(internet_quality.keys())
    )



# =====================================================
# COURSE INFORMATION
# =====================================================

st.header("📚 Course Information")


course_name = st.selectbox(
    "Course Name",
    [
        "Data Analysis with Python",
        "Digital Marketing Essentials",
        "Excel for Business",
        "Introduction to AI",
        "Machine Learning A-Z",
        "Python Basics",
        "Statistics for Data Science",
        "UI/UX Design Fundamentals"
    ]
)



category = st.selectbox(
    "Course Category",
    [
        "Business",
        "Design",
        "Marketing",
        "Math",
        "Programming"
    ]
)



course_level = st.selectbox(
    "Course Level",
    list(course_levels.keys())
)



# =====================================================
# LEARNING BEHAVIOUR
# =====================================================

st.header("📈 Learning Behaviour")


video_completion = st.slider(
    "Video Completion Rate (%)",
    0,
    100,
    50
)


progress = st.slider(
    "Progress Percentage (%)",
    0,
    100,
    50
)


time_spent = st.slider(
    "Time Spent Hours",
    0,
    200,
    30
)


quiz_score = st.slider(
    "Quiz Score Average",
    0,
    100,
    60
)


days_since_login = st.slider(
    "Days Since Last Login",
    0,
    100,
    5
)


assignments = st.slider(
    "Assignments Submitted",
    0,
    20,
    5
)



# =====================================================
# PAYMENT
# =====================================================

st.header("💳 Payment Information")


fee = st.selectbox(
    "Fee Paid",
    ["Yes","No"]
)


discount = st.selectbox(
    "Discount Used",
    ["Yes","No"]
)


payment_amount = st.number_input(
    "Payment Amount",
    min_value=0,
    value=1000
)



# =====================================================
# PREDICTION
# =====================================================

if st.button(
    "🔍 Predict Course Completion",
    use_container_width=True
):

    # Your existing preprocessing pipeline goes here:
    # encoding
    # one-hot
    # scaling
    # prediction


    prediction = model.predict(
        student_scaled
    )


    probability = model.predict_proba(
        student_scaled
    )[0]



    st.divider()

    st.header("📊 Prediction Result")


    if prediction[0] == 1:

        st.success(
            "✅ Student is likely to complete the course"
        )


        st.metric(
            "Completion Probability",
            f"{probability[1]*100:.1f}%"
        )


        st.info(
            """
            Recommended actions:

            - Continue monitoring engagement
            - Encourage consistent learning
            - Provide positive feedback
            """
        )


    else:

        st.warning(
            "⚠️ Student may be at risk of non-completion"
        )


        st.metric(
            "Risk Probability",
            f"{probability[0]*100:.1f}%"
        )


        st.error(
            """
            Recommended interventions:

            - Contact student
            - Encourage course participation
            - Provide additional support
            - Monitor progress
            """
        )