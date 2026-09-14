import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Course Completion Early Warning System",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .result {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# =====================================================
# LOAD MODEL FILES
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

st.markdown(
    '<div class="title">🎓 Course Completion Early Warning System</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
    Machine learning system that predicts student course completion risk
    and supports early intervention.
    </div>
    """,
    unsafe_allow_html=True
)



# =====================================================
# STUDENT PROFILE
# =====================================================

st.header("👤 Student Profile")


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


col1, col2, col3 = st.columns(3)


with col1:

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


with col2:

    category = st.selectbox(
        "Category",
        [
            "Business",
            "Design",
            "Marketing",
            "Math",
            "Programming"
        ]
    )


with col3:

    course_level = st.selectbox(
        "Course Level",
        list(course_levels.keys())
    )




# =====================================================
# LEARNING BEHAVIOUR
# =====================================================

st.header("📈 Learning Behaviour")


col1, col2, col3 = st.columns(3)


with col1:

    login_frequency = st.slider(
        "Login Frequency",
        0,
        100,
        20
    )


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


with col2:

    session_duration = st.slider(
        "Average Session Duration (Minutes)",
        0,
        500,
        60
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


with col3:

    assignments = st.slider(
        "Assignments Submitted",
        0,
        20,
        5
    )


    discussion = st.slider(
        "Discussion Participation",
        0,
        100,
        20
    )


    days_since_login = st.slider(
        "Days Since Last Login",
        0,
        100,
        5
    )




# =====================================================
# PAYMENT INFORMATION
# =====================================================

st.header("💳 Payment Information")


col1, col2 = st.columns(2)


with col1:

    fee = st.selectbox(
        "Fee Paid",
        [
            "Yes",
            "No"
        ]
    )


with col2:

    discount = st.selectbox(
        "Discount Used",
        [
            "Yes",
            "No"
        ]
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


    # Create dataframe

    student = pd.DataFrame({

        "Age":[age],

        "Education_Level":[education],
        "Internet_Connection_Quality":[internet],
        "Course_Level":[course_level],

        "Fee_Paid":[fee],
        "Discount_Used":[discount],

        "Gender":[gender],
        "Employment_Status":[employment],
        "Device_Type":[device],

        "Course_Name":[course_name],
        "Category":[category],


        "Login_Frequency":[login_frequency],
        "Average_Session_Duration_Min":[session_duration],
        "Video_Completion_Rate":[video_completion],
        "Discussion_Participation":[discussion],
        "Time_Spent_Hours":[time_spent],
        "Days_Since_Last_Login":[days_since_login],
        "Assignments_Submitted":[assignments],
        "Quiz_Score_Avg":[quiz_score],
        "Progress_Percentage":[progress],

        "Payment_Amount":[payment_amount]

    })



    # -----------------------------
    # Encoding
    # -----------------------------


    student["Fee_Paid"] = student["Fee_Paid"].map(
        fees_paid
    )


    student["Discount_Used"] = student["Discount_Used"].map(
        discount_used
    )


    student["Education_Level"] = student["Education_Level"].map(
        education_levels
    )


    student["Internet_Connection_Quality"] = student[
        "Internet_Connection_Quality"
    ].map(
        internet_quality
    )


    student["Course_Level"] = student["Course_Level"].map(
        course_levels
    )


    student = pd.get_dummies(
        student,
        columns=[
            "Gender",
            "Employment_Status",
            "Device_Type",
            "Course_Name",
            "Category"
        ],
        dtype=int
    )



    # Match training data

    student = student.reindex(
        columns=training_columns,
        fill_value=0
    )



    # Scale

    student_scaled = scaler.transform(
        student
    )



    # Prediction

    prediction = model.predict(
        student_scaled
    )


    probability = model.predict_proba(
        student_scaled
    )[0]




    # =====================================================
    # RESULTS
    # =====================================================


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


        st.progress(
            float(probability[1])
        )


        st.info(
            """
            Recommended action:

            • Continue monitoring progress
            • Maintain student engagement
            • Encourage consistent learning activity
            """
        )


    else:


        st.warning(
            "⚠️ Student is at risk of non-completion"
        )


        st.metric(
            "Risk Probability",
            f"{probability[0]*100:.1f}%"
        )


        st.progress(
            float(probability[0])
        )


        st.error(
            """
            Recommended interventions:

            • Contact student for support
            • Encourage course activity
            • Monitor progress closely
            • Encourage assignment completion
            """
        )