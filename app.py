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
# HOW TO USE
# =====================================================

st.header("How to Use the System")


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

if st.button("🔍 Predict Course Completion"):

    # 1. Create input dataframe
    student = pd.DataFrame({
        "Age": [age],
        "Education_Level": [education],
        "Internet_Connection_Quality": [internet],
        "Course_Level": [course_level],
        "Fee_Paid": [fee],
        "Discount_Used": [discount],
        "Gender": [gender],
        "Employment_Status": [employment],
        "Device_Type": [device],
        "Course_Name": [course_name],
        "Category": [category],
        "Video_Completion_Rate": [video_completion],
        "Progress_Percentage": [progress],
        "Time_Spent_Hours": [time_spent],
        "Quiz_Score_Avg": [quiz_score],
        "Days_Since_Last_Login": [days_since_login],
        "Assignments_Submitted": [assignments],
        "Payment_Amount": [payment_amount]
    })


    # 2. Apply encodings

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


    # 3. One-hot encoding

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


    # 4. Match training columns

    student = student.reindex(
        columns=training_columns,
        fill_value=0
    )


    # 5. Scale

    student_scaled = scaler.transform(
        student
    ) 

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