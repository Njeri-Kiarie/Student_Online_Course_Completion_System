# Machine_Learning_Capstone_Project

# Predicting Student Course Completion in Online Learning Platforms.

## Project Overview

Online learning platforms have transformed access to education by enabling learners to acquire new skills anytime and anywhere. Despite this growth, one of the biggest challenges faced by EdTech platforms is low course completion rates. Many learners enroll in courses but fail to complete them due to factors such as low engagement, poor study habits, or competing commitments.

This project aims to develop a machine learning classification model that predicts whether a student is likely to complete an online course based on their learning behavior, engagement, demographic characteristics, and course-related features. By identifying learners at risk of dropping out early, educational institutions and online learning platforms can implement targeted interventions to improve student success and course completion rates.

---

## Problem Statement

Online learning providers often struggle to identify students who are at risk of not completing their courses before they disengage. Traditional monitoring methods rely on manual tracking or identifying students only after they have already fallen behind, limiting opportunities for timely intervention.

The objective of this project is to build a predictive machine learning model that can classify whether a student will **complete** or **not complete** a course using historical student engagement and performance data. The insights generated from this model can support proactive decision-making, improve learner retention, and enhance the overall effectiveness of online education.

---

## Dataset

The dataset contains **100,000 student records** with **40 features** describing student demographics, engagement, learning behavior, assessments, and course characteristics.

## Research Questions

This project is guided by the following research questions:

1. **Can student engagement and learning behavior be used to accurately predict whether a student will complete an online course?**

2. **Which factors are the strongest predictors of course completion?**
   - For example, are features such as login frequency, video completion rate, progress percentage, or assignment submission more influential than demographic characteristics?

3. **How does student engagement influence the likelihood of course completion?**
   - Does increased participation in learning activities lead to higher completion rates?

4. **How do assessment performance and learning progress impact course completion?**
   - Are students with higher quiz scores, project grades, and assignment completion rates more likely to finish their courses?

5. **Which machine learning classification model provides the best predictive performance for this problem?**
   - The project will compare multiple models to determine the most effective approach for predicting course completion.

6. **How can the insights from the predictive model support EdTech platforms in improving student retention and course completion rates?**
   - What actionable recommendations can be derived to help educators and platform administrators intervene early and support at-risk learners?

## Key Features

### Student Information
- Age
- Gender
- Education Level
- Employment Status

### Learning Behaviour
- Login Frequency
- Average Session Duration
- Total Time Spent
- Video Completion Rate
- Progress Percentage
- Days Since Last Login
- Rewatch Count

### Assessment Performance
- Quiz Attempts
- Average Quiz Score
- Assignments Submitted
- Assignments Missed
- Project Grade

### Student Engagement
- Discussion Participation
- Peer Interaction
- Reminder Emails Clicked
- Satisfaction Rating

### Course Information
- Course Category
- Course Level
- Course Duration
- Instructor Rating

### Target Variable
- **Completed**
  - Completed - `1`
  - Not Completed - `0`

---

## Machine Learning Workflow

The project will follow a structured machine learning pipeline:

### 1. Data Understanding
- Load and inspect the dataset
- Explore data types and feature distributions
- Identify missing values and duplicates

### 2. Exploratory Data Analysis (EDA)
- Analyze the distribution of the target variable
- Explore relationships between features
- Identify trends, patterns, and potential predictors
- Detect outliers and data inconsistencies

### 3. Data Preprocessing
- Handle missing values
- Encode categorical variables
- Scale numerical features where necessary
- Split the dataset into training and testing sets

### 4. Model Development
Train and compare multiple classification models, including:
- Logistic Regression
- Decision Tree
- Random Forest

### 5. Model Evaluation
Evaluate model performance using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 6. Model Interpretation
- Analyze feature importance
- Identify the key factors influencing course completion
- Generate actionable insights for educators and platform administrators

---

## Tools and Technologies

- **Python** - Interactive development environment
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning and model evaluation

---

## Expected Outcome and Impact

The final outcome of this project will be a machine learning model capable of accurately predicting whether a student is likely to complete an online course.

The project aims to support EdTech platforms and educational institutions by enabling:

- Early identification of students at risk of dropping out
- Personalized learning interventions and support
- Improved student engagement and retention
- Higher course completion rates
- Better allocation of academic support resources
- Data-driven decision-making for instructors and administrators

Ultimately, this project demonstrates how machine learning can be applied to improve learner success and enhance the effectiveness of online education through predictive analytics.
