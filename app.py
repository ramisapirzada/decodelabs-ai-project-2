import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL + DATA
# ============================================================

model = joblib.load("model.pkl")
data = pd.read_csv("student_data.csv")

FEATURES = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed"
]

# ============================================================
# HEADER
# ============================================================

st.title("🎓 Student Performance AI")
st.markdown(
    "#### Intelligent Student Outcome Classification System"
)

st.write(
    "A machine learning application that predicts whether a student "
    "is likely to **PASS** or **FAIL** based on academic performance "
    "and attendance indicators."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🎯 Project Information")

    st.write("**Model:** Decision Tree Classifier")
    st.write("**Learning:** Supervised Learning")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.subheader("📊 Dataset")

    st.metric("Total Students", len(data))
    st.metric("Features", len(FEATURES))

    st.divider()

    st.caption("DecodeLabs Artificial Intelligence Internship")
    st.caption("Project 2 — Data Classification Using AI")

# ============================================================
# TABS
# ============================================================

prediction_tab, dataset_tab, model_tab = st.tabs(
    [
        "🔮 Prediction",
        "📊 Dataset Explorer",
        "🤖 Model Information"
    ]
)

# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    st.subheader("📝 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=60,
            value=24,
            step=1
        )

    with col2:
        study_hours = st.number_input(
            "Study Hours / Day",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

    with col3:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=1.0
        )

    col4, col5 = st.columns(2)

    with col4:
        previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with col5:
        assignments = st.number_input(
            "Assignments Completed",
            min_value=0,
            max_value=20,
            value=8,
            step=1
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Performance",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame({
            "Age": [age],
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Score": [previous_score],
            "Assignments_Completed": [assignments]
        })

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        fail_probability = probabilities[0]
        pass_probability = probabilities[1]

        st.divider()
        st.subheader("📌 Prediction Result")

        if prediction == 1:
            st.success(
                "### 🎉 PASS\n\n"
                "The model predicts that this student is likely to pass."
            )
        else:
            st.error(
                "### ⚠️ FAIL\n\n"
                "The model predicts that this student is at risk of failing."
            )

        st.write("### Probability Distribution")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "PASS Probability",
                f"{pass_probability:.1%}"
            )

        with metric2:
            st.metric(
                "FAIL Probability",
                f"{fail_probability:.1%}"
            )

        chart_data = pd.DataFrame(
            {
                "Outcome": ["PASS", "FAIL"],
                "Probability": [
                    pass_probability,
                    fail_probability
                ]
            }
        )

        st.bar_chart(
            chart_data.set_index("Outcome")
        )

        st.divider()

        st.subheader("👤 Student Profile")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Age", age)
        c2.metric("Study Hours", f"{study_hours:g}")
        c3.metric("Attendance", f"{attendance:g}%")
        c4.metric("Previous Score", f"{previous_score:g}")
        c5.metric("Assignments", assignments)

        st.info(
            "This prediction is generated by the trained Decision Tree "
            "classification model and should be treated as a model estimate, "
            "not a definitive academic outcome."
        )

# ============================================================
# DATASET TAB
# ============================================================

with dataset_tab:

    st.subheader("📊 Dataset Explorer")

    st.write(
        "The model uses student academic and attendance information "
        "to perform binary classification."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", data.shape[0])
    c2.metric("Input Features", len(FEATURES))
    c3.metric("Target Classes", data["Pass"].nunique())

    st.write("### Dataset Preview")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Dataset Statistics")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )

    st.write("### Class Distribution")

    class_counts = (
        data["Pass"]
        .map({0: "FAIL", 1: "PASS"})
        .value_counts()
    )

    st.bar_chart(class_counts)

# ============================================================
# MODEL INFORMATION TAB
# ============================================================

with model_tab:

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "The application uses a **Decision Tree Classifier**, "
        "a supervised machine learning algorithm used for classification."
    )

    st.write("### Input Features")

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Description": [
                "Student age",
                "Average daily study hours",
                "Attendance percentage",
                "Previous academic score",
                "Number of completed assignments"
            ]
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Feature Importance")

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.write("### Classification Output")

    st.write(
        "**0 → FAIL**  \n"
        "**1 → PASS**"
    )

    st.divider()

    st.info(
        "The model was trained using a train/test split and evaluated "
        "before being used for predictions."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)

st.caption(
    "DecodeLabs Artificial Intelligence Internship — Project 2"
)