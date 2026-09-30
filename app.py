import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f6f8fb;
    }

    .hero {
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a5f 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("student_data.csv")


@st.cache_data
def load_metrics():
    try:
        return joblib.load("metrics.pkl")
    except FileNotFoundError:
        return None


try:
    model = load_model()
    data = load_dataset()
    metrics = load_metrics()

except Exception as error:

    st.error(
        "The application could not load the trained model or dataset."
    )

    st.code(str(error))

    st.stop()

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    column
    for column in data.columns
    if column != "Pass"
]

categorical_features = data[FEATURES].select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = data[FEATURES].select_dtypes(
    exclude=["object", "string"]
).columns.tolist()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Performance AI")

    st.caption(
        "Machine Learning Classification System"
    )

    st.divider()

    st.markdown("### Model")

    st.info(
        "**Decision Tree Classifier**\n\n"
        "Supervised machine-learning model trained "
        "on real student performance data."
    )

    st.markdown("### Dataset")

    st.metric(
        "Students",
        len(data)
    )

    st.metric(
        "Features",
        len(FEATURES)
    )

    st.divider()

    st.markdown("### Technology")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🤖 Scikit-learn")
    st.write("⚡ Streamlit")
    st.write("💾 Joblib")

    st.divider()

    st.caption(
        "DecodeLabs • Artificial Intelligence"
    )

    st.caption(
        "Project 2 — Data Classification"
    )

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Performance AI</h1>
        <p>
            Predict student academic outcomes using supervised
            machine learning and real-world student data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Students",
        len(data)
    )

with col2:
    st.metric(
        "Features",
        len(FEATURES)
    )

with col3:

    if metrics:
        st.metric(
            "Test Accuracy",
            f"{metrics['accuracy'] * 100:.1f}%"
        )
    else:
        st.metric(
            "Model",
            "Decision Tree"
        )

with col4:

    pass_rate = data["Pass"].mean()

    st.metric(
        "Pass Rate",
        f"{pass_rate * 100:.1f}%"
    )

st.divider()

# ============================================================
# TABS
# ============================================================

prediction_tab, analytics_tab, dataset_tab, model_tab = st.tabs(
    [
        "🔮 Prediction",
        "📈 Analytics",
        "📊 Dataset",
        "🤖 Model"
    ]
)

# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    st.markdown(
        '<div class="section-title">Student Outcome Prediction</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Provide student information below. The trained model "
        "will estimate the probability of passing."
    )

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_values = {}

    st.markdown("### 👤 Personal Information")

    personal_features = [
        feature
        for feature in FEATURES
        if feature in [
            "school",
            "sex",
            "age",
            "address",
            "famsize",
            "Pstatus"
        ]
    ]

    if personal_features:

        columns = st.columns(3)

        for index, feature in enumerate(
            personal_features
        ):

            with columns[index % 3]:

                if feature in categorical_features:

                    options = sorted(
                        data[feature]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    input_values[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        options
                    )

                else:

                    min_value = float(
                        data[feature].min()
                    )

                    max_value = float(
                        data[feature].max()
                    )

                    default_value = float(
                        data[feature].median()
                    )

                    input_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value
                    )

    st.markdown("### 📚 Academic & Study Information")

    academic_features = [
        feature
        for feature in FEATURES
        if feature not in personal_features
    ]

    # Split remaining features into two columns
    left_features = academic_features[:len(academic_features)//2]
    right_features = academic_features[len(academic_features)//2:]

    left_column, right_column = st.columns(2)

    for column, feature_group in [
        (left_column, left_features),
        (right_column, right_features)
    ]:

        with column:

            for feature in feature_group:

                label = feature.replace(
                    "_",
                    " "
                ).title()

                if feature in categorical_features:

                    options = sorted(
                        data[feature]
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    input_values[feature] = st.selectbox(
                        label,
                        options,
                        key=f"input_{feature}"
                    )

                else:

                    min_value = float(
                        data[feature].min()
                    )

                    max_value = float(
                        data[feature].max()
                    )

                    default_value = float(
                        data[feature].median()
                    )

                    # Integer-like fields get integer inputs
                    if (
                        pd.api.types.is_integer_dtype(
                            data[feature]
                        )
                    ):

                        input_values[feature] = st.number_input(
                            label,
                            min_value=int(min_value),
                            max_value=int(max_value),
                            value=int(default_value),
                            step=1,
                            key=f"input_{feature}"
                        )

                    else:

                        input_values[feature] = st.number_input(
                            label,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_value,
                            step=0.1,
                            key=f"input_{feature}"
                        )

    st.write("")

    predict_button = st.button(
        "🚀 Predict Student Outcome",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_df = pd.DataFrame(
            [input_values]
        )

        # Ensure exact training column order
        input_df = input_df[FEATURES]

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = list(
            model.classes_
        )

        fail_probability = (
            probabilities[classes.index(0)]
            if 0 in classes
            else 0
        )

        pass_probability = (
            probabilities[classes.index(1)]
            if 1 in classes
            else 0
        )

        st.divider()

        st.markdown("### Prediction Result")

        if prediction == 1:

            st.success(
                "## ✅ Predicted Outcome: PASS"
            )

        else:

            st.error(
                "## ❌ Predicted Outcome: FAIL"
            )

        result1, result2 = st.columns(2)

        with result1:

            st.metric(
                "Pass Probability",
                f"{pass_probability * 100:.1f}%"
            )

        with result2:

            st.metric(
                "Fail Probability",
                f"{fail_probability * 100:.1f}%"
            )

        # Probability chart

        probability_df = pd.DataFrame(
            {
                "Outcome": [
                    "FAIL",
                    "PASS"
                ],
                "Probability": [
                    fail_probability * 100,
                    pass_probability * 100
                ]
            }
        )

        fig, ax = plt.subplots(
            figsize=(8, 3.5)
        )

        ax.bar(
            probability_df["Outcome"],
            probability_df["Probability"]
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_title(
            "Model Prediction Confidence"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.info(
            "This prediction is a machine-learning estimate "
            "based on patterns learned from the training dataset. "
            "It should not be treated as a definitive assessment "
            "of an individual student's academic future."
        )

# ============================================================
# ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.markdown(
        '<div class="section-title">Model Performance Analytics</div>',
        unsafe_allow_html=True
    )

    if metrics:

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy'] * 100:.2f}%"
            )

        with metric2:
            st.metric(
                "Precision",
                f"{metrics['precision'] * 100:.2f}%"
            )

        with metric3:
            st.metric(
                "Recall",
                f"{metrics['recall'] * 100:.2f}%"
            )

        with metric4:
            st.metric(
                "F1 Score",
                f"{metrics['f1_score'] * 100:.2f}%"
            )

        st.divider()

        st.markdown("### What do these metrics mean?")

        explanation1, explanation2 = st.columns(2)

        with explanation1:

            st.markdown(
                """
                **Accuracy**  
                Percentage of all test predictions that were correct.

                **Precision**  
                Among students predicted as PASS, how many were actually PASS.
                """
            )

        with explanation2:

            st.markdown(
                """
                **Recall**  
                Among actual PASS students, how many the model identified.

                **F1 Score**  
                A combined measure of precision and recall.
                """
            )

    else:

        st.warning(
            "Model metrics are unavailable."
        )

# ============================================================
# DATASET TAB
# ============================================================

with dataset_tab:

    st.markdown(
        '<div class="section-title">Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore the real student-performance dataset used by the model."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Records",
            len(data)
        )

    with c2:
        st.metric(
            "Features",
            len(FEATURES)
        )

    with c3:
        st.metric(
            "PASS",
            int(data["Pass"].sum())
        )

    with c4:
        st.metric(
            "FAIL",
            int((data["Pass"] == 0).sum())
        )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    st.markdown("### Target Distribution")

    distribution = data["Pass"].value_counts()

    distribution_df = pd.DataFrame(
        {
            "Outcome": [
                "FAIL",
                "PASS"
            ],
            "Students": [
                distribution.get(0, 0),
                distribution.get(1, 0)
            ]
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 3.5)
    )

    ax.bar(
        distribution_df["Outcome"],
        distribution_df["Students"]
    )

    ax.set_ylabel(
        "Students"
    )

    ax.set_title(
        "Student Outcome Distribution"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

# ============================================================
# MODEL TAB
# ============================================================

with model_tab:

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    info1, info2 = st.columns(2)

    with info1:

        st.markdown("### 🤖 Algorithm")

        st.write(
            "Decision Tree Classifier"
        )

        st.write(
            "A supervised learning algorithm that learns "
            "decision rules from labelled training data."
        )

        st.markdown("### 🎯 Target")

        st.write(
            "**PASS:** Final grade ≥ 10"
        )

        st.write(
            "**FAIL:** Final grade < 10"
        )

    with info2:

        st.markdown("### 📐 Dataset")

        st.write(
            f"**Records:** {len(data)}"
        )

        st.write(
            f"**Original features:** {len(FEATURES)}"
        )

        st.write(
            f"**Categorical features:** {len(categorical_features)}"
        )

        st.write(
            f"**Numeric features:** {len(numeric_features)}"
        )

    st.divider()

    st.markdown(
        "### Top Feature Importance"
    )

    try:

        preprocessor_model = model.named_steps[
            "preprocessor"
        ]

        classifier_model = model.named_steps[
            "classifier"
        ]

        feature_names = (
            preprocessor_model
            .get_feature_names_out()
        )

        importances = (
            classifier_model
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        ).sort_values(
            "Importance",
            ascending=False
        )

        top = importance_df.head(15).sort_values(
            "Importance"
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            top["Feature"],
            top["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Top 15 Model Features"
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    except Exception:

        st.warning(
            "Feature importance could not be displayed."
        )

    st.caption(
        "Feature importance describes the relative contribution "
        "of features to the trained decision tree. It does not "
        "establish causation."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>Student Performance AI</strong><br>
        DecodeLabs Artificial Intelligence Internship • Project 2<br><br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)