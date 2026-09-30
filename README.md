# 🎓 Student Performance AI

## DecodeLabs Artificial Intelligence Internship — Project 2

A machine-learning classification system that predicts whether a student is likely to **PASS or FAIL** based on academic, demographic, family, and study-related information.

The project uses the **UCI Student Performance dataset** containing **649 student records** and applies a **Decision Tree Classifier** with preprocessing for both numerical and categorical features.

---

## 📌 Project Overview

This project demonstrates a complete supervised machine-learning workflow:

* Loading and exploring a real-world dataset
* Preparing numerical and categorical features
* Creating a binary classification target
* Splitting data into training and testing sets
* Preprocessing categorical variables using One-Hot Encoding
* Training a Decision Tree classification model
* Evaluating model performance
* Saving the trained model
* Building an interactive Streamlit prediction dashboard

The project follows the core requirements of DecodeLabs Project 2: loading a dataset, splitting it into training/testing sets, and applying a classification algorithm.

---

## 🎯 Objective

The objective is to build a classification model capable of predicting student academic outcomes.

The original final-grade target is converted into two classes:

| Class | Definition       |
| ----- | ---------------- |
| PASS  | Final grade ≥ 10 |
| FAIL  | Final grade < 10 |

---

## 📊 Dataset

The project uses the **UCI Student Performance dataset**.

### Dataset characteristics

* **Records:** 649
* **Input features:** 30
* **Target:** Student final grade converted into PASS/FAIL
* **Learning type:** Supervised Learning
* **Problem type:** Binary Classification

The dataset contains information related to:

* School
* Gender
* Age
* Address
* Family information
* Parents' education
* Parents' jobs
* Study time
* Previous failures
* School support
* Family support
* Extra activities
* Internet access
* Social relationships
* Absences
* Health and lifestyle factors

---

## 🤖 Machine Learning Model

### Decision Tree Classifier

The project uses a Decision Tree Classifier because it can learn decision rules from labelled data and can work effectively with a mixture of numerical and encoded categorical features.

The model uses:

* Maximum tree depth: `6`
* Minimum samples required for a split: `10`
* Random state: `42`

### Preprocessing

Categorical variables are transformed using:

```text
OneHotEncoder
```

with:

```text
handle_unknown="ignore"
```

Numerical features are passed directly to the model.

The preprocessing and classifier are combined into a single Scikit-learn Pipeline.

---

## 📈 Model Evaluation

The dataset is divided into:

* **80% training data**
* **20% testing data**

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

The current trained model achieved approximately:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 76.15% |
| Precision | 86.00% |
| Recall    | 86.00% |
| F1 Score  | 86.00% |

The exact class-level performance can be viewed in the model training output and Streamlit dashboard.

> Note: The dataset contains more PASS examples than FAIL examples. Therefore, accuracy alone should not be used to judge the model's performance.

---

## 🖥️ Interactive Streamlit Dashboard

The project includes a complete web interface built with Streamlit.

### Dashboard sections

#### 🔮 Prediction

Users can enter student information and receive:

* Predicted outcome
* PASS probability
* FAIL probability
* Prediction confidence chart

#### 📈 Analytics

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* Metric explanations

#### 📊 Dataset Explorer

Allows users to inspect:

* Dataset size
* Feature count
* PASS/FAIL counts
* Dataset preview
* Statistical summary
* Outcome distribution

#### 🤖 Model Information

Displays:

* Machine-learning algorithm
* Dataset information
* Feature types
* Feature importance visualization

---

## 📁 Project Structure

```text
decodelabs_project2/
│
├── app.py
├── project2.py
├── student_data.csv
├── model.pkl
├── metrics.pkl
├── feature_importance.csv
├── feature_importance.png
├── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib
* UCI Machine Learning Repository

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/decodelabs-ai-project-2.git
```

### 2. Open the project

```bash
cd decodelabs-ai-project-2
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python project2.py
```

This generates:

* `model.pkl`
* `metrics.pkl`
* `student_data.csv`
* `confusion_matrix.png`
* `feature_importance.png`
* `feature_importance.csv`

### 7. Launch the dashboard

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📷 Project Visualizations

The project generates:

### Confusion Matrix

Shows the relationship between actual and predicted PASS/FAIL classes.

### Feature Importance

Shows which transformed input features contributed most to the trained Decision Tree.

### Streamlit Dashboard

Provides an interactive interface for predictions, dataset exploration, analytics, and model information.

---

## ⚠️ Important Note

This project is an educational machine-learning application.

The predictions represent patterns learned from the training dataset and should not be interpreted as definitive judgments about an individual student's academic future.

Feature importance indicates model behavior and does not establish that a feature causes a particular academic outcome.

---

## 🎓 DecodeLabs Requirements

This project was developed as part of the **DecodeLabs Artificial Intelligence Internship — Project 2: Data Classification Using AI**.

The project requirements include:

* Loading and understanding a dataset
* Splitting data into training and testing sets
* Applying a simple classification algorithm
* Practicing data handling
* Applying supervised learning fundamentals
* Training and evaluating a classification model

---

## 👩‍💻 Author

**Ramisa Pirzada**

Artificial Intelligence Intern
DecodeLabs — Batch 2026

---

## ⭐ Project Status

**Completed**

The project includes:

* ✅ Real-world dataset
* ✅ Data preprocessing
* ✅ Train/test split
* ✅ Machine-learning classification
* ✅ Model evaluation
* ✅ Saved trained model
* ✅ Confusion matrix
* ✅ Feature importance
* ✅ Interactive Streamlit dashboard
* ✅ GitHub repository
