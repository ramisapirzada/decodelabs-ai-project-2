import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# DECODELABS AI INTERNSHIP - PROJECT 2
# DATA CLASSIFICATION USING AI
# ============================================================

print("=" * 60)
print("DECODELABS AI INTERNSHIP")
print("PROJECT 2: DATA CLASSIFICATION USING AI")
print("=" * 60)


# ============================================================
# 1. CREATE DATASET
# ============================================================

data = {
    "Age": [
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        19, 21, 23, 25, 28, 20, 22, 24, 26, 29,
        18, 20, 22, 24, 27, 19, 21, 23, 25, 30
    ],

    "Study_Hours": [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        2, 6, 7, 8, 9, 3, 5, 7, 8, 10,
        1, 4, 6, 8, 9, 2, 5, 6, 8, 11
    ],

    "Attendance": [
        60, 65, 70, 75, 80, 85, 90, 92, 95, 96,
        62, 82, 88, 91, 94, 68, 78, 86, 90, 95,
        55, 72, 84, 89, 93, 61, 76, 83, 92, 97
    ],

    "Previous_Score": [
        48, 52, 55, 60, 65, 70, 78, 82, 88, 91,
        45, 68, 74, 80, 86, 50, 63, 76, 81, 89,
        40, 58, 69, 77, 85, 46, 62, 71, 84, 94
    ],

    "Assignments_Completed": [
        5, 6, 7, 8, 9, 9, 10, 10, 10, 10,
        5, 8, 9, 9, 10, 6, 8, 9, 9, 10,
        4, 7, 8, 9, 10, 5, 7, 8, 10, 10
    ],

    "Pass": [
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 2. UNDERSTAND THE DATASET
# ============================================================

print("\nDATASET OVERVIEW")
print("-" * 60)

print("Number of records:", len(df))
print("Number of features:", len(df.columns) - 1)

print("\nFirst five records:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

features = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed"
]

X = df[features]
y = df["Pass"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATA SPLIT")
print("-" * 60)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. TRAIN CLASSIFICATION MODEL
# ============================================================

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree Classifier trained successfully.")


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 60)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot()

plt.title("Confusion Matrix - Student Performance")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(9, 5))

importance.plot(kind="barh")

plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Study_Hours"],
    df["Attendance"],
    c=df["Pass"]
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")

plt.title(
    "Student Performance Classification"
)

plt.tight_layout()

plt.savefig(
    "student_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. PREDICT A NEW STUDENT
# ============================================================

new_student = pd.DataFrame({
    "Age": [24],
    "Study_Hours": [6],
    "Attendance": [82],
    "Previous_Score": [70],
    "Assignments_Completed": [8]
})

prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0]

result = "PASS" if prediction == 1 else "FAIL"


print("\nNEW STUDENT PREDICTION")
print("-" * 60)

print(new_student.to_string(index=False))

print("\nPrediction:", result)

print(
    f"Pass probability: {probability[1] * 100:.2f}%"
)

print(
    f"Fail probability: {probability[0] * 100:.2f}%"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- student_performance.png")