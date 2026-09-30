import pandas as pd
import joblib
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 65)
print("STUDENT PERFORMANCE AI — MODEL TRAINING")
print("=" * 65)

# ============================================================
# LOAD UCI DATASET
# ============================================================

print("\nLoading UCI Student Performance dataset...")

student_performance = fetch_ucirepo(id=320)

X = student_performance.data.features.copy()
y_raw = student_performance.data.targets.copy()

print("Dataset loaded successfully.")
print(f"Records: {len(X)}")
print(f"Features: {len(X.columns)}")

# ============================================================
# CREATE BINARY TARGET
# ============================================================

# G3 is the final grade from 0–20.
# PASS = G3 >= 10
# FAIL = G3 < 10

y = y_raw["G3"].apply(
    lambda grade: 1 if grade >= 10 else 0
)

# Save dataset for Streamlit
dataset = X.copy()
dataset["Pass"] = y

dataset.to_csv(
    "student_data.csv",
    index=False
)

print("\nTarget distribution:")
print(
    dataset["Pass"]
    .value_counts()
    .rename(index={0: "FAIL", 1: "PASS"})
)

# ============================================================
# PREPARE FEATURES
# ============================================================

# G3 is the target and is therefore not included in X.
#
# G1 and G2 are not present in the feature dataframe returned
# by the current UCI repository interface, so no additional
# removal is required here.

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()

print(f"\nCategorical features: {len(categorical_features)}")
print(f"Numeric features: {len(numeric_features)}")

# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ============================================================
# MODEL
# ============================================================

classifier = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    random_state=RANDOM_STATE
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining model...")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

pipeline.fit(
    X_train,
    y_train
)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = pipeline.predict(X_test)

# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

print(f"\nAccuracy : {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall   : {recall:.2%}")
print(f"F1 Score : {f1:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["FAIL", "PASS"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "model.pkl"
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "total_records": len(X),
    "feature_count": len(X.columns)
}

joblib.dump(
    metrics,
    "metrics.pkl"
)

# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    [0, 1],
    ["FAIL", "PASS"]
)

plt.yticks(
    [0, 1],
    ["FAIL", "PASS"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        plt.text(
            col,
            row,
            cm[row, col],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig(
    "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

classifier_model = pipeline.named_steps["classifier"]
preprocessor_model = pipeline.named_steps["preprocessor"]

feature_names = (
    preprocessor_model
    .get_feature_names_out()
)

importances = classifier_model.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
).sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

top_features = importance_df.head(15).sort_values(
    "Importance"
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.title("Top 15 Feature Importances")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nFiles generated:")
print("  ✓ student_data.csv")
print("  ✓ model.pkl")
print("  ✓ metrics.pkl")
print("  ✓ confusion_matrix.png")
print("  ✓ feature_importance.png")
print("  ✓ feature_importance.csv")

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)