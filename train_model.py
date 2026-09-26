"""
Diabetes Prediction Using ML - Model Training Script
Algorithm: Decision Tree Classifier (Supervised Machine Learning)
Dataset: Pima Indians Diabetes Dataset (diabetes.csv)
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Define expected feature list and target column
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_COLUMN = "Outcome"

def main():
    # Resolve file paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "diabetes.csv")
    model_path = os.path.join(script_dir, "diabetes_model.pkl")

    print("Loading dataset...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please check the file location.")

    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Verify that all required features and the target column exist in the CSV
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Separate features (X) and target variable (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split dataset into training set (80%) and testing set (20%)
    # Using stratify=y preserves the class ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Decision Tree...")
    # Initialize and train Decision Tree Classifier with a fixed random state for reproducibility
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Generate predictions on the unseen test set
    y_pred = model.predict(X_test)

    # Compute actual evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Display evaluation metrics in the terminal
    print("\nModel Evaluation")
    print("-------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Save trained model to disk
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully:\n{model_path}")

if __name__ == "__main__":
    main()
