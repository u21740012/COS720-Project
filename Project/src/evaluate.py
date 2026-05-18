import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(test_csv_path: str, model_path: str):
    df = pd.read_csv(test_csv_path)
    target_column = "is_malicious"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in testing data.")

    X = df.drop(columns=[target_column])
    y_true = df[target_column]
    X = X.copy()
    if "trip_day_number" in X.columns:
        X["trip_day_number"] = X["trip_day_number"].fillna(0)

    saved = joblib.load(model_path)

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.reindex(columns=feature_columns,
        fill_value=0
    )

    y_pred = model.predict(X_encoded)

    print("=== Evaluation Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-Score:", f1_score(y_true, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    evaluate_model(test_csv_path="data/processed/testing_data.csv",
        model_path="models/trained_model.pkl"
    )