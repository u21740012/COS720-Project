import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


def train_model(csv_path: str, model_output_path: str):
    df = pd.read_csv(csv_path)

    target_column = "is_malicious"

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.copy()

    if "trip_day_number" in X.columns:
        X["trip_day_number"] = X["trip_day_number"].fillna(0)

    X_encoded = pd.get_dummies(X, drop_first=True)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_encoded, y)

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X_encoded.columns)
        },
        model_output_path
    )

    print("=== Training Complete ===")
    print(f"Training records: {len(df)}")
    print(f"Features after encoding: {len(X_encoded.columns)}")
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    train_model(
        csv_path="data/processed/training_data.csv",
        model_output_path="models/trained_model.pkl"
    )