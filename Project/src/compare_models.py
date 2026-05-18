import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

TRAIN_PATH = "data/processed/training_data.csv"
TEST_PATH = "data/processed/testing_data.csv"
MODEL_OUTPUT_DIR = Path("models")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, save=False):
    print(f"\n==============================")
    print(f"Model: {name}")
    print(f"==============================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if save:
        joblib.dump(model, MODEL_OUTPUT_DIR / f"{name.lower().replace(' ', '_')}.pkl")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    target_column = "is_malicious"
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        )
    }
    results = []

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        result = evaluate_model(name=name,
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)

    print("\n==============================")
    print("MODEL COMPARISON SUMMARY")
    print("==============================")
    print(results_df)
    results_df.to_csv("models/model_comparison_results.csv", index=False)
    print("\nSaved comparison results to models/model_comparison_results.csv")


if __name__ == "__main__":
    main()