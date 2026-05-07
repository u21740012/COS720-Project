import joblib
import pandas as pd


class ThreatPredictor:
    def __init__(self, model_path: str):
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.feature_columns = saved["feature_columns"]

    def prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        input_df = df.copy()

        # Handle missing trip_day_number
        if "trip_day_number" in input_df.columns:
            input_df["trip_day_number"] = input_df["trip_day_number"].fillna(0)

        # One-hot encode input
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align to training columns
        input_encoded = input_encoded.reindex(columns=self.feature_columns, fill_value=0)

        return input_encoded

    def predict(self, df: pd.DataFrame):
        X = self.prepare_input(df)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_
            }).sort_values(by="importance", ascending=False)
            return importance_df
        return None