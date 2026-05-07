import os
import pandas as pd
from datetime import datetime


def log_prediction(record: dict, prediction: str, confidence: float, explanation: str, log_path: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        **record
    }

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        updated = pd.concat([existing, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        updated = pd.DataFrame([log_entry])

    updated.to_csv(log_path, index=False)