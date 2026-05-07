import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = "data/raw/insider_threat_clean_dataset.csv"

PROCESSED_DIR = Path("data/processed")
UPLOAD_DIR = Path("data/test_uploads")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Split into 50% train / 50% test
train_df, test_df = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify=df["is_malicious"]
)

# Save labelled training and testing datasets
train_df.to_csv(PROCESSED_DIR / "training_data.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "testing_data.csv", index=False)

# Keep labels temporarily for controlled sampling
test_with_labels = test_df.copy()

# ---------------- Normal upload ----------------
normal_labeled = (
    test_with_labels[test_with_labels["is_malicious"] == 0]
    .sample(10, random_state=42)
    .reset_index(drop=True)
)

normal_upload = normal_labeled.drop(columns=["is_malicious"])

normal_labeled.to_csv(UPLOAD_DIR / "normal_upload_with_labels.csv", index=False)
normal_upload.to_csv(UPLOAD_DIR / "normal_upload.csv", index=False)

# ---------------- Malicious upload ----------------
malicious_labeled = (
    test_with_labels[test_with_labels["is_malicious"] == 1]
    .sample(10, random_state=42)
    .reset_index(drop=True)
)

malicious_upload = malicious_labeled.drop(columns=["is_malicious"])

malicious_labeled.to_csv(UPLOAD_DIR / "malicious_upload_with_labels.csv", index=False)
malicious_upload.to_csv(UPLOAD_DIR / "malicious_upload.csv", index=False)

# ---------------- Mixed upload ----------------
# Force a true mix instead of random sampling from an imbalanced dataset
mixed_normal = (
    test_with_labels[test_with_labels["is_malicious"] == 0]
    .sample(8, random_state=42)
)

mixed_malicious = (
    test_with_labels[test_with_labels["is_malicious"] == 1]
    .sample(7, random_state=42)
)

mixed_labeled = pd.concat(
    [mixed_normal, mixed_malicious],
    ignore_index=True
)

# Shuffle so malicious/normal records are mixed together
mixed_labeled = (
    mixed_labeled
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

mixed_upload = mixed_labeled.drop(columns=["is_malicious"])

mixed_labeled.to_csv(UPLOAD_DIR / "mixed_upload_with_labels.csv", index=False)
mixed_upload.to_csv(UPLOAD_DIR / "mixed_upload.csv", index=False)

# ---------------- Summary ----------------
print("Done.")
print("Training data:", PROCESSED_DIR / "training_data.csv")
print("Testing data:", PROCESSED_DIR / "testing_data.csv")
print()
print("Upload files:")
print(" -", UPLOAD_DIR / "normal_upload.csv")
print(" -", UPLOAD_DIR / "malicious_upload.csv")
print(" -", UPLOAD_DIR / "mixed_upload.csv")
print()
print("Labelled versions for testing report:")
print(" -", UPLOAD_DIR / "normal_upload_with_labels.csv")
print(" -", UPLOAD_DIR / "malicious_upload_with_labels.csv")
print(" -", UPLOAD_DIR / "mixed_upload_with_labels.csv")
print()
print("Mixed upload label distribution:")
print(mixed_labeled["is_malicious"].value_counts())