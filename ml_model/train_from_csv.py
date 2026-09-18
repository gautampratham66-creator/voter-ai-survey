"""
ml_model/train_from_csv.py

Trains the Voter ID risk model on REAL survey data (a CSV like the one
your survey/upload pipeline produces), instead of the synthetic generator
in predictor.py.

Expected CSV columns (case-insensitive, extra columns are ignored):
    age              int, required
    gender           "Male" / "Female" / "Other", required
    district         string, required
    area_type        "Rural" / "Urban" / "Semi-Urban", required
    has_voter_id     "Yes" / "No" / "Not Applicable", required
                      -> rows marked "Not Applicable" (i.e. under 18 /
                         not eligible) are dropped before training, since
                         there is nothing to predict for them.

Optional columns (kept for reference, not used as model features):
    record_id, name, phone_number, nearest_seva_camp, risk_probability,
    risk_level

Usage:
    python ml_model/train_from_csv.py path/to/survey.csv

    or from code:
        from ml_model.train_from_csv import train_model_from_csv
        model, accuracy, report = train_model_from_csv("survey.csv")
"""

import sys
import os
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "voter_rf_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

REQUIRED_COLUMNS = ["age", "gender", "district", "area_type", "has_voter_id"]


def load_survey_csv(csv_path: str) -> pd.DataFrame:
    """Load and lightly validate/clean a survey CSV for training."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows that aren't eligible for a Voter ID (nothing to predict there)
    before = len(df)
    df = df[df["has_voter_id"].isin(["Yes", "No"])].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[ML] Dropped {dropped} row(s) not eligible ('Not Applicable') for training")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age", "gender", "district", "area_type", "has_voter_id"])
    df["age"] = df["age"].astype(int)

    # family_size is optional; if the CSV doesn't have it, default to 4
    if "family_size" not in df.columns:
        df["family_size"] = 4

    return df


def train_model_from_csv(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train the RandomForestClassifier on a real survey CSV and persist the
    model + label encoders to ml_model/voter_rf_model.pkl and encoders.pkl
    (same file names predictor.py already expects).
    """
    print(f"[ML] Loading survey data from {csv_path} ...")
    df = load_survey_csv(csv_path)
    print(f"[ML] {len(df)} eligible (age>=18) rows available for training")

    if df["has_voter_id"].nunique() < 2:
        raise ValueError(
            "Training data needs both 'Yes' and 'No' has_voter_id examples "
            "to train a classifier."
        )

    le_gender = LabelEncoder()
    le_district = LabelEncoder()
    le_village = LabelEncoder()  # kept name for compatibility with predictor.py

    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["district_enc"] = le_district.fit_transform(df["district"])
    df["village_enc"] = le_village.fit_transform(df["area_type"])
    df["target"] = (df["has_voter_id"] == "Yes").astype(int)

    features = ["age", "gender_enc", "district_enc", "family_size", "village_enc"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    print("[ML] Training Random Forest Classifier on real survey data...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["No Voter ID", "Has Voter ID"]
    )
    print(f"[ML] Model Accuracy: {accuracy:.2%}")
    print(report)

    importances = dict(zip(features, model.feature_importances_))
    print("\n[ML] Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {"gender": le_gender, "district": le_district, "village": le_village},
        ENCODERS_PATH,
    )
    print(f"\n[ML] Model saved to {MODEL_PATH}")
    print(f"[ML] Encoders saved to {ENCODERS_PATH}")

    return model, accuracy, report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_from_csv.py path/to/survey.csv")
        sys.exit(1)
    train_model_from_csv(sys.argv[1])
