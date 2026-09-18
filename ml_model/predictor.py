"""
ml_model/predictor.py
Machine Learning Model for Voter ID Prediction + Anomaly Detection
Uses: Scikit-learn Random Forest + Isolation Forest

CHANGE FROM ORIGINAL: the model is now trained on real survey data
(see train_from_csv.py) instead of only synthetic data. The synthetic
generator below is kept only as a fallback so predict_voter_id() still
works out of the box if no model has been trained yet.
"""

# pip install scikit-learn pandas numpy joblib

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import json

from ml_model.train_from_csv import (
    train_model_from_csv,
    MODEL_PATH,
    ENCODERS_PATH,
)

# ─── Synthetic fallback (only used if no CSV / no trained model exists) ────
def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Fallback synthetic data generator, used only when the caller hasn't
    trained on real survey data yet. Prefer train_model_from_csv() with
    your real survey export.
    """
    np.random.seed(42)

    ages = np.random.randint(18, 80, n_samples)
    genders = np.random.choice(["Male", "Female"], n_samples, p=[0.52, 0.48])
    districts = np.random.choice(["Muzaffarnagar", "Shamli", "Hapur", "Haridwar"], n_samples)
    family_sizes = np.random.randint(2, 8, n_samples)
    village_types = np.random.choice(["Urban", "Rural", "Semi-Urban"], n_samples, p=[0.3, 0.5, 0.2])

    prob = (
        0.3
        + (ages - 18) / 62 * 0.4
        + (genders == "Male") * 0.08
        + (village_types == "Urban") * 0.1
        + (village_types == "Rural") * -0.05
        + np.random.uniform(-0.1, 0.1, n_samples)
    ).clip(0.1, 0.95)

    has_voter_id = (np.random.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "age": ages,
        "gender": genders,
        "district": districts,
        "family_size": family_sizes,
        "area_type": village_types,
        "has_voter_id": np.where(has_voter_id == 1, "Yes", "No"),
    })


def train_model(csv_path: str = None):
    """
    Train the model.
    - If csv_path is given, trains on real survey data (recommended).
    - Otherwise falls back to synthetic data (demo/testing only).
    """
    if csv_path:
        model, accuracy, _ = train_model_from_csv(csv_path)
        return model, accuracy

    print("[ML] No CSV given — training on SYNTHETIC data (demo only).")
    df = generate_training_data(2000)
    tmp_path = "/tmp/_synthetic_survey.csv"
    df.to_csv(tmp_path, index=False)
    model, accuracy, _ = train_model_from_csv(tmp_path)
    return model, accuracy


# ─── Predict for a Citizen ───────────────────────────────────────────────────
def predict_voter_id(age: int, gender: str, district: str,
                      family_size: int = 4, village_type: str = "Rural") -> dict:
    """
    Predict likelihood of a citizen having / not having Voter ID.
    Returns probability and risk level.
    """
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
    except FileNotFoundError:
        # Train on synthetic data if nothing exists yet
        model, _ = train_model()
        encoders = joblib.load(ENCODERS_PATH)

    try:
        gender_enc = encoders["gender"].transform([gender])[0]
        district_enc = encoders["district"].transform([district])[0]
        village_enc = encoders["village"].transform([village_type])[0]
    except ValueError:
        # Unseen category at prediction time — fall back to 0 rather than crash
        gender_enc = district_enc = village_enc = 0

    features = [[age, gender_enc, district_enc, family_size, village_enc]]
    proba = model.predict_proba(features)[0]

    prob_has_id = round(proba[1] * 100, 1)
    prob_missing = round(proba[0] * 100, 1)
    risk = "HIGH" if prob_missing > 50 else "MEDIUM" if prob_missing > 30 else "LOW"

    return {
        "age": age, "gender": gender, "district": district,
        "prob_has_voter_id": prob_has_id,
        "prob_missing_voter_id": prob_missing,
        "risk_level": risk,
        "recommendation": "Immediate outreach needed" if risk == "HIGH" else
                           "Include in next drive" if risk == "MEDIUM" else
                           "Likely registered"
    }


# ─── Anomaly Detection ───────────────────────────────────────────────────────
def detect_anomalies(families: list) -> list:
    """
    Uses Isolation Forest to detect duplicate entries, suspicious
    age/gender combinations, and data entry errors.
    """
    rows = []
    for fam in families:
        for m in fam["members"]:
            rows.append({
                "age": m["age"],
                "has_voter_id": int(m["has_voter_id"]),
                "eligible": int(m["eligible"]),
                "gender_num": 0 if m["gender"] == "Male" else 1
            })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso.fit_predict(df)

    anomalies = df[df["anomaly"] == -1].index.tolist()
    return anomalies


# ─── Batch Prediction for All Eligible Non-Registered ───────────────────────
def prioritize_outreach(families: list) -> list:
    """
    Runs prediction on all eligible members without Voter ID.
    Returns sorted list by risk (highest risk first) for outreach prioritization.
    """
    targets = []
    for fam in families:
        for m in fam["members"]:
            if m["eligible"] and not m["has_voter_id"]:
                pred = predict_voter_id(m["age"], m["gender"], fam["district"])
                targets.append({
                    "family_id": fam["family_id"],
                    "name": m["name"],
                    "district": fam["district"],
                    **pred
                })
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    targets.sort(key=lambda x: risk_order[x["risk_level"]])
    return targets


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    model, accuracy = train_model(csv_arg)

    result = predict_voter_id(age=23, gender="Female", district="Varanasi", village_type="Rural")
    print("\n=== PREDICTION ===")
    print(json.dumps(result, indent=2))
