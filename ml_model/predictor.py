"""
ml_model/predictor.py
Machine Learning Model for Voter ID Prediction + Anomaly Detection
Uses: Scikit-learn Random Forest + Isolation Forest
"""

# pip install scikit-learn pandas numpy joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# ─── Generate Training Data ─────────────────────────────────────────────────
def generate_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    In production: use real survey data from PostgreSQL.
    Here: synthetic data for demonstration.
    """
    np.random.seed(42)

    ages    = np.random.randint(18, 80, n_samples)
    genders = np.random.choice(["Male", "Female"], n_samples, p=[0.52, 0.48])
    districts = np.random.choice(["Muzaffarnagar","Shamli","Hapur","Haridwar"], n_samples)
    family_sizes = np.random.randint(2, 8, n_samples)
    village_types = np.random.choice(["Urban","Rural","Semi-Urban"], n_samples, p=[0.3,0.5,0.2])

    # Simulate has_voter_id with realistic patterns:
    # Older people more likely to have it; males slightly more; urban more than rural
    prob = (
        0.3
        + (ages - 18) / 62 * 0.4                          # age factor
        + (genders == "Male") * 0.08                       # gender factor
        + (village_types == "Urban") * 0.1                 # urban factor
        + (village_types == "Rural") * -0.05               # rural factor
        + np.random.uniform(-0.1, 0.1, n_samples)          # noise
    ).clip(0.1, 0.95)

    has_voter_id = (np.random.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "age": ages,
        "gender": genders,
        "district": districts,
        "family_size": family_sizes,
        "village_type": village_types,
        "has_voter_id": has_voter_id
    })

# ─── Train Model ─────────────────────────────────────────────────────────────
def train_model():
    """Train Random Forest classifier and save to disk"""
    print("[ML] Generating training data...")
    df = generate_training_data(2000)

    # Encode categorical variables
    le_gender   = LabelEncoder()
    le_district = LabelEncoder()
    le_village  = LabelEncoder()

    df['gender_enc']   = le_gender.fit_transform(df['gender'])
    df['district_enc'] = le_district.fit_transform(df['district'])
    df['village_enc']  = le_village.fit_transform(df['village_type'])

    features = ['age', 'gender_enc', 'district_enc', 'family_size', 'village_enc']
    X = df[features]
    y = df['has_voter_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[ML] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[ML] Model Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=["No Voter ID","Has Voter ID"]))

    # Feature importance
    importances = dict(zip(features, model.feature_importances_))
    print("\n[ML] Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Save model and encoders
    joblib.dump(model, "ml_model/voter_rf_model.pkl")
    joblib.dump({"gender": le_gender, "district": le_district, "village": le_village},
                "ml_model/encoders.pkl")
    print("[ML] Model saved!")
    return model, accuracy

# ─── Predict for a Citizen ───────────────────────────────────────────────────
def predict_voter_id(age: int, gender: str, district: str,
                     family_size: int = 4, village_type: str = "Rural") -> dict:
    """
    Predict likelihood of a citizen having / not having Voter ID.
    Returns probability and risk level.
    """
    try:
        model    = joblib.load("ml_model/voter_rf_model.pkl")
        encoders = joblib.load("ml_model/encoders.pkl")
    except FileNotFoundError:
        # Train if not found
        model, _ = train_model()
        encoders = joblib.load("ml_model/encoders.pkl")

    # Encode input
    try:
        gender_enc   = encoders["gender"].transform([gender])[0]
        district_enc = encoders["district"].transform([district])[0]
        village_enc  = encoders["village"].transform([village_type])[0]
    except ValueError:
        gender_enc = district_enc = village_enc = 0

    features = [[age, gender_enc, district_enc, family_size, village_enc]]
    proba = model.predict_proba(features)[0]

    prob_has_id  = round(proba[1] * 100, 1)
    prob_missing = round(proba[0] * 100, 1)
    risk = "HIGH" if prob_missing > 50 else "MEDIUM" if prob_missing > 30 else "LOW"

    return {
        "age": age, "gender": gender, "district": district,
        "prob_has_voter_id": prob_has_id,
        "prob_missing_voter_id": prob_missing,
        "risk_level": risk,
        "recommendation": "Immediate outreach needed" if risk=="HIGH" else
                          "Include in next drive" if risk=="MEDIUM" else
                          "Likely registered"
    }

# ─── Anomaly Detection ───────────────────────────────────────────────────────
def detect_anomalies(families: list) -> list:
    """
    Uses Isolation Forest to detect:
    - Duplicate entries
    - Suspicious age/gender combinations
    - Data entry errors
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
    return anomalies  # Returns indices of suspicious records

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
    # Sort by risk
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    targets.sort(key=lambda x: risk_order[x["risk_level"]])
    return targets

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train model
    model, accuracy = train_model()

    # Test prediction
    result = predict_voter_id(age=23, gender="Female", district="Muzaffarnagar")
    print("\n=== PREDICTION ===")
    print(json.dumps(result, indent=2))
