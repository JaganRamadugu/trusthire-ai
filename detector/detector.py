
import joblib
from pathlib import Path

from detector.feature_extractor import extract_features
from detector.risk_scorer import calculate_risk


# Load trained ML model
MODEL_PATH = Path(__file__).with_name("fraud_model.joblib")

model = joblib.load(MODEL_PATH)


def analyze_job(job_description):

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    # 1. Rule-based analysis
    features = extract_features(job_description)

    rule_result = calculate_risk(features)

    rule_score = rule_result["risk_score"]

    # 2. ML prediction
    ml_probability = model.predict_proba(
        [job_description]
    )[0][1]

    ml_score = round(ml_probability * 100)

    # 3. Combine both signals
    # This is a risk index, NOT a fraud probability.
    combined_score = max(ml_score, rule_score)

    # 4. Determine risk level
    if combined_score >= 60:
        status = "High Risk"

    elif combined_score >= 30:
        status = "Suspicious"

    else:
        status = "Low Risk"

    # 5. Build explanation
    reasons = rule_result["reasons"].copy()

    if ml_score >= 30:
        reasons.append({
            "category": "ml_detection",
            "evidence": [
                "ML model detected patterns associated with fraudulent postings."
            ],
            "points": None
        })

    return {
        "risk_score": combined_score,
        "status": status,
        "ml_score": ml_score,
        "rule_score": rule_score,
        "reasons": reasons,
        "features": features
    }
