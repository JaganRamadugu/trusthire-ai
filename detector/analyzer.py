
import joblib
from pathlib import Path

from detector.feature_extractor import extract_features
from detector.risk_scorer import calculate_risk


MODEL_PATH = Path(__file__).with_name("fraud_model.joblib")

model = joblib.load(MODEL_PATH)


def analyze_job(job_description):
    """
    Analyze a job or internship description using a hybrid fraud detection approach.

    Combines:
    1. Heuristic rule-based pattern matching (weights across known scam categories).
    2. ML model output from a TF-IDF + Logistic Regression pipeline trained exclusively on descriptions.

    Note:
    - ml_score is a heuristic model output (0-100), not a statistically calibrated fraud probability.
    - combined_score is a heuristic risk index = max(ml_score, rule_score).
    - Thresholds: 0-29 Low Risk, 30-59 Suspicious, 60-100 High Risk.
    """
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    # 1. Rule-based detection
    features = extract_features(job_description)
    rule_result = calculate_risk(features)

    rule_score = rule_result["risk_score"]

    # 2. ML model prediction (input is the complete job description)
    ml_probability = model.predict_proba(
        [job_description]
    )[0][1]

    # Heuristic ML signal score (0-100)
    ml_score = round(ml_probability * 100)

    # 3. Hybrid combined risk score
    combined_score = max(ml_score, rule_score)

    # 4. Prototype decision thresholds
    if combined_score >= 60:
        status = "High Risk"
    elif combined_score >= 30:
        status = "Suspicious"
    else:
        status = "Low Risk"

    reasons = rule_result["reasons"].copy()

    # Append ML detection signal if above suspicious threshold
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
