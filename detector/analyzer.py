
from detector.feature_extractor import extract_features
from detector.risk_scorer import calculate_risk


def analyze_job(job_description):

    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    features = extract_features(job_description)

    result = calculate_risk(features)

    result["features"] = features

    return result
