
def calculate_risk(features):

    score = 0
    reasons = []

    weights = {
        "payment_request": 35,
        "unrealistic_promise": 20,
        "suspicious_urgency": 10,
        "sensitive_information": 35
    }

    for category, weight in weights.items():

        matches = features.get(category, [])

        if matches:

            score += weight

            reasons.append({
                "category": category,
                "evidence": matches,
                "points": weight
            })

    score = min(score, 100)

    if score >= 60:
        status = "High Risk"

    elif score >= 30:
        status = "Suspicious"

    else:
        status = "Low Risk"

    return {
        "risk_score": score,
        "status": status,
        "reasons": reasons
    }
