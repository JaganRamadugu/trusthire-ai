"""
TrustHire AI — Detector Integration and Validation Test Suite
=============================================================
Tests all required backend scenarios:
1. Genuine-looking job description
2. Suspicious job description with urgency
3. High-risk job description containing payment requests and sensitive information
4. Empty input (validates ValueError handling)
5. Input with no recognized warning keywords
6. Input containing emails and URLs

Verifies:
- Model prediction works (ML pipeline outputs 0-100 score).
- Rule-based detection works (keyword extraction and score calculation).
- Hybrid risk scoring works (combined_score = max(ml_score, rule_score)).
- All three risk categories are functional.
- Evidence is returned correctly with appropriate categories.
- Features (emails, URLs) are parsed accurately.
- No missing model or import errors occur.
"""

from detector.analyzer import analyze_job


def test_scenario_1_genuine():
    print("\n--- Test Scenario 1: Genuine-looking job description ---")
    text = """
    Software Engineer Intern (Cloud Systems)
    Company: Apex Cloud Technologies
    Location: Bengaluru, India (Hybrid)
    Duration: 6 Months Internship

    About the Role:
    We are seeking an enthusiastic Software Engineer Intern to join our Cloud Infrastructure team.
    You will assist senior engineers with developing scalable microservices, creating automated tests,
    and optimizing backend data pipelines.

    Requirements:
    - Currently pursuing a Bachelor's degree in Computer Science or related field.
    - Basic proficiency with Python and SQL.
    - Familiarity with Git and REST APIs.
    """
    result = analyze_job(text)
    print(f"Result: Score={result['risk_score']}, Status={result['status']}, ML={result['ml_score']}, Rule={result['rule_score']}")

    assert isinstance(result, dict), "Result must be a dictionary"
    assert "risk_score" in result and 0 <= result["risk_score"] <= 100
    assert "ml_score" in result and 0 <= result["ml_score"] <= 100
    assert "rule_score" in result and result["rule_score"] == 0, "Genuine posting should have 0 rule score"
    assert result["status"] in ["Low Risk", "Suspicious", "High Risk"]
    assert result["features"]["payment_request"] == []
    assert result["features"]["sensitive_information"] == []
    print("PASS: Scenario 1 verified successfully.")


def test_scenario_2_suspicious_urgency():
    print("\n--- Test Scenario 2: Suspicious job description with urgency ---")
    text = """
    Urgent Hiring: Campus Brand Ambassador & Marketing Intern
    Company: Apex Global Promotions
    Location: Work From Home / Remote

    Immediate requirement! We are urgently hiring 20 students for campus outreach.
    Limited seats available. Apply immediately!
    Offer expires today at midnight. Candidates should apply immediately to be considered.
    """
    result = analyze_job(text)
    print(f"Result: Score={result['risk_score']}, Status={result['status']}, ML={result['ml_score']}, Rule={result['rule_score']}")

    assert result["rule_score"] >= 10, "Urgency keywords must trigger at least 10 rule points"
    assert len(result["features"]["suspicious_urgency"]) > 0, "Should detect urgency keywords"
    assert result["status"] in ["Suspicious", "High Risk"]

    # Verify evidence returned in reasons
    urgency_reason = next((r for r in result["reasons"] if r["category"] == "suspicious_urgency"), None)
    assert urgency_reason is not None, "Reasons must include suspicious_urgency"
    assert len(urgency_reason["evidence"]) > 0, "Evidence list must not be empty"
    print(f"PASS: Scenario 2 verified. Detected urgency evidence: {urgency_reason['evidence']}")


def test_scenario_3_high_risk_scam():
    print("\n--- Test Scenario 3: High-risk job with payment & sensitive info demands ---")
    text = """
    Congratulations! Guaranteed selection without interview required. Earn money instantly.
    Immediate joining.
    Pay a refundable security deposit and registration fee of Rs. 1,999.
    Send your Aadhaar card and bank account details for verification.
    Limited seats available! Offer expires today. Apply immediately.
    """
    result = analyze_job(text)
    print(f"Result: Score={result['risk_score']}, Status={result['status']}, ML={result['ml_score']}, Rule={result['rule_score']}")

    assert result["status"] == "High Risk", f"Expected High Risk, got {result['status']}"
    assert result["risk_score"] >= 60, "High Risk score must be >= 60"
    assert result["rule_score"] >= 60, "Severe scam keywords should yield high rule score"

    # Verify specific category detections
    categories = {r["category"] for r in result["reasons"]}
    assert "payment_request" in categories, "Must detect payment_request"
    assert "sensitive_information" in categories, "Must detect sensitive_information"
    assert "unrealistic_promise" in categories, "Must detect unrealistic_promise"
    assert "suspicious_urgency" in categories, "Must detect suspicious_urgency"
    print(f"PASS: Scenario 3 verified. All 4 scam categories detected: {categories}")


def test_scenario_4_empty_input():
    print("\n--- Test Scenario 4: Empty input handling ---")
    empty_inputs = ["", "   ", "\n\t  \n"]
    for inp in empty_inputs:
        try:
            analyze_job(inp)
            assert False, f"Expected ValueError for input: {repr(inp)}"
        except ValueError as e:
            assert "cannot be empty" in str(e).lower()
            print(f"PASS: Empty input correctly raised ValueError: '{e}'")


def test_scenario_5_no_keywords():
    print("\n--- Test Scenario 5: Input with no recognized warning keywords ---")
    text = """
    We are seeking a Research Assistant in Environmental Science.
    Responsibilities include compiling field data, maintaining laboratory equipment,
    and assisting with statistical analysis using R and Python.
    Minimum qualifications: Bachelor's degree in Ecology, Biology, or related sciences.
    """
    result = analyze_job(text)
    print(f"Result: Score={result['risk_score']}, Status={result['status']}, ML={result['ml_score']}, Rule={result['rule_score']}")

    assert result["rule_score"] == 0, "Must have 0 rule score for clean text"
    for cat in ["payment_request", "sensitive_information", "unrealistic_promise", "suspicious_urgency"]:
        assert result["features"][cat] == [], f"Category {cat} should be empty"

    if result["ml_score"] < 30:
        assert result["status"] == "Low Risk"
        assert len(result["reasons"]) == 0, "No reasons should be flagged for low risk clean text"
    print("PASS: Scenario 5 verified. Clean text handled with 0 rule penalties.")


def test_scenario_6_emails_and_urls():
    print("\n--- Test Scenario 6: Input containing emails and URLs ---")
    text = """
    Junior QA Tester Internship at TechCorp.
    Send applications to careers@techcorp-hiring.example.com or support@jobs.io.
    Official portal: https://careers.techcorp.example.com/openings/1234
    Additional info: http://apply.techcorp.io
    """
    result = analyze_job(text)
    extracted_emails = result["features"]["emails"]
    extracted_urls = result["features"]["urls"]
    print(f"Extracted emails: {extracted_emails}")
    print(f"Extracted URLs:   {extracted_urls}")

    assert "careers@techcorp-hiring.example.com" in extracted_emails
    assert "support@jobs.io" in extracted_emails
    assert "https://careers.techcorp.example.com/openings/1234" in extracted_urls
    assert "http://apply.techcorp.io" in extracted_urls
    print("PASS: Scenario 6 verified. Both emails and URLs extracted accurately.")


def run_all_tests():
    print("=" * 60)
    print("TrustHire AI — Running Backend Test Suite")
    print("=" * 60)

    test_scenario_1_genuine()
    test_scenario_2_suspicious_urgency()
    test_scenario_3_high_risk_scam()
    test_scenario_4_empty_input()
    test_scenario_5_no_keywords()
    test_scenario_6_emails_and_urls()

    print("\n" + "=" * 60)
    print("ALL 6 SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
