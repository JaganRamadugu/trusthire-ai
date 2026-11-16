
import re
from urllib.parse import urlparse


SCAM_KEYWORDS = {
    "payment_request": [
        "registration fee",
        "processing fee",
        "security deposit",
        "pay to apply",
        "training fee",
        "refundable fee",
        "payment required"
    ],
    "unrealistic_promise": [
        "guaranteed selection",
        "no interview required",
        "100% placement guarantee",
        "earn money instantly",
        "immediate joining"
    ],
    "suspicious_urgency": [
        "limited seats",
        "apply immediately",
        "urgent hiring",
        "offer expires today"
    ],
    "sensitive_information": [
        "share aadhaar",
        "share pan card",
        "send bank details",
        "provide otp"
    ]
}


def extract_features(text):

    text = text.lower()

    features = {}

    for category, keywords in SCAM_KEYWORDS.items():

        matches = [
            keyword for keyword in keywords
            if keyword in text
        ]

        features[category] = matches

    emails = re.findall(
        r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}',
        text
    )

    urls = re.findall(
        r'https?://[^\s]+',
        text
    )

    features["emails"] = emails
    features["urls"] = urls

    return features
