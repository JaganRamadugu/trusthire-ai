import streamlit as st
from pathlib import Path
from detector.analyzer import analyze_job

# Page configuration
st.set_page_config(
    page_title="TrustHire AI | AI Job & Internship Fraud Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

load_css()

# Sample job descriptions for quick demonstration
GENUINE_SAMPLE = """Software Engineer Intern (Cloud Systems)
Company: Apex Cloud Technologies
Location: Bengaluru, India (Hybrid)
Duration: 6 Months Internship

About the Role:
We are seeking an enthusiastic Software Engineer Intern to join our Cloud Infrastructure team. You will assist senior engineers with developing scalable microservices, creating automated tests, and optimizing backend data pipelines.

Key Responsibilities:
- Write clean and maintainable Python code for internal tools.
- Assist with Docker containerization and CI/CD pipelines.
- Collaborate with agile engineering teams in daily standups and sprint reviews.
- Write unit tests and maintain technical documentation.

Requirements:
- Currently pursuing a Bachelor's or Master's degree in Computer Science, IT, or related technical field.
- Basic proficiency with Python and SQL.
- Familiarity with version control systems (Git) and REST APIs.
- Good problem-solving and communication skills.

Stipend & Perks:
- Competitive monthly stipend: Rs. 35,000 / month.
- Certificate of internship and mentorship from senior staff.
- Opportunity for full-time pre-placement offer (PPO) based on evaluation.

How to Apply:
Submit your resume and GitHub profile through our official careers portal at https://careers.apexcloud.tech. We do not charge any application or recruitment fees at any stage."""

SUSPICIOUS_SAMPLE = """Urgent Hiring: Campus Brand Ambassador & Marketing Intern
Company: Apex Global Promotions
Location: Work From Home / Remote

Immediate requirement! We are urgently hiring 20 students for campus outreach.

Opportunity Details:
- Limited seats available. Apply immediately!
- Urgent hiring for this week's cohort.
- Flexible timings (1-2 hours per day).
- Certificate of completion and performance incentives provided.

Note: Offer expires today at midnight. Candidates should apply immediately to be considered."""

HIGH_RISK_SAMPLE = """Urgent Selection Letter: Executive AI Operations Associate
Status: Direct Selection Confirmed

Congratulations! You have been selected for an immediate joining remote internship with Global Tech Solutions.
Guaranteed selection without interview required. Earn money instantly from day one with a guaranteed monthly stipend of Rs. 60,000.

Mandatory Verification Steps:
1. Pay a refundable security deposit and registration fee of Rs. 1,999 to activate your employee ID and system credentials.
2. Send your Aadhaar card and bank account details for payroll compliance.
3. Limited seats available! Offer expires today. Apply immediately to avoid forfeiture.

Contact HR: recruitment@tech-onboarding.biz"""

# Initialize session state keys
if "job_text_area" not in st.session_state:
    st.session_state["job_text_area"] = ""

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

def set_sample(text):
    st.session_state["job_text_area"] = text
    st.session_state["analysis_result"] = None

def clear_input():
    st.session_state["job_text_area"] = ""
    st.session_state["analysis_result"] = None

# ==============================================================================
# SECTION A: Navigation / Header
# ==============================================================================
st.html("""
<header class="nav-container">
    <div class="nav-brand">
        <div class="nav-logo-badge">🛡️</div>
        <div>
            <div class="nav-brand-title">TrustHire AI</div>
        </div>
        <span class="nav-brand-tag">FRAUD SHIELD v1.0</span>
    </div>
    <div class="nav-links">
        <a href="#hero" class="nav-link-item">Home</a>
        <a href="#analyzer" class="nav-link-item">Opportunity Analyzer</a>
        <a href="#how-it-works" class="nav-link-item">How It Works</a>
        <a href="#safety-advisory" class="nav-link-item">Safety Advisory</a>
    </div>
    <div class="nav-status">
        <span class="status-dot-pulse"></span>
        <span>AI Engine Operational</span>
    </div>
</header>
""")

# ==============================================================================
# SECTION B: Hero Section
# ==============================================================================
st.html("""
<div class="hero-section" id="hero">
    <div class="hero-pill">
        <span>⚡</span> AI-POWERED RECRUITMENT FRAUD DETECTION
    </div>
    <h1 class="hero-headline">Don't Let Fake Opportunities Steal Your Future.</h1>
    <p class="hero-subtext">
        Analyze job and internship opportunities using AI-powered fraud detection and identify suspicious warning signs before you apply.
    </p>
    <a href="#analyzer" class="hero-cta-btn">
        🛡️ Analyze an Opportunity
    </a>
    <div class="hero-highlights">
        <div class="highlight-item">
            <span>🔍</span> Hybrid Heuristic & Machine Learning Engine
        </div>
        <div class="highlight-item">
            <span>⚡</span> Real-Time Risk Scoring (0–100)
        </div>
        <div class="highlight-item">
            <span>🎓</span> Dedicated Defense for Students & Job Seekers
        </div>
    </div>
</div>
""")

# ==============================================================================
# SECTION C: Opportunity Analyzer
# ==============================================================================
st.html("""
<div class="card-container" id="analyzer">
    <div class="section-header-wrap">
        <span class="section-eyebrow">INPUT OPPORTUNITY CONTENT</span>
        <h2 class="section-heading">Opportunity Analyzer</h2>
        <p class="section-subheading">
            Paste the complete text from a job posting, internship offer, recruiter email, or message.
        </p>
    </div>
</div>
""")

# Quick sample buttons
col_lbl, col_s1, col_s2, col_s3, col_clr = st.columns([1.1, 1.2, 1.2, 1.2, 0.8])

with col_lbl:
    st.html("<div style='padding-top: 8px; font-weight: 700; font-size: 0.82rem; color: #64748b; text-transform: uppercase;'>Quick Samples:</div>")

with col_s1:
    st.button("📋 Genuine Job", help="Load a legitimate engineering internship listing", on_click=set_sample, args=(GENUINE_SAMPLE,), use_container_width=True)

with col_s2:
    st.button("⚠️ Suspicious Offer", help="Load an offer with artificial urgency and pressure", on_click=set_sample, args=(SUSPICIOUS_SAMPLE,), use_container_width=True)

with col_s3:
    st.button("🚨 High-Risk Scam", help="Load a fraudulent posting demanding fees and identity documents", on_click=set_sample, args=(HIGH_RISK_SAMPLE,), use_container_width=True)

with col_clr:
    st.button("🗑️ Clear", help="Clear the input text area and results", on_click=clear_input, use_container_width=True)

# Large text area for job description
user_text = st.text_area(
    label="Job / Internship Description",
    key="job_text_area",
    height=240,
    placeholder="Paste the complete job description here...\n\nExample:\n\"We are hiring a Python Developer Intern. Company: ABC Technologies. Skills: Python, SQL, Machine Learning. Duration: 3 months. Apply through our official careers website.\"",
    label_visibility="collapsed"
)

# Action buttons
btn_col1, btn_col2, btn_spacer = st.columns([1.8, 1, 3])

with btn_col1:
    analyze_clicked = st.button("🔍 Analyze Opportunity", type="primary", use_container_width=True)

with btn_col2:
    st.button("Clear Input", type="secondary", on_click=clear_input, use_container_width=True)

# Execute analysis when triggered
if analyze_clicked:
    current_input = st.session_state.get("job_text_area", "")
    if not current_input.strip():
        st.warning("⚠️ Please paste or type a job description to analyze.")
    else:
        with st.spinner("Analyzing opportunity with dual ML classifier and heuristic rule engine..."):
            try:
                result = analyze_job(current_input)
                st.session_state["analysis_result"] = result
            except Exception as e:
                st.error(f"Analysis encountered an error: {e}")

# ==============================================================================
# SECTION D: Analysis Results Dashboard
# ==============================================================================
result = st.session_state.get("analysis_result")

if result:
    risk_score = result.get("risk_score", 0)
    status = result.get("status", "Low Risk")
    ml_score = result.get("ml_score", 0)
    rule_score = result.get("rule_score", 0)
    reasons = result.get("reasons", [])
    features = result.get("features", {})

    # CSS classes and configuration based on status
    if status == "High Risk":
        status_class = "high-risk"
        status_icon = "🚨"
        status_tagline = "Severe threat detected. Strong indicators of recruitment fraud found in this posting."
        risk_color = "#dc2626"
    elif status == "Suspicious":
        status_class = "suspicious"
        status_icon = "⚠️"
        status_tagline = "Moderate threat level. Questionable patterns or artificial pressure tactics detected."
        risk_color = "#d97706"
    else:
        status_class = "low-risk"
        status_icon = "✅"
        status_tagline = "Minimal suspicious patterns detected. The listing matches standard genuine recruitment formats."
        risk_color = "#059669"

    # Clamp pointer position between 3% and 97% for clean visual representation
    pointer_pos = max(3, min(risk_score, 97))

    # Build warning cards HTML
    category_meta = {
        "payment_request": {
            "title": "Payment & Advance Fee Request",
            "icon": "💳",
            "points": "+35 Risk Points",
            "badge_class": "danger",
            "desc": "Demands payment for registration, training, verification, or application fees. Legitimate employers never charge candidates to apply or work."
        },
        "sensitive_information": {
            "title": "Sensitive Identity & Banking Demands",
            "icon": "🪪",
            "points": "+35 Risk Points",
            "badge_class": "danger",
            "desc": "Requests confidential documents like Aadhaar, PAN card, bank account numbers, or OTPs before formal employment offers."
        },
        "unrealistic_promise": {
            "title": "Unrealistic Promises & Guarantees",
            "icon": "🎁",
            "points": "+20 Risk Points",
            "badge_class": "warning",
            "desc": "Guarantees job selection without an interview or promises instant high income for minimal effort."
        },
        "suspicious_urgency": {
            "title": "Artificial Urgency & Pressure Tactics",
            "icon": "⏳",
            "points": "+10 Risk Points",
            "badge_class": "warning",
            "desc": "Uses high-pressure phrases like 'limited seats' or 'offer expires today' to rush candidates into hasty decisions."
        },
        "ml_detection": {
            "title": "Machine Learning Pattern Detection",
            "icon": "🤖",
            "points": "AI Classifier Signal",
            "badge_class": "info",
            "desc": "The trained machine learning model recognized semantic patterns characteristic of known fraudulent postings."
        }
    }

    warning_cards_html = []
    if reasons:
        for reason in reasons:
            cat_key = reason.get("category", "")
            meta = category_meta.get(cat_key, {
                "title": cat_key.replace("_", " ").title(),
                "icon": "⚠️",
                "points": f"+{reason.get('points')} Points" if reason.get("points") is not None else "Flagged",
                "badge_class": "warning",
                "desc": "Potential warning indicator detected during automated pattern scan."
            })

            badge_text = f"+{reason.get('points')} Points" if reason.get("points") is not None else meta["points"]
            evidence_list = reason.get("evidence", [])
            evidence_pills_html = "".join([f'<span class="evidence-pill">🔍 "{ev}"</span>' for ev in evidence_list])

            warning_cards_html.append(f"""
            <div class="warning-card {cat_key}">
                <div class="warning-top-row">
                    <div class="warning-category-group">
                        <span class="warning-cat-icon">{meta['icon']}</span>
                        <h4 class="warning-cat-title">{meta['title']}</h4>
                    </div>
                    <span class="warning-points-badge {meta['badge_class']}">{badge_text}</span>
                </div>
                <p class="warning-desc">{meta['desc']}</p>
                <div class="evidence-group">
                    <span class="evidence-label">Detected Evidence:</span>
                    {evidence_pills_html}
                </div>
            </div>
            """)
        warnings_section_content = "".join(warning_cards_html)
    else:
        warnings_section_content = """
        <div class="no-warning-card">
            <div class="no-warning-icon">✅</div>
            <div>
                <h4 class="no-warning-title">No Predefined Warning Patterns Detected</h4>
                <p class="no-warning-text">
                    TrustHire AI scanned the posting against all known fraud categories (advance fees, credential harvesting, excessive urgency, and unrealistic guarantees). No matching warning patterns were found.
                </p>
            </div>
        </div>
        """

    # Extracted Metadata
    emails = features.get("emails", [])
    urls = features.get("urls", [])
    emails_display = ", ".join(emails) if emails else "None detected in text"
    urls_display = ", ".join(urls) if urls else "None detected in text"

    # Assemble complete, cohesive Results Card
    full_results_dashboard_html = f"""
    <div class="results-container" id="results">
        <!-- Top Status Banner -->
        <div class="status-banner {status_class}">
            <div class="status-info-group">
                <div class="status-icon-box {status_class}">
                    <span>{status_icon}</span>
                </div>
                <div>
                    <h3 class="status-title {status_class}">{status}</h3>
                    <p class="status-description">{status_tagline}</p>
                </div>
            </div>
            <span class="status-pill-badge {status_class}">
                {status_icon} {status}
            </span>
        </div>

        <!-- Visual Risk Meter -->
        <div class="meter-card">
            <div class="meter-header">
                <div>
                    <span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Visual Threat Assessment</span>
                    <h4 class="meter-title">Comprehensive Risk Meter</h4>
                </div>
                <div>
                    <span class="meter-score-large {status_class}">{risk_score}</span>
                    <span class="meter-score-total">/100</span>
                </div>
            </div>

            <div class="meter-track">
                <div class="meter-segment zone-low" title="Low Risk Zone (0-29)"></div>
                <div class="meter-segment zone-suspicious" title="Suspicious Zone (30-59)"></div>
                <div class="meter-segment zone-high" title="High Risk Zone (60-100)"></div>
                <div class="meter-indicator-pin" style="left: {pointer_pos}%;">
                    <div class="pin-bubble {status_class}">{risk_score}/100</div>
                    <div class="pin-pointer {status_class}"></div>
                </div>
            </div>

            <div class="meter-legend">
                <span>0 • Low Risk Safe (0–29)</span>
                <span>30 • Suspicious Threshold (30–59)</span>
                <span>60 • High Risk Scam Alert (60–100)</span>
            </div>
        </div>

        <!-- Score Breakdown Grid -->
        <div class="metrics-grid">
            <!-- Metric 1: Overall Risk Index -->
            <div class="metric-card-box">
                <div class="metric-header">
                    <span class="metric-label">Overall Risk Index</span>
                    <span class="metric-icon-small">🎯</span>
                </div>
                <div class="metric-value-row">
                    <span class="metric-value-num" style="color: {risk_color};">{risk_score}</span>
                    <span class="metric-value-denom">/ 100</span>
                </div>
                <div class="metric-progress-bg">
                    <div class="metric-progress-fill" style="width: {risk_score}%; background-color: {risk_color};"></div>
                </div>
                <p class="metric-desc">Maximum evaluated index from ML probability and heuristic scoring.</p>
            </div>

            <!-- Metric 2: ML Probability -->
            <div class="metric-card-box">
                <div class="metric-header">
                    <span class="metric-label">Machine Learning Score</span>
                    <span class="metric-icon-small">🤖</span>
                </div>
                <div class="metric-value-row">
                    <span class="metric-value-num" style="color: #2563eb;">{ml_score}</span>
                    <span class="metric-value-denom">/ 100</span>
                </div>
                <div class="metric-progress-bg">
                    <div class="metric-progress-fill" style="width: {ml_score}%; background-color: #2563eb;"></div>
                </div>
                <p class="metric-desc">TF-IDF Vectorizer + Scikit-Learn classifier fraud probability.</p>
            </div>

            <!-- Metric 3: Rule-Based Score -->
            <div class="metric-card-box">
                <div class="metric-header">
                    <span class="metric-label">Rule-Based Score</span>
                    <span class="metric-icon-small">📑</span>
                </div>
                <div class="metric-value-row">
                    <span class="metric-value-num" style="color: #0f172a;">{rule_score}</span>
                    <span class="metric-value-denom">/ 100</span>
                </div>
                <div class="metric-progress-bg">
                    <div class="metric-progress-fill" style="width: {rule_score}%; background-color: #0f172a;"></div>
                </div>
                <p class="metric-desc">Weighted scam heuristic index across known fraud categories.</p>
            </div>
        </div>

        <!-- Detected Warning Signs Section -->
        <div class="warning-signs-section">
            <div class="warning-section-header">
                <div>
                    <span class="section-eyebrow">DETAILED SIGNAL ANALYSIS</span>
                    <h3 class="section-heading" style="font-size: 1.25rem;">Detected Warning Signs</h3>
                </div>
            </div>
            {warnings_section_content}
        </div>

        <!-- Extracted Metadata -->
        <div class="metadata-box">
            <div class="metadata-title">Extracted Posting Metadata</div>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>📧 Extracted Email Addresses</strong>
                    <span>{emails_display}</span>
                </div>
                <div class="metadata-item">
                    <strong>🔗 Extracted Links & Portals</strong>
                    <span>{urls_display}</span>
                </div>
            </div>
        </div>
    </div>
    """

    st.html(full_results_dashboard_html)

# ==============================================================================
# SECTION E: How It Works
# ==============================================================================
st.html("""
<div class="how-it-works-section" id="how-it-works">
    <div class="section-title-center">
        <span class="section-eyebrow">SIMPLE 3-STEP INSPECTION</span>
        <h2 class="section-heading">How TrustHire AI Works</h2>
        <p class="section-subheading">
            A hybrid security workflow built to protect students and career seekers from modern job scams.
        </p>
    </div>

    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num-badge">01</div>
            <div class="step-icon">📝</div>
            <h3 class="step-card-title">1. Paste Job Description</h3>
            <p class="step-card-desc">
                Paste any job or internship posting, offer letter snippet, or recruiter communication into the analyzer.
            </p>
        </div>

        <div class="step-card">
            <div class="step-num-badge">02</div>
            <div class="step-icon">⚙️</div>
            <h3 class="step-card-title">2. AI Scans Patterns</h3>
            <p class="step-card-desc">
                Our dual-engine system evaluates text using heuristic rule matching alongside a trained TF-IDF machine learning model.
            </p>
        </div>

        <div class="step-card">
            <div class="step-num-badge">03</div>
            <div class="step-icon">📊</div>
            <h3 class="step-card-title">3. Review Risk & Warnings</h3>
            <p class="step-card-desc">
                Get an instant 0–100 risk score, risk status tier, category breakdowns, and concrete evidence snippets before applying.
            </p>
        </div>
    </div>
</div>
""")

# ==============================================================================
# SECTION F: Safety Disclaimer
# ==============================================================================
st.html("""
<div class="disclaimer-card" id="safety-advisory">
    <div class="disclaimer-icon">🛡️</div>
    <div>
        <h4 class="disclaimer-title">Safety & Verification Advisory</h4>
        <p class="disclaimer-text">
            This is an automated risk assessment prototype, not a guarantee that an opportunity is genuine or fraudulent. Fraudulent actors continually alter phrasing to evade filters. Always independently verify companies, check official corporate domains and LinkedIn profiles, and never send money or sensitive identification details (such as Aadhaar, PAN, or banking credentials) as part of an application.
        </p>
    </div>
</div>
""")

# ==============================================================================
# SECTION G: Footer
# ==============================================================================
st.html("""
<footer class="footer-container">
    <div class="footer-brand">
        <span style="font-size: 18px;">🛡️</span>
        <div>
            <div class="footer-title">TrustHire AI</div>
            <div class="footer-subtitle">Student-built AI prototype for job and internship fraud prevention</div>
        </div>
    </div>
    <div class="footer-tags">
        <span class="footer-tag">Python</span>
        <span class="footer-tag">Streamlit</span>
        <span class="footer-tag">Scikit-Learn</span>
        <span class="footer-tag">NLP Classification</span>
    </div>
</footer>
""")
