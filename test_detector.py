
from detector.analyzer import analyze_job


genuine_job = """
We are hiring a Python Developer Intern.

Company: ABC Technologies

Skills: Python, SQL, Machine Learning.

Duration: 3 months.

Apply through our official careers website.
"""


suspicious_job = """
Congratulations! You are selected for an AI internship.

Guaranteed selection without interview.

Pay a registration fee of Rs. 1500.

Limited seats available. Apply immediately.

Send your Aadhaar and bank details.
"""


print("\n========== GENUINE JOB ==========")

print(analyze_job(genuine_job))


print("\n========== SUSPICIOUS JOB ==========")

print(analyze_job(suspicious_job))

from pprint import pprint

pprint(analyze_job(genuine_job))
pprint(analyze_job(suspicious_job))