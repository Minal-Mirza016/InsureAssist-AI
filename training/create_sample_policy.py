
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from xml.sax.saxutils import escape


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "data" / "documents" / "sample_insurance_policy.pdf"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

pages = [
    {
        "title": "Sample Insurance Policy Guide",
        "sections": [
            (
                "1. Introduction",
                "This fictional guide demonstrates how an insurance "
                "document assistant retrieves information. It is "
                "not an actual insurance contract or a company's policy."
            ),
            (
                "2. Example Coverage",
                "Under this fictional example, eligible policyholders "
                "may request reimbursement for listed eligible medical "
                "services, subject to their selected plan, benefit limits, "
                "waiting periods, exclusions, and required documentation."
            ),
            (
                "3. Coverage Limits",
                "Coverage amounts and annual limits depend on the plan "
                "schedule. A user should check the schedule for the "
                "applicable benefit limit before assuming an expense "
                "will be reimbursed."
            ),
        ],
    },
    {
        "title": "Claims and Required Documents",
        "sections": [
            (
                "4. Claim Submission",
                "To request reimbursement in this fictional example, "
                "the policyholder completes a claim form and submits it "
                "through the designated claims channel. The insurer "
                "reviews the submitted information before making a decision."
            ),
            (
                "5. Required Claim Documents",
                "A sample claim submission may require a completed claim "
                "form, proof of identity, original itemized bills or "
                "receipts, and relevant medical reports. Additional "
                "documents may be requested depending on the claim type."
            ),
            (
                "6. Claim Status",
                "A policyholder may use the insurer's designated service "
                "channel to ask about claim progress. A status enquiry "
                "should include the appropriate claim reference and "
                "follow the insurer's identity-verification process."
            ),
        ],
    },
    {
        "title": "Premiums, Exclusions and Renewal",
        "sections": [
            (
                "7. Premium Payments",
                "Premium amount, due dates, accepted payment methods, "
                "and payment frequency are determined by the applicable "
                "plan terms and payment instructions."
            ),
            (
                "8. Exclusions and Waiting Periods",
                "Some services or circumstances may be excluded or "
                "subject to waiting periods. The applicable policy "
                "wording and schedule must be consulted to determine "
                "whether a particular expense is covered."
            ),
            (
                "9. Eligibility",
                "Eligibility depends on the requirements specified "
                "in the applicable plan. Age limits, applicant conditions, "
                "and other restrictions must be verified against the "
                "relevant policy documentation."
            ),
            (
                "10. Policy Renewal",
                "Renewal depends on the plan's renewal conditions, "
                "payment requirements, and any applicable deadlines. "
                "Policyholders should review their renewal instructions "
                "before the policy expires."
            ),
        ],
    },
]


styles = getSampleStyleSheet()
story = []

for page_index, page in enumerate(pages):
    story.append(Paragraph(escape(page["title"]), styles["Title"]))
    story.append(Spacer(1, 18))

    for heading, content in page["sections"]:
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(escape(content), styles["BodyText"]))
        story.append(Spacer(1, 14))

    if page_index < len(pages) - 1:
        story.append(PageBreak())


document = SimpleDocTemplate(
    str(OUTPUT_PATH),
    pagesize=A4,
    title="Fictional Sample Insurance Policy Guide",
)

document.build(story)

print(f"Created sample PDF: {OUTPUT_PATH}")
