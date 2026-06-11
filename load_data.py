"""
load_data.py

Reads the job tracker Excel file and returns a list of cleaned
text documents — one per job description. Each document includes
the company, role, location, and the full JD text so the RAG
system has context when it retrieves chunks.
"""

import pandas as pd

TRACKER_PATH = "../../Getting Hired/[C] Job Tracker.xlsx"


def load_job_documents():
    """Load JDs from the tracker. Returns a list of strings."""
    df = pd.read_excel(TRACKER_PATH)

    # Only keep rows that have an actual job description
    df = df[df["Job Description"].notna()].copy()

    documents = []
    for _, row in df.iterrows():
        company = row.get("Company", "Unknown Company")
        role = row.get("Role Title", "Unknown Role")
        location = row.get("Location", "Unknown Location")
        jd = row.get("Job Description", "")

        # Format as one clean text block
        text = f"""Company: {company}
Role: {role}
Location: {location}

Job Description:
{jd}
"""
        documents.append(text)

    print(f"Loaded {len(documents)} job descriptions.")
    return documents


if __name__ == "__main__":
    docs = load_job_documents()
    # Print a preview of the first document
    print("\n--- Preview of first document ---")
    print(docs[0][:500])
