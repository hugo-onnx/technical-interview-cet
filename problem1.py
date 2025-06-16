import pandas as pd
from transformers import pipeline

# Load job postings
df = pd.read_csv('data/job_postings.csv')

# Load publicly available QA model
qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert-base-cased-distilled-squad"
)

# Function to extract fields using QA
def extract_info_from_description(description: str) -> dict:
    try:
        job_title = qa_pipeline(question="What is the job title?", context=description)['answer']
        location = qa_pipeline(question="Where is the job located?", context=description)['answer']
        salary = qa_pipeline(question="What is the salary range?", context=description)['answer']
        return {
            "job_title": job_title,
            "location": location,
            "salary_range": salary
        }
    except Exception as e:
        return {"error": str(e)}

# Apply to a few rows
results = df.apply(
    lambda row: {
        "posting_date": row["posting_date"],
        "company_name": row["company_name"],
        "extracted_info": extract_info_from_description(row["job_description"])
    }, axis=1
)

# Output as JSON
final_df = pd.DataFrame(results.tolist())
print(final_df.to_json(orient="records", indent=2))
