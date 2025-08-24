import pandas as pd
import re
import nltk

# 1. Load data
df = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/job_title_des.csv')

# 2. Clean data
df = df.drop_duplicates().dropna(subset=['Job Title', 'Job Description'])

# 3. Normalize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special chars/numbers
    text = re.sub('\s+', ' ', text).strip()
    return text

df['Job Title'] = df['Job Title'].apply(clean_text)
df['Job Description'] = df['Job Description'].apply(clean_text)

# 4. Filter out rows with Job Description less than 100 characters
df = df[df['Job Description'].str.len() >= 100]

# 5. Export the processed dataset
df.to_csv('processed_job_data.csv', index=False)
print(f"Processed dataset exported successfully!")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
