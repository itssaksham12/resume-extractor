import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from sentence_transformers import SentenceTransformer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
df_ai_resume = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/AI_Resume_Screening.csv')
df_job = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/processed_job_data.csv')
df_updated = pd.read_csv('/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/UpdatedResumeDataSet.csv')

# Preprocessing for merging
df_ai_resume.rename(columns=lambda x: x.strip(), inplace=True)
df_job.rename(columns=lambda x: x.strip(), inplace=True)
df_updated.rename(columns=lambda x: x.strip(), inplace=True)
df_updated.rename(columns={'Category': 'Job Title', 'Resume': 'Resume Text'}, inplace=True)
if 'Unnamed: 0' in df_job.columns:
    df_job.drop(columns=['Unnamed: 0'], inplace=True)

# Lowercase for merging
df_ai_resume['Job Role'] = df_ai_resume['Job Role'].astype(str).str.strip().str.lower()
df_job['Job Title'] = df_job['Job Title'].astype(str).str.strip().str.lower()
df_ai_merged = df_ai_resume.merge(df_job, left_on='Job Role', right_on='Job Title', how='inner')

# Label encoding: 1 for hire, 0 for no hire
df_ai_merged['label'] = df_ai_merged['Recruiter Decision'].apply(
    lambda x: 1 if str(x).strip().lower() in ['hire', 'yes', '1', 'true'] else 0
)

# ------ Advanced Feature Engineering -------

# Skill overlap (Jaccard similarity)
def jaccard_sim(a, b):
    set_a = set(str(a).lower().replace(';', ',').split(','))
    set_b = set(str(b).lower().replace(';', ',').split(','))
    set_a = set(s.strip() for s in set_a if s.strip())
    set_b = set(s.strip() for s in set_b if s.strip())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0

print("Calculating skill Jaccard overlap...")
df_ai_merged['skill_jaccard'] = [
    jaccard_sim(sk, jd) for sk, jd in zip(df_ai_merged['Skills'], df_ai_merged['Job Description'])
]

skills_text = df_ai_merged['Skills'].fillna('').astype(str).tolist()
job_desc_ai = df_ai_merged['Job Description'].fillna('').astype(str).tolist()
education_text = df_ai_merged['Education'].fillna('').astype(str).tolist()
certs_text = df_ai_merged['Certifications'].fillna('').astype(str).tolist()

print("Loading all-mpnet-base-v2 transformer...")
sentence_model = SentenceTransformer('all-mpnet-base-v2')

print("Extracting embeddings for skills...")
emb_skills = sentence_model.encode(skills_text, convert_to_tensor=True, show_progress_bar=True).cpu()
print("Extracting embeddings for JD...")
emb_job_ai = sentence_model.encode(job_desc_ai, convert_to_tensor=True, show_progress_bar=True).cpu()
print("Extracting embeddings for education...")
emb_education = sentence_model.encode(education_text, convert_to_tensor=True, show_progress_bar=True).cpu()
print("Extracting embeddings for certifications...")
emb_certs = sentence_model.encode(certs_text, convert_to_tensor=True, show_progress_bar=True).cpu()

# Semantic similarity calculation
print("Calculating semantic similarity...")
semantic_similarity = [
    float(torch.cosine_similarity(emb_skills[i].unsqueeze(0), emb_job_ai[i].unsqueeze(0)).item())
    for i in range(len(skills_text))
]

# Prepare structured features + new engineered features
structured_cols = ['Experience (Years)', 'Projects Count', 'Salary Expectation ($)']
structured_feats = df_ai_merged[structured_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
structured_feats['skill_jaccard'] = df_ai_merged['skill_jaccard'].replace([np.inf, -np.inf], np.nan).fillna(0)
structured_feats['semantic_similarity'] = pd.Series(semantic_similarity).replace([np.inf, -np.inf], np.nan).fillna(0)

# Normalize features using min-max normalization safely
structured_feats = (structured_feats - structured_feats.min()) / (structured_feats.max() - structured_feats.min())
structured_feats = structured_feats.replace([np.inf, -np.inf], np.nan).fillna(0)
structured_feats = torch.tensor(structured_feats.values, dtype=torch.float32)

# Labels tensor
labels_task3 = torch.tensor(df_ai_merged['label'].values, dtype=torch.float32).unsqueeze(1)

# Concatenate all features into input tensor
X_features = torch.cat([emb_skills, emb_job_ai, emb_education, emb_certs, structured_feats], dim=1).to(device)
Y = labels_task3.to(device)

# Remove rows with any nan or inf in X or Y
print("Before nan removal:", X_features.shape)
mask = (
    ~torch.isnan(X_features).any(dim=1)
    & ~torch.isinf(X_features).any(dim=1)
    & ~torch.isnan(Y).any(dim=1)
    & ~torch.isinf(Y).any(dim=1)
)
X_features_clean = X_features[mask]
Y_clean = Y[mask]
print("After nan removal:", X_features_clean.shape)

# Split into train and validation sets (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_features_clean.cpu(), Y_clean.cpu(), test_size=0.2, random_state=42, stratify=Y_clean.cpu()
)
X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

# Define Neural Network Model
class EnhancedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.activation(x)
        return x

# Instantiate model and optimizer/criterion
model = EnhancedNN(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

def train_eval(model, X_train, y_train, X_val, y_val, epochs=15, patience_limit=4):
    best_auc = 0
    patience = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        # Assert no nan/inf in outputs
        assert not torch.isnan(outputs).any(), "Outputs contain NaN"
        assert not torch.isinf(outputs).any(), "Outputs contain Inf"
        
        print(f"Epoch {epoch+1}: outputs.min={outputs.min().item():.6f}, outputs.max={outputs.max().item():.6f}")
        
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).cpu().numpy()
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_preds)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val AUC: {val_auc:.6f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
        else:
            patience += 1
            if patience > patience_limit:
                print("Early stopping triggered.")
                break
    return best_auc

print("\nStarting model training...")
final_auc = train_eval(model, X_train, y_train, X_val, y_val, epochs=15)
print(f"\nFinal Validation AUC-ROC: {final_auc:.6f}")
print(f"Validation set hire/no-hire counts: Hire={int(y_val.sum().item())}, No Hire={len(y_val)-int(y_val.sum().item())}")
print("\nTraining completed.")
