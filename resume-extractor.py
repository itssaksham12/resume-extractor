import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
import numpy as np

# ---- Load and Parse New Dataset ----
# Try this first in your script:
file_path = '/Users/sakshamfaujdar/Developer/NGIL/resume_reviewer/resume_samples.txt'
with open(file_path, 'r', encoding='latin1') as f:
    lines = f.read().strip().split('\n')

ids, labels, texts = [], [], []
for line in lines:
    parts = line.split(':::')
    if len(parts) == 3:
        ids.append(parts[0].strip())
        labels.append([lab.strip() for lab in parts[1].split(';') if lab.strip()])
        texts.append(parts[2].strip())



ids, labels, texts = [], [], []
for line in lines:
    parts = line.split(':::')
    if len(parts) == 3:
        ids.append(parts[0].strip())
        labels.append([lab.strip() for lab in parts[1].split(';') if lab.strip()])
        texts.append(parts[2].strip())

df = pd.DataFrame({'id': ids, 'labels': labels, 'text': texts})

# ---- Prepare Multi-labels ----
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['labels'])

# ---- Sentence Embeddings ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentence_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = sentence_model.encode(df['text'].tolist(), convert_to_tensor=True)
X = embeddings.cpu()
Y = torch.tensor(Y, dtype=torch.float32)

# ---- Split ----
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y.numpy())

# ---- Model ----
class MultiLabelNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = MultiLabelNN(X_train.shape[1], Y.shape[1]).to(device)
X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

def train_eval(model, X_train, y_train, X_val, y_val, epochs=15, patience_limit=4):
    best_auc, patience = 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).cpu().numpy()
            aucs = []
            for i in range(y_val.shape[1]):
                try:
                    auc = roc_auc_score(y_val.cpu().numpy()[:, i], val_preds[:, i])
                    aucs.append(auc)
                except:
                    pass
            mean_auc = np.mean(aucs) if aucs else float('nan')
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val Macro AUC: {mean_auc:.6f}')
        if mean_auc > best_auc:
            best_auc = mean_auc
            patience = 0
        else:
            patience += 1
            if patience > patience_limit:
                print('Early stopping triggered.')
                break
    return best_auc

print('\nStarting model training...')
final_auc = train_eval(model, X_train, y_train, X_val, y_val, epochs=15)
print(f'\nFinal Validation Macro AUC: {final_auc:.4f}')
