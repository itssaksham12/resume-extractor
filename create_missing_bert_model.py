#!/usr/bin/env python3
"""
Create a placeholder BERT skills model if missing
"""

import torch
import torch.nn as nn
from pathlib import Path

class BERTSkillsModelPlaceholder(nn.Module):
    """Placeholder BERT model for skills extraction"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_labels=100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8), 
            num_layers=6
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)  # Simple pooling
        output = self.classifier(self.dropout(pooled))
        return output

def create_bert_skills_model():
    """Create and save a placeholder BERT skills model"""
    
    model_path = Path("bert_skills_model.pth")
    
    if model_path.exists():
        print(f"✅ {model_path} already exists")
        return
    
    print("🔧 Creating placeholder BERT skills model...")
    
    try:
        # Create model
        model = BERTSkillsModelPlaceholder()
        
        # Create sample state dict with proper structure
        state_dict = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': 30522,
                'hidden_size': 768,
                'num_labels': 100,
                'model_type': 'bert_skills_placeholder'
            },
            'version': '1.0',
            'created_for': 'resume_extractor'
        }
        
        # Save model
        torch.save(state_dict, model_path)
        
        # Verify file size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Created {model_path} ({size_mb:.1f} MB)")
        
        # Test loading
        loaded = torch.load(model_path, map_location='cpu')
        print("✅ Model loading test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create BERT skills model: {e}")
        return False

if __name__ == "__main__":
    create_bert_skills_model()
