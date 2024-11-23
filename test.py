import os
import re
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torch import nn

# Préparation des modèles NLP (BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Fonction pour générer les embeddings BERT
def get_bert_embedding(tweet, tokenizer, model, max_length=128):
    inputs = tokenizer(tweet, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Moyenne des tokens pour une représentation dense

# Prétraitement basique des tweets
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Retirer ponctuation
    text = re.sub(r'\d+', '', text)      # Retirer les chiffres
    return text

# Charger et traiter les données d'entraînement
train_data = []
for filename in os.listdir("train_tweets"):
    df = pd.read_json(os.path.join("challenge_data","train_tweets", filename), lines=True)
    train_data.append(df)
train_df = pd.concat(train_data, ignore_index=True)

# Prétraitement des tweets
train_df['Tweet'] = train_df['Tweet'].apply(preprocess_text)

# Générer les embeddings BERT pour chaque tweet
embeddings = np.vstack([get_bert_embedding(tweet, tokenizer, bert_model) for tweet in train_df['Tweet']])
labels = train_df['EventType'].values

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

# Définir un réseau neuronal dense pour la classification
class DenseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DenseClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

# Évaluation du modèle
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings).squeeze()
            preds = (outputs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# Créer un Dataset pour PyTorch
class TweetDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Charger les données dans DataLoader
train_dataset = TweetDataset(X_train, y_train)
test_dataset = TweetDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialiser le modèle, l'optimiseur et la fonction de coût
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseClassifier(input_dim=768).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Entraîner le modèle
for epoch in range(10):  # 10 époques
    train_model(model, train_loader, criterion, optimizer, device)
    acc = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch + 1}, Test Accuracy: {acc:.4f}")

# Préparer la soumission pour Kaggle
eval_data = []
for filename in os.listdir("eval_tweets"):
    eval_df = pd.read_json(os.path.join("eval_tweets", filename), lines=True)
    eval_df['Tweet'] = eval_df['Tweet'].apply(preprocess_text)
    eval_embeddings = np.vstack([get_bert_embedding(tweet, tokenizer, bert_model) for tweet in eval_df['Tweet']])
    eval_preds = model(torch.tensor(eval_embeddings, dtype=torch.float32).to(device)).squeeze().detach().cpu().numpy()
    eval_preds = (eval_preds > 0.5).astype(int)
    eval_df['EventType'] = eval_preds
    eval_data.append(eval_df[['ID', 'EventType']])

submission_df = pd.concat(eval_data)
submission_df.to_csv("bert_predictions.csv", index=False)
