import os
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load and combine all CSV files from the directory
print("Loading data...")
data_dir = Path("./challenge_data/eval_tweets")
data_files = list(data_dir.glob("*.csv"))

if not data_files:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")

dataframes = [pd.read_csv(file) for file in data_files]
df = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(data_files)} files.")

# Preprocess tweets
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = " ".join([word for word in text.split() if word != "rt"])  # Remove "RT"
    return text

print("Preprocessing tweets...")
df["CleanTweet"] = df["Tweet"].apply(preprocess_text)

print("Preprocessing complete.")

# Aggregate tweets by ID and EventType
new_data = df.groupby(["ID"], as_index=False).agg({"CleanTweet": lambda x: " ".join(x)})

# 1. Charger le modèle et le tokenizer sauvegardés
model_path = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Mettre le modèle en mode évaluation
model.eval()

# 3. Prétraiter les tweets (tokenization)
tweets = new_data["CleanTweet"].tolist()
encodings = tokenizer(tweets, truncation=True, padding=True, max_length=512, return_tensors="pt")

# 4. Effectuer les prédictions
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)  # Prédictions des classes

# Ajouter les prédictions dans le dataframe
new_data["EventType"] = predictions.numpy()

# 5. Sauvegarder le dataframe sous forme CSV
output_csv_path = "predictions.csv"
new_data[["ID", "EventType"]].to_csv(output_csv_path, index=False)
print(f"Prédictions sauvegardées dans {output_csv_path}")