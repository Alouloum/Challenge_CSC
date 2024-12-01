import os
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from tqdm import tqdm



max_length = 4096
device = torch.device("cuda")
save = True
load = False
model_path = "./results/model_stropwords/checkpoint-1000"
output_csv_path = "all_predictions.csv"

# Load and combine all CSV files from the directory
if not(load):
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


if save:
    new_data.to_csv("./data/preprocessed.csv", index=False)
    print("Preprocessed data saved in preprocessed.csv")
if load:
    new_data = pd.read_csv("./data/preprocessed.csv")
    print("Data loaded from preprocessed.csv")

# 1. Charger le modèle et le tokenizer sauvegardés

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Mettre le modèle en mode évaluation
model.eval()

# 3. Prétraiter les tweets (tokenization)
tweets = new_data["CleanTweet"].tolist()
#encodings = tokenizer(tweets, truncation=True, padding=True, max_length=max_length, return_tensors="pt")#.to(device)

# 4. Effectuer les prédictions

print("Predicting...")
predictions = []
for tweet in tqdm(tweets, desc="Processing Tweets"):
    encodings = tokenizer(tweet, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        prediction = torch.argmax(outputs.logits, dim=1).item()  # Prédictions des classes
    predictions.append(prediction)
# with torch.no_grad():
#     outputs = model(**encodings)
#     predictions = torch.argmax(outputs.logits, dim=1)  # Prédictions des classes

print("Predictions complete.")
# Ajouter les prédictions dans le dataframe
new_data["EventType"] = predictions

# 5. Sauvegarder le dataframe sous forme CSV

new_data[["ID", "EventType"]].to_csv(output_csv_path, index=False)
print(f"Prédictions sauvegardées dans {output_csv_path}")