import os
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from tqdm import tqdm
from nltk.corpus import stopwords
from torch.nn.functional import softmax

max_length = 4096
device = torch.device("cuda")

load = False
model_path = "/users/eleves-b/2022/mohamed.aloulou/Desktop/Challenge_CSC/results/model_stropwords/checkpoint-1060"
output_csv_path = "bert" + str(max_length)+"p_avg_predictions.csv"

# Load and combine all CSV files from the directory

print("Loading data...")
data_dir = Path("./challenge_data/eval_tweets")
data_files = list(data_dir.glob("*.csv"))

if not data_files:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")

dataframes = [pd.read_csv(file) for file in data_files]
df = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(data_files)} files.")
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"rt", "", text)
     # Supprimer les espaces en début et fin
    text = text.strip()

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    # text = text.split()
    stop_words = set(stopwords.words('english'))

    text = " ".join([word for word in text if (word not in stop_words)])
    return text

print("Preprocessing tweets...")
tqdm.pandas()
df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text)

group_size = 400

grouped = df.groupby(['ID'])['CleanTweet'].progress_apply(lambda x: list(x)).reset_index()

# Function to split tweets into quotas
def split_into_quotas(tweets_list, quota_size):
    return [" ".join(tweets_list[i:i + quota_size]) for i in range(0, len(tweets_list), quota_size)]

# Apply the function to create quotas
grouped['TweetQuotas'] = grouped['CleanTweet'].apply(lambda tweets: split_into_quotas(tweets, group_size))

# Explode the quotas to have one quota per row
new_data = grouped.explode('TweetQuotas').reset_index(drop=True)


# 1. Charger le modèle et le tokenizer sauvegardés

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Mettre le modèle en mode évaluation
model.eval()

# 3. Prétraiter les tweets (tokenization)
tweets = new_data["TweetQuotas"].tolist()
#encodings = tokenizer(tweets, truncation=True, padding=True, max_length=max_length, return_tensors="pt")#.to(device)
print(tweets[0])
# 4. Effectuer les prédictions

print("Predicting...")
predictions = []

predictions = []

# Traitement des tweets
for tweet in tqdm(tweets, desc="Processing Tweets"):
    encodings = tokenizer(tweet, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        probabilities = softmax(outputs.logits, dim=1)  # Convertir les logits en probabilités
    predictions.append(probabilities.cpu().numpy())  # Sauvegarder les probabilités

# Convertir les probabilités en DataFrame
probabilities_df = pd.DataFrame(
    {
        "ID": new_data["ID"],  # Associer chaque tweet à son ID
        "Probabilities": predictions,  # Liste des probabilités
    }
)

# 4. Fusionner les probabilités par ID en calculant la moyenne
# On suppose que chaque ID peut avoir plusieurs tweets (groupage par ID)
def average_probabilities(group):
    prob_array = torch.tensor(list(group["Probabilities"].values))  # Transformer en tenseur
    mean_probs = torch.mean(prob_array, dim=0)  # Moyenne des probabilités
    predicted_class = torch.argmax(mean_probs).item()  # Classe finale par probabilité max
    return predicted_class

# Appliquer la fonction sur chaque groupe
new_data = probabilities_df.groupby("ID").apply(average_probabilities).reset_index()
new_data.columns = ["ID", "EventType"]  # Renommer les colonnes

# 5. Sauvegarder le DataFrame sous forme CSV
new_data.to_csv(output_csv_path, index=False)
print(f"Prédictions sauvegardées dans {output_csv_path}")

# for tweet in tqdm(tweets, desc="Processing Tweets"):
#     encodings = tokenizer(tweet, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**encodings)
#         prediction = torch.argmax(outputs.logits, dim=1).item()  # Prédictions des classes
#     predictions.append(prediction)


# print("Predictions complete.")
# # Ajouter les prédictions dans le dataframe
# new_data["EventType"] = predictions

# # 4. Fusionner les prédictions par ID en utilisant la valeur majoritaire de 'EventType'
# new_data = new_data.groupby("ID", as_index=False).agg({'EventType': lambda x: x.value_counts().idxmax()})

# # 5. Sauvegarder le DataFrame sous forme CSV
# new_data[["ID", "EventType"]].to_csv(output_csv_path, index=False)
# print(f"Prédictions sauvegardées dans {output_csv_path}")