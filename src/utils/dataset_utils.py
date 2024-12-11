import pandas as pd
from pathlib import Path
import re
import numpy as np
import torch

def import_data(directory):
    data_dir = Path(directory)
    data_files = list(data_dir.glob("*.csv"))

    if not data_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dataframes = [pd.read_csv(file) for file in data_files]
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(data_files)} files.")
    return df


# Preprocess tweets
def preprocess_text_bert(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"http\S+", "[URL]", text)  # Remove URLs
    text = re.sub(r"@\w+", "[USER]", text)  # Remove mentions
    text = re.sub(r"\s+", " ", text)# Remove multiple spaces
    text = re.sub(r"rt", "", text) # Remove retweet indicator
    text = text.strip()

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    return text


# Preprocess tweets and return embeddings
def preprocess_text_embed(text, embeddings_model, embedding_dim, lemmatizer):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)      # Remove hashtags
    text = re.sub(r"http\S+", "", text)   # Remove URLs
    text = re.sub(r"@\w+", "", text)      # Remove mentions
    text = re.sub(r"\brt\b", "", text)    # Remove retweet indicator
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text)      # Remove multiple spaces
    text = text.strip()
    tokens = text.split()   
    # Calcul de l'embedding
    embeddings = []
    for token in tokens:
        token = lemmatizer.lemmatize(token)
        if token in embeddings_model:
            embeddings.append(embeddings_model[token])
    if embeddings == []:
        embeddings = [np.zeros(embedding_dim)]
    return embeddings  # Retourne l'embedding du tweet              # Tokenize by splitting on spaces


# Function to split tweets into quotas
def split_into_quotas(tweets_list, quota_size):
    return [" ".join(tweets_list[i:i + quota_size]) for i in range(0, len(tweets_list), quota_size)]

#Class TweetDataset for Bert
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

