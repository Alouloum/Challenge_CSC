import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import gensim.downloader as api
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments


from lstm import TweetClassifier


MAX_TWEETS_PER_GROUP = 2000
model_path = "./results/model_stropwords/checkpoint-1000"
output_csv_path = "lstm_pred.csv"


lstm_hidden_dim = 64
max_n_tweets = MAX_TWEETS_PER_GROUP  # Nombre maximum de tweets par groupe
max_n_words = 20   # Nombre maximum de mots par tweet
embedding_dim = 200  # Dimension des embeddings
num_classes = 2  # Nombre de classes pour les labels


print("Loading model...")
print("model loading...")
embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
print("model loaded.")
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
    text = re.sub(r"#\w+", "", text)      # Remove hashtags
    text = re.sub(r"http\S+", "", text)   # Remove URLs
    text = re.sub(r"@\w+", "", text)      # Remove mentions
    text = re.sub(r"\brt\b", "", text)    # Remove retweet indicator
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text)      # Remove multiple spaces
    text = text.strip()                   # Remove leading/trailing spaces
    tokens = text.split()   
    # Calcul de l'embedding
    embeddings = []
    for token in tokens:
        if token in embeddings_model:
            embeddings.append(embeddings_model[token])
    if embeddings == []:
        embeddings = [np.zeros(embedding_dim)]
    return embeddings  # Retourne l'embedding du tweet              # Tokenize by splitting on spaces

print("Preprocessing tweets...")
tqdm.pandas()
df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text)

print("Preprocessing complete.")


data = df.groupby(["ID"], as_index=False).agg({"CleanTweet": lambda x: list(x)[:MAX_TWEETS_PER_GROUP]})


class evalEmbeddingDataset(Dataset):
    def __init__(self, tweets_embeddings_list):
        self.tweets_embeddings_list = tweets_embeddings_list

    def __len__(self):
        return len(self.tweets_embeddings_list)

    def __getitem__(self, idx):
        tweets_embeddings = self.tweets_embeddings_list[idx]
        tweets_embeddings = [torch.tensor(embedding, dtype=torch.float) for embedding in tweets_embeddings]
        result = {
            'input_ids': tweets_embeddings,
            'n_tweets': torch.tensor(len(tweets_embeddings), dtype=torch.long),
            'n_words': torch.tensor([te.size(0) for te in tweets_embeddings], dtype=torch.long),
        }
        return result

eval_dataset = evalEmbeddingDataset(data["CleanTweet"].tolist())

def collate_fn(batch,max_n_tweets=max_n_tweets, max_n_words=max_n_words,embedding_dim=embedding_dim):
    batch_size = len(batch)
    input_ids = torch.zeros((batch_size, max_n_tweets, max_n_words,embedding_dim), dtype=torch.float)
    n_tweets = torch.zeros((batch_size,), dtype=torch.long)
    n_words = torch.zeros((batch_size, max_n_tweets), dtype=torch.long)

    for i,sample in enumerate(batch):
        n_tweets[i] = min(sample['n_tweets'].item(), max_n_tweets)
        for j,tweet in enumerate(sample['input_ids'][:n_tweets[i]]):
            n_words[i,j] = min(sample['n_words'][j].item(), max_n_words)

            input_ids[i, j, :n_words[i, j],:] = tweet[:n_words[i, j]]


    
    return {
        'input_ids': input_ids,  # List of lists of Tensors
        'n_tweets': n_tweets,  # Tensor of tweet counts
        'n_words': n_words,  # Tensor of word counts
    }


def evaluate_model(model, eval_dataset, collate_fn, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            n_tweets = batch['n_tweets'].to(device)
            n_words = batch['n_words'].to(device)

            outputs = model(input_ids=input_ids, n_tweets=n_tweets, n_words=n_words)
            probs = outputs['logits'].cpu().numpy()

            # Convert probabilities to binary predictions
            preds = (probs >= 0.5).astype(int)

            # Retrieve original IDs and labels (you may need to adjust this if IDs are not in eval_dataset)
            
            results.extend(preds)

    return results

# Charger le modèle entraîné (assurez-vous que le modèle est sauvegardé correctement)
trained_model = TweetClassifier(embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim,bidirectional=False)
trained_model.load_state_dict(torch.load("path_to_trained_model.pth"))

# Évaluer le modèle
eval_results = evaluate_model(trained_model, eval_dataset, collate_fn)

data["EventType"] = eval_results

# Sauvegarder les résultats sous forme de fichier CSV
data.to_csv(output_csv_path, index=False)
print("Results saved to evaluation_results.csv")
