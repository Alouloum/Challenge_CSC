import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import LongformerForSequenceClassification, LongformerTokenizer
from nltk.corpus import stopwords

max_length = 4096
model_path  = "./fine_tuned_model"
log_dir = "./logs/model_stropwords"
results_dir = "./results/model_stropwords"

# Load and combine all CSV files from the directory
print("Loading data...")
data_dir = Path("./challenge_data/train_tweets")
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
    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)
     # Supprimer les espaces en début et fin
    text = text.strip()

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.split()
    stopwords = set(stopwords.words('english'))

    words = [word for word in words if (word not in stopwords) or (word != "rt")]
    return text

print("Preprocessing tweets...")
df["CleanTweet"] = df["Tweet"].apply(preprocess_text)

print("Preprocessing complete.")

# Aggregate tweets by ID and EventType
tweets = df.groupby(["ID", "EventType"], as_index=False).agg({"CleanTweet": lambda x: " ".join(x)})
# n_tweets = 10
# tweets = df.groupby("ID", group_keys=False).apply(lambda x: x.sample(n=min(len(x), n_tweets)))
# Separate into train and test sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    tweets["CleanTweet"].tolist(),
    tweets["EventType"].tolist(),
    test_size=0.2,
    random_state=42
)

# Charger le modèle
print("Chargement du modèle...")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "cardiffnlp/twitter-roberta-base", 
#     num_labels=len(set(train_labels))  # Définir le nombre de classes
# )
# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
print("Modèle chargé.")

# 

# Tokeniser les données d'entraînement et de validation
print("Tokenisation des tweets...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=max_length)
print("Tokenisation terminée.")

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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Convertir les labels en entiers (si nécessaire)
train_labels = [int(label) for label in train_labels]
eval_labels = [int(label) for label in eval_labels]

# Créer les datasets
train_dataset = TweetDataset(train_encodings, train_labels)
eval_dataset = TweetDataset(eval_encodings, eval_labels)

training_args = TrainingArguments(
    output_dir=results_dir,           # Répertoire pour sauvegarder le modèle
    eval_strategy="epoch",     # Évaluer après chaque époque
    learning_rate=2e-5,              # Taux d'apprentissage
    per_device_train_batch_size=1,  # Taille des lots pour l'entraînement
    per_device_eval_batch_size=1,   # Taille des lots pour l'évaluation
    gradient_accumulation_steps=16, # Accumuler les gradients


    num_train_epochs=10,              # Nombre d'époques
    weight_decay=0.01,               # Régularisation L2
    save_total_limit=2,              # Limiter les checkpoints sauvegardés
    logging_dir=log_dir,            # Répertoire pour les journaux
    logging_steps=10,                # Enregistrer toutes les 10 étapes
    save_steps=1000,                      # Sauvegarder le modèle toutes les 50 étapes  
)


# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Entraînement du modèle...")
trainer.train()
print("Entraînement terminé.")

# Sauvegarder le modèle
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Modèle sauvegardé.")