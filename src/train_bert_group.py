import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import LongformerForSequenceClassification, LongformerTokenizer

from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords


max_length = 512
group_size = 50
model_path  = "./model_bert_avg_groupn_size_"+str(group_size)+"max_length_"+str(max_length)
log_dir = "./logs/model_bert_avg_groupn_size_"+str(group_size)+"max_length_"+str(max_length)
results_dir = "./results/model_bert_avg_groupn_size_"+str(group_size)+"max_length_"+str(max_length)





# Load and combine all CSV files from the directory
print("Loading data...")
data_dir = Path("./challenge_data/train_tweets")
data_files = list(data_dir.glob("*.csv"))

if not data_files:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")

dataframes = [pd.read_csv(file) for file in data_files]
df = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(df)} rows from {len(data_files)} files.")




## Preprocess tweets

# Preprocess tweets
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"http\S+", "[URL]", text)  # Remove URLs
    text = re.sub(r"@\w+", "[USER]", text)  # Remove mentions
    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"rt", "", text)
     # Supprimer les espaces en début et fin
    text = text.strip()

    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    # text = text.split()
    # stop_words = set(stopwords.words('english'))

    # text = " ".join([lemmatizer.lemmatize(word)for word in text if (word not in stop_words) or (word != "rt")])
    return text

print("Preprocessing tweets...")
tqdm.pandas()
df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text)

grouped = df.groupby(['ID', 'EventType'])['CleanTweet'].progress_apply(lambda x: list(x)).reset_index()

# Function to split tweets into quotas
def split_into_quotas(tweets_list, quota_size):
    return [" ".join(tweets_list[i:i + quota_size]) for i in range(0, len(tweets_list), quota_size)]

# Apply the function to create quotas
grouped['TweetQuotas'] = grouped['CleanTweet'].apply(lambda tweets: split_into_quotas(tweets, group_size))

# Explode the quotas to have one quota per row
tweets = grouped.explode('TweetQuotas').reset_index(drop=True)

# print(grouped.head())

# print(grouped['TweetQuotas'][0])
print("Preprocessing complete.")

#Separate into train and test sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    tweets["TweetQuotas"].tolist(),
    tweets["EventType"].tolist(),
     test_size=0.2,
    random_state=42
 )


# Charger le modèle
print("Chargement du modèle...")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base", 
    num_labels=len(set(train_labels))  # Définir le nombre de classes
)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
# model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

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
    per_device_train_batch_size=4,  # Taille des lots pour l'entraînement
    per_device_eval_batch_size=1,   # Taille des lots pour l'évaluation
    gradient_accumulation_steps=4, # Accumuler les gradients


    num_train_epochs=7,              # Nombre d'époques
    weight_decay=0.01,               # Régularisation L2
                 # Limiter les checkpoints sauvegardés
    logging_dir=log_dir,            # Répertoire pour les journaux
    logging_steps=100,                # Enregistrer toutes les 10 étapes
    save_steps=2500,            # Sauvegarder le modèle à chaque époque
    save_total_limit=1,  # Utiliser la perte d'évaluation pour déterminer le meilleur modèle
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
checkpoint_path = "/users/eleves-b/2022/mohamed.aloulou/Desktop/Challenge_CSC/results/model_bert_avg_groupn_size_50max_length_512/checkpoint-16000"
trainer.train(resume_from_checkpoint=True)
print("Entraînement terminé.")


# Sauvegarder le modèle
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Modèle sauvegardé.")


