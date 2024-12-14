import sys
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from torch import nn
from transformers import TrainingArguments, Trainer
import nltk
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.models.lstm import TweetClassifier, PrecomputedEmbeddingDataset, collate_lstm
from utils.train_utils import compute_metrics_lstm
from utils.dataset_utils import import_data, preprocess_text_embed


# Load the configuration file
with open("configs/lstmbi32.yaml", "r") as file:
    config = yaml.safe_load(file)

model_config = config['model']

lstm_hidden_dim = model_config['lstm_hidden_dim']
bidirectional = model_config['bidirectional']
embedding_dim = model_config['embedding_dim']
num_classes = model_config['num_classes']
max_tweets_per_group = model_config['max_tweets_per_group']
max_n_words = model_config['max_n_words']



if __name__ == "__main__":
    # Load the dataset
    df = import_data("./challenge_data/train_tweets/")
    
    print("glove loading...")
    embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
    print("glove loaded.")
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()                 # Remove leading/trailing spaces
    embedding_dim = 200
    
    print("Preprocessing tweets...")
    tqdm.pandas()
    preprocess_text_lstm = lambda x: preprocess_text_embed(x, embeddings_model, embedding_dim, lemmatizer)
    df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text_lstm)
    print("Preprocessing complete.")

    # Aggregate tweets by ID and EventType
    tweets = df.groupby(["ID", "EventType"], as_index=False).agg({"CleanTweet": lambda x: list(x)[:max_tweets_per_group]})

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        tweets["CleanTweet"].tolist(),
        tweets["EventType"].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    train_dataset = PrecomputedEmbeddingDataset(train_texts, train_labels)
    eval_dataset = PrecomputedEmbeddingDataset(eval_texts, eval_labels)

    model = TweetClassifier(embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim,bidirectional=bidirectional)

    criterion = nn.BCELoss()
    training_args = TrainingArguments(**config["training_args"])
    collate_fn = lambda batch: collate_lstm(batch, max_tweets_per_group, max_n_words, embedding_dim)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics_lstm
    )

    trainer.train()
    print("Training complete.")
    results_dir = config["training_args"]["output_dir"]
    trainer.save_model(results_dir+'/bestmodel')
    print("Model saved.")
    