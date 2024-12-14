import sys
import os
from tqdm import tqdm
import yaml
import nltk
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from safetensors.torch import load_file


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.models.lstm import evalEmbeddingDataset, collate_lstm_eval, TweetClassifier
from src.utils.eval_utils import evaluate_model
from src.utils.dataset_utils import import_data, preprocess_text_embed


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

weights_path = config["training_args"]["output_dir"]+"/bestmodel/model.safetensors"
weights = load_file(weights_path)


# Load the dataset
df = import_data("./challenge_data/test_tweets/")

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
data = df.groupby(["ID"], as_index=False).agg({"CleanTweet": lambda x: list(x)[:max_tweets_per_group]})
eval_dataset = evalEmbeddingDataset(data["CleanTweet"].tolist())

collate_fn = lambda batch : collate_lstm_eval(batch, max_n_tweets=max_tweets_per_group, max_n_words=max_n_words, embedding_dim=embedding_dim)


# Charger le modèle entraîné (assurez-vous que le modèle est sauvegardé correctement)
print("Loading model...")
model = TweetClassifier(embedding_dim=embedding_dim, lstm_hidden_dim=lstm_hidden_dim,bidirectional=bidirectional)
model.load_state_dict(weights)
print("Model loaded.")


eval_results = evaluate_model(model, eval_dataset, collate_fn)
data["EventType"] = eval_results
new_data = data[["ID", "EventType"]]

#Save the results
new_data.to_csv("./results/lstm_predictions.csv", index=False)
print("Results saved ")