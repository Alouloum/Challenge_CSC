import pandas as pd
from pathlib import Path
import torch
from torch.nn.functional import softmax

import sys
import os
from tqdm import tqdm
import yaml
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.dataset_utils import import_data, preprocess_text_bert, split_into_quotas




# Load the configuration file
with open("configs/roberta512.yaml", "r") as file:
    config = yaml.safe_load(file)

model_config = config['model']

max_length = model_config['max_length']
group_size = model_config['group_size']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = config['training_args']['output_dir']+"/bestmodel"

# Load and preprocess data
df = import_data("./challenge_data/eval_tweets/")
print("Preprocessing tweets...")
tqdm.pandas()
df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text_bert)
grouped = df.groupby(['ID'])['CleanTweet'].progress_apply(lambda x: list(x)).reset_index()
grouped['TweetQuotas'] = grouped['CleanTweet'].apply(lambda tweets: split_into_quotas(tweets, group_size)) # Split the tweets into quotas
new_data = grouped.explode('TweetQuotas').reset_index(drop=True) # Explode the quotas to have one quota per row
tweets = new_data["TweetQuotas"].tolist()
print("Preprocessing complete.")


model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Predictions
print("Predicting...")
predictions = []

for tweet in tqdm(tweets, desc="Predicting proabilities"):
    encodings = tokenizer(tweet, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        probabilities = softmax(outputs.logits, dim=1)  # Convertir logits to probabilities
    predictions.append(probabilities.cpu().numpy().squeeze())  # Save probabilities

# Convert probabilities to DataFrame
probabilities_df = pd.DataFrame(
    {
        "ID": new_data["ID"],  
        "Probabilities": predictions,  
    }
)

# We will now aggregate the probabilities by ID and take the average of the probabilities to get the final prediction
def average_probabilities(group):
    prob_array = torch.tensor(list(group["Probabilities"].values))  # Transformer en tenseur
    mean_probs = torch.mean(prob_array, dim=0)  # Moyenne des probabilités
    predicted_class = torch.argmax(mean_probs).item()  # Classe finale par probabilité max
    return predicted_class

# Average probabilities
new_data = probabilities_df.groupby("ID").apply(average_probabilities).reset_index()
new_data.columns = ["ID", "EventType"]  

# Save the predictions
new_data.to_csv("./results/bert_predictions.csv", index=False)
print("Predictions saved to results/bert_predictions.csv")

