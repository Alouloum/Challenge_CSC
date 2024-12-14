import sys
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.train_utils import compute_metrics_bert
from src.utils.dataset_utils import import_data, preprocess_text_bert, split_into_quotas, TweetDataset


# Load the configuration file
with open("configs/roberta512.yaml", "r") as file:
    config = yaml.safe_load(file)

model_config = config['model']

max_length = model_config['max_length']
group_size = model_config['group_size']


# Load the dataset
df = import_data("./challenge_data/train_tweets/")

print("Preprocessing tweets...")
tqdm.pandas()
df["CleanTweet"] = df["Tweet"].progress_apply(preprocess_text_bert)
grouped = df.groupby(['ID', 'EventType'])['CleanTweet'].progress_apply(lambda x: list(x)).reset_index()
grouped['TweetQuotas'] = grouped['CleanTweet'].apply(lambda tweets: split_into_quotas(tweets, group_size)) # Split the tweets into quotas
tweets = grouped.explode('TweetQuotas').reset_index(drop=True) # Explode the quotas to have one quota per row
print("Preprocessing complete.")


#Separate into train and test sets
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    tweets["TweetQuotas"].tolist(),
    tweets["EventType"].tolist(),
     test_size=0.2,
    random_state=42
 )


print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_config['model_name'], num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
print("Model loaded.")


print("Tokenising tweets...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=max_length)
print("Tokenisation complete.")

train_dataset = TweetDataset(train_encodings, train_labels)
eval_dataset = TweetDataset(eval_encodings, eval_labels)



training_args = TrainingArguments(**config['training_args'])
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_bert
)

print("Entraînement du modèle...")
trainer.train(resume_from_checkpoint=False)
print("Entraînement terminé.")

# Sauvegarder le modèle
model_path = config["training_args"]["output_dir"]+"/bestmodel"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Modèle sauvegardé.")
