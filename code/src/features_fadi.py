import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from src.utils.features_utils import (
    count_repeated_letters,
    count_punctuation,
    count_uppercase_letters,
    weighted_average,
    calculate_total_football_terms,
)

# Directories
train_input_dir = "challenge_data/train_tweets"
eval_input_dir = "challenge_data/eval_tweets"
prob_train_file = "results/train_bert_probabilities.csv"
prob_eval_file = "results/eval_bert_probabilities.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Process probabilistic features
def process_probabilities(prob_file):
    probs = pd.read_csv(prob_file)
    probs['PeriodID'] = probs['ID'].apply(lambda x: int(x.split('_')[1]))
    probs['MatchID'] = probs['ID'].apply(lambda x: int(x.split('_')[0]))
    probs['Probabilities'] = probs['Probabilities'].str.replace(r'\s+', ',', regex=True)
    prob_features = probs.groupby(['ID', 'MatchID', 'PeriodID'])['Probabilities'].apply(lambda x: [eval(p) for p in x]).reset_index()
    prob_features['Avg_Prob'] = prob_features['Probabilities'].apply(lambda x: np.mean([p[1] for p in x]))
    prob_features['Var_Prob'] = prob_features['Probabilities'].apply(lambda x: np.var([p[1] for p in x]))
    prob_features['Weighted_Avg_Prob'] = prob_features['Probabilities'].apply(lambda x: weighted_average([p[1] for p in x]))
    prob_features['Proportion_High_Probs'] = prob_features['Probabilities'].apply(lambda x: np.mean([1 if p[1] > 0.9 else 0 for p in x]))
    prob_features['Proportion_Low_Probs'] = prob_features['Probabilities'].apply(lambda x: np.mean([1 if p[1] < 0.1 else 0 for p in x]))
    prob_features.drop(columns=['Probabilities'], inplace=True)
    return prob_features

# Process tweet-based features
def process_tweets(input_dir, is_train):
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    results = []

    for file in tqdm(files, desc="Processing Tweet Files"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        grouped = df.groupby(["MatchID", "PeriodID"])
        for (match_id, period_id), group in grouped:
            combined_text = " ".join(group["Tweet"].astype(str))
            total_terms = calculate_total_football_terms(combined_text)
            num_tweets = len(group)
            avg_repeated = group["Tweet"].apply(count_repeated_letters).mean()
            avg_punctuation = group["Tweet"].apply(count_punctuation).mean()
            avg_uppercase = group["Tweet"].apply(count_uppercase_letters).mean()

            result = {
                "ID": f"{match_id}_{period_id}",
                "MatchID": match_id,
                "PeriodID": period_id,
                "TotalFootballTerms": total_terms,
                "NumTweets": num_tweets,
                "AvgRepeatedLetters": avg_repeated,
                "AvgPunctuation": avg_punctuation,
                "AvgUppercase": avg_uppercase
            }

            if is_train:
                result["EventType"] = group["EventType"].iloc[0]

            results.append(result)

    result_df = pd.DataFrame(results)
    return result_df

# Add PeriodID_Ratio
def calculate_period_id_ratio(df):
    max_periods = df.groupby("MatchID")["PeriodID"].transform("max")
    df["PeriodID_Ratio"] = df["PeriodID"] / max_periods
    return df.drop(columns=["MatchID"])

# Process train and eval data
print("Processing train tweet features...")
train_tweet_features = process_tweets(train_input_dir, is_train=True)
print("Processing eval tweet features...")
eval_tweet_features = process_tweets(eval_input_dir, is_train=False)

print("Processing train probabilistic features...")
train_prob_features = process_probabilities(prob_train_file)
print("Processing eval probabilistic features...")
eval_prob_features = process_probabilities(prob_eval_file)

# Merge features into final datasets
print("Merging train features...")
train_final = pd.merge(train_tweet_features, train_prob_features, on=["ID", "MatchID", "PeriodID"])
train_final = calculate_period_id_ratio(train_final)
print("Merging eval features...")
eval_final = pd.merge(eval_tweet_features, eval_prob_features, on=["ID", "MatchID", "PeriodID"])
eval_final = calculate_period_id_ratio(eval_final)

# Reorder columns
columns_order = [
    "ID", "EventType", "PeriodID", "PeriodID_Ratio", "TotalFootballTerms",
    "NumTweets", "AvgRepeatedLetters", "AvgPunctuation", "AvgUppercase",
    "Avg_Prob", "Var_Prob", "Weighted_Avg_Prob", "Proportion_High_Probs", "Proportion_Low_Probs"
]
train_final = train_final[columns_order]
eval_final = eval_final[[col for col in columns_order if col != "EventType"]]

# Save final datasets
train_final.to_csv(os.path.join(output_dir, "train_features_fadi.csv"), index=False)
eval_final.to_csv(os.path.join(output_dir, "eval_features_fadi.csv"), index=False)

print("Final features saved.")
