import os
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np

# Adding utility functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.features_utils import (
    analyse_terms,
    get_proba,
    analyze_symbol_ratio,
    analyse_repeated_letters,
    analyse_sentiments,
)

# Directories
train_input_dir = "challenge_data/train_tweets"
eval_input_dir = "challenge_data/eval_tweets"
prob_train_file = "results/train_bert_probabilities.csv"
prob_eval_file = "results/eval_bert_probabilities.csv"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Define term sets
football_terms = {
    "goal", "penalty", "offside", "yellow", "red", "card",
    "corner", "free", "kick", "var", "kickoff"
}
no_event_terms = {
    "fan", "boring", "possession", "streaming", "slow", "waiting",
    "strategy", "control", "passing", "defense", "midfield", "formation",
    "scoreless", "break", "die", "supporter", "giveaway", "retweet",
    "follow", "comment", "like", "share", "tag", "mention", "trend",
    "hashtag", "subscribe", "reply", "repost", "dm", "link", "vote",
    "promo", "view", "click"
}

# Process tweets and features
def process_tweets(input_dir, prob_file, is_train, output_file):
    # Load and combine data
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    print(f"Found {len(files)} {'training' if is_train else 'evaluation'} files to process...")
    
    dataframes = []
    for file in tqdm(files, desc=f"Reading {'Train' if is_train else 'Eval'} Tweet Files"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Analyze sentiments
    print("Analyzing sentiments...")
    combined_df = analyse_sentiments(combined_df)
    
    # Analyze football-related terms
    print("Analyzing football terms...")
    combined_df = analyse_terms(combined_df, football_terms, name="FootballTerms")
    
    # Analyze no-event-related terms
    print("Analyzing no-event terms...")
    combined_df = analyse_terms(combined_df, no_event_terms, name="NoEventTerms")
    
    # Analyze symbol ratios and repeated letters
    print("Analyzing symbol ratios...")
    combined_df = analyze_symbol_ratio(combined_df, '!')
    print("Analyzing repeated letters...")
    combined_df = analyse_repeated_letters(combined_df)
    
    # Drop unnecessary columns
    combined_df.drop(columns=["Timestamp", "MatchID", "PeriodID", "Tweet"], inplace=True)

    # Aggregate features by ID
    agg_dict = {
        'FootballTerms': 'mean',
        'NoEventTerms': 'mean',
        '!Ratio': 'mean',
        'RepeatedLettersRatio': 'mean',
        'pos': ['mean', 'std'],
        'neg': ['mean', 'std'],
        'neu': ['mean', 'std'],
        'compound': ['mean', 'std']
    }
    
    print("Aggregating features by ID...")
    if is_train:
        grouped_df = combined_df.groupby(["ID","EventType"]).agg(agg_dict).reset_index()
        grouped_df.columns = [
        "ID",
        "EventType",
        "FootballTerms_mean", "NoEventTerms_mean", "!Ratio_mean",
        "RepeatedLettersRatio_mean",
        "pos_mean", "pos_std", "neg_mean", "neg_std",
        "neu_mean", "neu_std", "compound_mean", "compound_std",
    ]
    else:
        grouped_df = combined_df.groupby("ID").agg(agg_dict).reset_index()
        grouped_df.columns = [
        "ID",
        "FootballTerms_mean", "NoEventTerms_mean", "!Ratio_mean",
        "RepeatedLettersRatio_mean",
        "pos_mean", "pos_std", "neg_mean", "neg_std",
        "neu_mean", "neu_std", "compound_mean", "compound_std",
    ]

    
    
    # Add probabilistic features
    print("Processing probabilistic features...")
    probas = get_proba(prob_file)
    final_df = pd.merge(grouped_df, probas, on="ID", how="left")
    
    # Save final features
    final_df.to_csv(output_file, index=False)
    print(f"{'Training' if is_train else 'Evaluation'} features saved to {output_file}")

# Process training and evaluation datasets
process_tweets(train_input_dir, prob_train_file, is_train=True, output_file=os.path.join(output_dir, "train_features_elie.csv"))
process_tweets(eval_input_dir, prob_eval_file, is_train=False, output_file=os.path.join(output_dir, "eval_features_elie.csv"))
