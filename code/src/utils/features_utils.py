from tqdm import tqdm
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# Define football terms
football_terms = {
    "goal", "penalty", "offside", "yellow", "red", "card",
    "corner", "kick", "var", "kickoff", "shot", "pass",
    "save", "score", "foul", "header", "assist", "dribble",
    "cross", "freekick", "throw", "tackle",
    "owngoal", "referee", "substitution", "halftime",
    "fulltime"
}

lemmatizer = WordNetLemmatizer()
lemmatized_terms = {lemmatizer.lemmatize(term) for term in football_terms}

def calculate_total_football_terms(combined_text):
    """
    Calculate the total number of football terms in the given text.
    """
    words = combined_text.lower().split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return sum(1 for word in lemmatized_words if word in lemmatized_terms)

def count_repeated_letters(text):
    return sum(len(re.findall(r'(\w)\1{2,}', word.lower())) for word in text.split())

def count_punctuation(text):
    return sum(1 for char in text if char in "!?")

def count_uppercase_letters(text):
    return sum(1 for char in text if char.isupper())

def weighted_average(probs):
    return np.average(probs, weights=np.square(probs))

def analyse_terms( df: pd.DataFrame,football_terms,name):
    def match_football_terms(text):
        # Create a pattern to match variations of football terms
        text.lower()
        pattern = r'\b(?:' + '|'.join([f'{term}+' for term in football_terms]) + r')\b'
        matches = re.findall(pattern, text.lower())
        return matches

    # Apply preprocessing and count words in football terms for the whole dataset
    #print(df.head())
    df['TotalWords'] = df['Tweet'].apply(lambda x: len(x.split()))
    df['MatchedTerms'] = df['Tweet'].apply(match_football_terms)
    df['FootballWords'] = df['MatchedTerms'].apply(len)

    df[name] = df['FootballWords'] / df['TotalWords']
    df.drop(columns=['TotalWords', 'MatchedTerms', 'FootballWords'], inplace=True)
    
    return df

def analyze_symbol_ratio(df, char):
    
   
    # Fonction pour compter le caractère dans chaque texte
    def count_character(text):
        return text.count(char)
    
    # Calculer le nombre total de caractères par tweet
    df['TotalChars'] = df['Tweet'].apply(len)
    
    # Calculer le nombre d'occurrences du caractère spécifié
    df[f'{char}Count'] = df['Tweet'].apply(count_character)
    
    # Calculer le ratio du caractère par rapport au nombre total de caractères
    df[f'{char}Ratio'] = df[f'{char}Count'] / df['TotalChars']

    # Supprimer les colonnes intermédiaires
    df.drop(columns=['TotalChars', f'{char}Count'], inplace=True)


    return df

def analyse_repeated_letters(df):
    
    
    def count_repeated_letters(text):
        repeated_count = 0
        words = text.split()
        for word in words:
            # Find letter repetitions using regex (e.g., "goooal", "yeeees")
            matches = re.findall(r'(\w)\1{2,}', word.lower())  # Match any letter repeated 3+ times
            repeated_count += len(matches)  # Count the number of words with repetitions
        return repeated_count

    # Apply the function to count repeated letters in each tweet
    tqdm.pandas()
    df['RepeatedLetters'] = df['Tweet'].progress_apply(count_repeated_letters)


    
    # Calculer le nombre total de caractères par tweet
    df['TotalWords'] = df['Tweet'].apply(lambda x: len(x.split()))


    df['RepeatedLettersRatio'] = df['RepeatedLetters'] / df['TotalWords']
    df.drop(columns=['TotalWords', 'RepeatedLetters'], inplace=True)

    return df


def analyse_sentiments(df):
    analyzer = SentimentIntensityAnalyzer()
    tqdm.pandas()
    tweet_sentiment_vectors = np.vstack([np.array(list(analyzer.polarity_scores(tweet).values())) for tweet in tqdm(df['Tweet'])])


    df['pos'] = tweet_sentiment_vectors[:, 0]
    df['neg'] = tweet_sentiment_vectors[:, 1]
    df['neu'] = tweet_sentiment_vectors[:, 2]
    df['compound'] = tweet_sentiment_vectors[:, 3]

    return df



def get_proba(csv):
    probas_quota = pd.read_csv(csv)

    # Convert the 'Probabilities' column from string representation to numpy arrays
    probas_quota['Probabilities'] = probas_quota['Probabilities'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    probas_quota['Probabilities'] = probas_quota['Probabilities'].apply(lambda x: x[0])

    result = probas_quota.groupby('ID').agg(
    ProbaMean=('Probabilities', 'mean'),
    ProbaStd=('Probabilities', 'std')
    ).reset_index()

    return result
