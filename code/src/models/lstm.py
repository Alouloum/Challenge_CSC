
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils


## LSTM/attention model
class TweetClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, bidirectional=True):
        super(TweetClassifier, self).__init__()

        # Attention 
        self.word_attention = nn.Linear(embedding_dim, 1)

        # LSTM for tweets
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        
        # Classifier head
        if bidirectional:
            self.classifier = nn.Linear(2 * lstm_hidden_dim, 1)
        else:
            self.classifier = nn.Linear(lstm_hidden_dim, 1)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def attention_aggregation(self, word_embeddings):
        attn_scores = self.word_attention(word_embeddings)  # (n_words, 1)
        attn_scores = F.softmax(attn_scores, dim=0)
        tweet_embedding = torch.sum(attn_scores * word_embeddings, dim=0)
        return tweet_embedding

    def forward(self, input_ids=None,  n_tweets=None, n_words=None,labels=None):
      
        batch_size = input_ids.size(0)
        tweet_embeddings = []

        for i in range(batch_size):
            tweets = input_ids[i] # (max_n_tweets, max_n_words, embedding_dim)
            tweet_embeds = []
            for j in range(n_tweets[i]):
                word_embeddings = tweets[j][:n_words[i][j]]  # (n_words, embedding_dim)
                # Attention
                tweet_embedding = self.attention_aggregation(word_embeddings)
                tweet_embeds.append(tweet_embedding)

            tweet_embeds = torch.stack(tweet_embeds)  # (n_tweets, embedding_dim)
            tweet_embeddings.append(tweet_embeds)

        # Padding 
        tweet_embeddings_padded = rnn_utils.pad_sequence(tweet_embeddings, batch_first=True)
        lengths = torch.tensor([te.size(0) for te in tweet_embeddings], dtype=torch.long)

        # Pack the padded sequence
        packed_input = rnn_utils.pack_padded_sequence(
            tweet_embeddings_padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, _) = self.lstm(packed_input)

        # Use the hidden state of the LSTM as the tweet embedding
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, 2 * lstm_hidden_dim)
        else:
            hidden = hidden[-1]

        #Classifier
        logits = self.classifier(hidden).squeeze(1)  # (batch_size)
        probs = torch.sigmoid(logits)  # Probabilités entre 0 et 1

        outputs = {'logits': probs}

        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs['loss'] = loss

        return outputs
    
    
## Dataset class for the LSTM model
class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, tweets_embeddings_list, labels):
        self.tweets_embeddings_list = tweets_embeddings_list
        self.labels = labels

    def __len__(self):
        return len(self.tweets_embeddings_list)

    def __getitem__(self, idx):
        tweets_embeddings = self.tweets_embeddings_list[idx]
        label = self.labels[idx]
        tweets_embeddings = [torch.tensor(embedding, dtype=torch.float) for embedding in tweets_embeddings]
        result = {
            'input_ids': tweets_embeddings,
            'n_tweets': torch.tensor(len(tweets_embeddings), dtype=torch.long),
            'n_words': torch.tensor([te.size(0) for te in tweets_embeddings], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }
        return result

## Collate function for the LSTM model
def collate_lstm(batch,max_n_tweets, max_n_words,embedding_dim):
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
        'labels':  torch.stack([item['labels'] for item in batch])           # Tensor of labels
    }

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
    


def collate_lstm_eval(batch,max_n_tweets, max_n_words,embedding_dim):
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