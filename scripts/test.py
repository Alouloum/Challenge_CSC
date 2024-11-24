import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from tqdm.auto import tqdm
import time

# Problème perso de GPU : à modifier si nécessaire
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

sys.path.append("/home/maloulou/stage_mamba/testmamba/")
from src.model import AudioMamba
from src.dataloader import AudioSpectrogram, get_splits, get_stats

# Initialisation des graines
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Configuration des données
CSV_FILE = '/home/maloulou/stage_mamba/stage2024/data/VoxCeleb_small.csv'
data_config = get_stats(CSV_FILE)

df = pd.read_csv(CSV_FILE)
print("CSV READ")
label2id, id2label = dict(), dict()
for i, label in enumerate(df.speaker_id.unique()):
    label2id[label] = i
    id2label[i] = label

# Paramètres du modèle
embed_dim = 256
depth = 4
num_classes = len(label2id)

# Initialisation du modèle
model = AudioMamba(num_classes=num_classes, depth=depth, embed_dim=embed_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device : {device}")

# Chargement des données
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(CSV_FILE)

Datasets = {
    "train": AudioSpectrogram(X_train, y_train, label2id, mstd=data_config),
    "val": AudioSpectrogram(X_val, y_val, label2id, mstd=data_config),
    "test": AudioSpectrogram(X_test, y_test, label2id, mstd=data_config)
}
batch_sizes = {
    "train": 8,
    "val": 1,
    "test": 1
}
Dataloaders = {
    'train': DataLoader(Datasets['train'], batch_size=batch_sizes['train'], shuffle=True),
    'val': DataLoader(Datasets['val'], batch_size=batch_sizes['val'], shuffle=False),
    'test': DataLoader(Datasets['test'], batch_size=batch_sizes['test'], shuffle=False)
}

# Charger le meilleur modèle
checkpoint_path = '/home/maloulou/stage_mamba/testmamba/checkpoints/best_AudioMamba_256_4_N36.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
print(f'Model loaded from {checkpoint_path}')

# Évaluation sur l'ensemble de test
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(Dataloaders['test'], desc="Evaluating on Test Set"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calcul de la perte et de l'exactitude sur le test
test_loss /= len(Dataloaders['test'])
test_accuracy = 100 * correct / total       
print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
