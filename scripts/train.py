import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from tqdm.auto import tqdm
import time
import pandas as pd

import sys
import os



PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_DIR)

from src.model import TweetMamba
from src.dataloader import AudioSpectrogram, get_splits, get_stats
from src.training_utilities import save_history, import_history


#Training Data 
EXP_NAME = 'TweetMamba'
CSV_dataset = 'VoxCeleb_large.csv'
checkpoint = False

#Model parameters
embed_dim = 512
depth = 2

#Training parameters
num_epochs = 35
batch_size = 8
lr=1e-5

weight_decay= 5e-7
b1 = 0.95
b2 = 0.999
eps = 1e-8


CSV_FILE = os.path.join(PROJECT_DIR, 'data', CSV_dataset)
df = pd.read_csv(CSV_FILE)
print("CSV READ")
label2id, id2label = dict(), dict()

EXP_NAME = EXP_NAME + '_' + str(depth) + '_N'
print(EXP_NAME)

#Seed for reproductibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

#Directories
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
EXP_DIR = os.path.join(PROJECT_DIR, 'experiments')


#Initialize or import model
start_epoch, training_losses, validation_losses, validation_accuracies, best_acc, best_epoch = import_history(EXP_DIR, EXP_NAME,checkpoint=checkpoint)

# Device and Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

model = TweetMamba(num_classes =2  ,depth=depth, embed_dim = embed_dim)
model.to(device)
print(f"Number of model parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
if checkpoint:
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR , 'best_'+EXP_NAME+'.pth')))
    print(f'Model imported from checkpoint. Current epoch : {start_epoch}, Accuracy : {best_acc}.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(b1,b2), eps=eps)
milestones = list(range(5, num_epochs, 2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= milestones, gamma=0.75)


#Data loading
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(CSV_FILE)

Datasets = {
    "train": AudioSpectrogram(X_train, y_train, label2id, mstd = data_config),
    "val": AudioSpectrogram(X_val, y_val, label2id, mstd = data_config),
    "test": AudioSpectrogram(X_test, y_test, label2id, mstd = data_config)
}
batch_sizes = {
    "train": batch_size,
    "val": 1,
    "test": 1
}
Dataloaders = {
    'train': DataLoader(Datasets['train'], batch_size=batch_sizes['train'], shuffle=True),
    'val': DataLoader(Datasets['val'], batch_size=batch_sizes['val'], shuffle=False),
    'test': DataLoader(Datasets['test'], batch_size=batch_sizes['test'], shuffle=False)
}


#Training loop
print("Start Training")
for epoch in range(start_epoch,num_epochs+1):
    start_time = time.time()
    model.train()
    running_loss = 0.0
   
    loop=tqdm(Dataloaders['train'])
    loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
    for counter, (audio, labels) in enumerate(loop, start=1):
        audio, labels = audio.to(device), labels.to(device) 
        
        outputs = model(audio)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # TQDM progression bar update 
        loop.set_postfix(loss=loss.item())
        
    l = len(Dataloaders['train'])
    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch}/{num_epochs}], Running Loss: {running_loss/l:.4f}, Time: {epoch_time:.2f}s')
    
    training_losses.append(running_loss / l)
   
    
    # Evaluation on the validation dataset
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in Dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # Loss computation
    val_loss /= len(Dataloaders['val'])
    val_accuracy = 100 * correct / total  
    
    validation_accuracies.append(val_accuracy)     
    validation_losses.append(val_loss)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')        
        
 
    scheduler.step()
    
    #Save best model
    with torch.no_grad():
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model = model.state_dict()
            best_epoch = epoch
            torch.save(best_model, os.path.join(CHECKPOINT_DIR, 'best_'+EXP_NAME+'.pth'))
            print(f'Best model saved at epoch {epoch} with accuracy {best_acc:.2f}%')
    
    #Save training data 
    save_history(epoch, training_losses, validation_losses, validation_accuracies, EXP_DIR, EXP_NAME,plots=(epoch>=5))
    
print('Finished Training')