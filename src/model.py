import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.dirname(__file__))
from mamba.mamba_encoder import MambaConfig, MambaEncoder
#from utilities import PatchEmbed, PosEmbed



class TweetMamba(nn.Module):
    def __init__(self, num_classes, depth=4, embed_dim=200 ,bidirectional = True):
        super().__init__()
        
        #self.pos_embed =  PosEmbed(num_patches = self.num_patches,embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        config  = MambaConfig(d_model = embed_dim, n_layers= depth,bidirectional=bidirectional)
        self.mamba = MambaEncoder(config)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),          # Layer normalization
            nn.Linear(embed_dim, embed_dim),  # Hidden layer
            nn.ReLU(),                        # Non-linearity
            nn.Dropout(0.1),                  # Dropout for regularization
            nn.Linear(embed_dim, num_classes) # Output layer
        )) 


    def forward(self,x):

        B, N, _ = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)
        token_position = N // 2
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
        #x = self.pos_embed(x, cls_pos = N//2) #(B, N+1, embed_dim)
        x = self.mamba(x) #Mamba Encoder : (B, N+1, embed_dim) -> (B, N+1, embed_dim) 

        #  We keep only CLS token at position N//2
        x = x[:, token_position, :]
        x = self.head(x)
        return x
    
    

if __name__ == "__main__": 
    print("Test TweetMamba")
    print('test')
    device = torch.device("cpu")
    num_classes = 2
    model = TweetMamba(depth = 24, embed_dim= 200, num_classes= 2)
    model.to(device)
    B, N, D = 3, 200, 200
    x = torch.randn(B, N, D).to(device)
    with torch.no_grad():
        y = model(x)
    print("Output generated")
    print(f"Size comparaison  : {y.size()==torch.Size([B,num_classes])}")
    print(f"y : {y}")
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    