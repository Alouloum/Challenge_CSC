import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_



class PosEmbed(nn.Module):
    def __init__(self, num_patches, embed_dim,pos_embed_load = None, verbose = True):
        super().__init__()
        pos_embed_shape = (1, num_patches +1, embed_dim)
        self.num_patches = num_patches
        
        if pos_embed_load is not None: #Import existant weights 
            if verbose:
                print('Loading position embedding!')
                print(f'The number of patches of the current grid is : {num_patches}')

            self.pos_embed = nn.Parameter(pos_embed_load)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(*pos_embed_shape))
            trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self,x,cls_pos):
        # Precaution to import weights from other models
        # Move the CLS token to the beginning
        cls_token = x[:, cls_pos:cls_pos+1]
        x = torch.cat([cls_token, x[:, :cls_pos], x[:, cls_pos+1:]], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Move the CLS token back to the middle 
        x = torch.cat([x[:, 1:cls_pos+1], x[:, 0:1], x[:, cls_pos+1:]], dim=1)
        
        return x
        