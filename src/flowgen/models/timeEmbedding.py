import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class time_embedding(nn.Module):
    
    def __init__(self, scale, n_features):
        
        super().__init__()
        self.B = 2 * np.pi * scale * torch.randn([n_features]).view(1,-1)
    
    def forward(self,t):
        B = self.B.type_as(t)
        t = t.unsqueeze(-1)
        return torch.cat([torch.sin(B *t),torch.cos(B * t)], dim=1)