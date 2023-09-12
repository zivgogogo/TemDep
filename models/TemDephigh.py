__all__ = ['TemDephigh']
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from layers.RevIN import RevIN
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, d_model,seq_kernel):
        super(TokenEmbedding, self).__init__()
        self.kernel = seq_kernel
        self.tokenConv = nn.Conv2d(in_channels=1, out_channels=d_model,
                                   kernel_size=(seq_kernel,1),stride=(1,1),padding=((seq_kernel-1)//2,0))

    def forward(self, x):
        if x.size(2)%2==0 and self.kernel%2==0:
            x = torch.nn.functional.pad(x,(0,0,1,0))
        x = self.tokenConv(x)  # B, C, L ,V
        return x

class DataEmbedding(nn.Module):
    def __init__(self, d_model,seq_kernel=1, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(d_model=d_model,seq_kernel = seq_kernel)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.value_embedding(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x   
    
class feature_linear(nn.Module):
    def __init__(self,seq_lenth):
        super(feature_linear, self).__init__()
        self.Linear = nn.Linear(seq_lenth,seq_lenth)

    def forward(self, x):
        y = x.clone()
        x = self.Linear(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = x + y
        return x
     
class Model(nn.Module):
    """  
    Conv2d
    """
    def __init__(self,configs):
        super(Model,self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.revin_layer = RevIN(configs.enc_in,affine=configs.affine,subtract_last=configs.subtract_last)
        self.embed = DataEmbedding(configs.d_model,configs.seq_kernel,configs.dropout)
        self.multi_var = feature_linear(self.seq_len)
        self.F = nn.Flatten(start_dim=1,end_dim=2)
        self.FC = nn.Linear(self.seq_len*configs.d_model,self.pred_len)
        
    def forward(self,x):
        x = self.revin_layer(x,'norm')
        x = self.embed(x.unsqueeze(1))
        x = self.multi_var(x)
        x = self.F(x)
        x = self.FC(x.transpose(1,2)).transpose(1,2)
        x = self.revin_layer(x,'denorm')
        return x