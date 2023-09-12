__all__ = ['TemDeplow']
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

# class RevIN(nn.Module):
#     def __init__(self,norm_num=1):
#         """
#         :param num_features: the number of features or channels
#         :param eps: a value added for numerical stability
#         :param affine: if True, RevIN has learnable affine parameters
#         """
#         super(RevIN, self).__init__()
#         self.norm_num = norm_num

#     def forward(self, x, mode:str):
#         if mode == 'norm':
#             self._get_statistics(x)
#             x = self._normalize(x)
#         elif mode == 'denorm':
#             x = self._denormalize(x)
#         else: raise NotImplementedError
#         return x

#     def _get_statistics(self, x):
#         self.last = x[:,-self.norm_num:,:].mean(dim=1).unsqueeze(1).detach()

#     def _normalize(self, x):
#         x = x - self.last
#         return x

#     def _denormalize(self, x):
#         x = x + self.last
#         return x
       
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
    
    
class ConvLayer(nn.Module):
    def __init__(self, c_in, kernel=3, dropout=0):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv2d(in_channels=c_in,
                                              out_channels=c_in,
                                              kernel_size=(kernel, 1),
                                              padding=((kernel - 1) // 2, 0)))
        self.activation1 = nn.GELU()

    def forward(self, x):
        y = x.clone()
        if x.size(2) % 2 == 0 and self.kernel%2==0:
            x = torch.nn.functional.pad(x, (0, 0, 1, 0))
        x = self.dropout(self.downConv(x))
        x = self.activation1(x)
        # x = x + y
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
        self.convlayer = ConvLayer(configs.d_model,configs.kernel_size,configs.fc_dropout)
        self.F = nn.Flatten(start_dim=1,end_dim=2)
        self.FC = nn.Linear(self.seq_len*configs.d_model,self.pred_len)
        
    def forward(self,x):
        x = self.revin_layer(x,'norm')
        x = self.embed(x.unsqueeze(1))
        x = self.convlayer(x)
        x = self.F(x)
        x = self.FC(x.transpose(1,2)).transpose(1,2)
        x = self.revin_layer(x,'denorm')
        return x