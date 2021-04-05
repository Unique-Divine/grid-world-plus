# PyTorch
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping 

# Built-in
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import copy
import random
import pickle
from typing import List, Tuple, Dict, Any
import logging
import collections

import numpy as np


class MultiHeadAttention(nn.Module):
    """TODO: docs  """
    def __init__(self, embed_dim, num_heads: int = 1, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.scaling = self.head_dim ** -0.5
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        tgt_len, batch_size, embed_dim = x.size()
        x = self.in_proj(x)
        q, k, v = x.chunk(3, dim=-1)
        q *= self.scaling

        q = q.contiguous().view(
            tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(
            -1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # attn weight [batch_size * num_heads, tgt_len, src_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights - torch.max(attn_weights, dim=-1, keepdim=True)[0]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        return attn

class LitSelfAttention(pl.LightningModule):
    """
    SAGAN: https://arxiv.org/abs/1805.08318

    Args:
        pl ([type]): [description]
    """
    def __init__(self, 
                 lr: float, 
                 batch_size: int,
                 attention_dropout: float = 0.1, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass # TODO
        self.lr = lr
        self.BATCH_SIZE = batch_size

    def forward(self, x):
        pass # TODO
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params = self.parameters(), 
            lr = self.lr)
        return optimizer

