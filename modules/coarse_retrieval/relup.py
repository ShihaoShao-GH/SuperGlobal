# written by Shihao Shao (shaoshihao@pku.edu.cn)


import torch
from torch import nn
import numpy as np



class relup(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, alpha=0.014):
        super(relup, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        x = x.clamp(self.alpha)
        return x
    