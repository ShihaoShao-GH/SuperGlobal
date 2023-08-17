# written by Shihao Shao (shaoshihao@pku.edu.cn)


import torch
from torch import nn
import numpy as np



class RerankwMDA(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, M=400, K = 9, beta = 0.15):
        super(RerankwMDA, self).__init__()
        self.M = M 
        self.K = K + 1 # including oneself
        self.beta = beta
    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):

        ranks_trans_1000 = torch.stack(rerank_dba_final,0) # 70 400
        ranks_value_trans_1000 = -torch.sort(-res_top1000_dba,-1)[0] # 70 400
        

        ranks_trans = torch.unsqueeze(ranks_trans_1000_pre[:,:self.K],-1) # 70 10 1
        ranks_value_trans = torch.unsqueeze(ranks_value_trans_1000[:,:self.K].clone(),-1) # 70 10 1
        ranks_value_trans[:,:,:] *=self.beta
        
        X1 =torch.take_along_dim(x_dba, ranks_trans,1) # 70 10 2048
        X2 =torch.take_along_dim(x_dba, torch.unsqueeze(ranks_trans_1000_pre,-1),1) # 70 400 2048
        X1 = torch.max(X1, 1, True)[0] # 70 1 2048
        res_rerank = torch.sum(torch.einsum(
            'abc,adc->abd',X1,X2),1) # 70 400
        

        res_rerank = (ranks_value_trans_1000 + res_rerank) / 2. # 70 400
        res_rerank_ranks = torch.argsort(-res_rerank, axis=-1) # 70 400
        
        rerank_qe_final = []
        ranks_transpose = torch.transpose(ranks,1,0)[:,self.M:] # 70 6322-400
        for i in range(res_rerank_ranks.shape[0]):
            temp_concat = torch.concat([ranks_trans_1000[i][res_rerank_ranks[i]],ranks_transpose[i]],0)
            rerank_qe_final.append(temp_concat) # 6322
        ranks = torch.transpose(torch.stack(rerank_qe_final,0),1,0) # 70 6322
        
        return ranks
