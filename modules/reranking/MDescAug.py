# written by Shihao Shao (shaoshihao@pku.edu.cn)


import torch
from torch import nn
import numpy as np



class MDescAug(nn.Module):
    """ Top-M Descriptor Augmentation"""
    def __init__(self, M = 400, K = 9, beta = 0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1 # including oneself
        self.beta = beta
    def forward(self, X, Q, ranks):

        #ranks = torch.argsort(-sim, axis=0) # 6322 70
        
        
        ranks_trans_1000 = torch.transpose(ranks,1,0)[:,:self.M] # 70 400 
        
        
        X_tensor1 = torch.tensor(X[ranks_trans_1000]).cuda()
        
        res_ie = torch.einsum('abc,adc->abd',
                X_tensor1,X_tensor1) # 70 400 400

        res_ie_ranks = torch.unsqueeze(torch.argsort(-res_ie.clone(), axis=-1)[:,:,:self.K],-1) # 70 400 10 1
        res_ie_ranks_value = torch.unsqueeze(-torch.sort(-res_ie.clone(), axis=-1)[0][:,:,:self.K],-1) # 70 400 10 1
        res_ie_ranks_value = res_ie_ranks_value
        res_ie_ranks_value[:,:,1:,:] *= self.beta
        res_ie_ranks_value[:,:,0:1,:] = 1.
        res_ie_ranks = torch.squeeze(res_ie_ranks,-1) # 70 400 10
        x_dba = X[ranks_trans_1000] # 70 1 400 2048
        
        
        x_dba_list = []
        for i,j in zip(res_ie_ranks,x_dba):
            # we should avoid for-loop in python, 
            # thus even make the numbers in paper look nicer, 
            # but i just want to go to bed.
            # i 400 10 j # 400 2048
            x_dba_list.append(j[i])
        
        x_dba = torch.stack(x_dba_list,0) # 70 400 10 2048
        
        x_dba = torch.sum(x_dba * res_ie_ranks_value, 2) / torch.sum(res_ie_ranks_value,2) # 70 400 2048
        res_top1000_dba = torch.einsum('ac,adc->ad', Q, x_dba) # 70 400 
 
        ranks_trans_1000_pre = torch.argsort(-res_top1000_dba,-1) # 70 400
        rerank_dba_final = []
        for i in range(ranks_trans_1000_pre.shape[0]):
            temp_concat = ranks_trans_1000[i][ranks_trans_1000_pre[i]]
            rerank_dba_final.append(temp_concat) # 400
        return rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba