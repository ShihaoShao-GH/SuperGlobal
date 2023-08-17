r""" Correlation Verification Network """
# written by Seongwon Lee (won4113@yonsei.ac.kr)
# Original code: HSNet (https://github.com/juhongm999/hsnet)

from functools import reduce
from operator import add

import torch
import torch.nn as nn

from model.resnet import ResNet

from .base.feature import extract_feat_res_pycls
from .base.correlation import Correlation
from .CVlearner import CVLearner
import torch.nn as nn

class CVNet_Rerank(nn.Module):
    def __init__(self, RESNET_DEPTH, REDUCTION_DIM, relup):
        super(CVNet_Rerank, self).__init__()

        self.encoder_q = ResNet(RESNET_DEPTH, REDUCTION_DIM, relup)
        self.encoder_q.eval()

        self.scales = [0.25, 0.5, 1.0]
        self.num_scales = len(self.scales)

        feat_dim_l3 = 1024
        self.channel_compressed = 256

        self.softmax = nn.Softmax(dim=1)
        self.extract_feats = extract_feat_res_pycls

        if RESNET_DEPTH == 50:
            nbottlenecks = [3, 4, 6, 3]
            self.feat_ids = [13]
        elif RESNET_DEPTH == 101:
            nbottlenecks = [3, 4, 23, 3]
            self.feat_ids = [30]
        else:
            raise Exception('Unavailable RESNET_DEPTH %s' % RESNET_DEPTH)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        self.conv2ds = nn.ModuleList([nn.Conv2d(feat_dim_l3, 256, kernel_size=3, padding=1, bias=False) for _ in self.scales])

        self.cv_learner = CVLearner([self.num_scales*self.num_scales, self.num_scales*self.num_scales, self.num_scales*self.num_scales])
        
    
    
    def forward(self, query_img, key_img):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            key_feats = self.extract_feats(key_img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
            
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[:,1]
            
        return score
    def extract_global_descriptor(self, im_q, gemp, rgem, sgem, scale_list):
        # compute query features
        res = self.encoder_q(im_q, scale_list, gemp, rgem, sgem)
        return res
    
    def extract_featuremap(self, img):
        with torch.no_grad():
            feats = self.extract_feats(img, self.encoder_q, self.feat_ids, self.bottleneck_ids, self.lids)
        return feats

    def extract_score_with_featuremap(self, query_feats, key_feats):
        with torch.no_grad():
            corr_qk = Correlation.build_crossscale_correlation(query_feats[0], key_feats[0], self.scales, self.conv2ds)
            logits_qk = self.cv_learner(corr_qk)
            score = self.softmax(logits_qk)[0][1]
        return score
