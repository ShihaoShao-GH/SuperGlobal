#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ResNe(X)t models."""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from modules.coarse_retrieval.rgem import rgem
from modules.coarse_retrieval.gemp import gemp
from modules.coarse_retrieval.sgem import sgem
from modules.coarse_retrieval.relup import relup
# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
TRANS_FUN = "bottleneck_transform"
NUM_GROUPS = 1
WIDTH_PER_GROUP = 64
STRIDE_1X1 = False
BN_EPS = 1e-5
BN_MOM = 0.1
RELU_INPLACE = True

def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.fc = nn.Linear(w_in, nc, bias=True)
        self.pool = GeneralizedMeanPoolingP(norm=pp)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class BasicTransform(nn.Module):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.a_relu = relup(0.)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x



        
class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs, relup_):
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if STRIDE_1X1 else (1, stride)
        self.a = nn.Conv2d(w_in, w_b, 1, stride=s1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        if w_out == 256 and relup_:
            self.a_relu = relup(0.014)
        else:
            self.a_relu = relup(0.)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=BN_EPS, momentum=BN_MOM)
        if w_out == 256 and relup_:
            self.b_relu = relup(0.014)
        else:
            self.b_relu = relup(0.)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBlock(nn.Module):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1, relup_=True):
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs, relup)
        self.relu = relup(0.)
        self.w_in = w_in
        self.w_out = w_out
        if relup_:
            self.relup = relup(0.014)
        else:
            self.relup = relup(0.)
        self.relu = relup(0.)
    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        if self.w_out!=2048:
            x = self.relup(x)
        else:
            x = self.relu(x)
        
        return x

class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1, relup=True):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs, relup)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class ResStage_basetransform(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = "basic_transform"
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=BN_EPS, momentum=BN_MOM)
        self.relu = nn.ReLU(RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, RESNET_DEPTH, REDUCTION_DIM, relup):
        super(ResNet, self).__init__()
        self.RESNET_DEPTH = RESNET_DEPTH
        self.REDUCTION_DIM = REDUCTION_DIM
        self.RELU_P = relup
        self._construct()

    def _construct(self):
        g, gw = NUM_GROUPS, WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[self.RESNET_DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g, relup=self.RELU_P)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g, relup=self.RELU_P)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g, relup=self.RELU_P)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g, relup=self.RELU_P)
        self.head = GlobalHead(2048, nc=self.REDUCTION_DIM)
        self.seb1 = nn.Conv2d(2048, 512, 7, 1, 'same')
        self.seb2 = nn.Conv2d(512, 2048, 7, 1, 'same')
        self.sefc = nn.Linear(2048,1)
        self.pool = nn.MaxPool2d(3,1,padding=1)
        self.rgem = rgem()
        self.root = 2.5
        self.area = 5
        self.gemp = gemp()
        self.sgem = sgem()
    def _forward_singlescale(self, x, gemp = True, rgem = True):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2) # 1 C H W
        x4 = self.s4(x3)
        if rgem:
            x5 = self.rgem(x4)
        else:
            x5 = x4
        x5 = x5.view(x5.shape[0], x5.shape[1], -1) 
        if gemp:
            x6 = self.gemp(x5)
        else:
            x6 = self.head.pool(x5)
        x6 = x6.view(x6.size(0), -1)
        x6 = F.normalize(x6, p=2, dim=-1)
        x7 = self.head.fc(x6)
        return x7
    def forward(self, x_, scale = 3, gemp = True, rgem = True, sgem = True):

        assert scale in [1, 3, 5], "scale must be in [1, 3, 5]"

        feature_list = []
        if scale == 1:
            scale_list = [1.]
        elif scale == 3:
            scale_list = [0.7071, 1., 1.4142]
        elif scale == 5:
            scale_list = [0.5, 0.7071, 1., 1.4142, 2.]
        else:
            raise 
        for _, scl in zip(range(scale), scale_list):
            x = torchvision.transforms.functional.resize(x_, [int(x_.shape[-2]*scl),int(x_.shape[-1]*scl)])
            x = self._forward_singlescale(x, gemp, rgem)
            feature_list.append(x)
        if sgem:
            x_out = self.sgem(feature_list)
        else:
            x_out = torch.stack(feature_list, 0)
            x_out = torch.mean(x_out, 0)
        

        return x_out

