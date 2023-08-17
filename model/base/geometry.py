r""" Provides functions that manipulate boxes and points """
# Original code: HSNet (https://github.com/juhongm999/hsnet)

import math

import torch.nn.functional as F
import torch

class Geometry(object):

    @classmethod
    def initialize(cls, img_size):
        cls.img_size = img_size

        cls.spatial_side = int(img_size / 8)
        norm_grid1d = torch.linspace(-1, 1, cls.spatial_side).cuda()

        cls.norm_grid_x = norm_grid1d.view(1, -1).repeat(cls.spatial_side, 1).view(1, 1, -1)
        cls.norm_grid_y = norm_grid1d.view(-1, 1).repeat(1, cls.spatial_side).view(1, 1, -1)
        cls.grid = torch.stack(list(reversed(torch.meshgrid(norm_grid1d, norm_grid1d)))).permute(1, 2, 0)

        cls.feat_idx = torch.arange(0, cls.spatial_side).float().cuda()

    @classmethod
    def normalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] -= (cls.img_size // 2)
        kps[kps != -2] /= (cls.img_size // 2)
        return kps

    @classmethod
    def unnormalize_kps(cls, kps):
        kps = kps.clone().detach()
        kps[kps != -2] *= (cls.img_size // 2)
        kps[kps != -2] += (cls.img_size // 2)
        return kps

    @classmethod
    def attentive_indexing(cls, kps, thres=0.1):
        r"""kps: normalized keypoints x, y (N, 2)
            returns attentive index map(N, spatial_side, spatial_side)
        """
        nkps = kps.size(0)
        kps = kps.view(nkps, 1, 1, 2)

        eps = 1e-5
        attmap = (cls.grid.unsqueeze(0).repeat(nkps, 1, 1, 1) - kps).pow(2).sum(dim=3)
        attmap = (attmap + eps).pow(0.5)
        attmap = (thres - attmap).clamp(min=0).view(nkps, -1)
        attmap = attmap / attmap.sum(dim=1, keepdim=True)
        attmap = attmap.view(nkps, cls.spatial_side, cls.spatial_side)

        return attmap

    @classmethod
    def apply_gaussian_kernel(cls, corr, sigma=17):
        bsz, side, side = corr.size()

        center = corr.max(dim=2)[1]
        center_y = center // cls.spatial_side
        center_x = center % cls.spatial_side

        y = cls.feat_idx.view(1, 1, cls.spatial_side).repeat(bsz, center_y.size(1), 1) - center_y.unsqueeze(2)
        x = cls.feat_idx.view(1, 1, cls.spatial_side).repeat(bsz, center_x.size(1), 1) - center_x.unsqueeze(2)

        y = y.unsqueeze(3).repeat(1, 1, 1, cls.spatial_side)
        x = x.unsqueeze(2).repeat(1, 1, cls.spatial_side, 1)

        gauss_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        filtered_corr = gauss_kernel * corr.view(bsz, -1, cls.spatial_side, cls.spatial_side)
        filtered_corr = filtered_corr.view(bsz, side, side)

        return filtered_corr

    @classmethod
    def transfer_kps(cls, confidence_ts, src_kps, n_pts, normalized):
        r""" Transfer keypoints by weighted average """

        if not normalized:
            src_kps = Geometry.normalize_kps(src_kps)
        confidence_ts = cls.apply_gaussian_kernel(confidence_ts)

        pdf = F.softmax(confidence_ts, dim=2)
        prd_x = (pdf * cls.norm_grid_x).sum(dim=2)
        prd_y = (pdf * cls.norm_grid_y).sum(dim=2)

        prd_kps = []
        for idx, (x, y, src_kp, np) in enumerate(zip(prd_x, prd_y, src_kps, n_pts)):
            max_pts = src_kp.size()[1]
            prd_xy = torch.stack([x, y]).t()

            src_kp = src_kp[:, :np].t()
            attmap = cls.attentive_indexing(src_kp).view(np, -1)
            prd_kp = (prd_xy.unsqueeze(0) * attmap.unsqueeze(-1)).sum(dim=1).t()
            pads = (torch.zeros((2, max_pts - np)).cuda() - 2)
            prd_kp = torch.cat([prd_kp, pads], dim=1)
            prd_kps.append(prd_kp)

        return torch.stack(prd_kps)

    @staticmethod
    def get_coord1d(coord4d, ksz):
        i, j, k, l = coord4d
        coord1d = i * (ksz ** 3) + j * (ksz ** 2) + k * (ksz) + l
        return coord1d

    @staticmethod
    def get_distance(coord1, coord2):
        delta_y = int(math.pow(coord1[0] - coord2[0], 2))
        delta_x = int(math.pow(coord1[1] - coord2[1], 2))
        dist = delta_y + delta_x
        return dist

    @staticmethod
    def interpolate4d(tensor4d, size):
        bsz, h1, w1, h2, w2 = tensor4d.size()
        ha, wa, hb, wb = size
        tensor4d = tensor4d.view(bsz, h1, w1, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, (ha, wa), mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, h2, w2, -1).permute(0, 3, 1, 2)
        tensor4d = F.interpolate(tensor4d, (hb, wb), mode='bilinear', align_corners=True)
        tensor4d = tensor4d.view(bsz, ha, wa, hb, wb)

        return tensor4d
    @staticmethod
    def init_idx4d(ksz):
        i0 = torch.arange(0, ksz).repeat(ksz ** 3)
        i1 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz).view(-1).repeat(ksz ** 2)
        i2 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz ** 2).view(-1).repeat(ksz)
        i3 = torch.arange(0, ksz).unsqueeze(1).repeat(1, ksz ** 3).view(-1)
        idx4d = torch.stack([i3, i2, i1, i0]).t().numpy()

        return idx4d

