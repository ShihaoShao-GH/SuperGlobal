#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data

import pickle as pkl
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]
def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist

class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path_base):
        assert os.path.exists(
            data_path_base), "Data path '{}' not found".format(data_path_base)
        self.data_path_base = data_path_base
        self._scale_list = [1.]
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        # Compile the split data path
        self.data_path = read_imlist(os.path.join(self.data_path_base, "revisitop1m.txt"))
        self.n = len(self.data_path)
        

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        try:
            im = cv2.imread(os.path.join(self.data_path_base, self.data_path[index]))

            im_list = []

            for scale in self._scale_list:
                if scale == 1.0:
                    im_np = im.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale < 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale > 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)      
                else:
                    assert()
      
        except:
            print('error: ', self._db[index]["im_path"])

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
        return im_list

    def __len__(self):
        return len(self.data_path)
