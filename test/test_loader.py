#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from test.dataset import DataSet


# Default data directory (/path/pycls/pycls/datasets/data)

def _construct_loader(_DATA_DIR, dataset_name, fn, split, scale_list, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = DataSet(_DATA_DIR, dataset_name, fn, split, scale_list)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=4,
        pin_memory=False,
        drop_last=drop_last,
    )
    return loader

def construct_loader(_DATA_DIR, dataset_name, fn, split, scale_list):
    """Test loader wrapper."""
    return _construct_loader(
        _DATA_DIR=_DATA_DIR,
        dataset_name=dataset_name,
        fn=fn,
        split=split,
        scale_list = scale_list,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
