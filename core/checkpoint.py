#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os

import torch
from config import cfg

def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    
    
    model_dict = model.state_dict()
    state_dict = {k : v for k, v in state_dict.items()}
    weight_dict = {k : v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    

    model_dict.update(weight_dict)
    model.load_state_dict(model_dict)

    return checkpoint
