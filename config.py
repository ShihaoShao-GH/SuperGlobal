#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode


# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C


_C.MODEL_NAME = ""
# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = "RESNET"

# Number of weight layers
_C.MODEL.DEPTH = 50

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSSES = CfgNode()
_C.MODEL.LOSSES.NAME = "cross_entropy"

# ------------------------------------------------------------------------------------ #
# Heads options
# ------------------------------------------------------------------------------------
_C.MODEL.HEADS = CfgNode()
_C.MODEL.HEADS.NAME = "LinearHead"
# Normalization method for the convolution layers.

# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 2048

# ------------------------------------------------------------------------------------ #
# Testing options
# ------------------------------------------------------------------------------------ #
_C.TEST = CfgNode()

_C.TEST.WEIGHTS = ""
_C.TEST.DATA_DIR = ""
_C.TEST.DATASET_LIST = ["roxford5k"]
_C.TEST.SCALE_LIST = 3
_C.TEST.TOPK_LIST = [400]

# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()
# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# ------------------------------------------------------------------------------------ #
# Batch norm options
# ------------------------------------------------------------------------------------ #
_C.BN = CfgNode()


# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ------------------------------------------------------------------------------------ #
# CUDNN options
# ------------------------------------------------------------------------------------ #
_C.CUDNN = CfgNode()
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True

# ------------------------------------------------------------------------------------ #
# SuperGlobal options
# ------------------------------------------------------------------------------------ #
_C.SupG = CfgNode()

_C.SupG.gemp = True
_C.SupG.sgem = True
_C.SupG.rgem = True
_C.SupG.relup = True
_C.SupG.rerank = True
_C.SupG.onemeval = True
# ------------------------------------------------------------------------------------ #
# Deprecated keys
# ------------------------------------------------------------------------------------ #

_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_list(args.opts)
