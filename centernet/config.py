# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.CENTERNET = CN()
