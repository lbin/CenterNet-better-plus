# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.DECONV_CHANNEL = [512, 256, 128, 64]
    _C.MODEL.CENTERNET.DECONV_KERNEL = [4, 4, 4]
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.MODULATE_DEFORM = True
    _C.MODEL.CENTERNET.BIAS_VALUE = -2.19
    _C.MODEL.CENTERNET.DOWN_SCALE = 4
    _C.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    _C.MODEL.CENTERNET.TENSOR_DIM = 128
    _C.MODEL.CENTERNET.IN_FEATURES = ["p5"]
    _C.MODEL.CENTERNET.OUTPUT_SIZE = [128, 128]
    _C.MODEL.CENTERNET.TRAIN_PIPELINES = [
        ("CenterAffine",
         dict(boarder=128, output_size=(512, 512), random_aug=True)),
        ("RandomFlip", dict()),
        ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
        ("RandomLighting", dict(scale=0.1)),
    ]
    _C.MODEL.CENTERNET.TEST_PIPELINES = []
    _C.MODEL.CENTERNET.LOSS = CN()
    _C.MODEL.CENTERNET.LOSS.CLS_WEIGHT = 1
    _C.MODEL.CENTERNET.LOSS.WH_WEIGHT = 0.1
    _C.MODEL.CENTERNET.LOSS.REG_WEIGHT = 1
    _C.INPUT.FORMAT = 'RGB'
