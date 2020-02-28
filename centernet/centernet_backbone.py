# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

__all__ = ["build_centernet_backbone"]


@BACKBONE_REGISTRY.register()
def build_centernet_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)

    return bottom_up
