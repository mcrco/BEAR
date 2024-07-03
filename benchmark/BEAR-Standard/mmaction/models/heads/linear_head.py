import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class LinearHead(BaseHead):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss')):
        super().__init__(num_classes, in_channels, loss_cls)
        self.mean = lambda x : torch.mean(x, 1) # first dim is video, second dim is frames
        self.flatten = nn.Flatten()
        self.cls_head = nn.Linear(320 * 64 * 64, 6) # 6 classes in tsh-mpii

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        x = self.mean(x)
        x = self.flatten(x)
        x = self.cls_head(x)
        return x

