from ..builder import HEADS
from .base import BaseHead

from TADP.tadv_heads import LinearHead, NeeharHead, RogerioHead

@HEADS.register_module()
class LinearHeadMM(LinearHead, BaseHead):
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
        BaseHead.__init__(self, num_classes, in_channels, loss_cls)
        LinearHead.__init__(self, num_classes, in_channels)

@HEADS.register_module()
class NeeharHeadMM(NeeharHead, BaseHead):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes=6,
                 in_channels=320,
                 num_heads=8,
                 num_layers=6,
                 hidden_dim=2048,
                 dropout=0.1,
                 num_frames=8,
                 loss_cls=dict(type='CrossEntropyLoss')):

        BaseHead.__init__(self, num_classes, in_channels, loss_cls)
        NeeharHead.__init__(self, num_classes, in_channels, num_heads, num_layers, hidden_dim, dropout, num_frames)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        x = self.mean(x)
        x = self.classifier(x)
        return x

@HEADS.register_module()
class RogerioHeadMM(RogerioHead, BaseHead):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes=6,
                 in_channels=320,
                 embed_dim=512,
                 num_heads=4,
                 num_layers=4,
                 hidden_dim=1024,
                 dropout=0.1,
                 num_frames=8,
                 loss_cls=dict(type='CrossEntropyLoss')):

        BaseHead.__init__(self, num_classes, in_channels, loss_cls)
        RogerioHead.__init__(self, num_classes, in_channels, embed_dim, num_heads, num_layers, hidden_dim, dropout, num_frames, init_super=False)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def loss(self, cls_score, labels, **kwargs):
        return BaseHead.loss(self, cls_score, labels)
