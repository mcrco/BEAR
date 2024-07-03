from ..builder import BACKBONES
from TADP.tadp_video import TADPVid

@BACKBONES.register_module()
class TADPVidMM(TADPVid):
    def __init__(self, cfg, class_names):
        super().__init__(cfg=cfg, use_decode_head=False, class_names=class_names)

    def forward(self, x):
        return super().forward(x, img_metas=None)

    def init_weights(self):
        pass
