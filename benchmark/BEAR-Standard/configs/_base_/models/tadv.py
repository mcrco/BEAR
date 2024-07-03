# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TADPVidMM'),
    cls_head=dict(
        type='LinearHead'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# This setting refers to https://github.com/open-mmlab/mmaction/blob/master/mmaction/models/tenons/backbones/resnet_i3d.py#L329-L332  # noqa: E501
