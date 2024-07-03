proj_dir = '/home/marco/dev/diffusion_ar/'
config_root_dir = proj_dir + 'BEAR/benchmark/BEAR-Standard/configs/'
i3d_convnext_path = config_root_dir + '_base_/models/i3d_convnext.py'
default_runtime_path = config_root_dir + '_base_/default_runtime.py'

_base_ = [
        i3d_convnext_path, default_runtime_path
]

# model settings
model=dict(cls_head=dict(num_classes=6))
# dataset settings
dataset_type = 'RawframeDataset'
data_root = proj_dir + 'BEAR/datasets/ToyotaSmarthome/frames/'
data_root_val = proj_dir + 'BEAR/datasets/MPII-Cooking/frames/'
ann_file_test = proj_dir + 'BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/mpii_cooking_da_test.csv'
ann_file_train = proj_dir + 'BEAR/benchmark/BEAR-UDA/data/toyota_smarthome_mpii_cooking/toyota_smarthome_da_train.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=16, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=16,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05d}.jpg',
        start_index=0,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        filename_tmpl='{:05d}.jpg',
        start_index=0,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    #lr=0.01,  # this lr is used for 8 gpus
    lr=0.01 / 8,  # for 1 gpu
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[20, 40],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
    
total_epochs = 50

# logger settings
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend')
]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# runtime settings
work_dir = './work_dirs/i3d/mpii_cooking_8frame/'
log_config = dict(interval=50)
checkpoint_config = dict(interval=5)
