dataset_type = 'VOCDataset'
data_root = '/home/wubw/voc-tire-fbb/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=[(320, 320), (400, 400)],
                    multiscale_mode='range',
                    keep_ratio=True,
                    backend='pillow'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(400, 400),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, backend='pillow'),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='CocoDataset',
            ann_file='../data/neu_det_coco/annotations/train_annotation.json',
            img_prefix='../data/neu_det_coco/train/',
        pipeline=train_pipeline,
        classes=('crazing', 'inclusion', 'patches', 'pitted_surface',
                 'rolled-in_scale', 'scratches'))),
    val=dict(
        type='CocoDataset',
        ann_file='../data/neu_det_coco/annotations/test_annotation.json',
        img_prefix='../data/neu_det_coco/test/',
        pipeline=test_pipeline,
        classes=('crazing', 'inclusion', 'patches', 'pitted_surface',
                 'rolled-in_scale', 'scratches')),
    test=dict(
        type='CocoDataset',
        ann_file='../data/neu_det_coco/annotations/test_annotation.json',
        img_prefix='../data/neu_det_coco/test/',
        pipeline=test_pipeline,
        classes=('crazing', 'inclusion', 'patches', 'pitted_surface',
                 'rolled-in_scale', 'scratches')))
evaluation = dict(interval=1, metric='bbox')
optimizer_config = dict(grad_clip=None)
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from ='checkpoints/swin_epoch_18.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='TOOD',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=
        #     'checkpoints/swin_small_patch4_window7_224.pth'
        # )
        ),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5)
    ],
    bbox_head=dict(
        type='TOODHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        num_dcn=2),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
work_dir = './tutorial_exps'
seed = 10086
gpu_ids = [0, 1]
device = 'cuda'