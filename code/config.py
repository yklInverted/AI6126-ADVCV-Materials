# Modified by Wang Jianyi for AI6126 Project2
# srresnet_ffhq_300k_(for_MMEditing's_version>=v1.0)

_base_ = 'configs/_base_/default_runtime.py'

experiment_name = 'srresnet_ffhq_300k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

# DistributedDataParallel
model_wrapper_cfg = dict(type='MMSeparateDistributedDataParallel')

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MSRResNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    train_cfg=dict(),
    test_cfg=dict(),
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='gt',
        channel_order='rgb'),
    dict(type='CopyValues', src_keys=['gt'], dst_keys=['img']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[41],
            kernel_list=['iso', 'aniso'],
            kernel_prob=[0.5, 0.5],
            sigma_x=[0.2, 5],
            sigma_y=[0.2, 5],
            rotate_angle=[-3.1416, 3.1416],
        ),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0, 1, 0],  # up, down, keep
            resize_scale=[0.0625, 1],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian'],
            noise_prob=[1],
            gaussian_sigma=[0, 25],
            gaussian_gray_noise_prob=0),
        keys=['img'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[50, 95]),
        keys=['img']),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(512, 512),
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
        keys=['img'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            target_size=(128, 128), resize_opt=['area'], resize_prob=[1]),
        keys=['img'],
    ),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='PackEditInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

# dataset settings
dataset_type = 'BasicImageDataset'

train_dataloader = dict(
    num_workers=12,
    batch_size=12,  # gpus 4
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='face_sr'),
        data_root='/content/drive/MyDrive/2023-AI6126-Project2/train/GT',
        data_prefix=dict(gt='', img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(dataset_type='ffhq', task_name='face_sr'),
        data_root='/content/drive/MyDrive/2023-AI6126-Project2/val/',
        data_prefix=dict(img='LQ', gt='GT'),
        filename_tmpl=dict(img='{}', gt='{}'),
        pipeline=val_pipeline))

# For test, since you do not have GT, you just need to resize the LQ image using opencv
# and use it as GT. Then you can run the code. The GT is only used to make sure the code
# can run. Remember to change the data root accordingly.
test_dataloader = val_dataloader

val_evaluator = dict(
    type='EditEvaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300_000, val_interval=5000)
val_cfg = dict(type='EditValLoop')
test_cfg = dict(type='EditTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999)))

# learning policy, Shangchen added
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[150000, 150000],
    restart_weights=[1, 1],
    eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)


# custom hook
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=True)
custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.001),
    )
]