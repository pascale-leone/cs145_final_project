
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='BNInception',
        pretrained='https://download.openmmlab.com/mmaction/recognition/'
                   'tsn/tsn_bninception_imagenet_rgb_20210219-66bc2e4e.pth'
    ),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=1024,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[104, 117, 128],
        std=[1, 1, 1],
        format_shape='NCHW'
    )
)

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True
    ),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

dataset_type = 'RawframeDataset'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/tad_video_list.txt',
        data_prefix=dict(img='/Users/pascaleleone/.cache/kagglehub/datasets/nikanvasei/traffic-anomaly-dataset-tad/versions/1/TAD/frames'),
        pipeline=test_pipeline,
        test_mode=True
    )
)
