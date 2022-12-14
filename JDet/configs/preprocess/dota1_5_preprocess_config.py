type = 'DOTA1_5'
source_dataset_path = 'workspace/JAD/datasets/DOTA1_5/'
target_dataset_path = 'workspace/JAD/datasets/processed_DOTA1_5/'

# available labels: train, val, test, trainval
tasks = [
    dict(
        label='trainval',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.]
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.]
        )
    )
]
