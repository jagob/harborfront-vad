import os
import shutil

import numpy as np
import pandas as pd

import src.data.object_annotations.utils as utils
import src.data.datasets.harbor as harbor_dataset

TAMPERING_VIDEOS = [
    '20200819/clip_23_1041',  # crane opertaing in front of camera
    '20210213/clip_36_1627',  # camera freezes, stutters
    '20210213/clip_37_1655',  # camera freezes, stutters
    '20210305/clip_28_1238',  # camera freezes, stutters
    '20210306/clip_40_2312',  # camera freezes, stutters
    '20210308/clip_13_0604',  # camera freezes, stutters
    '20210308/clip_14_0632',  # camera freezes, stutters
    '20210307/clip_20_0923',  # camera freezes, stutters
    '20210307/clip_21_0951',  # camera freezes, stutters
    '20210307/clip_26_1211',  # camera freezes, stutters
    '20210307/clip_27_1239',  # camera freezes, stutters
    '20210307/clip_28_1434',  # camera freezes, stutters
    '20210307/clip_23_1047',  # camera freezes, stutters
    '20210425/clip_9_0412',   # camera freezes, stutters
    # 20200516_clip_39_1726
    ]


def tampering_videos(dataset):
    df_videos = pd.DataFrame(TAMPERING_VIDEOS, columns=['joindir'])
    df_videos['folder_name'] = df_videos.joindir.apply(lambda x: x.split(os.sep)[0])
    df_videos['clip_name'] = df_videos.joindir.apply(lambda x: x.split(os.sep)[1])
    df_videos = df_videos.drop(columns='joindir')
    df_videos = df_videos.sort_values(['folder_name', 'clip_name'])
    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    csv_videos_path = os.path.join(r'src/data/split/videos/', dataset.csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_videos.to_csv(csv_videos_path)


def tampering_test_set(dataset, nr_normal_videos: int = 5):
    ''' add normal videos to make test set '''
    split_videos_path = r'src/data/split/videos'
    csv_videos_path = os.path.join(split_videos_path, dataset.csv_videos_name)
    df_tamp = pd.read_csv(csv_videos_path, index_col='index')

    # TODO: where to get normal train videos
    # normal_csv_path = os.path.join(split_videos_path, 'harbor_train_low_density.csv')
    normal_csv_path = os.path.join(split_videos_path, 'harbor_train.csv')
    df_normal = pd.read_csv(normal_csv_path, index_col='index')
    df_normal = df_normal.sample(nr_normal_videos, random_state=0).sort_index()
    df_tamp_test = pd.concat([df_tamp, df_normal])

    csv_videos_path = os.path.join(split_videos_path, dataset.csv_test)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_tamp_test.to_csv(csv_videos_path)


def frame_level_ground_truth(dataset, overwrite: bool = False):
    gt_video_dir = dataset.frame_gt
    if overwrite:
        if os.path.exists(gt_video_dir):
            shutil.rmtree(gt_video_dir)
    os.makedirs(gt_video_dir, exist_ok=False)

    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df = pd.read_csv(csv_path, dtype={'folder_name': str})
    for idx, row in df.iterrows():
        video_name = row.folder_name + os.sep + row.clip_name
        normal = True if video_name not in TAMPERING_VIDEOS else False
        seq_gt = ground_truth_video_high_density(dataset, row.folder_name, row.clip_name, normal)

        video_name = video_name.replace('/', '_')
        gt_video_path = os.path.join(gt_video_dir, f'{video_name}.txt')
        np.savetxt(gt_video_path, seq_gt, '%d')


def ground_truth_video_high_density(dataset, folder_name, clip_name, normal):
    img_dir = os.path.join(dataset.images, folder_name, clip_name)
    frame_names = sorted(os.listdir(img_dir))

    if normal:
        seq_gt = np.zeros(len(frame_names), dtype=np.int8)
    else:
        seq_gt = np.ones(len(frame_names), dtype=np.int8)
    return seq_gt


if __name__ == "__main__":
    dataset = harbor_dataset.Tampering()

    tampering_videos(dataset)
    tampering_test_set(dataset, nr_normal_videos=6)

    utils.create_image_datasplit(dataset, dataset.csv_test)
    utils.object_datasplit(dataset, dataset.csv_test, gt=False)
    utils.object_datasplit(dataset, dataset.csv_test, gt=True)

    frame_level_ground_truth(dataset, overwrite=True)
    # no r/tbdc ground-truth due to no specific localization
