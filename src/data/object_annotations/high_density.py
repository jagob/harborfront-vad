import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.data.extract_frames import videocsv2splitcsv
import src.data.datasets.harbor as harbor_dataset
import src.data.object_annotations.utils as utils
from src.evaluation.plot_anomaly_score import get_anomaly_intervals

pd.set_option('mode.chained_assignment', None)
sns.set_theme()

IGNORE_VIDEOS = ['20200526/clip_36_1617',  # faulty bounding boxes
                 '20200526/clip_38_1713',  # faulty bounding boxes
                 '20200614/clip_47_2057',  # faulty bounding boxes
                 '20200521/clip_31_1359',  # faulty bounding boxes
                 '20200614/clip_39_1727',  # faulty bounding boxes
                 ]


def create_high_density(dataset):
    object_annotations_path = dataset.object_annotations
    csv_annotations = os.path.join(object_annotations_path, 'harbor_annotations.csv')

    dtypes = {"folder_name": str, "clip_name": str,
              "object_id": np.int32, "y2": np.uint16}
    df = pd.read_csv(csv_annotations, usecols=dtypes.keys(), dtype=dtypes)

    # ignore far away ounding boxes
    y_max_dist = 130
    df = df[df.y2 > y_max_dist]

    df_grp = df.groupby(['folder_name', 'clip_name'], as_index=False).object_id.nunique()
    df_grp = utils.add_metadata_index(dataset.metadata, df_grp)
    df_grp = df_grp.sort_values(by=['object_id', 'folder_name', 'clip_name'],
                                ascending=[False, True, True])

    plot_high_density_distribution(df_grp)

    # filter manual inspected videos
    df_grp['joindir'] = df_grp.folder_name + os.sep + df_grp.clip_name
    df_grp = df_grp[~df_grp.joindir.isin(IGNORE_VIDEOS)]
    df_grp = df_grp.drop(columns='joindir')

    # save the 3% most dense videos for ignore training
    df_high_dense = df_grp.iloc[:int(0.03 * 8940)]
    df_high_dense = utils.add_metadata_index(dataset.metadata, df_high_dense)
    csv_videos_path = os.path.join(r'src/data/split/videos/', dataset.csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_high_dense.to_csv(csv_videos_path)
    print(f'high-density videos: {len(df_high_dense)}')


def plot_high_density_distribution(df):
    sns.histplot(data=df, x='object_id', binwidth=5)
    save_path = 'src/data/tmp/high_density_distribution.png'
    plt.xlabel("unique id's in video")
    plt.ylabel('frequency')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()


def select_test_videos(dataset, nr_test_videos: int = 10, nr_normal_videos: int = 5):
    split_videos_path = r'src/data/split/videos'
    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_videos_name)
    df_high_dense = pd.read_csv(csv_videos_path, dtype={'folder_name': str}, index_col='index')

    excludes = ['harbor_empty.csv',
                'harbor_appearance.csv',
                'harbor_fast_moving.csv',
                'harbor_near_edge.csv',
                # 'harbor_high_density.csv',
                'harbor_tampering.csv']
    for ex in excludes:
        csv_ex_path = os.path.join(split_videos_path, ex)
        df_high_dense = utils.exclude_videos(df_high_dense, csv_ex_path)

    df_80_vehicle_videos = df_high_dense.iloc[:10]

    # TODO: low density? or empty?
    # add 20 random scenes from low density set
    normal_csv_path = 'src/data/split/videos/harbor_train_low_density.csv'
    df_normal = pd.read_csv(normal_csv_path, index_col='index')
    df_20_normal = df_normal.sample(nr_normal_videos, random_state=0).sort_index()
    df_100_vehicle_videos = pd.concat([df_80_vehicle_videos, df_20_normal])

    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_100_vehicle_videos.to_csv(csv_videos_path)

    # # save high-density training set
    # df_train = df_grp.copy()
    # excludes = ['harbor_empty.csv',
    #             'harbor_vehicles.csv',
    #             'harbor_near_edge.csv',
    #             'harbor_high_density.csv',
    #             'harbor_test_100_high_density.csv',
    #             ]
    # for ex in excludes:
    #     csv_ex_path = os.path.join(split_videos_path, ex)
    #     df_train = exclude_videos(df_train, csv_ex_path)
    # # df_trn = df_trn.drop(columns='object_id')
    # df_train_100 = df_train.iloc[:100]
    # csv_train_name = 'harbor_train_high_density_0100.csv'
    # csv_train_path = os.path.join(split_videos_path, csv_train_name)
    # df_train_100.to_csv(csv_train_path, index=False)
    # utils.create_image_datasplit(cfg, csv_train_name, 'train')
    # utils.object_datasplit(cfg, 'harbor', csv_train_name, gt=False)
    # utils.object_datasplit(cfg, 'harbor', csv_train_name, gt=True)


def frame_level_ground_truth(dataset, sub_dir='image_dataset', overwrite: bool = False):
    gt_video_dir = dataset.frame_gt
    if overwrite:
        if os.path.exists(gt_video_dir):
            shutil.rmtree(gt_video_dir)
    os.makedirs(gt_video_dir, exist_ok=False)

    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df = pd.read_csv(csv_path)
    df['folder_name'] = df['folder_name'].astype(str)
    for idx, row in df.iterrows():
        video_name = row.folder_name + os.sep + row.clip_name
        normal = True if np.isnan(row.object_id) else False
        seq_gt = ground_truth_video_high_density(dataset, row.folder_name, row.clip_name, normal)

        # save to txt
        if sub_dir == 'image_dataset':
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
    dataset = harbor_dataset.HighDensity()

    create_high_density(dataset)
    select_test_videos(dataset, nr_test_videos=10, nr_normal_videos=5)

    utils.create_image_datasplit(dataset, dataset.csv_test)
    utils.object_datasplit(dataset, dataset.csv_test, gt=False)
    utils.object_datasplit(dataset, dataset.csv_test, gt=True)

    frame_level_ground_truth(dataset, overwrite=True)
    # no r/tbdc ground-truth due to no specific localization
