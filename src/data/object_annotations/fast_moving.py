# TODO: rename motion, both fast, or irregular
import os
import shutil

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.data.object_annotations.utils as utils
from src.utils.config import load_config
from src.utils.ipm import transform_df_points
from src.data.datasets.dataset import Dataset, get_dataset
from src.data.extract_frames import videocsv2splitcsv
from src.evaluation.plot_anomaly_score import get_anomaly_intervals

pd.set_option('mode.chained_assignment', None)
sns.set_theme()

# 20200627/clip_43_1926
# 20200628/clip_51_2244

IGNORE_VIDEOS = ['20200530/clip_29_2104',
                 '20210223/clip_18_1234',
                 '20200609/clip_15_0705',
                 '20210413/clip_39_1737',
                 '20200708/clip_31_1339',
                 '20200609/clip_15_0705',
                 '20210321/clip_19_1115',
                 '20200624/clip_9_1725',
                 '20200811/clip_40_1734',
                 '20200612/clip_19_2238',
                 '20210315/clip_17_0756',
                 '20200819/clip_17_0753',
                 '20200710/clip_43_1914',
                 '20210330/clip_32_1432',
                 '20210202/clip_29_1434',
                 '20210305/clip_20_0854',
                 '20200531/clip_39_1658',
                 '20200617/clip_17_0739',  # maybe
                 '20200630/clip_17_0749',  # maybe
                 '20210304/clip_26_1401',  # maybe
                 ]

FAST_MOVING_TEST_VIDEOS = [
    # '20200515/clip_29_1318',
    # '20210317/clip_37_1813',
    # '20210318/clip_43_2203',
    # '20210401/clip_3_0124',
    # '20210407/clip_42_1855',
    # '20210414/clip_43_1924',
    # '20210428/clip_14_0612',
    # '20210115_clip_14_0605',
    # '20200706_clip_10_1433',
    # '20200617_clip_17_0739',
]


def create_datasplit_fast_moving(dataset, velo_thresh):
    tmp_path = 'src/data/tmp/motion.csv'
    if os.path.exists(tmp_path):
    # if False:
        df_motion = pd.read_csv(tmp_path)
    else:
        csv_path = os.path.join(dataset.object_annotations, 'harbor_annotations.csv')
        dtypes = {"folder_name": str, "clip_name": str,
                  "object_id": np.int32, "name": str,
                  "x1": np.int16, "x2": np.int16, "y2": np.int16}
        df = pd.read_csv(csv_path, usecols=dtypes.keys(), dtype=dtypes)

        # ignore far away bounding boxes
        y_max_dist = 120
        df = df[df.y2 > y_max_dist]

        # filter for minimum nr of observations
        df = df.groupby(by=['folder_name', 'clip_name', 'object_id']).filter(
                lambda x: x['object_id'].count() >= 5)

        # x_center, use bottom center as point
        df['xc'] = (df.x2 + df.x1) // 2
        df = transform_df_points(df)

        data = []
        tic = time.time()

        # for name in tqdm(df.name.unique()):
        #     df_name = df[df.name == name]
        for folder in tqdm(df.folder_name.unique()):
            df_folder = df[df.folder_name == folder]
            for clip in df_folder.clip_name.unique():
                df_clip = df_folder[df_folder.clip_name == clip]
                for object_id in df_clip.object_id.unique():
                    df_object = df_clip[df_clip.object_id == object_id]
                    name = df_object.iloc[0]['name']

                    # # image coordinates
                    # x_diff = df_object.xc.diff()
                    # y_diff = df_object.y2.diff()

                    # world coordinates
                    x_diff = df_object.x_ipm.diff()
                    y_diff = df_object.y_ipm.diff()

                    dist = np.sqrt(x_diff**2 + y_diff**2)
                    # dist = dist.rolling(3).median()  # TODO: refactor
                    mean_velo = np.mean(abs(dist))
                    max_velo = np.max(abs(dist))
                    observations = len(df_object)
                    speed_observations = sum(dist > velo_thresh)
                    data.append([folder, clip, object_id, name, mean_velo, max_velo, observations, speed_observations])
        print(f'time: {(time.time() - tic):.0f} s')  # 221 s
        df_motion = pd.DataFrame(data,
                columns=['folder_name', 'clip_name', 'object_id', 'name', 'mean_velo', 'max_velo', 'observations', 'speed_observations'])
        df_motion.to_csv(tmp_path, float_format='%.2f', index=False)

    df_motion.folder_name = df_motion.folder_name.astype(str)
    df_motion = filter_motion(df_motion)
    # TODO: filter for normal-abnormal-normal pattern
    # df_motion = filter_anomaly_pattern_motion(cfg, df_motion, velo_thresh=23, nr_anom_threshold=1)

    # use plot to determine fast moving threshold
    plot_fast_moving_distribution(df_motion)
    df_motion = df_motion[df_motion.mean_velo > velo_thresh]

    # aggregate to video level
    df_videos = df_motion.groupby(['folder_name', 'clip_name']).max().reset_index()
    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_videos.to_csv(csv_videos_path)


def plot_fast_moving_distribution(df):
    sns.kdeplot(data=df, x='mean_velo', hue='name',
                hue_order=['human', 'bicycle', 'motorcycle', 'vehicle'])
                # legend='full', palette='deep')
    # plt.xlim(-5, 30)
    plt.savefig('src/data/tmp/kde_motion.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    # plt.show()
    plt.close()


def filter_motion(df_motion):
    df_motion = df_motion[df_motion.mean_velo > 0]
    # df_motion = df_motion[df_motion.max_velo < 100]
    # df_motion = df_motion[df_motion.observations >= 10]

    df_motion = df_motion[df_motion.observations >= 5]
    df_motion = df_motion[df_motion.speed_observations >= 5]
    return df_motion


def select_test_videos(dataset, nr_test_videos: int = 10):
    df_videos = pd.DataFrame([x.split(os.sep) for x in FAST_MOVING_TEST_VIDEOS],
                             columns=['folder_name', 'clip_name'])
    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    if len(df_videos) < nr_test_videos:
        csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_videos_name)
        df_motion = pd.read_csv(csv_videos_path, dtype={'folder_name': str}, index_col='index')

        df_motion['joindir'] = df_motion.folder_name + os.sep + df_motion.clip_name
        df_motion = df_motion[~df_motion.joindir.isin(IGNORE_VIDEOS)]
        # df_motion = df_motion[~df_motion.joindir.isin(TAMPERING_VIDEOS)]
        df_motion = df_motion.drop(columns='joindir')

        split_videos_path = r'src/data/split/videos'
        excludes = ['harbor_empty.csv',
                    'harbor_appearance.csv',
                    # 'harbor_fast_moving.csv',
                    'harbor_near_edge.csv',
                    'harbor_high_density.csv',
                    'harbor_tampering.csv']
        for ex in excludes:
            csv_ex_path = os.path.join(split_videos_path, ex)
            df_motion = utils.exclude_videos(df_motion, csv_ex_path)

        df_motion = df_motion.sort_values(by='mean_velo', ascending=False)
        df_test = df_motion.iloc[:nr_test_videos - len(df_videos)]
        for col in ['object_id', 'name', 'mean_velo', 'max_velo', 'observations', 'speed_observations']:
            df_videos[col] = ''
        df_videos = pd.concat([df_videos, df_test]).sort_index()

    df_videos = df_videos.sort_values(by='mean_velo', ascending=False)
    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_videos.to_csv(csv_videos_path)


# def filter_anomaly_pattern_motion(cfg, df: pd.DataFrame, velo_thresh: float, nr_anom_threshold: int = 1):
#     # df['nr_anom'] = -1

#     mask = []
#     for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#         # if row.clip_name != 'clip_29_2104':
#         #     continue
#         df_video = df[(df.folder_name == row.folder_name) &
#                       (df.clip_name == row.clip_name)]
#         anom_obj_ids = set(df_video[df_video.mean_velo > velo_thresh].object_id)
#         seq_gt = ground_truth_video_motion(cfg, row.folder_name, row.clip_name, anom_obj_ids)
#         anomaly_intervals = get_anomaly_intervals(seq_gt)

#         # # ignore anomalies smaller than 5 frames
#         anomaly_intervals = [[x1, x2] for x1, x2 in anomaly_intervals if x2 - x1 > 5]

#         # calc nr transitions
#         # # TODO: zero out small anomalies in array as well
#         # # TODO: consider removing starting and ending with anomalies
#         # # TODO: ignore anomalies starting at 0, or 1 in det case

#         keep_mask = True if len(anomaly_intervals) >= 1 else False
#         mask.append(keep_mask)
#         __import__('ipdb').set_trace()
#     # assert not (df.nr_anom == -1).any()
#     __import__('ipdb').set_trace()
#     df = df[df.nr_anom > nr_anom_threshold]
#     return df


def ground_truth_frame_level_object(cfg, csv_videos_path, velo_thresh: float):
    gt_video_dir = os.path.join(cfg['dataset_path'], 'harbor-fast-moving', 'gt')
    # TODO: OBS XXX
    os.makedirs(gt_video_dir, exist_ok=True)
    # os.makedirs(gt_video_dir, exist_ok=False)

    df_motion = pd.read_csv(csv_videos_path)
    df_motion.folder_name = df_motion.folder_name.astype(str)
    df_motion = filter_motion(df_motion)

    for _, row in df_motion.iterrows():
        df_video = df_motion[(df_motion.folder_name == row.folder_name) &
                             (df_motion.clip_name == row.clip_name)]
        anom_obj_ids = set(df_video[df_video.mean_velo > velo_thresh].object_id)
        # TODO obs, set of float
        seq_gt = ground_truth_video_motion(cfg, row.folder_name, row.clip_name, anom_obj_ids)

        # save to txt
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(gt_video_dir, f'{video_name}.txt')
        np.savetxt(gt_video_path, seq_gt, '%d')


def ground_truth_frame_level(cfg, dataset, velo_thresh: float, overwrite: bool = False):
    if overwrite:
        if os.path.exists(dataset.frame_gt):
            shutil.rmtree(dataset.frame_gt)
    os.makedirs(dataset.frame_gt, exist_ok=False)

    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df_motion = pd.read_csv(csv_path)
    df_motion.folder_name = df_motion.folder_name.astype(str)
    # df_motion = filter_motion(df_motion)

    for _, row in df_motion.iterrows():
        # df_video = df_motion[(df_motion.folder_name == row.folder_name) &
        #                      (df_motion.clip_name == row.clip_name)]
        # anom_obj_ids = set(df_video[df_video.mean_velo > velo_thresh].object_id)
        seq_gt = ground_truth_video_motion(cfg, dataset, row.folder_name, row.clip_name, velo_thresh)

        # save to txt
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(dataset.frame_gt, f'{video_name}.txt')
        np.savetxt(gt_video_path, seq_gt, '%d')


def ground_truth_video_motion(cfg, dataset, folder_name, clip_name, velo_thresh: float):
    video_annotations_path = os.path.join(dataset.object_annotations, 'video_annotations')
    video_csv_path = os.path.join(video_annotations_path, folder_name, f'{clip_name}.csv')
    df_video = pd.read_csv(video_csv_path)

    # add velocity
    # x_center, use bottom center as point
    df_video['xc'] = (df_video.x2 + df_video.x1) // 2
    df_video = transform_df_points(df_video)
    df_video['velo'] = np.nan
    for object_id in df_video.object_id.unique():
        df_object = df_video[df_video.object_id == object_id]

        # # image coordinates
        # x_diff = df_object.xc.diff()
        # y_diff = df_object.y2.diff()

        # world coordinates
        x_diff = df_object.x_ipm.diff()
        y_diff = df_object.y_ipm.diff()

        dist = np.sqrt(x_diff**2 + y_diff**2)
        # dist = dist.rolling(3).median()
        df_object['velo'] = dist
        df_video.update(df_object)

    # get frames from image folder
    img_dir = os.path.join(dataset.images, folder_name, clip_name)
    frame_names = sorted(os.listdir(img_dir))
    frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
    seq_gt = np.zeros(len(frame_ids), dtype=np.int8)
    for idx, frame_id in enumerate(frame_ids):
        # no motion for first frame
        if frame_id == 0:
            continue

        df_frame = df_video[df_video.frame_id == frame_id]

        # try:
        #     if not df_frame.velo.isnull().all():
        #         df_frame.velo.max() > velo_thresh
        # except:
        #     __import__('ipdb').set_trace()
        #     pass

        if not df_frame.velo.isnull().all():
            if df_frame.velo.max() > velo_thresh:
                seq_gt[frame_id] = 1
    return seq_gt


# def ground_truth_video_motion_object(cfg, folder_name, clip_name, anom_obj_ids):
#     img_dir = os.path.join(cfg['dataset']['harbor']['path'],
#                            cfg['dataset']['harbor']['img_dir'],
#                            folder_name, clip_name)

#     object_annotations_path = cfg['dataset']['harbor']['object_annotations']
#     video_annotations_path = os.path.join(object_annotations_path, 'video_annotations')
#     video_csv_path = os.path.join(
#             video_annotations_path, folder_name, f'{clip_name}.csv')
#     df_video = pd.read_csv(video_csv_path)

#     # offset between object annotations and images
#     offset = 1

#     # get frames from image folder
#     frame_names = sorted(os.listdir(img_dir))
#     frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
#     seq_gt = np.zeros(len(frame_ids), dtype=np.int8)
#     for frame_id in frame_ids:
#         df_frame = df_video[df_video.frame_id == (frame_id - offset)]
#         if len(anom_obj_ids.intersection(set(df_frame.object_id))) > 0:
#             seq_gt[frame_id] = 1

#     # object annotations not annotated for frame 121?
#     # assume the last frame is similar to next last
#     # seq_gt[-1] = seq_gt[-2]
#     # assume the first frame is similar to second frame
#     seq_gt[0] = seq_gt[1]
#     return seq_gt


if __name__ == "__main__":
    cfg = load_config('config.yaml')
    dataset = get_dataset(Dataset.harbor_fast_moving)

    csv_videos_name = dataset.csv_videos_name
    csv_name = dataset.csv_test
    # csv_name = r'harbor_test_100_fast_moving.csv'

    create_datasplit_fast_moving(dataset, velo_thresh=750)
    select_test_videos(dataset, nr_test_videos=10)

    # create image datasplit
    utils.create_image_datasplit(dataset, dataset.csv_test)
    utils.object_datasplit(dataset, dataset.csv_test, gt=False)
    utils.object_datasplit(dataset, dataset.csv_test, gt=True)

    # frame-level gt
    # csv_videos_path = os.path.join(r'src/data/split/videos', csv_name)
    # ground_truth_frame_level_object(cfg, csv_videos_path, velo_thresh=23)
    ground_truth_frame_level(cfg, dataset, velo_thresh=750, overwrite=True)

    # region and track based gt
    # ground_truth_from_object_annotation_motion(cfg, dataset, csv_videos_path)
    # rbdc_tbdc_ground_truth()
