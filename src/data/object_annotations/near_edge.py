import os
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

from src.utils.config import load_config
from src.data.datasets.dataset import Dataset, get_dataset
import src.data.datasets.harbor as harbor_dataset
import src.data.object_annotations.utils as utils
from src.evaluation.plot_anomaly_score import get_anomaly_intervals

pd.set_option('mode.chained_assignment', None)
sns.set_theme()

NEAR_EDGE_TEST_VIDEOS = ['20200612/clip_11_1921',  # kid far away, use to threshold
                         '20210121/clip_42_2132',  # walking close, and throwing stuff in water
                         '20210316/clip_18_1104',  # back and forth
                         '20210425/clip_32_1449',  # walking close, and throwing stuff in water
                         '20210321/clip_8_0344',   # walking close
                         '20210316/clip_16_1008',  # walk to and sit on edge, get back up leave
                         '20200831/clip_26_1143',  # walk to edge look down
                         '20200624/clip_10_1753',  # walk to and sit on lamp
                         '20200625/clip_10_0440',  # jumping off van into water
                         '20200707/clip_46_2029',  # group leaving restaurant towards edge
                         # '20200520/clip_33_1437',  # potential, need closer fit
                         # '20200615/clip_48_2125',  # padle board
                         # '20210407/clip_25_1114',  # walking close
                         ]

IGNORE_VIDEOS = ['20200531/clip_39_1658',  # wrong annotations
                 '20200601/clip_36_1614',  # imprecise annotations
                 '20210219/clip_44_2110',  # borderline
                 '20210326/clip_50_2230',  # borderline
                 '20210414/clip_23_1030',  # borderline
                 '20210416/clip_32_1416',  # borderline
                 '20210424/clip_27_1210',  # borderline
                 '20210425/clip_39_1754',  # borderline
                 '20210426/clip_32_1427',  # wrong annotations
                 '20210427/clip_31_1401',  # borderline
                 '20210430/clip_31_1350',  # borderline
                 '20200526/clip_38_1713',  # wrong annotations
                 '20210420/clip_35_1534',  # borderline
                 '20210425/clip_40_1759',  # borderline
                 '20210427/clip_30_1333',  # borderline
                 '20210427_clip_20_0916',  # borderline fishing
                 '20200519_clip_37_1642',  # borderline fishing
                 '20200520_clip_46_2014',  # wrong annotations
                 '20200523_clip_38_2010',  # borderline fishing
                 '20200612_clip_11_1921',  # borderline
                 ]


def create_datasplit_edge(dataset):
    # near-edge image line using two points
    # TODO: on change, manually insert a and b in near_edge function
    x1, y1, x2, y2 = (15, 0, 152, 287)  # near-edge
    # x1, y1, x2, y2 = (15+10, 0, 152+10, 287)  # near-near-edge

    # # y = ax + b
    # a = (y2 - y1) / float(x2 - x1)
    # b = y1 - (a * x1)
    # b2 = y2 - (a * x2)
    # np.testing.assert_almost_equal(b, b2)
    # __import__('ipdb').set_trace()

    # # create image with boundaries
    # plot_edge_in_image(x1, y1, x2, y2)

    # load dataset csv
    dtypes = {"Folder name": str, "Clip Name": str}
    df_meta = pd.read_csv(dataset.metadata, usecols=dtypes.keys(), dtype=dtypes)
    # df_meta = pd.read_csv(meta_path, dtype={'Folder name': str})
    df_meta = df_meta.rename(columns={"Folder name": "folder_name", "Clip Name": "clip_name"})
    df_meta['near_edge'] = -1
    df_meta['near_near_edge'] = -1
    df_meta['nr_anom'] = -1
    df_meta['first_anom_frame'] = -1
    df_meta['longest_anomaly'] = -1

    tmp_path = 'src/data/tmp/metadata_near_edge.csv'
    # 40 min for loop
    if not os.path.exists(tmp_path):
    # if True:
        for idx, row in tqdm(df_meta.iterrows(), total=df_meta.shape[0]):
            near_near_edge = is_near_near_edge(cfg, row.folder_name, row.clip_name)
            df_meta.at[idx, 'near_near_edge'] = near_near_edge

            seq_gt = frame_level_ground_truth_single_vid1(cfg, row.folder_name, row.clip_name)
            if sum(seq_gt) > 0:
                df_meta.at[idx, 'near_edge'] = 1
            else:
                df_meta.at[idx, 'near_edge'] = 0

            first_anom_frame = -1 if sum(seq_gt) < 1 else np.argmax(seq_gt > 0)
            df_meta.at[idx, 'first_anom_frame'] = first_anom_frame

            anomaly_intervals = get_anomaly_intervals(seq_gt)

            nr_anomalies = len(anomaly_intervals)
            df_meta.at[idx, 'nr_anom'] = nr_anomalies

            if anomaly_intervals:
                longest_anomaly = max([x2 - x1 for x1, x2 in anomaly_intervals])
                longest_anomaly += 1
            else:
                longest_anomaly = 0
            df_meta.at[idx, 'longest_anomaly'] = longest_anomaly

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        df_meta.to_csv(tmp_path, index=False)
        assert not (df_meta.near_edge == -1).any()
        assert not (df_meta.nr_anom == -1).any()
        assert not (df_meta.longest_anomaly == -1).any()

    df_videos = pd.read_csv(tmp_path, dtype={'folder_name': str})
    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    # save all near-edge videos
    df_near_edge_videos = df_videos[df_videos.near_near_edge == True]
    csv_videos_path = os.path.join(r'src/data/split/videos/', dataset.csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_near_edge_videos.to_csv(csv_videos_path)
    print(f'near-edge videos: {len(df_near_edge_videos)}')


def plot_edge_in_image(x1, y1, x2, y2):
    img_path = r'src/data/20200816_clip_1_0028_image_0043.jpg'
    img = cv2.imread(img_path)
    img = cv2.line(img, (0, 130), (383, 130), (0, 0, 255), 1)
    img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # near-edge
    img = cv2.line(img, (x1 + 10, y1), (x2 + 10, y2), (0, 255, 255), 1)  # near-near-edge
    cv2.imwrite('src/data/tmp/edge.png', img)


def near_edge(x, y, a=2.102189781021898, b=-31.532846715328468):
    if y < 130 or y > 245:
        return False

    res_y = a * x + b
    res_x = (y - b) / a
    if y > res_y and x < res_x:
        return True
    else:
        return False


def is_near_near_edge(cfg, folder_name: str, clip_name: str) -> int:
    img_dir = os.path.join(cfg['dataset']['harbor']['path'],
                           cfg['dataset']['harbor']['img_dir'],
                           folder_name, clip_name)

    # offset between object annotations and images
    offset = 0

    object_annotations_path = cfg['dataset']['harbor']['object_annotations']
    video_annotations_path = os.path.join(object_annotations_path, 'video_annotations')
    video_csv_path = os.path.join(
            video_annotations_path, folder_name, f'{clip_name}.csv')
    df_video = pd.read_csv(video_csv_path)

    frame_names = sorted(os.listdir(img_dir))
    frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
    for frame_id in frame_ids:
        df_frame = df_video[df_video.frame_id == (frame_id - offset)]
        if not df_frame.empty:
            for idx, row in df_frame.iterrows():
                is_near_edge = near_edge((row.x2 + row.x1) // 2, row.y2,
                                         a=2.0948905109489053, b=-52.37226277372263)
                if is_near_edge:
                    return 1
    return 0


def select_test_videos(dataset, nr_test_videos=10):
    df_videos = pd.DataFrame([x.split(os.sep) for x in NEAR_EDGE_TEST_VIDEOS],
                             columns=['folder_name', 'clip_name'])
    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    if len(df_videos) < nr_test_videos:
        csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_videos_name)
        df_edge = pd.read_csv(csv_videos_path, dtype={'folder_name': str}, index_col='index')

        df_edge['joindir'] = df_edge.folder_name + os.sep + df_edge.clip_name
        df_edge = df_edge[~df_edge.joindir.isin(IGNORE_VIDEOS)]
        df_edge = df_edge.drop(columns='joindir')

        df_edge2 = df_edge.copy()

        split_videos_path = r'src/data/split/videos'
        excludes = ['harbor_empty.csv',
                    'harbor_appearance.csv',
                    'harbor_fast_moving.csv',
                    # 'harbor_near_edge.csv',
                    'harbor_high_density.csv',
                    'harbor_tampering.csv']
        for ex in excludes:
            csv_ex_path = os.path.join(split_videos_path, ex)
            df_edge = utils.exclude_videos(df_edge, csv_ex_path)

        df_edge = df_edge[df_edge.first_anom_frame > 9]
        # df_edge = df_edge[df_edge.nr_anom == 1]
        df_edge = df_edge[df_edge.longest_anomaly >= 5]

        df_test = df_edge.sample(nr_test_videos - len(df_videos), random_state=0)
        df_videos = pd.concat([df_videos, df_test]).sort_index()

    # df_videos = df_videos.sort_values(by='mean_velo', ascending=False)
    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_videos.to_csv(csv_videos_path)


def frame_level_ground_truth(dataset, overwrite: bool = False):
    if overwrite:
        if os.path.exists(dataset.frame_gt):
            shutil.rmtree(dataset.frame_gt)
    os.makedirs(dataset.frame_gt, exist_ok=False)

    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df = pd.read_csv(csv_path, dtype={'folder_name': str})
    for idx, row in df.iterrows():
        seq_gt = frame_level_ground_truth_single_vid1(dataset, row.folder_name, row.clip_name)

        # save to txt
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(dataset.frame_gt, f'{video_name}.txt')
        np.savetxt(gt_video_path, seq_gt, '%d')


def frame_level_ground_truth_single_vid1(dataset, folder_name: str, clip_name: str):
    video_annotations_path = os.path.join(dataset.object_annotations, 'video_annotations')
    video_csv_path = os.path.join(video_annotations_path, folder_name, f'{clip_name}.csv')
    df_video = pd.read_csv(video_csv_path)

    img_dir = os.path.join(dataset.images, folder_name, clip_name)
    frame_names = sorted(os.listdir(img_dir))
    frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
    seq_gt = np.zeros(len(frame_ids), dtype=np.int8)
    for frame_id in frame_ids:
        df_frame = df_video[df_video.frame_id == frame_id]
        if not df_frame.empty:
            # TODO: ignore small anomalies
            df_frame['near_edge'] = df_frame.apply(
                    lambda x: near_edge((x.x2 + x.x1) // 2, x.y2), axis=1)
            df_near_edge = df_frame[df_frame.near_edge == True]

            if not df_near_edge.empty:
                seq_gt[frame_id] = 1

    return seq_gt


def rbdc_tbdc_ground_truth(dataset, overwrite: bool = False):
    if overwrite:
        if os.path.exists(dataset.rtbdc_gt):
            shutil.rmtree(dataset.rtbdc_gt)
    os.makedirs(dataset.rtbdc_gt, exist_ok=False)

    video_annotations_path = os.path.join(dataset.object_annotations, 'video_annotations')

    # get frame names from split csv
    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df = pd.read_csv(csv_path, dtype={'folder_name': str})
    num_frames = 0
    for idx, row in df.iterrows():

        video_csv_path = os.path.join(
                video_annotations_path, row.folder_name, f'{row.clip_name}.csv')
        df_video = pd.read_csv(video_csv_path)

        img_dir = os.path.join(dataset.images, row.folder_name, row.clip_name)
        frame_names = sorted(os.listdir(img_dir))
        frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
        num_frames += len(frame_names)
        df_anom = pd.DataFrame().reindex(columns=df_video.columns)
        data = []
        for frame_id in frame_ids:
            df_frame = df_video[df_video.frame_id == frame_id]
            for _, det in df_frame.iterrows():
                if near_edge((det.x2 + det.x1) // 2, det.y2):
                    data.append(det)
        df_anom = pd.DataFrame(data, columns=['object_id', 'frame_id', 'x1', 'y1', 'x2', 'y2'])
        df_anom.frame_id = df_anom.frame_id

        # TODO: does sorting matter?
        df_anom = df_anom.sort_values(by=['object_id', 'frame_id'])

        # re-number object_ids, to count from 0...
        object_ids = df_anom.object_id.unique()
        obj_remap = {i: j for i, j in zip(object_ids, range(len(object_ids)))}
        df_anom.object_id = df_anom.object_id.replace(obj_remap)

        # save to txt
        # track_id,frame_idx,x_min,y_min,x_max,y_max
        # 0,102,0,232,37,298
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(dataset.rtbdc_gt, f'{video_name}.txt')
        df_anom.to_csv(gt_video_path, header=None, index=None, sep=',')
    print(f'num_frames: {num_frames}')


if __name__ == "__main__":
    dataset = harbor_dataset.NearEdge()

    create_datasplit_edge(dataset)
    select_test_videos(dataset, nr_test_videos=10)

    utils.create_image_datasplit(dataset, dataset.csv_test)
    utils.object_datasplit(dataset, dataset.csv_test, gt=False)
    utils.object_datasplit(dataset, dataset.csv_test, gt=True)

    frame_level_ground_truth(dataset, overwrite=True)
    rbdc_tbdc_ground_truth(dataset, overwrite=True)
