import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.data.object_annotations.utils as utils
from src.utils.config import load_config
from src.data.datasets.dataset import Dataset, get_dataset
from src.data.object_annotations.tampering import TAMPERING_VIDEOS
from src.evaluation.plot_anomaly_score import get_anomaly_intervals

pd.set_option('mode.chained_assignment', None)
sns.set_theme()


IGNORE_VIDEOS = []


def find_appearance_videos(dataset):
    # load dataset csv
    csv_path = os.path.join(dataset.object_annotations, 'harbor_annotations.csv')
    df_anot = pd.read_csv(csv_path, dtype={'folder_name': str})
    df_anot['size'] = (df_anot.x2 - df_anot.x1) * (df_anot.y2 - df_anot.y1)

    # plot_save_path = 'src/data/tmp/size_distribution.png'
    # plot_size_distribution(df_anot, plot_save_path)

    df_videos = df_anot.groupby(['folder_name', 'clip_name'])['size']\
        .max().reset_index(name='size_max')
    df_videos = df_videos.sort_values(['size_max'], ascending=False)
    df_videos = df_videos[df_videos.size_max > 10e3]

    df_videos = utils.add_metadata_index(dataset.metadata, df_videos)

    csv_videos_path = os.path.join(r'src/data/split/videos/', dataset.csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_videos.to_csv(csv_videos_path)
    return


def select_test_videos(dataset):
    csv_videos_path = os.path.join(r'src/data/split/videos', dataset.csv_videos_name)
    df_videos = pd.read_csv(csv_videos_path, dtype={'folder_name': str}, index_col='index')

    tmp_path = 'src/data/tmp/appearance.csv'
    if not os.path.exists(tmp_path):
    # if True:
        # filter videos where gt is not starting with normal
        df_videos = filter_gt_transition(dataset, df_videos)

        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        df_videos.to_csv(tmp_path)
    df_videos = pd.read_csv(tmp_path, dtype={'folder_name': str}, index_col='index')
    df_videos = df_videos[df_videos.first_anom_frame > 9]
    df_videos = df_videos[df_videos.longest_anomaly >= 5]
    print(f'{len(df_videos)} appearance videos to select from')
    df_videos = df_videos.drop(columns=['first_anom_frame', 'longest_anomaly'])

    # sub sample to have some leiroom to manually ignore videos
    # df_sub = df_videos.sample(20, random_state=0)
    df_sub = df_videos.iloc[:20]
    # filter manual inspected videos
    df_sub['joindir'] = df_sub.folder_name + os.sep + df_sub.clip_name
    df_sub = df_sub[~df_sub.joindir.isin(IGNORE_VIDEOS)]
    df_sub = df_sub[~df_sub.joindir.isin(TAMPERING_VIDEOS)]
    df_sub = df_sub.drop(columns='joindir')

    # sample final test set
    df_tst = df_sub.iloc[:10]
    df_tst = df_tst.sort_values(['size_max', 'folder_name', 'clip_name'],
                                ascending=[False, True, True])
    csv_videos_path = os.path.join(split_videos_path, dataset.csv_test)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_tst.to_csv(csv_videos_path)


def plot_size_distribution(df_anot, save_path):
    sns.set_theme()
    # sns.displot(df_anot, x="size", hue="name", kind="kde", fill=True)
    sns.kdeplot(data=df_anot, x='size',
                hue='name', hue_order=['human', 'bicycle', 'motorcycle', 'vehicle'])

    plt.xlim([-1000, 15000])
    plt.ylim([-1e-6, 5.0e-5])
    # plt.grid()
    plt.xlabel('object size')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="upper right")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def filter_gt_transition(dataset, df_vids: pd.DataFrame) -> pd.DataFrame:
    object_annotations_path = dataset.object_annotations
    video_annotations_path = os.path.join(object_annotations_path, 'video_annotations')

    df_vids['first_anom_frame'] = -1
    df_vids['longest_anomaly'] = -1
    for idx, row in tqdm(df_vids.iterrows(), total=df_vids.shape[0]):
        video_csv_path = os.path.join(
                video_annotations_path, row.folder_name, f'{row.clip_name}.csv')
        df_anot_video = pd.read_csv(video_csv_path)

        seq_gt = ground_truth_single_video(row.folder_name, row.clip_name, df_anot_video)
        first_anom_frame = -1 if sum(seq_gt) < 1 else np.argmax(seq_gt > 0)
        df_vids.at[idx, 'first_anom_frame'] = first_anom_frame

        anomaly_intervals = get_anomaly_intervals(seq_gt)
        if anomaly_intervals:
            longest_anomaly = max([x2 - x1 for x1, x2 in anomaly_intervals]) + 1
        else:
            longest_anomaly = 0
        df_vids.at[idx, 'longest_anomaly'] = longest_anomaly

    # assert (df_vids.first_anom_frame != -1).all()
    assert (df_vids.longest_anomaly != -1).all()

    return df_vids


def frame_level_ground_truth(dataset, overwrite: bool = False):
    gt_video_dir = dataset.frame_gt
    if overwrite:
        if os.path.exists(gt_video_dir):
            shutil.rmtree(gt_video_dir)
    os.makedirs(gt_video_dir, exist_ok=False)

    object_annotations_path = dataset.object_annotations
    video_annotations_path = os.path.join(object_annotations_path, 'video_annotations')

    csv_name = dataset.csv_test
    csv_path = os.path.join(r'src/data/split/videos', csv_name)
    df_videos = pd.read_csv(csv_path, dtype={'folder_name': str})

    for idx, row in df_videos.iterrows():
        video_csv_path = os.path.join(
                video_annotations_path, row.folder_name, f'{row.clip_name}.csv')
        df_video = pd.read_csv(video_csv_path)
        df_vehicle = df_video[df_video.name == 'vehicle']

        seq_gt = ground_truth_single_video(dataset, row.folder_name, row.clip_name, df_vehicle)

        # save to txt
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(gt_video_dir, f'{video_name}.txt')
        np.savetxt(gt_video_path, seq_gt, '%d')


def ground_truth_single_video(dataset, folder_name, clip_name, df_anot, thresh=10e3):
    df_anot['size'] = (df_anot.x2 - df_anot.x1) * (df_anot.y2 - df_anot.y1)

    img_dir = os.path.join(dataset.images, folder_name, clip_name)

    frame_names = sorted(os.listdir(img_dir))
    frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
    seq_gt = np.zeros(len(frame_ids), dtype=np.int8)
    for frame_id in frame_ids:
        df_anot_frame = df_anot[df_anot.frame_id == frame_id]
        df_anot_frame = df_anot_frame[df_anot_frame['size'] > thresh]
        if not df_anot_frame.empty:
            seq_gt[frame_id] = 1

    return seq_gt


def rbdc_tbdc_ground_truth(dataset, thresh=10e3, overwrite: bool = False):
    gt_video_dir = dataset.rtbdc_gt
    if overwrite:
        if os.path.exists(gt_video_dir):
            shutil.rmtree(gt_video_dir)
    os.makedirs(gt_video_dir, exist_ok=False)

    object_annotations_path = dataset.object_annotations
    video_annotations_path = os.path.join(object_annotations_path, 'video_annotations')

    # get frame names from split csv
    num_frames = 0
    csv_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    df = pd.read_csv(csv_path, dtype={'folder_name': str})
    for idx, row in df.iterrows():
        img_dir = os.path.join(dataset.images, row.folder_name, row.clip_name)

        video_csv_path = os.path.join(
                video_annotations_path, row.folder_name, f'{row.clip_name}.csv')
        df_anot = pd.read_csv(video_csv_path)
        df_anot['size'] = (df_anot.x2 - df_anot.x1) * (df_anot.y2 - df_anot.y1)
        df_anot = df_anot[df_anot['size'] > thresh]

        # re-number object_ids, to count from 0...
        object_ids = df_anot.object_id.unique()
        obj_remap = {i: j for i, j in zip(object_ids, range(len(object_ids)))}
        df_anot.object_id = df_anot.object_id.replace(obj_remap)

        frame_names = sorted(os.listdir(img_dir))
        frame_ids = [int(x.lstrip('image_').rstrip('.jpg')) for x in frame_names]
        num_frames += len(frame_names)
        data = []
        for frame_id in frame_ids:
            df_frame = df_anot[df_anot.frame_id == frame_id]
            for idx, veh in df_frame.iterrows():
                # track_id,frame_idx,x_min,y_min,x_max,y_max
                # 0,102,0,232,37,298
                data.append([veh.object_id, frame_id, veh.x1, veh.y1, veh.x2, veh.y2])

        # TODO sort track_id/object_id, to be inline with ShanghaiTech
        # save to txt
        video_name = row.folder_name + '_' + row.clip_name
        gt_video_path = os.path.join(gt_video_dir, f'{video_name}.txt')
        data_arr = np.array(data)
        np.savetxt(gt_video_path, data_arr, '%d', delimiter=',')
    print(f'num_frames: {num_frames}')


if __name__ == "__main__":
    cfg = load_config('config.yaml')
    dataset = get_dataset(Dataset.harbor_appearance)

    find_appearance_videos(dataset)
    select_test_videos(dataset)

    utils.create_image_datasplit(dataset, dataset.csv_test)
    utils.object_datasplit(dataset, dataset.csv_test, gt=False)
    utils.object_datasplit(dataset, dataset.csv_test, gt=True)

    frame_level_ground_truth(dataset, overwrite=True)
    rbdc_tbdc_ground_truth(dataset, overwrite=True)
