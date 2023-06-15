import os
import pandas as pd
from src.utils.config import load_config
import src.data.object_annotations.utils as utils
import src.data.datasets.harbor as harbor_dataset

pd.set_option('mode.chained_assignment', None)

IGNORE_VIDEOS = [
                 '20200819/clip_23_1041',  # crane opertaing in front of camera
                 '20210213/clip_36_1627',  # camera freezes, stutters
                 '20210213/clip_37_1655',  # camera freezes, stutters
                 '20210305/clip_28_1238',  # camera freezes, stutters
                 '20210306/clip_40_2312',  # camera freezes, stutters
                 '20210307/clip_27_1239',  # camera freezes, stutters
                 '20210308/clip_14_0632',  # camera freezes, stutters
                 '20210425/clip_9_0412',  # camera freezes, stutters
                 ]


def low_density_videos(dataset):
    split_videos_path = r'src/data/split/videos'
    dtypes = {"Folder name": str, "Clip Name": str}
    df = pd.read_csv(dataset.metadata, usecols=dtypes.keys(), dtype=dtypes)
    df = df.rename(columns={"Folder name": "folder_name", "Clip Name": "clip_name"})

    # filter manual inspected videos
    df['joindir'] = df.folder_name + os.sep + df.clip_name
    df = df[~df.joindir.isin(IGNORE_VIDEOS)]
    df = df.drop(columns='joindir')

    excludes = ['harbor_empty.csv',
                'harbor_appearance.csv',
                'harbor_fast_moving.csv',
                'harbor_near_edge.csv',
                'harbor_high_density.csv',
                'harbor_tampering.csv']
    for ex in excludes:
        csv_ex_path = os.path.join(split_videos_path, ex)
        df = utils.exclude_videos(df, csv_ex_path)

    # exclude near empty videos / low activity, ignoring far away objects
    df = exclude_low_activity(dataset, df)
    df = exclude_top_only_activity(dataset, df)

    csv_save_path = os.path.join(split_videos_path, dataset.csv_train)
    df.to_csv(csv_save_path, index_label='index')


def exclude_low_activity(dataset, df):
    csv_path = os.path.join(dataset.object_annotations, 'harbor_annotations.csv')

    dtypes = {"folder_name": str, "clip_name": str, 'y2': int}
    df_anot = pd.read_csv(csv_path, usecols=dtypes.keys(), dtype=dtypes)

    y_max_dist = 130
    df_anot = df_anot[df_anot.y2 > y_max_dist]
    df_anot = df_anot.groupby(['folder_name', 'clip_name']).size().reset_index(name='det_count')
    df_anot = df_anot[df_anot.det_count < 5*120]
    df_anot = df_anot.drop(columns=['det_count'])

    df = utils.exclude_videos(df, df_anot)
    return df


def exclude_top_only_activity(datasetcfg, df):
    csv_path = os.path.join(dataset.object_annotations, 'harbor_annotations.csv')

    dtypes = {"folder_name": str, "clip_name": str, 'y2': int}
    df_anot = pd.read_csv(csv_path, usecols=dtypes.keys(), dtype=dtypes)

    y_max_dist = 130
    df_top = df_anot[df_anot.y2 <= y_max_dist]
    df_bot = df_anot[df_anot.y2 > y_max_dist]

    df_top = df_top.groupby(['folder_name', 'clip_name']).size().reset_index(name='det_count')
    df_bot = df_bot.groupby(['folder_name', 'clip_name']).size().reset_index(name='det_count')
    df_bot = df_bot[df_bot.det_count > 5]

    df_top = df_top.drop(columns=['det_count'])
    df_bot = df_bot.drop(columns=['det_count'])

    df_top = utils.exclude_videos(df_top, df_bot, verbose=False)
    df = utils.exclude_videos(df, df_top)
    return df


def subsets(dataset):
    split_videos_path = r'src/data/split/videos'
    df = pd.read_csv(os.path.join(split_videos_path, dataset.csv_train), index_col='index')
    csv_name = os.path.splitext(dataset.csv_train)[0]

    for nr_samples in [1000, 500, 200, 100, 50, 20, 10, 5]:
        df = df.sample(nr_samples, random_state=0).sort_index()
        csv_file = csv_name + f'_{len(df):04d}.csv'
        csv_save_path = os.path.join(split_videos_path, csv_file)
        df.to_csv(csv_save_path, index_label='index')

        # # if nr_samples <= 100:
        # #     utils.create_image_datasplit(cfg, 'harbor_05fps', csv_file, 'train', suffix='_05fps')  # fps test
        utils.create_image_datasplit(dataset, csv_file, 'train')
        utils.object_datasplit(dataset, csv_file, gt=False)
        utils.object_datasplit(dataset, csv_file, gt=True)


if __name__ == "__main__":
    dataset = harbor_dataset.Harbor()

    # low_density_videos(dataset)
    # utils.create_image_datasplit(dataset, dataset.csv_train, 'train')
    utils.object_datasplit(dataset, dataset.csv_train, gt=False)
    utils.object_datasplit(dataset, dataset.csv_train, gt=True)

    subsets(dataset)
