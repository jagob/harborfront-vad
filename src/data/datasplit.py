import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.config import load_config

np.random.seed(seed=1234)


def videocsv2splitcsv(df_videos, dataset_img_dir, split, split_ratio=0.85):
    # TODO: pass in list of videos instead of joindir
    try:
        video_list = df_videos.joindir
    except:
        video_list = df_videos
    img_paths = []
    # for video_path in df_videos.joindir:
    for video_path in video_list:
        video_img_paths = sorted(glob.glob(os.path.join(dataset_img_dir, video_path, '*.jpg')))
        if not video_img_paths:
            print(video_path)
            __import__('ipdb').set_trace()
            raise
        img_paths.extend(video_img_paths)

    df = pd.DataFrame(img_paths, columns=['img_path'])
    df.img_path = df.img_path.str.replace(dataset_img_dir + '/', '')

    # todo: optimize to split on video already
    if split == 'train':
        df = split_dataframe(df, split_ratio)
    return df


def split_dataframe(df, split_ratio=None):
    # optimize needed
    df['split'] = ''
    if split_ratio is not None:
        videos = df.img_path.apply(lambda x: os.path.dirname(x)).unique()
        mask = np.random.rand(len(videos)) < split_ratio
        print('Creating datasplit')
        for idx, video in enumerate(tqdm(videos)):
            df.loc[df['img_path'].str.startswith(video), 'split'] = 'train' if mask[idx] else 'val'
    return df


if __name__ == "__main__":
    cfg = load_config('config.yaml')
    dataset = 'avenue'

    dataset_path = cfg.dataset[dataset].path
    for split in ['train', 'test']:
        split_path = split
        split_path = split_path.replace('train', 'training')
        split_path = split_path.replace('test', 'testing')

        # video datasplit
        video_paths = sorted(glob.glob(os.path.join(dataset_path, split_path, 'frames', '*')))
        video_paths = [x.replace(dataset_path + os.sep, '') for x in video_paths]
        df_videos = pd.DataFrame(video_paths, columns=['joindir'])
        df_videos['folder_name'] = df_videos.joindir.apply(lambda x: os.path.dirname(x))
        df_videos['clip_name'] = df_videos.joindir.apply(lambda x: os.path.basename(x))
        df_videos = df_videos.drop(columns=['joindir'])

        csv_videos_name = f'{dataset}_{split}.csv'
        csv_videos_path = os.path.join(r'src/data/split/videos', csv_videos_name)
        df_videos.to_csv(csv_videos_path, index_label='index')

        # image datasplit
        df_split = videocsv2splitcsv(video_paths, dataset_path, split)
        df_split.to_csv(f'src/data/split/avenue_{split}.csv', index=False)
