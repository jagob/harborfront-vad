import os
import sys
import glob
import argparse
import yaml
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.config import load_config

np.random.seed(seed=1234)

parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('--harborsynthetic', type=str, help="Input path to harbor-synthetic dataset")
parser.add_argument('--datasets', type=str, help="Output path to datasets")
parser.add_argument('--dataset', type=str, help="Output path to datasets")
parser.add_argument('--archive_path', type=str, help="Output path to datasets")


def extract_frames(config, video_dir, dataset_save_path, split):
    video_names = sorted(os.listdir(video_dir))
    for video_name in tqdm(video_names):
        img_save_dir = os.path.join(dataset_save_path, split, os.path.splitext(video_name)[0])
        os.makedirs(img_save_dir, exist_ok=True)

        video_path = os.path.join(video_dir, video_name)
        video2images(video_path, img_save_dir)


def video2images(video_path, img_save_dir):
    cap = cv2.VideoCapture(video_path)
    frame_nr = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        img_name = f'{frame_nr:05d}.jpg'
        img_path = os.path.join(img_save_dir, img_name)
        cv2.imwrite(img_path, frame)
        frame_nr += 1
    cap.release()
    cv2.destroyAllWindows()


def videocsv2splitcsv(df_videos, dataset_img_dir, split, split_ratio=0.85):
    # TODO: pass in list of videos instead of joindir crap
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


def dataset2csv_split(dataset_save_path, split, split_ratio=None):
    if 'synthetic' in dataset_save_path:
        pattern = f'{split}/**/*.jpg'
    else:
        pattern = f'**/**/*.jpg'
    img_paths = sorted(glob.glob(os.path.join(dataset_save_path, pattern)))
    if not img_paths:
        raise

    df = pd.DataFrame(img_paths, columns=['img_path'])
    df.img_path = df.img_path.str.replace(dataset_save_path + '/', '')

    if split == 'train':
        df = split_dataframe(df)
    return df


# def split_dataframe(df, split_ratio=None):
#     # optimize needed
#     df['split'] = ''
#     if split_ratio is not None:
#         videos = df.img_path.apply(lambda x: os.path.dirname(x)).unique()
#         mask = np.random.rand(len(videos)) < split_ratio
#         print('Creating datasplit')
#         for idx, video in enumerate(tqdm(videos)):
#             df.loc[df['img_path'].str.startswith(video), 'split'] = 'train' if mask[idx] else 'val'
#     return df


if __name__ == "__main__":
    # python extract_frames.py --harborsynthetic /home/jacob/data/harbor-synthetic/Normal --datasets /home/jacob/data
    args = parser.parse_args()
    config = load_config('config.yaml')
    dataset_save_path = config['dataset_path']

    # https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/datasets/DATASET.md
    # ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"

    # if len(sys.argv) == 1:
    #     video_dir = r'/home/jacob/data/harbor-synthetic/Normal'
    # else:
    #     video_dir = args.harborsynthetic
    # extract_frames(video_dir, dataset_save_path, 'train')
    # extract_frames(r'/home/jacob/data/harbor-synthetic/OneDrive_2_19-08-2022/Bicycle Fall', dataset_save_path, 'test')
    # extract_frames(r'/home/jacob/data/harbor-synthetic/OneDrive_2_19-08-2022/Fall1', dataset_save_path, 'test')
    # extract_frames(r'/home/jacob/data/harbor-synthetic/OneDrive_2_19-08-2022/Fire In House', dataset_save_path, 'test')
    # extract_frames(r'/home/jacob/data/harbor-synthetic/OneDrive_2_19-08-2022/Skateboard Fall', dataset_save_path, 'test')
    # extract_frames(r'/home/jacob/data/harbor-synthetic/OneDrive_2_19-08-2022/Wheel Chair Fall', dataset_save_path, 'test')

    # # TODO: verify data is correct

    # df_trn = dataset2csv_split(dataset_save_path, 'train', split_ratio=0.9)
    # df_trn.to_csv('normal_split.csv', index=False)
    # df_tst = dataset2csv_split(dataset_save_path, 'test')
    # df_tst.to_csv('test_split.csv', index=False)

    # harbor_path = os.path.join(config['dataset']['harbor']['path'],
    #                            'image_dataset')
    # # TODO should use gt files to look up which videos to use
    # df_harbor_tst = dataset2csv_split(harbor_path, 'test')
    # df_harbor_tst.to_csv('src/data/harbor_test_split.csv', index=False)

    # # remove annotated test videos from training data
    # harbor_path = os.path.join(config['dataset']['harbor']['path'],
    #                            'image_dataset')
    # df_harbor_train = dataset2csv_split(harbor_path, '')
    # df_harbor_tst = pd.read_csv('src/data/harbor_test_split.csv')
    # dt_harbor_train = pd.concat([df_harbor_train, df_harbor_tst])
    # df_harbor_train = df_harbor_train.drop_duplicates(keep=False)
    # df_harbor_train = split_dataframe(df_harbor_train, split_ratio=0.9)
    # df_harbor_train.to_csv('src/data/harbor_train_split.csv', index=False)
    # df_harbor_train['dir'] = df_harbor_train.img_path.apply(lambda x: os.path.dirname(x).split('/')[0])
    # df_harbor_train['subdir'] = df_harbor_train.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    # df_harbor_train['joindir'] = df_harbor_train.dir + '/' + df_harbor_train.subdir
    # print(f'harbor train, videos: {len(df_harbor_train.joindir.unique())}, frames: {len(df_harbor_train)}')

    # # Optimize using groupby wip
    # harbor_path = os.path.join(config['dataset']['harbor']['path'],
    #                            'image_dataset',
    #                            'metadata_images.csv')
    # df_harbor_train = pd.read_csv(harbor_path)
    # df_harbor_train['Folder name', 'Clip Name', 'Image Number'].groupby(['Folder name', 'Clip Name']).count()
    # # https://stackoverflow.com/questions/12200693/python-pandas-how-to-assign-groupby-operation-results-back-to-columns-in-parent

    # # extract training/normal videos for harbor-rareevents
    # df_harbor_tst = pd.read_csv('src/data/harbor_test_split.csv')
    # df_harbor_tst['dir'] = df_harbor_tst.img_path.apply(lambda x: os.path.dirname(x).split('/')[0])
    # df_harbor_tst['subdir'] = df_harbor_tst.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    # df_harbor_tst['joindir'] = df_harbor_tst.dir + '/' + df_harbor_tst.subdir
    # print(f'rareevents test, videos: {len(df_harbor_tst.joindir.unique())}, frames: {len(df_harbor_tst)}')
    # video_dates = df_harbor_tst.img_path.apply( lambda x: x.split('/')[0]).unique()
    # df_harbor_train = pd.read_csv('src/data/harbor_train_split.csv')
    # df_harbor_train['dir'] = df_harbor_train.img_path.apply( lambda x: os.path.dirname(x).split('/')[0])
    # df_harbor_train['subdir'] = df_harbor_train.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    # df_harbor_train['joindir'] = df_harbor_train.dir + '/' + df_harbor_train.subdir
    # df_rareevents_train = df_harbor_train[df_harbor_train.dir.isin(video_dates)]
    # print(f'rareevents train, videos: {len(df_rareevents_train.joindir.unique())}, frames: {len(df_rareevents_train)}')
    # df_rareevents_train = df_rareevents_train.drop(columns=['dir', 'subdir', 'joindir'])
    # df_rareevents_train = split_dataframe(df_rareevents_train, split_ratio=0.9)
    # df_rareevents_train.to_csv('src/data/harbor_rareevents_train_split.csv', index=False)


    # # https://www.kaggle.com/ivannikolov/longterm-thermal-drift-dataset
    # # https://www.kaggle.com/datasets/ivannikolov/thermal-mannequin-fall-image-dataset
    # if len(sys.argv) == 1:
    #     dataset = 'harbor-mannequin'
    #     archive_path = r'/home/jacob/data/archive.zip'
    # else:
    #     dataset = args.dataset
    #     archive_path = args.archive_path
    # extract_dir = os.path.join(config['dataset_path'], dataset)
    # shutil.unpack_archive(archive_path, extract_dir)
    # video_dir = os.path.join(config['dataset_path'], dataset, 'videos')
    # extract_frames(config, video_dir, extract_dir, 'test')

    # mannequin_imgdir = os.path.join(config['dataset'][dataset]['path'],
    #                                 config['dataset'][dataset]['img_dir'])
    # mannequin_videos = sorted(os.listdir(mannequin_imgdir))
    # df_mannequin_videos = pd.DataFrame(mannequin_videos, columns=['clip_name'])
    # df_mannequin_videos['folder_name'] = ''
    # df_mannequin_videos = df_mannequin_videos[['folder_name', 'clip_name']]
    # df_mannequin_videos.to_csv('src/data/split/videos/harbor_test_mannequin.csv', index=False)

    # TODO: dont include imd_dir 'test' in img_path
    # df_mannequin = dataset2csv_split(extract_dir, 'test')
    # df_mannequin.to_csv('src/data/split/harbor_test_mannequin.csv', index=False)
    # df_mannequin['dir'] = df_mannequin.img_path.apply(lambda x: os.path.dirname(x).split('/')[0])
    # df_mannequin['subdir'] = df_mannequin.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    # df_mannequin['joindir'] = df_mannequin.dir + '/' + df_mannequin.subdir
    # print(f'mannequin test, videos: {len(df_mannequin.joindir.unique())}, frames: {len(df_mannequin)}')


    # if len(sys.argv) == 1:
    #     dataset = 'harbor-realfall'
    #     archive_path = r'/home/jacob/data/real_fall_dataset.tar.gz'
    # else:
    #     dataset = args.dataset
    #     archive_path = args.archive_path
    # extract_dir = os.path.join(config['dataset_path'], dataset)
    # os.rename(archive_path, archive_path.replace('.tar.gz', '.zip'))
    # archive_path = archive_path.replace('.tar.gz', '.zip')
    # shutil.unpack_archive(archive_path, extract_dir)
    # shutil.rmtree(os.path.join(extract_dir, 'real_fall', 'Eksporter 07-03-2016 09-46-19'))
    # delete_files = glob.glob(os.path.join(
    #     extract_dir, 'real_fall', 'all_mkv', 'location_a*.mkv'))
    # for file in sorted(delete_files):
    #     os.remove(file)
    # shutil.move(os.path.join(extract_dir, 'real_fall', 'all_mkv'),
    #             os.path.join(extract_dir, 'videos'))
    # shutil.rmtree(os.path.join(extract_dir, 'real_fall'))
    # video_dir = os.path.join(config['dataset_path'], dataset, 'videos')
    # extract_frames(config, video_dir, extract_dir, 'test')

    # realfall_imgdir = os.path.join(config['dataset'][dataset]['path'],
    #                                config['dataset'][dataset]['img_dir'])
    # realfall_videos = sorted(os.listdir(realfall_imgdir))
    # df_realfall_videos = pd.DataFrame(realfall_videos, columns=['clip_name'])
    # df_realfall_videos['folder_name'] = ''
    # df_realfall_videos = df_realfall_videos[['folder_name', 'clip_name']]
    # df_realfall_videos.to_csv('src/data/split/videos/harbor_test_realfall.csv', index=False)

    # TODO: dont include imd_dir 'test' in img_path
    # df_realfall = dataset2csv_split(extract_dir, 'test')
    # df_realfall.to_csv('src/data/split/harbor_test_realfall.csv', index=False)
    # df_realfall['dir'] = df_realfall.img_path.apply(lambda x: os.path.dirname(x).split('/')[0])
    # df_realfall['subdir'] = df_realfall.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    # df_realfall['joindir'] = df_realfall.dir + '/' + df_realfall.subdir
    # print(f'realfall test, videos: {len(df_realfall.joindir.unique())}, frames: {len(df_realfall)}')


    # # detection dataset to split csv
    # video_csv_paths = [r'src/data/video_annotations/empty_videos.csv']
    # dataset_path = config['dataset']['harbor']['path']
    # dataset_img_dir = os.path.join(dataset_path, config['dataset']['harbor']['img_dir'])
    # df_videos = pd.read_csv(video_csv_path)
    # df_empty = videocsv2splitcsv(df_videos, dataset_img_dir, 'train')
    # df_empty.to_csv('src/data/split/harbor_train_empty.csv', index=False)


    # # detection dataset to split csv
    # csv_name = 'harbor_train_low_density.csv'
    # video_csv_paths = [r'src/data/video_annotations/videos_10.csv',
    #                    r'src/data/video_annotations/videos_100.csv',
    #                    r'src/data/video_annotations/videos_500.csv',
    #                    r'src/data/video_annotations/videos_1000.csv']
    # dataset_path = config['dataset']['harbor']['path']
    # dataset_img_dir = os.path.join(dataset_path, config['dataset']['harbor']['img_dir'])
    # df_from_each_file = (pd.read_csv(f, index_col=0) for f in video_csv_paths)
    # df_videos = pd.concat(df_from_each_file)
    # df_videos.joindir = df_videos.joindir.str.replace('.mp4', '')
    # df_videos = df_videos[df_videos.comment.isna()]

    # csv_videos_path = os.path.join(r'src/data/split/videos', csv_name)
    # # df_videos['folder_name'] = df_videos.joindir.apply(lambda x: os.path.dirname(x).split('/')[0])
    # df_videos['folder_name'] = df_videos.joindir.apply(lambda x: os.path.dirname(x))
    # df_videos['clip_name'] = df_videos.joindir.apply(lambda x: os.path.basename(x))
    # df_videos = df_videos[['folder_name', 'clip_name', 'joindir', 'count', 'comment']]
    # os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    # df_videos.to_csv(csv_videos_path, index=False)

    # df_empty = videocsv2splitcsv(df_videos, dataset_img_dir, 'train')
    # csv_path = os.path.join('src/data/split', csv_name)
    # df_empty.to_csv(csv_path, index=False)


    # # detection dataset to split csv car|truck|boat
    # video_csv_paths = [r'src/data/video_annotations/videos_10.csv',
    #                    r'src/data/video_annotations/videos_100.csv',
    #                    r'src/data/video_annotations/videos_500.csv',
    #                    r'src/data/video_annotations/videos_1000.csv']
    # dataset_path = config['dataset']['harbor']['path']
    # dataset_img_dir = os.path.join(dataset_path, config['dataset']['harbor']['img_dir'])
    # df_from_each_file = (pd.read_csv(f, index_col=0) for f in video_csv_paths)
    # df_videos = pd.concat(df_from_each_file)
    # df_videos.joindir = df_videos.joindir.str.replace('.mp4', '')
    # # df_videos = df_videos[df_videos.comment.isna()]
    # regex_pattern = 'car|truck|boat'
    # # df_videos = df_videos.astype(str).replace(np.nan, '')
    # df_videos = df_videos.comment..replace(np.nan, '')
    # __import__('ipdb').set_trace()
    # # TODO: convert nan to string, empty probably
    # df_videos = df_videos.astype(str).replace(np.nan, '')
    # mask = df_videos.comment.str.contains(regex_pattern)

    # df_train = df_videos[mask]
    # df_train = videocsv2splitcsv(df_train, dataset_img_dir, 'train')
    # df_train.to_csv('src/data/split/harbor_train_large_objects.csv', index=False)

    # df_test = df_videos[~mask]
    # df_train = videocsv2splitcsv(df_train, dataset_img_dir, 'test')
    # df_train.to_csv('src/data/split/harbor_test_large_objects.csv', index=False)
    # df_videos.joindir = df_videos.joindir.str.replace('.mp4', '')
    # df_videos = df_videos[df_videos.comment.isna()]
    # df_empty = videocsv2splitcsv(df_videos, dataset_img_dir, 'train')
    # df_empty.to_csv('src/data/split/harbor_train_low_density.csv', index=False)

    # dataset = 'ucsdped2'
    # dataset_path = config['dataset'][dataset]['path']
    # for split in ['train', 'test']:
    #     pattern = os.path.join(dataset_path, f'{split}/frames/*')
    #     video_paths = sorted(glob.glob(pattern))
    #     video_paths = [x.replace(dataset_path + os.sep, '') for x in video_paths]
    #     df_split = videocsv2splitcsv(video_paths, dataset_path, split)
    #     df_split.to_csv(f'src/data/split/ucsdped2_{split}.csv', index=False)
