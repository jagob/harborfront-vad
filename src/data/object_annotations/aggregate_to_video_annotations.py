import os
import glob

from tqdm import tqdm
import pandas as pd

from src.utils.config import load_config


def read_txt(txt_filepath):
    names = ['object_id', 'name', 'x1', 'y1', 'x2', 'y2', 'occlusion']
    df = pd.read_csv(txt_filepath, sep=" ", names=names)
    return df


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    object_annotations_path = cfg['dataset']['harbor']['object_annotations']
    frame_annotations_path = os.path.join(object_annotations_path, 'annotations')
    date_folders = sorted(os.listdir(frame_annotations_path))
    df_dataset = pd.DataFrame({'folder_name': pd.Series(dtype='str'),
                               'clip_name': pd.Series(dtype='str'),
                               'frame_id': pd.Series(dtype='int'),
                               'object_id': pd.Series(dtype='int'),
                               'name': pd.Series(dtype='str'),
                               'x1': pd.Series(dtype='int'),
                               'y1': pd.Series(dtype='int'),
                               'x2': pd.Series(dtype='int'),
                               'y2': pd.Series(dtype='int'),
                               'occlusion': pd.Series(dtype='int')})
    for df in tqdm(date_folders):
        clip_folders = sorted(os.listdir(os.path.join(frame_annotations_path, df)))
        for cf in clip_folders:
            df_video = pd.DataFrame({'frame_id': pd.Series(dtype='int'),
                                     'object_id': pd.Series(dtype='int'),
                                     'name': pd.Series(dtype='str'),
                                     'x1': pd.Series(dtype='int'),
                                     'y1': pd.Series(dtype='int'),
                                     'x2': pd.Series(dtype='int'),
                                     'y2': pd.Series(dtype='int'),
                                     'occlusion': pd.Series(dtype='int')})
            frame_annotation_paths = sorted(glob.glob(os.path.join(
                frame_annotations_path, df, cf, '*.txt')))
            for fap in frame_annotation_paths:
                frame_id = os.path.splitext(os.path.basename(fap))[0]
                frame_id = int(frame_id.replace('annotations_', ''))
                df_frame = read_txt(fap)
                df_frame['frame_id'] = frame_id
                df_video = pd.concat([df_video, df_frame])

            # save video csv
            video_filename = f'{cf}.csv'
            video_csv_save_path = os.path.join(object_annotations_path,
                                               'video_annotations',
                                               df, video_filename)
            os.makedirs(os.path.dirname(video_csv_save_path), exist_ok=True)
            df_video = df_video.reset_index(drop=True)
            df_video.to_csv(video_csv_save_path, index=False)

            # prepare dataset csv
            df_video['clip_name'] = cf
            df_video['folder_name'] = df
            df_dataset = pd.concat([df_dataset, df_video])

    # save dataset csv
    dataset_filename = 'harbor_annotations.csv'
    dataset_csv_save_path = os.path.join(object_annotations_path, dataset_filename)
    df_dataset = df_dataset.reset_index(drop=True)
    df_dataset.to_csv(dataset_csv_save_path, index_label='index', index=False)
