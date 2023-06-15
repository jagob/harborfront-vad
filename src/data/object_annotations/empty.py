import os
import pandas as pd
from src.utils.config import load_config
from src.data.object_annotations.utils import add_metadata_index
pd.set_option('mode.chained_assignment', None)


def empty(cfg, csv_videos_name):
    # load dataset metadata csv
    meta_path = os.path.join(cfg['dataset']['harbor']['path'], 'metadata.csv')
    dtypes = {"Folder name": str, "Clip Name": str}
    df_meta = pd.read_csv(meta_path, usecols=dtypes.keys(), dtype=dtypes)
    df_meta = df_meta.rename(columns={"Folder name": "folder_name", "Clip Name": "clip_name"})
    df_meta['joindir'] = df_meta.folder_name + os.sep + df_meta.clip_name

    object_annotations_path = cfg['dataset']['harbor']['object_annotations']
    video_annotations_path = os.path.join(object_annotations_path)
    csv_path = os.path.join(video_annotations_path, 'harbor_annotations.csv')
    dtypes = {"folder_name": str, "clip_name": str}
    df = pd.read_csv(csv_path, usecols=dtypes.keys(), dtype=dtypes)
    df_anot_group = df[['folder_name', 'clip_name']] \
            .groupby(['folder_name', 'clip_name'], as_index=False) \
            .count()
    df_anot_group = add_metadata_index(cfg, df_anot_group)
    df_anot_group['joindir'] = df_anot_group.folder_name + os.sep + df_anot_group.clip_name

    df_merge = df_meta.merge(df_anot_group,
                             on='joindir', how='outer', indicator=True, suffixes=['', '_'])
    df_empty = df_merge[df_merge['_merge'] == 'left_only']
    df_empty = df_empty.drop(columns=['joindir', 'folder_name_', 'clip_name_', '_merge'])

    csv_videos_path = os.path.join(r'src/data/split/videos/', csv_videos_name)
    os.makedirs(os.path.dirname(csv_videos_path), exist_ok=True)
    df_empty.to_csv(csv_videos_path, index_label='index')
    print(f'empty videos: {len(df_empty)}')


if __name__ == "__main__":
    cfg = load_config('config.yaml')
    csv_videos_name = r'harbor_empty.csv'

    empty(cfg, csv_videos_name)
