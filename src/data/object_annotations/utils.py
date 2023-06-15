import os

import pandas as pd

from src.utils.config import load_config
from src.data.datasplit import videocsv2splitcsv


def add_metadata_index(csv_meta_path, df):
    dtypes = {"Folder name": str, "Clip Name": str}
    df_meta = pd.read_csv(csv_meta_path, usecols=dtypes.keys(), dtype=dtypes)
    df_meta = df_meta.rename(columns={"Folder name": "folder_name",
                                      "Clip Name": "clip_name"})

    # add index
    df['index'] = -1
    for idx, row in df.iterrows():
        match = df_meta.index[(df_meta.folder_name == row.folder_name) &
                              (df_meta.clip_name == row.clip_name)].tolist()
        assert len(match) == 1
        df.at[idx, 'index'] = match[0]
    df = df.set_index('index')
    assert (df.index != -1).all()
    return df


def add_metadata(csv_meta_path, df):
    df_meta = pd.read_csv(csv_meta_path, dtype={'Folder name': str})
    df_meta = df_meta.rename(columns={"Folder name": "folder_name",
                                      "Clip Name": "clip_name"})

    # add metadata
    # columns = ['DateTime',
    #            'Temperature',
    #            'Humidity',
    #            'Precipitation latest 10 min',
    #            'Dew Point',
    #            'Wind Direction',
    #            'Wind Speed',
    #            'Sun Radiation Intensity',
    #            'Min of sunshine latest 10 min']
    columns = ['DateTime', 'Temperature', 'Humidity']
    df = df.join(df_meta[columns])
    return df


def exclude_videos(df, csv_path, verbose=True):
    if isinstance(csv_path, str):
        df_ex = pd.read_csv(csv_path, dtype={'folder_name': str}, index_col='index')
        if csv_path.endswith('appearance.csv'):
            df_ex = df_ex.drop(columns=['size_max'])
        elif csv_path.endswith('fast_moving.csv'):
            df_ex = df_ex.drop(columns=['object_id', 'name', 'mean_velo', 'max_velo', 'observations', 'speed_observations'])
        elif csv_path.endswith('near_edge.csv'):
            df_ex = df_ex.drop(columns=['near_edge', 'near_near_edge', 'nr_anom', 'first_anom_frame', 'longest_anomaly'])
        elif csv_path.endswith('high_density.csv'):
            df_ex = df_ex.drop(columns=['object_id'])
    else:
        df_ex = csv_path
    df_ex['joindir'] = df_ex.folder_name + os.sep + df_ex.clip_name
    df['joindir'] = df.folder_name + os.sep + df.clip_name

    df['index'] = df.index
    df_merge = df.merge(df_ex,
                        on='joindir', how='outer', indicator=True, suffixes=['', '_'])
    df = df_merge[df_merge['_merge'] == 'left_only']
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')
    # TODO: remove merge columns without hardcoding
    df = df.drop(columns=['joindir', 'folder_name_', 'clip_name_', '_merge'])

    if verbose:
        print(len(df))
    return df


def create_image_datasplit(dataset, csv_name, split='test', suffix=''):
    video_csv_path = os.path.join(r'src/data/split/videos', csv_name)
    df_videos = pd.read_csv(video_csv_path, dtype={'folder_name': str})
    df_videos = df_videos.sort_values(['folder_name', 'clip_name'])

    videos_list = (df_videos.folder_name + os.sep + df_videos.clip_name).to_list()
    df_img = videocsv2splitcsv(videos_list, dataset.images, split)
    csv_save_name = csv_name.replace('.csv', suffix + '.csv')
    csv_path = os.path.join(r'src/data/split', csv_save_name)
    df_img.to_csv(csv_path, index=False)


def object_datasplit(dataset, csv_name: str, gt: bool = False, meta: bool = False):
    dtypes = {"folder_name": str, "clip_name": str, 'index': int}
    videos_csv_path = os.path.join(r'src/data/split/videos', csv_name)
    # df = pd.read_csv(videos_csv_path, usecols=dtypes.keys(), dtype=dtypes, keep_default_na=False)
    df = pd.read_csv(videos_csv_path, usecols=dtypes.keys(), dtype=dtypes, index_col='index')
    if meta:
        df = add_metadata(dataset.metadata, df)

    df_obj = pd.DataFrame({'image_id': pd.Series(dtype='str'),
                           'bbox_x1': pd.Series(dtype='int'),
                           'bbox_y1': pd.Series(dtype='int'),
                           'bbox_w': pd.Series(dtype='int'),
                           'bbox_h': pd.Series(dtype='int'),
                           'confidence': pd.Series(dtype='float'),
                           'class': pd.Series(dtype='int'),
                           'name': pd.Series(dtype='str'),
                           'object_id': pd.Series(dtype='int'),  # gt track id
                           })

    # for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    for _, row in df.iterrows():
        if gt:
            # TODO check image annodation offset
            annotations_path = dataset.object_annotations
            video_csv_path = os.path.join(annotations_path, 'video_annotations', row.folder_name, f'{row.clip_name}.csv')
            df_vid = pd.read_csv(video_csv_path)

            # format
            # image_id,bbox_x1,bbox_y1,bbox_w,bbox_h,confidence,class,name
            # 01/0000.jpg,479,115,37,126,0.8289,0,person
            # image_id,bbox_x1,bbox_y1,bbox_w,bbox_h,confidence,class,name
            # cyclefall-2022060717-4453203/00001.jpg,136,200,20,55,0.7613,0,person
            # TODO OBS OBS OBS, annotations offset +1
            # df_vid['image_path'] = df_vid['frame_id'].apply(lambda x: f'image_{x+1:04}.jpg')  # harbor specific
            df_vid['image_path'] = df_vid['frame_id'].apply(lambda x: f'image_{x:04}.jpg')  # harbor specific
            df_vid['image_path'] = row.clip_name + os.sep + df_vid['image_path']
            df_vid['image_path'] = row.folder_name + os.sep + df_vid['image_path']
            df_vid['bbox_w'] = df_vid.x2 - df_vid.x1
            df_vid['bbox_h'] = df_vid.y2 - df_vid.y1
            df_vid = df_vid.rename(columns={"image_path": "image_id", "x1": "bbox_x1", "y1": "bbox_y1"})
            df_vid['confidence'] = 1.0  # ground-truth boxes
            df_vid['class'] = -1  # TODO: lookup coco table
            df_vid = df_vid.drop(columns=['frame_id', 'x2', 'y2', 'occlusion'])
            df_vid = df_vid[['image_id', 'bbox_x1', 'bbox_y1', 'bbox_w', 'bbox_h',
                             'confidence', 'class', 'name', 'object_id']]
        else:
            detections_path = dataset.yolov5
            video_csv_path = os.path.join(detections_path, row.folder_name, f'{row.clip_name}.csv')
            df_vid = pd.read_csv(video_csv_path)

        # # add metadata
        # df_vid['DateTime'] = df.loc[row.name, 'DateTime']
        # df_vid['Temperature'] = df.loc[row.name, 'Temperature']
        # df_vid['Humidity'] = df.loc[row.name, 'Humidity']

        df_obj = pd.concat([df_obj, df_vid])

    ddirname = os.path.dirname(os.path.dirname(videos_csv_path))
    basename, _ = os.path.splitext(csv_name)
    obj_filename = f'{basename}_object_gt.csv' if gt else f'{basename}_object.csv'
    object_csv_path = os.path.join(ddirname, obj_filename)
    df_obj.to_csv(object_csv_path, index=False)


if __name__ == "__main__":
    # from video split to object split
    cfg = load_config('config.yaml')

    csv_names = [
                 r'harbor_train_low_density.csv',
                 r'harbor_test_100_vehicles.csv',
                 r'harbor_test_100_near_edge.csv',
                 r'harbor_test_100_high_density.csv',
                  ]
    for csv_name in csv_names:
        object_datasplit(cfg, 'harbor', csv_name, gt=False)
        object_datasplit(cfg, 'harbor', csv_name, gt=True)

    dataset = 'harbor-mannequin'
    videos_csv_path = r'src/data/split/videos/harbor_test_mannequin.csv'
    object_datasplit(cfg, dataset, videos_csv_path)

    dataset = 'harbor-realfall'
    videos_csv_path = r'src/data/split/videos/harbor_test_realfall.csv'
    object_datasplit(cfg, dataset, videos_csv_path)
