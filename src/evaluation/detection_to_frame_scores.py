import os
import pandas as pd
from tqdm import tqdm
from src.utils.config import load_config


def det_to_frame_scores(obj_csv):
    df_obj = pd.read_csv(obj_csv)

    # get frame csv from obj_csv
    dirname = os.path.dirname(obj_csv)
    basename, ext = os.path.splitext(os.path.basename(obj_csv))
    filename = basename.replace('_object', '') + ext
    results_filename = 'results_' + filename
    csv_save_path = os.path.join(dirname, results_filename)
    if os.path.exists(csv_save_path):
        print(f'FileExists: {csv_save_path}')
        return

    df_obj['joindir'] = df_obj.image_id.apply(lambda x: os.path.dirname(x))
    df_img2 = df_obj.groupby(['image_id'])['score'].max().reset_index(name='score_max')

    frame_csv = os.path.join(r'src/data/split', filename)
    df = pd.read_csv(frame_csv)
    df['score'] = 0.0

    # TODO: update requires index, loop it is
    for idx, row in tqdm(df_img2.iterrows(), total=df.shape[0]):
        df_matches = df[df.img_path == row.image_id]
        if len(df_matches) == 1:
            df_idx = df_matches.index[0]
            df.at[df_idx, 'score'] = row.score_max
        elif len(df_matches) > 1:
            __import__('ipdb').set_trace()

    df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    # single file
    # obj_csv = os.path.join(cfg['models_path'],
    #         r'ssmtl++/harbor_low_density_1_task_trans/per_epoch_predictions/1/harbor_test_100_high_density_object.csv')
    obj_csv = os.path.join(cfg['models_path'],
            r'/home/jacob/data/models/kde/harbor/low_density_0100/bw05.0/harbor_test_100_near_edge_object.csv')
    det_to_frame_scores(obj_csv)

    # # dataset = 'harbor-mannequin'
    # # dataset = 'harbor-realfall'
    # # dataset = 'harbor-near-edge'
    # # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_1_task')
    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_1_task_trans')
    # # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task')
    # epoch_dirs = sorted(os.listdir(os.path.join(model_dir, 'per_epoch_predictions', dataset)))
    # for epoch_dir in epoch_dirs:
    #     obj_csvs = sorted(glob.glob(os.path.join(model_dir, 'per_epoch_predictions', dataset, epoch_dir, '*_object.csv')))
    #     for obj_csv in obj_csvs:
    #         det_to_frame_scores(obj_csv)
