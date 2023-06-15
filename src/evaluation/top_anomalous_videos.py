import os
import glob
import pandas as pd
from src.utils.config import load_config
from src.evaluation.visualize import visualize_video


def top_anomalous_videos(results_path: str, top_x: int = 10):
    df = pd.read_csv(results_path)
    df['dir'] = df.img_path.apply(lambda x: os.path.dirname(x).split('/')[0])
    df['subdir'] = df.img_path.apply(lambda x: os.path.dirname(x).split('/')[1])
    g_max_mse = df.groupby(['dir', 'subdir']).max("MSE")
    g_max_mse = g_max_mse.MSE.sort_values(ascending=False)
    top_videos = g_max_mse[:top_x]
    return top_videos


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    dataset_type = test_dataset = 'harbor-original'
    model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_001/')
    
    results_path = os.path.join(model_dir, 'results_harbor_train_split.csv')

    top_videos = top_anomalous_videos(results_path, 100)
    selected_video_names = ['/'.join([dir, subdir]) for dir, subdir in top_videos.index]

    dataset_path = cfg['dataset_path']
    dataset_test = os.path.join(dataset_path, 'harbor', 'image_dataset')

    save_dir = os.path.join(model_dir, 'visualization', dataset_type, 'top_anomalous_videos')

    gt_dir = os.path.join(cfg['dataset'][test_dataset]['path'], 'gt')
    if test_dataset == 'harbor':
        test_videos = sorted(glob.glob(os.path.join(dataset_test, '**/*')))
        test_videos = [x.lstrip(dataset_test) for x in test_videos]
        results_path = os.path.join(model_dir, 'results_harbor_test_split.csv')
    elif test_dataset == 'harbor-synthetic':
        test_videos = sorted(os.listdir(dataset_test))
        results_path = os.path.join(model_dir, 'results_test_split.csv')
    elif test_dataset == 'harbor-mannequin':
        test_videos = sorted(os.listdir(dataset_test))
        results_path = os.path.join(model_dir, 'results_harbor_mannequin_test_split.csv')
    elif test_dataset == 'harbor-realfall':
        test_videos = sorted(os.listdir(dataset_test))
        results_path = os.path.join(model_dir, 'results_harbor_realfall_test_split.csv')
    elif test_dataset == 'harbor-original':
        # test_videos = sorted(os.listdir(dataset_test))
        test_videos = sorted(glob.glob(os.path.join(dataset_test, '**/*')))
        test_videos = [x.lstrip(dataset_test) for x in test_videos]
        results_path = os.path.join(model_dir, 'results_harbor_train_split.csv')
    else:
        __import__('ipdb').set_trace()
        raise
    df = pd.read_csv(results_path)
    df.MSE = (df.MSE - min(df.MSE)) / (max(df.MSE) - min(df.MSE))  # normalize

    for video_name, mse in zip(selected_video_names, top_videos):
        if video_name not in test_videos:
            continue

        sequence = video_name
        images_sequence = glob.glob(os.path.join(dataset_test, video_name, '*.jpg'))
        images_sequence = sorted(images_sequence)

        if test_dataset in ['harbor-synthetic', 'harbor-mannequin', 'harbor-realfall']:
            df_video = df[df['img_path'].apply(lambda x: x.startswith('test/' + video_name))]

            gt_filepath = os.path.join(gt_dir, f"{video_name.replace('test/', '')}.txt")
        elif test_dataset == 'harbor':
            df_video = df[df['img_path'].apply(lambda x: x.startswith(video_name))]
            gt_filename = video_name + '.txt'
            gt_filename = gt_filename.replace('/', '_')
            gt_filepath = os.path.join(gt_dir, gt_filename)
        elif test_dataset == 'harbor-original':
            df_video = df[df['img_path'].apply(lambda x: x.startswith(video_name))]
        else:
            __import__('ipdb').set_trace()
            raise

        try:
            gt_labels = np.loadtxt(gt_filepath, dtype=int).tolist()
        except Exception:
            gt_labels = None

        video_scores = df_video.MSE.to_list()

        video_name = video_name.replace('/', '_')
        save_video_path = os.path.join(save_dir, f'mse{int(mse*100):04d}_{video_name}.avi')
        visualize_video(cfg, test_dataset, model_dir, sequence, images_sequence,
                save_video_path=save_video_path, video_scores=video_scores, gt_labels=gt_labels)
