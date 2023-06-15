import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import figaspect
from src.utils.config import load_config

sns.set_theme()
# sns.set_style("whitegrid")
# sns.set_style("dark")
sns.set_style("white")
sns.set_context("paper")

COLORS = sns.color_palette("deep", 8)
COLORS_JET = sns.color_palette("coolwarm_r", 11)

TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12



def plot_anomaly_score(models: list, width: int, title: str, gt_labels: list = None, frame_idx: int = None):
    # figure = plt.figure()
    figure = plt.figure(figsize=(width/100, (width//2)/100))
    # figure = plt.figure(figsize=(640/100, (640//2)/100))  # avenue
    # figure = plt.figure(figsize=(384/100, (384//2)/100))  # harbor
    # figure = plt.figure(figsize=(384/100*2, (384//3)/100*2))
    # figure = plt.figure(figsize=(384/10, (384//3)/10))
    # figure, ax = plt.subplots(figsize=(384/100, (384//2)/100), layout='constrained')
    # figure, ax = plt.subplots()

    # figure = plt.figure()
    # ax = plt.subplot(111)

    # w, h = figaspect(1/3)
    # __import__('ipdb').set_trace()
    # figure, ax = plt.subplots(figsize=(w,h))
    # xsize, ysize = figure.get_size_inches()

    if gt_labels:
        anomaly_intervals = get_anomaly_intervals(np.array(gt_labels))
        ax = plt.gca()
        for interval in anomaly_intervals:
            width = interval[1] - interval[0]
            ax.add_patch(Rectangle((interval[0], 0), width, 1, color='r', alpha=0.2, linewidth=0))
        plt.xlim(0, len(gt_labels) - 1)

    if frame_idx:
        plt.axvline(x=frame_idx, color='black')

    for model in models:
        video_scores = model['video_scores']
        if gt_labels and video_scores:
            assert len(gt_labels) == len(video_scores)
        if video_scores:
            plt.plot(video_scores, label=model['label'] + f' AUC {model["video_auc"]*100:.0f}%')

    plt.xlabel('Frames')
    # plt.ylabel('Anomaly score')
    plt.ylabel('Score')
    plt.ylim(-0.01, 1)
    plt.grid()
    plt.title(title)
    sns.despine(left=True, bottom=True)

    if len(models) > 0:
        # plt.legend(loc="upper right")

        plt.gca().set_position([0.1, 0.1, .8, .75])
        plt.legend(loc='upper center',
                   # bbox_to_anchor=(0.5, 1.2), ncol=4, fancybox=True, shadow=False)
                   bbox_to_anchor=(0.5, 1.3), ncol=2, fancybox=True, shadow=False)


    plt.rc('font', size=TINY_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TINY_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=TINY_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    return figure


def get_anomaly_intervals(label_orig: np.array):
    anomaly_intervals = []
    idx_start = 0
    idx_end = 0
    while idx_end != len(label_orig) - 1 and 1 in label_orig[idx_start:]:
        idx_start = idx_end + np.where(label_orig[idx_start:] == 1)[0][0]

        if 0 in label_orig[idx_start:]:
            anomaly_length = np.where(label_orig[idx_start:] == 0)[0][0] - 1
            idx_end = idx_start + anomaly_length
        else:
            idx_end = len(label_orig) - 1
        anomaly_intervals.append([idx_start, idx_end])
        idx_end += 1
        idx_start = idx_end
    return anomaly_intervals


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    # # model_dir = '/home/jacob/data/models/mnad/harbor/recon/one_week_001'
    # # results_path = os.path.join(model_dir, 'results_feb_week_5000.csv')

    # # model_dir = '/home/jacob/data/models/mnad/harbor-synthetic/recon/normal_001'
    # model_dir = '/home/jacob/data/models/mnad/harbor-synthetic/pred/normal_001'
    # # model_dir = r'/home/jacob/data/models/ssmtl++/harbor_synthetic_1_task'

    # # test_dataset = 'harbor-synthetic'
    # # results_path = os.path.join(model_dir, 'results_test_split.csv')
    # # test_dataset = 'harbor-mannequin'
    # # results_path = os.path.join(model_dir, 'results_harbor_mannequin_test_split.csv')
    # # test_dataset = 'harbor-realfall'
    # # results_path = os.path.join(model_dir, 'results_harbor_realfall_test_split.csv')
    # test_dataset = 'harbor-rareevents'
    # results_path = os.path.join(model_dir, 'results_test_split.csv')

    # # test_dataset = 'harbor'
    # # results_path = os.path.join(model_dir, 'results_harbor_test_split.csv')

    # gt_dir = os.path.join(config['dataset'][test_dataset]['path'], 'gt')
    # df = pd.read_csv(results_path)

    # # df = df[~df.img_path.str.endswith('00000.jpg')]  # ignore first black frame
    # df.MSE = (df.MSE - min(df.MSE)) / (max(df.MSE) - min(df.MSE))  # normalize

    # videos = df.img_path.apply(lambda x: os.path.dirname(x)).unique()
    # all_gt_labels = []
    # save_dir = os.path.join(model_dir, 'img', test_dataset)
    # print(f'save_dir: {save_dir}')
    # for folder_name in videos:
    #     if test_dataset in ['harbor-synthetic', 'harbor-mannequin', 'harbor-realfall']:
    #         gt_filepath = os.path.join(gt_dir, f"{folder_name.replace('test/', '')}.txt")
    #     elif test_dataset in ['harbor', 'harbor-rareevents']:
    #         gt_filename = folder_name + '.txt'
    #         # gt_filename = gt_filename.replace('test/', '')
    #         gt_filename = gt_filename.replace('/', '_')
    #         gt_filepath = os.path.join(gt_dir, gt_filename)
    #     else:
    #         __import__('ipdb').set_trace()
    #         raise

    #     try:
    #         gt_labels = np.loadtxt(gt_filepath, dtype=int).tolist()
    #         all_gt_labels.extend(gt_labels)
    #     except Exception:
    #         gt_labels = None

    #     df_fn = df[df.img_path.str.startswith(folder_name)]
    #     title = f'{folder_name}'
    #     video_scores = df_fn.MSE.to_list()

    #     figure = plot_anomaly_score(video_scores, title, gt_labels)
    #     save_path = os.path.join(save_dir, f'{folder_name}.png')
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path)
    #     # plt.show()
    #     plt.close()

    # title = 'all'
    # video_scores = df['MSE'].to_list()
    # figure = plot_anomaly_score(video_scores, title, all_gt_labels)

    # plot_anomaly_score(models: list, width: int, title: str, gt_labels: list = None, frame_idx: int = None):

    # save_path = os.path.join(model_dir, 'img', 'all.png')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)
    # # plt.show()
    # plt.close()

    models = []
    dataset = 'harbor-fast-moving'
    video_name = r'20210318/clip_43_2203'
    img_width = cfg.dataset[dataset].img_width
    gt_dir = cfg.dataset[dataset].gt_path
    if os.path.exists(gt_dir):
        if 'avenue' in dataset:
            gt_filename = os.path.basename(video_name) + '.txt'
        elif 'harbor' in dataset:
            gt_filename = video_name + '.txt'
            gt_filename = gt_filename.replace('/', '_')
        gt_filepath = os.path.join(gt_dir, gt_filename)
        # gt_labels = np.loadtxt(gt_filepath, dtype=int).tolist()
        gt_labels = np.loadtxt(gt_filepath).astype(int).tolist()

    figure = plot_anomaly_score(models, img_width, '', gt_labels)
    plot_save_path = r'src/data/tmp/video_anomaly_score.png'
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    figure.tight_layout(pad=0.2)
    plt.savefig(plot_save_path)
