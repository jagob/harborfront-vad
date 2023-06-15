import os
import math
import glob

from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.config import load_config


def roundup(x, nearest):
    return int(math.ceil(x / float(nearest))) * nearest


if __name__ == "__main__":
    dataset = 'harbor'
    binwidth = 200
    cfg = load_config('config.yaml')

    detections_dir = os.path.join(cfg['detection']['dir'], 'yolov5', dataset)
    csv_paths = sorted(glob.glob(os.path.join(detections_dir, '**/*.csv')))
    if dataset == 'harbor' and len(csv_paths) != 8940:
        __import__('ipdb').set_trace()
        raise

    data = []
    for csv_path in tqdm(csv_paths):
        dir = os.path.basename(os.path.dirname(csv_path))
        subdir = os.path.basename(os.path.splitext(csv_path)[0])
        joindir = os.path.join(dir, subdir)

        det_count = len(pd.read_csv(csv_path))
        data.append([joindir, det_count])
    df = pd.DataFrame(data, columns=['joindir', 'count'])
    print(df)

    # # save video for annotation
    df['comment'] = ""
    # empty_videos = df[df['count'] == 0]
    # empty_videos.to_csv('empty_videos.csv', index=True)
    # 20210122/clip_12_0536.mp4  # not empty
    # videos_1 = df[df['count'] == 1]
    # videos_1.joindir = videos_1.joindir + '.mp4'
    # # videos_1.to_csv('videos_1.csv', index=True)
    # videos_10 = df[df['count'] == 10]
    # videos_10.joindir = videos_10.joindir + '.mp4'
    # videos_10.to_csv('videos_10.csv', index=True)
    # videos_100 = df[df['count'] >= 100]
    # videos_100 = videos_100[videos_100['count'] <= 110]
    # videos_100.joindir = videos_100.joindir + '.mp4'
    # videos_100.to_csv('videos_100.csv', index=True)
    # videos_500 = df[df['count'] >= 500]
    # videos_500 = videos_500[videos_500['count'] <= 520]
    # videos_500.joindir = videos_500.joindir + '.mp4'
    # videos_500.to_csv('videos_500.csv', index=True)
    # videos_1000 = df[df['count'] >= 1000]
    # videos_1000 = videos_1000[videos_1000['count'] <= 1020]
    # videos_1000.joindir = videos_1000.joindir + '.mp4'
    # videos_1000.to_csv('videos_1000.csv', index=True)
    __import__('ipdb').set_trace()

    sns.set_theme()
    max_bin = roundup(max(df['count']), binwidth) + 1
    bins = [-binwidth, 1] + list(range(binwidth, max_bin, binwidth))
    print(bins)
    sns.histplot(df, bins=bins, shrink=0.95, label=f'bin width {binwidth}')
    plt.xlabel("Video detections")
    plt.legend()

    plot_save_path = os.path.join(detections_dir, 'video_histogram.png')
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    print(f'plot_save_path: {plot_save_path}')
    plt.show()
    plt.close()
    __import__('ipdb').set_trace()
