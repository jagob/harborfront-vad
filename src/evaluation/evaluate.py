import os
import math
import shutil

# import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as skmetr

from src.utils.config import load_config
from src.evaluation.compute_tbdc_rbdc import compute_metrics
from src.evaluation.detection_to_frame_scores import det_to_frame_scores
import src.data.datasets.harbor as harbor_dataset

pd.set_option('mode.chained_assignment', None)


def evaluate(cfg, test_dataset: str, model_dir: str, epoch_dir: str = None, verbose: bool = True, alpha = 0.6):
    if epoch_dir is None:
        results_path = os.path.join(model_dir, 'results_' + cfg['dataset'][test_dataset]['csv_test'])

        csv_obj_name, ext = os.path.splitext(cfg['dataset'][test_dataset]['csv_test'])
        csv_obj_name = csv_obj_name + '_object' + ext
        csv_obj_path = os.path.join(model_dir, csv_obj_name)
        if not os.path.exists(results_path) and os.path.exists(csv_obj_path):
            print(f'object to frame level: {csv_obj_path}')
            det_to_frame_scores(csv_obj_path)
    else:
        if 'mnad' in model_dir:
            # results_path = os.path.join(model_dir, f'epoch_{int(epoch_dir):06}', 'results_' + cfg['dataset'][test_dataset]['csv_test'])
            results_path = os.path.join(model_dir, epoch_dir, 'results_' + cfg['dataset'][test_dataset]['csv_test'])
        elif 'ssmtl' in model_dir:
            results_path = os.path.join(model_dir, 'per_epoch_predictions', test_dataset, epoch_dir,
                                        'results_' + cfg['dataset'][test_dataset]['csv_test'])

            csv_obj_name, ext = os.path.splitext(cfg['dataset'][test_dataset]['csv_test'])
            csv_obj_name = csv_obj_name + '_object' + ext
            csv_obj_path = os.path.join(model_dir, 'per_epoch_predictions', test_dataset, epoch_dir,
                                        csv_obj_name)
            if not os.path.exists(results_path) and os.path.exists(csv_obj_path):
                print(f'object to frame level: {csv_obj_path}')
                det_to_frame_scores(csv_obj_path)
        elif 'kde' in model_dir:
            results_path = os.path.join(model_dir, epoch_dir, 'results_' + cfg['dataset'][test_dataset]['csv_test'])

            csv_obj_name, ext = os.path.splitext(cfg['dataset'][test_dataset]['csv_test'])
            csv_obj_name = csv_obj_name + '_object' + ext
            csv_obj_path = os.path.join(model_dir, epoch_dir, csv_obj_name)
            if not os.path.exists(results_path) and os.path.exists(csv_obj_path):
                print(f'object to frame level: {csv_obj_path}')
                det_to_frame_scores(csv_obj_path)

    if not os.path.exists(results_path):
        __import__('ipdb').set_trace()
        return -1, -1, -1, -1

    df = pd.read_csv(results_path)
    if 'mnad' in model_dir:
        df['psnr'] = df.apply(lambda x: psnr(x.MSE), axis=1)
        df.psnr[df.psnr < 0] = max(df.psnr)
        df.psnr = 1 - df.psnr

        # min-max normalization
        df.psnr = (df.psnr - df.psnr.min()) / (df.psnr.max() - df.psnr.min())
        df.feat_comp = (df.feat_comp - df.feat_comp.min()) / (df.feat_comp.max() - df.feat_comp.min())
        # alpha = 0.6
        df['scores'] = alpha * df.psnr + (1 - alpha) * df.feat_comp

        # df['scores'] = df.psnr
        # df['scores'] = df.feat_comp
        # df['scores'] = df.MSE
    else:
        # scores = df.MSE.to_numpy()
        # df['scores'] = df.MSE
        df['scores'] = df.score

    all_labels = []
    all_scores = []
    all_auc = []
    img_dir = os.path.dirname(os.path.dirname(df.iloc[0].img_path))
    # img_dir = os.path.join(cfg['dataset'][test_dataset]['path'], cfg['dataset'][test_dataset]['img_dir'])
    gt_dir = os.path.join(cfg['dataset'][test_dataset]['gt_path'])
    for gt_filename in sorted(os.listdir(gt_dir)):
        gt_path = os.path.join(gt_dir, gt_filename)
        if 'avenue' in gt_path:
            labels = np.loadtxt(gt_path)
        else:
            labels = np.loadtxt(gt_path, dtype=int)
        if test_dataset in ['harbor', 'harbor-rareevents', 'harbor-vehicles', 'harbor-appearance', 'harbor-fast-moving', 'harbor-near-edge', 'harbor-high-density', 'harbor-tampering']:
        # if est_dataset in ['harbor', 'harbor-rareevents', 'harbor-vehicles', 'harbor-high-density']:
            video_name, _ = os.path.splitext(gt_filename)
            video_name = video_name[:8] + '/' + video_name[9:]
        elif test_dataset in ['harbor-synthetic', 'harbor-mannequin', 'harbor-realfall']:
            video_name, _ = os.path.splitext(gt_filename)
        elif test_dataset == 'ucsdped2':
            video_name = f'test/frames/{gt_filename[:-4]}'  # gt_filename: 001.txt
        elif test_dataset == 'avenue':
            video_name = os.path.join(img_dir, os.path.splitext(gt_filename)[0])
        else:
            video_name = os.path.join(img_dir, os.path.splitext(gt_filename)[0])
            __import__('ipdb').set_trace()
            raise
        scores = df[df.img_path.str.startswith(video_name)].scores.to_numpy()

        # # dont evaluate first frames for prediction model. Fair?
        # if 'mnad' in model_dir and 'pred' in model_dir:
        #     t_length = 5
        #     labels = labels[t_length-1:]
        #     scores = scores[t_length-1:]

        if len(labels) != len(scores):
            __import__('ipdb').set_trace()

        all_labels.extend(labels)
        all_scores.extend(scores)

        # only compute macro auc if both 0 and 1 in gt labels
        if len(np.unique(labels)) < 2:
            if verbose:
                print(f'gt_filename: {gt_filename} skipped for macro auc')
            continue
        video_auc = compute_auc(labels, scores)[0]
        all_auc.append(video_auc)

    auc_micro, fpr, tpr = compute_auc(all_labels, all_scores)
    auc_macro = 0.0 if len(all_auc) == 0 else np.mean(all_auc)
    return auc_micro, auc_macro, fpr, tpr


# from mnad, changed a bit
def psnr(mse):
    if mse == 0.0:
        return -1.0
    return 10 * math.log10(1 / mse)


def compute_auc(label_after, score_after):
    try:
        assert len(label_after) == len(score_after)
    except:
        __import__('ipdb').set_trace()
    fpr, tpr, thresholds = skmetr.roc_curve(label_after, score_after)
    auc = skmetr.auc(fpr, tpr)
    return auc, fpr, tpr


def compute_rtbdc(cfg, test_dataset: str, model_dir: str, epoch_dir: int = None, verbose: bool = True, alpha = 0.6) -> [float, float]:
    tracks_path = cfg['dataset'][test_dataset]['rtbdc_gt_path']
    num_frames_in_video = cfg['dataset'][test_dataset]['num_test_frames']

    obj_csv_name, ext = os.path.splitext(cfg['dataset'][test_dataset]['csv_test'])
    # obj_csv_name = 'results_' + obj_csv_name + '_objects' + ext
    obj_csv_name = obj_csv_name + '_object' + ext
    if 'ssmtl' in model_dir:
        obj_csv_path = os.path.join(
                model_dir, 'per_epoch_predictions', dataset, str(epoch_dir), obj_csv_name)
    elif 'pgm' in model_dir:
        obj_csv_path = os.path.join(model_dir, obj_csv_name)
    elif 'kde' in model_dir:
        obj_csv_path = os.path.join(model_dir, epoch_dir, obj_csv_name)
    else:
        __import__('ipdb').set_trace()
        raise
    tmpdir = os.path.join(model_dir, 'tmp')
    csv_to_trbdc_format(obj_csv_path, tmpdir)

    anomalies_path = tmpdir
    rbdc, tbdc = compute_metrics(tracks_path=tracks_path,
                                 anomalies_path=anomalies_path,
                                 num_frames_in_video=num_frames_in_video,
                                 plot=False)
    return rbdc, tbdc


def csv_to_trbdc_format(obj_csv_path: str, tmpdir: str):
    df = pd.read_csv(obj_csv_path)
    video_names = df.image_id.apply(lambda x: os.path.dirname(x)).unique()

    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir, exist_ok=False)
    for video_name in tqdm(video_names):
        df_vid = df[df.image_id.str.startswith(video_name)]
        data = []

        # rtbdc anomaly format
        # frame_id, x_min, y_min, x_max, y_max, anomaly_score
        # 0,10,10,20,20,0.5
        for idx, row in df_vid.iterrows():
            frame_id = int(os.path.basename(row.image_id).lstrip('image_').rstrip('.jpg'))
            x_max = row.bbox_x1 + row.bbox_w
            y_max = row.bbox_y1 + row.bbox_h
            data.append([frame_id, row.bbox_x1, row.bbox_y1, x_max, y_max, row.score])

        # save to txt
        video_name = video_name.replace(os.sep, '_', 1) + '.txt'
        video_path = os.path.join(tmpdir, video_name)
        np.savetxt(video_path, np.array(data), '%d,%d,%d,%d,%d,%f')


if __name__ == "__main__":
    cfg = load_config('config.yaml')
    epoch_dir = None

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/recon/normal_001')
    # test_dataset = 'harbor-synthetic'
    # # auc_micro: 83.39, auc_macro: 63.46
    # test_dataset = 'harbor-mannequin'
    # # auc_micro: 40.32, auc_macro: 41.19
    # test_dataset = 'harbor-realfall'
    # # auc_micro: 71.26, auc_macro: 74.10
    # test_dataset = 'harbor'
    # # auc_micro: 51.00, auc_macro: 52.83

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/recon/normal_001/fine-tune-harbor-original')
    # test_dataset = 'harbor-synthetic'
    # # auc_micro: 43.11, auc_macro: 51.20
    # test_dataset = 'harbor-mannequin'
    # # auc_micro: 61.55, auc_macro: 68.95
    # test_dataset = 'harbor-realfall'
    # # auc_micro: 56.36, auc_macro: 54.00
    # test_dataset = 'harbor'
    # # auc_micro: 53.38, auc_macro: 54.63

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_001/')
    # test_dataset = 'harbor-synthetic'
    # # auc_micro: 93.96, auc_macro: 93.34
    # test_dataset = 'harbor'
    # # auc_micro: 49.34, auc_macro: 58.36
    # test_dataset = 'harbor-mannequin'
    # # auc_micro: 54.20, auc_macro: 64.92
    # test_dataset = 'harbor-realfall'
    # # auc_micro: 60.40, auc_macro: 63.65

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_001/fine-tune-harbor-old')
    # # harbor-synthetic  auc_micro: 90.36, auc_macro: 88.34
    # # harbor-mannequin  auc_micro: 64.43, auc_macro: 66.47
    # # harbor-realfall   auc_micro: 55.11, auc_macro: 62.45
    # # harbor-rareevents auc_micro: 49.96, auc_macro: 61.97

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_001/fine-tune-harbor')
    # harbor-synthetic  auc_micro: 91.89, auc_macro: 90.55
    # harbor-mannequin  auc_micro: 66.51, auc_macro: 67.75
    # harbor-realfall   auc_micro: 56.04, auc_macro: 63.17
    # harbor-rareevents auc_micro: 50.10, auc_macro: 61.93

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_001/fine-tune-harbor-rareevents')
    # # harbor-synthetic  auc_micro: 89.07, auc_macro: 87.54
    # # harbor-mannequin  auc_micro: 60.91, auc_macro: 65.31
    # # harbor-realfall   auc_micro: 53.44, auc_macro: 60.30
    # # harbor-rareevents auc_micro: 50.28, auc_macro: 60.50
    # # # epoch 40
    # # harbor-synthetic  auc_micro: 88.51, auc_macro: 86.87
    # # harbor-mannequin  auc_micro: 59.30, auc_macro: 63.65
    # # harbor-realfall   auc_micro: 51.57, auc_macro: 58.73
    # # harbor-rareevents auc_micro: 50.55, auc_macro: 60.23
    # # # epoch 10
    # # harbor-synthetic  auc_micro:      , auc_macro:
    # # harbor-mannequin  auc_micro: 61.12, auc_macro: 65.24
    # # harbor-realfall   auc_micro: 53.70, auc_macro: 60.94
    # # harbor-rareevents auc_micro: 49.74, auc_macro: 61.39

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor-synthetic/pred/normal_invert_008')
    # # harbor-synthetic  auc_micro: 93.96, auc_macro: 93.59
    # # harbor-mannequin  auc_micro: 63.76, auc_macro: 68.33
    # # harbor-realfall   auc_micro: 49.03, auc_macro: 62.98
    # # harbor-rareevents auc_micro: 49.40, auc_macro: 58.08


    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/empty_001')
    # harbor-mannequin  auc_micro: 62.43, auc_macro: 65.16
    # harbor-realfall   auc_micro: 51.38, auc_macro: 61.00
    # harbor-rareevents auc_micro: 49.44, auc_macro: 58.73
    # harbor-vehicles   auc_micro: 51.59, auc_macro: 62.20
    # harbor-near-edge  auc_micro: 53.28, auc_macro: 55.91

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_001')
    # epoch_dir = '20'
    # harbor-mannequin    auc_micro: 61.87, auc_macro: 65.06
    # harbor-realfall     auc_micro: 51.31, auc_macro: 65.17
    # harbor-vehicles     auc_micro: 61.76, auc_macro: 74.73
    # harbor-near-edge    auc_micro: 50.00, auc_macro: 50.71
    # harbor-high-density auc_micro: 91.06, auc_macro: 0.00

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/empty_001')
    # harbor-mannequin  auc_micro: 53.56, auc_macro: 63.42
    # harbor-realfall   auc_micro: 58.79, auc_macro: 63.70
    # harbor-rareevents auc_micro: 53.14, auc_macro: 56.66
    # harbor-vehicles   auc_micro: 57.23, auc_macro: 70.89
    # harbor-near-edge  auc_micro: 60.55, auc_macro: 60.55

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/low_density_001')
    # epoch_dir = '20'
    # # harbor-mannequin    auc_micro: 42.96, auc_macro: 43.99
    # # harbor-realfall     auc_micro: 54.89, auc_macro: 50.88
    # # harbor-vehicles     auc_micro: 52.81, auc_macro: 66.70
    # # harbor-near-edge    auc_micro: 50.66, auc_macro: 53.79
    # # harbor-high-density auc_micro: 81.30, auc_macro: 0.00


    # model_dir = r'/home/jacob/data/models/ssmtl++/harbor_synthetic_1_task'
    # test_dataset = 'harbor-synthetic'
    # # det2frame_scores(test_dataset, model_dir, epoch=19)
    # # auc_micro: 68.99, auc_macro: 78.28

    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_1_task')
    # epoch_dir = 4
    # harbor-mannequin    auc_micro: 57.07, auc_macro: 53.64  # epoch 2, epoch 3: 36.47
    # harbor-realfall     auc_micro: 48.10, auc_macro: 60.97  # epoch 4,
    # harbor-near-edge    auc_micro: 51.55, auc_macro: 46.90
    # harbor-high-density auc_micro: 88.61, auc_macro: 0.00
    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_1_task_trans')
    # epoch_dir = 9
    # harbor-mannequin    auc_micro: 64.29, auc_macro: 64.45  # epoch 09
    # harbor-realfall     auc_micro: 68.14, auc_macro: 59.87  # epoch 14  fluctuating
    # harbor-near-edge    auc_micro: 51.94, auc_macro: 53.41
    # harbor-high-density auc_micro: 87.99, auc_macro: nan

    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task')
    # epoch_dir = '19'
    # harbor-vehicles     auc_micro: 52.77, auc_macro: 58.65, rbdc: 7.16, tbdc: 33.57
    # harbor-near-edge    auc_micro: 50.70, auc_macro: 51.27, rbdc: 5.02, tbdc: 19.90
    # harbor-high-density auc_micro: 89.70, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 54.12, auc_macro: 53.77
    # harbor-realfall     auc_micro: 55.60, auc_macro: 50.19

    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task_gt')
    # epoch_dir = '19'
    # harbor-vehicles     auc_micro: 54.41, auc_macro: 59.79, rbdc: 7.48, tbdc: 26.66
    # harbor-near-edge    auc_micro: 54.08, auc_macro: 54.80, rbdc: 8.06, tbdc: 28.75
    # harbor-high-density auc_micro: 92.98, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 49.73, auc_macro: 45.45
    # harbor-realfall     auc_micro: 56.05, auc_macro: 51.76

    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/001')  # epoch 20
    # # ucsdped2          auc_micro: 87.01, auc_macro: 96.24
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/001')  # epoch 50
    # ucsdped2          auc_micro: 87.93, auc_macro: 96.52

    # model_dir = os.path.join(cfg['models_path'], r'mnad/avenue/pred/pretrained')
    # # avenue            a: 0.6 auc_micro: 87.98, auc_macro: 83.27
    # model_dir = os.path.join(cfg['models_path'], r'mnad/avenue/pred/001')

    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s01m04')
    # harbor-vehicles     auc_micro: 77.44, auc_macro: 79.24, rbdc: 57.43, tbdc: 81.43
    # harbor-near-edge    auc_micro: 59.16, auc_macro: 60.22, rbdc: 26.18, tbdc: 38.29
    # harbor-high-density auc_micro: 94.80, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 51.03, auc_macro: 51.42
    # harbor-realfall     auc_micro: 50.00, auc_macro: 50.00

    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s01m04_gt')
    # harbor-vehicles     auc_micro: 76.43, auc_macro: 80.96, rbdc: 57.36, tbdc: 79.88
    # harbor-near-edge    auc_micro: 61.39, auc_macro: 60.00, rbdc: 26.74, tbdc: 36.52
    # harbor-high-density auc_micro: 94.60, auc_macro: 0.00

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_005')
    # epoch_dir = 20
    # harbor-vehicles     auc_micro: 58.93, auc_macro: 74.44
    # harbor-near-edge    auc_micro: 51.57, auc_macro: 50.61
    # harbor-high-density auc_micro: 87.91, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 62.24, auc_macro: 65.72
    # harbor-realfall     auc_micro: 55.68, auc_macro: 64.79
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_020')
    # epoch_dir = 20
    # harbor-vehicles     auc_micro: 61.86, auc_macro: 75.25
    # harbor-near-edge    auc_micro: 51.47, auc_macro: 50.76
    # harbor-high-density auc_micro: 87.87, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 67.77, auc_macro: 67.62
    # harbor-realfall     auc_micro: 55.52, auc_macro: 66.05
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_050')
    # epoch_dir = 20
    # harbor-vehicles     auc_micro: 59.51, auc_macro: 74.67
    # harbor-near-edge    auc_micro: 51.42, auc_macro: 50.77
    # harbor-high-density auc_micro: 91.18, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 63.05, auc_macro: 66.84
    # harbor-realfall     auc_micro: 62.22, auc_macro: 67.18
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_100')
    # epoch_dir = 20
    # harbor-vehicles     auc_micro: 59.77, auc_macro: 74.38
    # harbor-near-edge    auc_micro: 55.01, auc_macro: 49.91
    # harbor-high-density auc_micro: 92.12, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 64.26, auc_macro: 68.41
    # harbor-realfall     auc_micro: 55.37, auc_macro: 70.46
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_200')
    # epoch_dir = 20
    # harbor-vehicles     auc_micro: 62.23, auc_macro: 74.66
    # harbor-near-edge    auc_micro: 51.37, auc_macro: 50.76
    # harbor-high-density auc_micro: 89.29, auc_macro: 0.00
    # harbor-mannequin    auc_micro: 66.11, auc_macro: 67.16
    # harbor-realfall     auc_micro: 54.50, auc_macro: 62.65

    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor')
    # harbor-near-edge    auc_micro: 62.06, auc_macro: 61.33
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_020')
    # harbor-vehicles     auc_micro: 49.56, auc_macro: 53.74
    # harbor-near-edge    auc_micro: 62.01, auc_macro: 61.14  # bw  1
    # harbor-near-edge    auc_micro: 73.54, auc_macro: 71.96  # bw  2
    # harbor-near-edge    auc_micro: 73.48, auc_macro: 74.48  # bw  5
    # harbor-near-edge    auc_micro: 65.80, auc_macro: 63.44  # bw 10
    # harbor-near-edge    auc_micro: 64.46, auc_macro: 62.27  # bw 15 <- grid search
    # harbor-high-density auc_micro: 92.99, auc_macro: 0.00
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_100')
    # harbor-vehicles     auc_micro: 89.36, auc_macro: 88.69  # bw  1
    # harbor-near-edge    auc_micro: 74.31, auc_macro: 73.23  # bw  1
    # harbor-near-edge    auc_micro: 64.45, auc_macro: 64.08  # bw 10
    # harbor-near-edge    auc_micro: 62.46, auc_macro: 61.96  # bw 15
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0100')
    # harbor-near-edge    auc_micro: 58.95, auc_macro: 58.38, rbdc: 18.99, tbdc: 19.05  # bw 2
    # harbor-near-edge    auc_micro: 64.80, auc_macro: 64.69, rbdc: 20.88, tbdc: 23.93  # bw 5
    # harbor-near-edge    auc_micro: 58.66, auc_macro: 58.57, rbdc: 7.11, tbdc: 6.64  # bw 10
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0200')
    # harbor-near-edge    auc_micro: 65.12, auc_macro: 64.42, rbdc: 21.74, tbdc: 24.23
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0500')
    # harbor-near-edge    auc_micro: 61.75, auc_macro: 62.66, rbdc: 19.78, tbdc: 21.50
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1000')
    # harbor-near-edge    auc_micro: 62.41, auc_macro: 63.80, rbdc: 19.55, tbdc: 21.39
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1500')
    # harbor-near-edge    auc_micro: 63.15, auc_macro: 64.04, rbdc: 19.66, tbdc: 21.15

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/low_density_0100')
    # harbor-vehicles     auc_micro: 46.79, auc_macro: 47.34
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/high_density_0100')
    # harbor-vehicles     auc_micro: 56.99, auc_macro: 67.12  # bw 5
    # epoch_dir = '20'

    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0005')
    # epoch_dir = 'bw05'
    # harbor-near-edge    auc_micro: 56.46, auc_macro: 56.35, rbdc: 15.20, tbdc: 15.03
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0200'); epoch_dir = 'bw01'
    # harbor-near-edge    auc_micro: 65.12, auc_macro: 64.42, rbdc: 21.74, tbdc: 24.23  # bw 05
    # harbor-near-edge    auc_micro: 58.59, auc_macro: 58.42, rbdc: 7.16, tbdc: 6.54  # bw 10
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1500'); epoch_dir = 'bw00.5'
    # harbor-near-edge    auc_micro: 50.27, auc_macro: 47.58, rbdc: 0.18, tbdc: 0.50
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1500'); epoch_dir = 'bw0100'
    # harbor-near-edge    auc_micro: 63.96, auc_macro: 63.37, rbdc: 22.88, tbdc: 22.67
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1500'); epoch_dir = 'bw0500'
    # harbor-near-edge    auc_micro: 63.15, auc_macro: 64.04, rbdc: 19.66, tbdc: 21.15
    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_1500'); epoch_dir = 'bw1000'
    # harbor-near-edge    auc_micro: 58.09, auc_macro: 57.49, rbdc: 7.17, tbdc: 6.81

    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0100')
    # epoch_dir = 'bw02.0'
    # harbor-near-edge    auc_micro: 58.95, auc_macro: 58.38, rbdc: 22.11, tbdc: 22.75  # 'bw02.0'
    # harbor-near-edge    auc_micro: 64.80, auc_macro: 64.69, rbdc: 24.23, tbdc: 25.94  # 'bw05.0'
    # harbor-near-edge    auc_micro: 58.66, auc_macro: 58.57, rbdc: 09.18, tbdc: 07.42  # 'bw10.0'

    # harbor-near-edge    auc_micro: 82.90, auc_macro: 81.67, rbdc: 76.81, tbdc: 78.75
    # harbor-near-edge    auc_micro: 58.06, auc_macro: 64.18, rbdc: 1.61, tbdc: 1.27
    # harbor-near-edge    auc_micro: 64.57, auc_macro: 65.18, rbdc: 25.35, tbdc: 26.62
    # epoch_dir = 'bw10.0'
    # harbor-near-edge    auc_micro: 63.37, auc_macro: 64.23, rbdc: 25.35, tbdc: 26.62
    # harbor-near-edge    auc_micro: 59.29, auc_macro: 63.07, rbdc: 10.50, tbdc:  9.62  # 05
    # harbor-near-edge    auc_micro: 54.65, auc_macro: 55.85, rbdc:  8.02, tbdc:  6.49  # 10

    # epoch_dir = 'bw02.0'
    # harbor-near-edge    auc_micro: 56.34, auc_macro: 61.47, rbdc:  1.61, tbdc:  1.27
    # harbor-near-edge    auc_micro: 75.31, auc_macro: 73.62, rbdc: 59.37, tbdc: 61.88  # no cars
    # harbor-near-edge    auc_micro: 77.55, auc_macro: 76.42, rbdc: 65.28, tbdc: 67.63  # no cars, no exp
    # harbor-near-edge    auc_micro: 87.17, auc_macro: 85.45, rbdc: 84.99, tbdc: 87.11  # no cars, no exp
    # harbor-near-edge    auc_micro: 88.27, auc_macro: 86.38, rbdc: 65.93, tbdc: 67.91  # no cars, exp_inv
    # harbor-near-edge    auc_micro: 87.17, auc_macro: 85.45, rbdc: 85.01, tbdc: 87.12  # no cars, inv_exp
    # harbor-near-edge    auc_micro: 79.91, auc_macro: 81.60, rbdc: 70.00, tbdc: 74.33  # stats, scott
    # harbor-near-edge    auc_micro: 86.22, auc_macro: 84.90, rbdc: 50.45, tbdc: 52.71
    # harbor-near-edge    auc_micro: 80.44, auc_macro: 82.77, rbdc: 72.22, tbdc: 75.72  # weighting
    # harbor-near-edge    auc_micro: 81.31, auc_macro: 83.54, rbdc: 74.57, tbdc: 77.79  # weighting no sqrt
    # harbor-near-edge    auc_micro: 84.61, auc_macro: 86.10, rbdc: 81.25, tbdc: 83.91  # weighting pow 2
    # harbor-near-edge    auc_micro: 87.12, auc_macro: 88.52, rbdc: 85.22, tbdc: 87.46  # weighting pow 3

    # model_dir = os.path.join(cfg['models_path'], r'kde/harbor/low_density_0005')
    # epoch_dir = 'bw02.0'
    # harbor-near-edge    auc_micro: 71.10, auc_macro: 75.32, rbdc: 47.83, tbdc: 49.49

    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s01m04_gt')
    # harbor-vehicles     auc_micro: 84.79, auc_macro: 85.56, rbdc: 65.04, tbdc: 66.46
    # harbor-near-edge    auc_micro: 60.49, auc_macro: 59.05, rbdc: 15.16, tbdc: 17.28
    # harbor-high-density auc_micro: 97.57, auc_macro:  0.00
    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s12_gt_temp')
    # harbor-vehicles     auc_micro: 83.60, auc_macro: 84.42, rbdc: 60.36, tbdc: 61.98
    # harbor-near-edge    auc_micro: 59.62, auc_macro: 58.38, rbdc: 14.03, tbdc: 16.09
    # harbor-high-density auc_micro: 97.34, auc_macro:  0.00
    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s12_gt_humi')
    # harbor-vehicles     auc_micro: 83.60, auc_macro: 84.42, rbdc: 60.36, tbdc: 61.98
    # harbor-near-edge    auc_micro: 59.62, auc_macro: 58.38, rbdc: 14.03, tbdc: 16.09
    # harbor-high-density auc_micro: 97.34, auc_macro:  0.00

    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/001')  # epoch 20
    # ucsdped2          auc_micro: 87.01, auc_macro: 96.24
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/001')  # epoch 50
    # ucsdped2          auc_micro: 87.93, auc_macro: 96.52
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/recon/001')
    # ucsdped2            auc_micro: 72.95, auc_macro: 89.58  # 20
    # ucsdped2            auc_micro: 74.00, auc_macro: 89.31  # 60
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/recon/002')
    # epoch_dir = 'epoch_000020'
    # ucsdped2            auc_micro: 70.04, auc_macro: 85.45  # 20
    # ucsdped2            auc_micro: 71.77, auc_macro: 88.51  # 60
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/008')
    # epoch_dir = 'epoch_000060'
    # ucsdped2            auc_micro: 87.11, auc_macro: 96.08
    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/009_rgb')
    # epoch_dir = 'epoch_000060'
    # ucsdped2            alpha:0.7, auc_micro: 89.45, auc_macro: 97.56

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/006')
    # epoch_dir = 'epoch_000020'
    # # harbor-appearance   auc_micro: 38.40, auc_macro: 43.76
    # # harbor-fast-moving  auc_micro: 57.99, auc_macro: 61.06
    # # harbor-near-edge    auc_micro: 70.30, auc_macro: 54.80
    # # harbor-high-density auc_micro: 99.86, auc_macro:  0.00
    # # harbor-tampering    auc_micro:  8.21, auc_macro:  0.00

    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/002')
    # epoch_dir = 'epoch_000020'
    # # harbor-appearance   auc_micro: 74.32, auc_macro: 93.89
    # # harbor-fast-moving  auc_micro: 60.43, auc_macro: 60.90
    # # harbor-near-edge    auc_micro: 63.02, auc_macro: 51.03
    # # harbor-high-density auc_micro: 92.62, auc_macro:  0.00
    # # harbor-tampering    auc_micro: 26.27, auc_macro:  0.00

    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s01m04')
    # # harbor-appearance   auc_micro: 61.98, auc_macro: 75.37, rbdc: 89.80, tbdc: 94.59
    # # harbor-fast-moving  auc_micro: 50.93, auc_macro: 58.39
    # # harbor-near-edge    auc_micro: 83.28, auc_macro: 76.28, rbdc: 48.87, tbdc: 60.63
    # # harbor-high-density auc_micro: 100.00, auc_macro:  0.00
    # # harbor-tampering    auc_micro: 97.23, auc_macro:  0.00

    # model_dir = os.path.join(cfg['models_path'], r'pgm/harbor/spatial/s01m04_gt')
    # # harbor-appearance   auc_micro: 70.39, auc_macro: 72.91, rbdc: 32.65, tbdc: 36.36
    # # harbor-fast-moving  auc_micro: 55.14, auc_macro: 60.72
    # # harbor-near-edge    auc_micro: 64.59, auc_macro: 62.83, rbdc: 39.75, tbdc: 36.84
    # # harbor-high-density auc_micro: 100.00, auc_macro:  0.00
    # # harbor-tampering    auc_micro: 95.62, auc_macro:  0.00

    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task')
    # epoch_dir = '19'
    # # harbor-appearance   auc_micro: 42.40, auc_macro: 56.62, rbdc:  1.14, tbdc: 10.83
    # # harbor-fast-moving  auc_micro: 58.54, auc_macro: 61.70
    # # harbor-near-edge    auc_micro: 56.46, auc_macro: 52.68, rbdc:  6.49, tbdc: 32.90
    # # harbor-high-density auc_micro: 87.28, auc_macro:  0.00
    # # harbor-tampering    auc_micro: 30.78, auc_macro:  0.00

    model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task_gt')
    epoch_dir = '19'
    # harbor-appearance   auc_micro: 47.57, auc_macro: 61.96, rbdc:  0.92, tbdc:  6.03
    # harbor-fast-moving  auc_micro: 54.43, auc_macro: 61.19
    # harbor-near-edge    auc_micro: 61.02, auc_macro: 57.65, rbdc: 11.83, tbdc: 44.67
    # harbor-high-density auc_micro: 87.99, auc_macro:  0.00
    # harbor-tampering    auc_micro: 31.19, auc_macro:  0.00


    # datasets = ['ucsdped2']
    # datasets = ['avenue']
    # datasets = ['harbor-vehicles']
    # dataset = [harbor_dataset.Appearance()]
    # datasets = ['harbor-appearance']
    # datasets = ['harbor-fast-moving']
    # datasets = ['harbor-near-edge']
    # datasets = ['harbor-high-density']
    # datasets = ['harbor-tampering']
    # datasets = ['harbor-mannequin']
    # datasets = ['harbor-realfall']
    # datasets = ['harbor-synthetic', 'harbor-mannequin', 'harbor-realfall', 'harbor-rareevents']
    # datasets = ['harbor-vehicles', 'harbor-near-edge']
    # datasets = ['harbor-high-density', 'harbor-mannequin', 'harbor-realfall', ]
    # datasets = ['harbor-vehicles', 'harbor-near-edge', 'harbor-high-density', 'harbor-mannequin', 'harbor-realfall']
    datasets = ['harbor-appearance', 'harbor-fast-moving', 'harbor-near-edge', 'harbor-high-density', 'harbor-tampering']

    for dataset in datasets:
        # for alpha in np.arange(0.0, 1.1, 0.1):
        #     auc_micro, auc_macro, *_ = evaluate(cfg, dataset, model_dir, epoch_dir, verbose=False, alpha=alpha)
        #     print(f'{dataset:19s} alpha:{alpha:.1f}, auc_micro: {auc_micro*100:5.2f}, auc_macro: {auc_macro*100:5.2f}')

        auc_micro, auc_macro, *_ = evaluate(cfg, dataset, model_dir, epoch_dir, verbose=False)
        if not 'rtbdc_gt_path' in cfg['dataset'][dataset] or 'mnad' in model_dir:
            print(f'{dataset:19s} auc_micro: {auc_micro*100:5.2f}, auc_macro: {auc_macro*100:5.2f}')
        else:
            rbdc, tbdc = compute_rtbdc(cfg, dataset, model_dir, epoch_dir)
            print(f'{dataset:19s} auc_micro: {auc_micro*100:5.2f}, auc_macro: {auc_macro*100:5.2f}, rbdc: {rbdc*100:5.2f}, tbdc: {tbdc*100:5.2f}')
