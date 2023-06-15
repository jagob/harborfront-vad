import glob
import os
from urllib.request import proxy_bypass

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils.config import load_config
from src.data.datasets.dataset import Dataset, get_dataset
import src.data.datasets.harbor as harbor_dataset
from src.evaluation.plot_anomaly_score import plot_anomaly_score
from src.evaluation.evaluate import compute_auc


COLORS = sns.color_palette("deep", 8)
COLORS_JET = sns.color_palette("coolwarm_r", 11)


def visualize_video(cfg, dataset, video_name, gt_pixel_mask_dir=None, rbdc_tbdc=True, draw_gt_bbox=True, gt_labels=None, models=None, save_video_path=None):
    # cv2.namedWindow('Combined', cv2.WINDOW_AUTOSIZE)
    scale = 1
    fps = 25
    delay_ms = int(1 / fps * 1000)
    delay_ms = 1  # override to speed up video

    if save_video_path:
        if os.path.exists(save_video_path):
            return

        print(save_video_path)
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        img_width = dataset.img_width
        img_height = dataset.img_height
        if gt_labels is None and not models:
            double_plot = False
            video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (img_width, img_height))
        else:
            double_plot = True
            anom_plot_height = img_height
            anom_plot_height = img_width // 2
            video_writer = cv2.VideoWriter(
                    save_video_path, fourcc, fps, (img_width, img_height + anom_plot_height))

    if gt_pixel_mask_dir:
        # if 'ucsdped' in rbdc_tbdc_dir:
        #     masks = scipy.io.loadmat(gt_path)['volLabel'][0]
        #     assert len(masks) == len(images_video_name)
        if 'avenue' in rbdc_tbdc_dir:
            # http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
            gt_path = os.path.join(gt_pixel_mask_dir, f'{int(video_name)}_label.mat')
            masks = scipy.io.loadmat(gt_path)['volLabel'][0]
            assert len(masks) == len(images_video_name)
        elif 'shanghaitech' in rbdc_tbdc_dir:
            gt_path = os.path.join(gt_pixel_mask_dir, f'{video_name}.npy')
            masks = np.load(gt_path)
            assert len(masks) == len(images_video_name)

    if hasattr(dataset, 'object_annotations'):
        gt_detections_dir = os.path.join(dataset.object_annotations, 'video_annotations')
        gt_detections_path = os.path.join(gt_detections_dir, video_name + '.csv')
        if os.path.exists(gt_detections_path):
            gt_detections = pd.read_csv(gt_detections_path)
        else:
            gt_detections = None
    else:
        gt_detections = None

    # if rbdc_tbdc and hasattr(dataset, 'rtbdc_gt'):
    if rbdc_tbdc and os.path.exists(dataset.rtbdc_gt):
        tracks_path = dataset.get_rtbdc_track_path(video_name)
        # if 'ucsdped' in rbdc_tbdc_dir:
        #     filename = f'{video_name.capitalize()}_gt.txt'
        #     tracks_path = os.path.join(rbdc_tbdc_dir, filename)
        # elif 'avenue' in rbdc_tbdc_dir:
        #     filename = f'Test{int(video_name):03d}_gt.txt'
        #     tracks_path = os.path.join(rbdc_tbdc_dir, filename)
        # elif 'shanghaitech' in rbdc_tbdc_dir:
        #     filename = f'{video_name}.txt'
        #     tracks_path = os.path.join(rbdc_tbdc_dir, filename)
        # elif 'streetscene' in rbdc_tbdc_dir:
        #     filename = f'{video_name}_gt.txt'
        #     tracks_path = os.path.join(rbdc_tbdc_dir, video_name, filename)
        # elif 'harbor' in rbdc_tbdc_dir:
        #     filename = f'{video_name}_gt.txt'
        #     tracks_path = os.path.join(rbdc_tbdc_dir, video_name, filename)
        #     __import__('ipdb').set_trace()
        # tracks_array = read_single_track_file(tracks_path)
        tracks_array = np.loadtxt(tracks_path, dtype=int, delimiter=',')
        columns = ['track_id', 'frame', 'x1', 'y1', 'x2', 'y2']
        if len(tracks_array.shape) == 1:
            rbdc_df = pd.DataFrame([tracks_array], columns=columns)
        else:
            rbdc_df = pd.DataFrame(tracks_array, columns=columns)
        rbdc_df['width'] = rbdc_df.x2 - rbdc_df.x1
        rbdc_df['height'] = rbdc_df.y2 - rbdc_df.y1

    images_video_name = glob.glob(os.path.join(dataset.images, video_name, '*.jpg'))
    images_video_name = sorted(images_video_name)
    if not images_video_name:
        __import__('ipdb').set_trace()
        raise ValueError

    for idx, img_path in enumerate(tqdm(images_video_name)):
        img_original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img_original.copy()
        # cv2.putText(img, f"{os.path.basename(img_path)}",
        #             (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img_combined = img

        # draw ground-truth detections manually annotated
        if draw_gt_bbox is True:
            if gt_detections is not None:
                frame_detections = gt_detections[gt_detections.frame_id == idx]
                y_margin = 2

                if len(frame_detections) != len(frame_detections.object_id.unique()):
                    breakpoint()
                    print("not unique id in frame?")

                if True:
                    for index, det in frame_detections.iterrows():
                        width = det.x2 - det.x1
                        height = det.y2 - det.y1
                        bbox = [det.x1, det.y1, width, height]
                        class_name = det['name']
                        img = draw_bounding_boxes(img, [bbox], class_name)
                        # draw_txt(img, f"{det.object_id}", det.x1, det.y1 - y_margin)
                        # if 'occlusion' in frame_detections.columns:
                        #     if det.occlusion:
                        #         draw_txt(img, "Occ", det.x1, det.y1 + 6)
                    img_combined = img

        # # detections from collected csv file
        # # TODO: function
        # csv_path = os.path.join(cfg['detection']['dir'], 'yolov5', dataset.name, video_name + '.csv')
        # if os.path.exists(csv_path):
        #     # frame_pattern = f'{video_name:02}/{idx:05}'
        #     # frame_pattern = f'{video_name}/{idx:05}.jpg'
        #     frame_pattern = f'{video_name}/{os.path.basename(img_path)}'
        #     detections = pd.read_csv(csv_path)
        #     # frame_detections = detections[detections.image_id.str.contains(frame_pattern)]
        #     frame_detections = detections[detections.image_id.str.endswith(frame_pattern)]
        #     y_margin = 0
        #     for index, detection in frame_detections.iterrows():
        #         x1 = detection.bbox_x1
        #         y1 = detection.bbox_y1
        #         bbox = [x1, y1, detection.bbox_w, detection.bbox_h]
        #         conf = detection.confidence
        #         class_name = detection['name']
        #         img = draw_bounding_boxes(img, [bbox], class_name)
        #         # draw_txt(img, f"{class_name} {conf*100:3.0f}", x1, y1-y_margin)
        #     # img = cv2.resize(img, None, fx=scale, fy=scale)
        #     # img = cv2.resize(img, None, fx=scale, fy=scale)
        #     img_combined = img

        # #  draw object-centric anomaly scores
        # if len(models) == 1:
        #     model = models[0]
        #     if 'df_obj' in model:
        #         df_obj = model['df_obj']

        #         y_margin = 2
        #         idx_offset = 1
        #             # lambda x: x.startswith(video_name + f'/image_{idx - index_offset:40}'))]
        #         df_obj_frm = df_obj[df_obj['image_id'].apply(
        #             lambda x: x.startswith(video_name + f'/image_{idx-idx_offset:04d}'))]
        #         for index, obj in df_obj_frm.iterrows():
        #             bbox = [obj['bbox_x1'], obj['bbox_y1'], obj['bbox_w'], obj['bbox_h']]
        #             img = draw_bounding_boxes_anomaly(img, [bbox], obj.score)
        #             if obj.score > 0.05:
        #                 draw_txt(img, f"{obj.score:.2f}", obj.bbox_x1, obj.bbox_y1 - y_margin)

        # # draw detections with colored rectangles depending on anomaly score
        # # frame_pattern = f'{video_name:02}/{idx:05}'
        # frame_pattern = f'{video_name}/{idx:05}_'
        # frame_detections = detections[detections['image_id'].str.contains(frame_pattern)]
        # for index, detection in frame_detections.iterrows():
        #     # bbox = detection['bbox']
        #     bbox = [detection['bbox_x1'], detection['bbox_y1'], detection['bbox_w'], detection['bbox_h']]
        #     # anomaly_score_norm = (detection['anomaly_score'] - min_val) / (max_val - min_val)
        #     anomaly_score_norm = detection['anomaly_score_norm']
        #     img = draw_bounding_boxes_anomaly(img, [bbox], anomaly_score_norm)
        # img = cv2.resize(img, None, fx=scale, fy=scale)
        # # cv2.putText(img, f"{os.path.basename(img_path)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # # draw kde score
        # y_margin = 24
        # frame_pattern = f'{video_name}/{os.path.basename(img_path)}'
        # frame_detections = detections[detections.image_id.str.contains(frame_pattern)]
        # for index, detection in frame_detections.iterrows():
        #     x1 = detection.bbox_x1
        #     y1 = detection.bbox_y1
        #     bbox = [x1, y1, detection.bbox_w, detection.bbox_h]
        #     kde_score = detection.kde_score
        #     draw_txt(img, f"kde {kde_score*100:3.0f}", x1, y1-y_margin)

        # draw pixel masks
        if gt_pixel_mask_dir:
            if dataset.name == 'ucsdped2':
                gt_pixel_mask_path = os.path.join(gt_pixel_mask_dir, video_name[4:7], f'{idx+1:03d}.bmp')
                if not os.path.exists(gt_pixel_mask_path):  # pixel mask 6, 9 and 12 use other naming
                    gt_pixel_mask_path = os.path.join(gt_pixel_mask_dir, video_name[4:7], f'frame{idx+1:03d}.bmp')
                mask = cv2.imread(gt_pixel_mask_path, cv2.IMREAD_GRAYSCALE)
            elif 'avenue' in dataset.rtbdc_gt or 'shanghaitech' in dataset.rtbdc_gt:
                mask = masks[idx]
            img = draw_pixel_mask(img, mask)

        # draw rbdc bounding boxes
        # if rbdc_tbdc and hasattr(dataset, 'rtbdc_gt'):
        if rbdc_tbdc and os.path.exists(dataset.rtbdc_gt):
            rbdc_frame_df = rbdc_df[rbdc_df.frame == idx]  # 0 index
            #     rbdc_frame_df = rbdc_df[rbdc_df.frame == idx + 1]  # 1 index
            for index, rbdc_bbox in rbdc_frame_df.iterrows():
                bbox = [rbdc_bbox.x1, rbdc_bbox.y1, rbdc_bbox.width, rbdc_bbox.height]
                img = draw_bounding_boxes(img, [bbox], None, rbdc=True)
            img_combined = img  # override for single image output

        if double_plot:
            figure = plot_anomaly_score(models, img_width, '', gt_labels, idx)
            plot_save_path = r'src/data/tmp/video_anomaly_score.png'
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            figure.tight_layout(pad=0.2)
            plt.savefig(plot_save_path)
            # plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
            # plt.show()
            plt.close()

            time_score_img = cv2.imread(plot_save_path, cv2.IMREAD_COLOR)
            # time_score_img = cv2.resize(time_score_img, (img_width, anom_plot_height))
            img_combined = np.concatenate((img, time_score_img))

        if save_video_path:
            video_writer.write(img_combined)

        # cv2.imshow("Combined", img_combined)  # Manual break by escape or q
        # returnKey = cv2.waitKey(delay_ms)
        # if returnKey == 27 or returnKey == ord('q'):
        #     break

    if save_video_path:
        video_writer.release()
        cv2.destroyAllWindows()


def draw_txt(img, txt, x1, y1):
    cv2.putText(img, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 2)
    cv2.putText(img, txt, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)


def draw_bounding_boxes(frame, bboxes, class_name, rbdc=False):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        # x2 = int(bbox[2])
        # y2 = int(bbox[3])
        w = int(bbox[2])
        h = int(bbox[3])
        x2 = x1 + w
        y2 = y1 + h
        # bb = annotation['bbox']
        # cv2.rectangle(frame, (bb.left, bb.top), (bb.right, bb.bot), (0, 0, 255), 2)

        # if cls:
        #     cls -= 1
        #     color = (COLORS[cls][0]*255, COLORS[cls][1]*255, COLORS[cls][2]*255)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        if rbdc:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        else:
            if class_name in ['background', 'optical_flow']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            elif class_name in ['person', 'human']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            elif class_name == 'bicycle':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            elif class_name == 'vehicle':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # category = annotation['category']
        # if category:
        #     cv2.putText(frame, category, (bb.left, bb.top),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def draw_bounding_boxes_anomaly(frame, bboxes, anomaly_score):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        x2 = x1 + w
        y2 = y1 + h

        color_index = round(anomaly_score * 10)
        color_index = max(color_index, 0)
        color_index = min(color_index, 10)
        # color_index = l if x < l else u if x > u else x
        color = (COLORS_JET[color_index][0]*255,
                 COLORS_JET[color_index][1]*255,
                 COLORS_JET[color_index][2]*255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    return frame


def draw_pixel_mask(frame, mask, alpha=0.70):
    mask_inv = cv2.bitwise_not(mask)
    red = np.ones(frame.shape, dtype=float) * (0, 0, 1)
    red = np.array(red*255, dtype=np.uint8)
    frame2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    red = cv2.bitwise_and(red, red, mask=mask)
    opaque_red = cv2.add(frame2, red)

    cv2.addWeighted(frame, alpha, opaque_red, 1-alpha, 0, frame)
    return frame


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    selected_video_names = None
    # dataset = 'avenue'
    # dataset = 'harbor-synthetic'
    # dataset = 'harbor-vehicles'
    # dataset = 'harbor-appearance'
    # dataset = 'harbor-fast-moving'
    # dataset = 'harbor-near-edge'
    # selected_video_names = [
    #     '20200603/clip_12_0522',
    #     '20200625/clip_10_0440',
    #     '20200707/clip_46_2029',
    #     '20200831/clip_26_1143',
    #     '20210110/clip_2_0056',
    # ]
    # dataset = 'harbor-high-density'
    # dataset = 'harbor-low-density'
    # dataset = 'harbor-tampering'
    # dataset = 'harbor'
    # dataset = 'harbor-mannequin'
    # dataset = 'harbor-realfall'
    # dataset = 'harbor-rareevents'
    # dataset = 'harbor-original'
    # dataset = get_dataset('harbor-fast-moving', cfg)
    # dataset = dataset.get_dataset(dataset.Dataset.harbor_fast_moving, cfg)

    # dataset = harbor_dataset.Harbor()
    # dataset = get_dataset(Dataset.harbor_appearance)
    # dataset = get_dataset(Dataset.harbor_fast_moving)
    # dataset = get_dataset(Dataset.harbor_near_edge)
    # dataset = get_dataset(Dataset.harbor_near_edge)
    # dataset = harbor_dataset.Appearance()
    # dataset = harbor_dataset.Fastmotion()
    # dataset = harbor_dataset.NearEdge()
    # dataset = harbor_dataset.HighDensity()
    dataset = harbor_dataset.Tampering()

    model_dir = r''
    # model_dir = r'/home/jacob/data/models/mnad/harbor/pred/empty_001'
    # model_dir = r'/home/jacob/data/models/mnad/harbor-synthetic/pred/normal_001/'
    # model_dir = r'/home/jacob/data/models/mnad/harbor-synthetic/pred/normal_001/fine-tune-harbor-rareevents/'
    # model_dir = r'/home/jacob/data/models/ssmtl++/harbor_synthetic_1_task'
    save_dir = os.path.join(model_dir, 'visualization', dataset.name)

    gt_pixel_mask_dir = None
    draw_gt_bbox = False
    plot_rtbdc_bbox = True

    csv_vid_path = os.path.join(r'src/data/split/videos', dataset.csv_test)
    # csv_vid_path = os.path.join(r'src/data/split/videos', dataset.csv_train)
    # csv_vid_path = os.path.join(r'src/data/split/videos', 'harbor_train_0100.csv')
    # csv_vid_path = r'src/data/split/videos/harbor_appearance.csv'
    # csv_vid_path = r'src/data/split/videos/harbor_appearance_test.csv'
    # csv_vid_path = r'src/data/split/videos/harbor_tampering.csv'
    df_vid = pd.read_csv(csv_vid_path, dtype={'folder_name': str, 'clip_name': str})
    test_videos = (df_vid.folder_name + os.sep + df_vid.clip_name).to_list()
    # test_videos = test_videos.sort_values(['folder_name', 'clip_name'])
    # test_videos = sorted(test_videos)

    models = [
              # {'label': 'MNAD-Pred', 'model_dir': r'mnad/avenue/pred/pretrained', 'epoch_dir': ''},
              {'label': 'MNAD-Recon', 'model_dir': r'mnad/harbor/recon/006', 'epoch_dir': 'epoch_000020'},
              {'label': 'MNAD-Pred', 'model_dir': r'mnad/harbor/pred/002', 'epoch_dir': 'epoch_000020'},
              {'label': 'SSMTL', 'model_dir': r'ssmtl++/harbor_low_density_3_task', 'epoch_dir': f'per_epoch_predictions/{dataset.name}/19'},
              {'label': 'PGM', 'model_dir': r'pgm/harbor/spatial/s01m04', 'epoch_dir': ''},
              # {'label': 'KDE', 'model_dir': r'kde/harbor/low_density_0100', 'epoch_dir': 'bw02.0'},
              # {'label': 'KDE exp_inv', 'model_dir': r'kde/harbor/low_density_0100', 'epoch_dir': 'bw02.0_exp_inv'},
              # {'label': 'KDE inv_exp', 'model_dir': r'kde/harbor/low_density_0100', 'epoch_dir': 'bw02.0'},
              ]

    # for idx, (video_name, size_max) in tqdm(enumerate(zip(test_videos, df_vid.size_max))):
    # for idx, (video_name, mean_velo) in tqdm(enumerate(zip(test_videos, df_vid.mean_velo))):
    # for idx, (video_name, object_id) in tqdm(enumerate(zip(test_videos, df_vid.object_id))):
    for video_name in tqdm(test_videos):
        if selected_video_names:
            if video_name not in selected_video_names:
                continue

        # read video and gt
        gt_labels = None
        if hasattr(dataset, 'frame_gt'):
            gt_dir = dataset.frame_gt
            if os.path.exists(gt_dir):
                if 'avenue' in dataset.name:
                    gt_filename = os.path.basename(video_name) + '.txt'
                elif 'harbor' in dataset.name:
                    gt_filename = video_name + '.txt'
                    gt_filename = gt_filename.replace('/', '_')
                gt_filepath = os.path.join(gt_dir, gt_filename)
                # gt_labels = np.loadtxt(gt_filepath, dtype=int).tolist()
                gt_labels = np.loadtxt(gt_filepath).astype(int).tolist()

        video_scores = None
        video_scores = []
        for model in models:
            if dataset.name in ['harbor-synthetic', 'harbor-mannequin', 'harbor-realfall']:
                df_video = df[df['img_path'].apply(lambda x: x.startswith('test/' + video_name))]
                gt_filepath = os.path.join(gt_dir, f"{video_name.replace('test/', '')}.txt")
            elif dataset.name in ['avenue', 'harbor', 'harbor-rareevents', 'harbor-vehicles', 'harbor-appearance', 'harbor-near-edge', 'harbor-fast-moving', 'harbor-high-density', 'harbor-tampering']:
                csv_test_name = 'results_' + dataset.csv_test
                results_path = os.path.join(cfg['models_path'], model['model_dir'], model['epoch_dir'], csv_test_name)
                df = pd.read_csv(results_path)

                if 'mnad' in model['model_dir']:
                    # TODO combine scores
                    key = 'MSE'
                elif 'ssmtl' in model['model_dir'] or \
                     'pgm' in model['model_dir'] or \
                     'kde' in model['model_dir']:
                    key = 'score'
                else:
                    __import__('ipdb').set_trace()
                    key = ''
                df[key] = (df[key] - min(df[key])) / (max(df[key]) - min(df[key]))  # normalize
                df_video = df[df['img_path'].apply(lambda x: x.startswith(video_name))]
                model['video_scores'] = df_video[key].to_list()
                # if len(model['video_scores']) == 121:
                #     model['video_scores'] = model['video_scores'][1:]

                video_auc = compute_auc(gt_labels, model['video_scores'])[0]
                # model['label'] += model['label'] + f' AUC {video_auc*100:.0f}%'
                # model['auc'] += f' AUC {video_auc*100:.0f}%'
                model['video_auc'] = video_auc

                # # object-centric
                # obj_csv_name = os.path.splitext(dataset.csv_test)[0] + '_object.csv'
                # obj_csv_path = os.path.join(cfg['models_path'], model['model_dir'], model['epoch_dir'], obj_csv_name)
                # df_obj = pd.read_csv(obj_csv_path)
                # df_obj_vid = df_obj[df_obj['image_id'].apply(lambda x: x.startswith(video_name))]
                # model['df_obj'] = df_obj_vid
            elif dataset.name == 'harbor-original':
                df_video = df[df['img_path'].apply(lambda x: x.startswith(video_name))]

        # save_video_path = os.path.join(save_dir, f'{video_name}.avi')
        save_video_path = os.path.join(save_dir, f"{video_name.replace('/', '_')}.avi")
        # save_video_path = os.path.join(save_dir, f"{size_max}_{video_name.replace('/', '_')}.avi")
        # if np.isnan(mean_velo):
        #     mean_velo = 0
        # save_video_path = os.path.join(save_dir, f"{int(mean_velo)}_{video_name.replace('/', '_')}.avi")
        # if np.isnan(object_id):
        #     object_id = 0
        # save_video_path = os.path.join(save_dir, f"{int(object_id):03d}_{video_name.replace('/', '_')}.avi")

        visualize_video(cfg, dataset, video_name, False, plot_rtbdc_bbox, draw_gt_bbox, gt_labels, models, save_video_path)
