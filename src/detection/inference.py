import os
import glob
import time

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
pd.set_option('mode.chained_assignment', None)


def list_dataset_images(data_path, save_file):
    image_paths = glob.glob(os.path.join(data_path, '**/*.jpg'))
    image_paths.sort()
    with open(save_file, 'w') as f:
        f.write("\n".join(image_paths))


def run_inference(img_file_paths, model):
    # Images
    if isinstance(img_file_paths, str):
        with open(img_file_paths) as f:
            img_paths = f.readlines()
        img_paths = [img_path.rstrip('\n') for img_path in img_paths]
    else:
        img_paths = img_file_paths
    imgs = [cv2.imread(img_path)[..., ::-1] for img_path in img_paths]

    # Inference
    results = model(imgs)

    # Results
    # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.))))
    # results.save()  # or .show()
    # results.xyxy[0]  # img1 predictions (tensor)
    # results.pandas().xyxy[0]  # img1 predictions (pandas)
    results_list_df = results.pandas().xyxy
    for idx, image_df in enumerate(results_list_df):
        results_list_df[idx]['image_id'] = img_paths[idx]
    detections = pd.concat(results_list_df)
    return detections


def chunks(lst, n):
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_bboxes(video_path, model):
    img_paths = sorted(glob.glob(video_path + '/*.jpg'))
    batch_generator = chunks(img_paths, n=250)  # batching to avoid running out of memory
    results_list = []
    for batch_imgs in batch_generator:
        results_batch = run_inference(batch_imgs, model)
        results_list.append(results_batch)
    detections = pd.concat(results_list)
    return detections


def filter_bboxes(detections, dataset_path, common_classes_only=False):
    # filter out non-generic classes
    if common_classes_only:
    # names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #         'hair drier', 'toothbrush']  # class names
        # detections = detections[detections.name.isin(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])]
        detections = detections[detections.name.isin(['person', 'bicycle', 'skateboard',
                                                      'car', 'motorcycle', 'bus', 'train', 'truck',
                                                      'backpack', 'handbag', 'suitcase',
                                                      'bird', 'dog', 'cat', 'horse', 'sheep', 'cow'])]

    # rearrange columns and names
    detections.image_id = detections.image_id.str.replace(dataset_path + '/', '')
    detections['bbox_w'] = detections.xmax - detections.xmin
    detections['bbox_h'] = detections.ymax - detections.ymin
    detections = detections.drop(columns=['xmax', 'ymax'])
    detections = detections.rename(columns={"xmin": "bbox_x1", "ymin": "bbox_y1"})
    detections = detections[['image_id', 'bbox_x1', 'bbox_y1', 'bbox_w', 'bbox_h', 'confidence', 'class', 'name']]
    for column in ['bbox_x1', 'bbox_y1', 'bbox_w', 'bbox_h']:
        detections[column] = np.round(detections[column]).astype(int)
    return detections


if __name__ == "__main__":
    # dataset = 'harbor'
    # dataset = 'harbor-mannequin'
    dataset = 'harbor-realfall'

    model = r'yolov5/harborfront_thermal_yolov5_baseline/best.pt'

    cfg = load_config('config.yaml')
    save_dir = os.path.join(cfg['detection']['dir'], 'yolov5')

    torch.hub.set_dir(cfg['detection']['model_dir'])   # default: /home/jacob/.cache/torch/hub
    # model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or yolov3-spp, yolov3-tiny, custom
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5m, yolov5l, yolov5x, custom
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # model.conf = 0.25  # NMS confidence threshold
    hub_dir = cfg['detection']['hub_dir']
    model_path = os.path.join(cfg['detection']['model_dir'], model)
    model = torch.hub.load(hub_dir, 'custom', path=model_path, source='local')  # local repo
    model.eval()

    dataset_path = os.path.join(cfg['dataset'][dataset]['path'],
                                cfg['dataset'][dataset]['img_dir'])
    if dataset in ['harbor']:
        video_paths = sorted(glob.glob(os.path.join(dataset_path, '**/**')))
    elif dataset in ['harbor-mannequin',  'harbor-realfall']:
        video_paths = sorted(glob.glob(os.path.join(dataset_path, '*')))
    else:
        __import__('ipdb').set_trace()
        raise

    for video_path in (pbar := tqdm(video_paths, ncols=89)):
        detections = extract_bboxes(video_path, model)
        detections = filter_bboxes(detections, dataset_path)

        video_path = video_path.replace(dataset_path + '/', '')
        csv_path = os.path.join(save_dir, dataset, video_path + '.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        detections.to_csv(csv_path, float_format='%.4f', index=False)
        pbar.set_description(f'{video_path:22s}')

        # # time the first sequence
        # if split == 'test':
        #     first_sequence = sorted(os.listdir(data_split))[0]
        #     dataset_img_paths = sorted(glob.glob(os.path.join(data_split, first_sequence, '*.jpg')))
        #     times_ms = []
        #     for idx, img_path in enumerate(dataset_img_paths):
        #         tic = time.time()
        #         img = cv2.imread(img_path)[::-1]
        #         results = model(img)
        #         toc = time.time()
        #         times_ms.append((toc-tic)*1000)
        #     print(f'split: {split}, mean: {np.mean(times_ms):.2f}ms, std: {np.std(times_ms):.2f}, min: {np.min(times_ms):.2f}ms, max: {np.max(times_ms):.2f}ms')
