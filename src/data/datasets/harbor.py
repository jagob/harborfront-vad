import os
from src.utils.config import load_config


class Harbor():
    name: str = 'harbor'
    img_width: int = 384
    img_height: int = 288
    img_dir: str = 'Image Dataset'

    def __init__(self):
        self.cfg = load_config('config.yaml')

        # csv_train: harbor_train_split.csv
        # csv_train: harbor_train_empty.csv
        self.csv_train: str = 'harbor_train.csv'

        self.path = os.path.join(self.cfg.dataset_path, 'LTD Dataset')
        self.images = os.path.join(self.path, self.img_dir)
        self.metadata = os.path.join(self.path, 'metadata.csv')
        self.object_annotations = self.cfg.dataset[Harbor.name].object_annotations
        self.yolov5 = os.path.join(self.cfg.detection.dir, 'yolov5', Harbor.name)

        self.config_override()

    # TODO: consider moving to a general dataset class
    def config_override(self):
        # if 'dataset' in self.cfg:
        if self.name in self.cfg.dataset:
            if 'path' in self.cfg.dataset[self.name]:
                self.path = self.cfg.dataset[self.name].path

    def get_rtbdc_track_path(self, video_name):
        rtbdc_filename = video_name.replace('/', '_') + '.txt'
        tracks_path = os.path.join(self.rtbdc_gt, rtbdc_filename)
        return tracks_path


class TestSet(Harbor):
    name: str = 'harbor-testname'
    csv_videos_name: str = 'harbor_testname.csv'
    csv_test: str = 'harbor_testname_test.csv'
    num_test_frames: int = 1200

    def __init__(self):
        super().__init__()

        # override
        self.path = os.path.join(self.cfg.dataset_path, self.name)
        if self.name in self.cfg.dataset:
            Harbor.config_override(self)

        self.frame_gt: str = os.path.join(self.path, 'frame-gt')
        self.rtbdc_gt: str = os.path.join(self.path, 'rbdc-tbdc-gt')


class Appearance(TestSet):
    name: str = 'harbor-appearance'
    csv_videos_name: str = 'harbor_appearance.csv'
    csv_test: str = 'harbor_appearance_test.csv'
    num_test_frames: int = 1200


class Vehicles(TestSet):
    name: str = 'harbor-vehicles'
    csv_videos_name: str = 'harbor_vehicles.csv'
    csv_test: str = 'harbor_test_100_vehicles.csv'
    num_test_frames: int = 12100


class FastMoving(TestSet):
    name: str = 'harbor-fast-moving'
    csv_videos_name: str = 'harbor_fast_moving.csv'
    csv_test: str = 'harbor_fast_moving_test.csv'
    num_test_frames: int = 1200


class NearEdge(TestSet):
    name: str = 'harbor-near-edge'
    csv_videos_name: str = 'harbor_near_edge.csv'
    csv_test: str = 'harbor_near_edge_test.csv'
    num_test_frames: int = 1200


class HighDensity(TestSet):
    name: str = 'harbor-high-density'
    csv_videos_name: str = 'harbor_high_density.csv'
    csv_test: str = 'harbor_high_density_test.csv'
    num_test_frames: int = 1200


class Tampering(TestSet):
    name: str = 'harbor-tampering'
    csv_videos_name: str = 'harbor_tampering.csv'
    csv_test: str = 'harbor_tampering_test.csv'
    num_test_frames: int = 1200
