from src.utils.config import load_config
from src.data.harbor_ground_truth import anomaly_intervals_to_frame_level_gt


ANOMALY_INTERVALS = [
        ['fall_01', [235, 298]],
        ['fall_02', [22, 147]],
        ['fall_03', [43, 130]],
        ['fall_04', [127, 210]],
        ['fall_05', [104, 249]],
        ['fall_06', [192, 347]],
        ['fall_07', [246, 346]],
        ['fall_08', [65, 200]],
        ['fall_09', [282, 352]],
        ['fall_10', [0, 149]],
        ['fall_11', [135, 230]],
        ['fall_12', [263, 319]],
        ['fall_13', [122, 196]],
        ['fall_14', [265, 344]],
        ['fall_15', [101, 177]],
        ['fall_16', [200, 258]],
        ['fall_17', [163, 267]],
        ['fall_18', [75, 157]],
        ['fall_19', [69, 153]],
        ['fall_20', [132, 210]],
        ['fall_21', [7, 143]],
        ['fall_22', [28, 154]],
    ]


if __name__ == "__main__":
    config = load_config('config.yaml')

    dataset = 'harbor-mannequin'
    data_path = config.dataset[dataset].path

    # download and extract data

    # extract object-detections

    # ground-truth
    anomaly_intervals_to_frame_level_gt(
            ANOMALY_INTERVALS, data_path, sub_dir='test', offset=1)
