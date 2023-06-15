from enum import Enum
import src.data.datasets.harbor as harbor_datasets


class Dataset(Enum):
    harbor = 1
    harbor_appearance = 2
    harbor_vehicles = 3
    harbor_fast_moving = 4
    harbor_near_edge = 5


def get_dataset(dataset_name):
    if dataset_name == Dataset.harbor:
        dataset = harbor_datasets.Harbor()
    elif dataset_name == Dataset.harbor_appearance:
        dataset = harbor_datasets.Appearance()
    elif dataset_name == Dataset.harbor_vehicles:
        dataset = harbor_datasets.Vehicles()
    elif dataset_name == Dataset.harbor_fast_moving:
        dataset = harbor_datasets.FastMoving()
    elif dataset_name == Dataset.harbor_near_edge:
        dataset = harbor_datasets.NearEdge()
    else:
        raise ValueError
    return dataset


if __name__ == "__main__":
    harbor = get_dataset(Dataset.harbor)
