import yaml
from munch import Munch


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    config = Munch.fromDict(config)
    return config
