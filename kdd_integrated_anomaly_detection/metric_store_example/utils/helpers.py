import pathlib

import yaml

ROOT = pathlib.Path(__file__).parent.parent.absolute()  # Metric Store folder
CONFIG_DIR = ROOT.joinpath("configs/")
DATA_DIR = ROOT.joinpath("data/")
TIMESTAMP_COLUMN = "ts"
METRIC_TIME_COLUMN = "time"
DIALECT = "spark"


def load_config(filepath) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
