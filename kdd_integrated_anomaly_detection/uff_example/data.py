from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .tstypes import TimeIndexedData
from .utils import time_range

_M4_DATE_FORMATS = [
    "%d-%m-%y %I:%M",
    "%Y-%m-%d %I:%M:%S",
]


def _url_prefix() -> str:
    return "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"


def _url(frequency: str, train_test: str) -> str:
    frequency = frequency.lower()
    train_test = train_test.lower()
    prefix = _url_prefix()
    return f"{prefix}/{train_test.title()}/{frequency.title()}-{train_test}.csv"


def _info_url() -> str:
    return _url_prefix() + "/M4-info.csv"


def _parse_time(value: Any) -> pd.Timestamp:
    for fmt in _M4_DATE_FORMATS:
        try:
            dt = pd.to_datetime(value, format=fmt)
            return pd.Timestamp(dt)
        except Exception:
            pass

    raise ValueError(f"Could not parse timestamp from {value}")


def _get_start_times() -> Dict[str, pd.Timestamp]:
    return {row["M4id"]: _parse_time(row["StartingDate"]) for _, row in pd.read_csv(_info_url()).iterrows()}


def download_m4_dataset(
    frequency: str,
    max_time_series: Optional[int] = None,
) -> Dict[str, Tuple[TimeIndexedData, TimeIndexedData]]:
    """Download all univariate m4 competition datasets of a particular frequency

    Parameters
    ----------
    frequency: str
        One of the M4 competition frequencies: "hourly", "daily", "weekly", "monthly", "quarterly", "yearly"
    max_time_series: Optional[int]
        If specified, the maximum number of unique time series that will be returned.

    Returns
    -------
    Dict[str, Tuple[TimeIndexedData, TimeIndexedData]]
        A dictionary mapping the M4 UUID (see M4-info.csv in the public repo) to a tuple containing the `Train` and
        `Test` data

    Raises
    ------
    ValueError
        If `frequency` is not a recognized option.
    """
    frequency = frequency.lower()
    uff_granularity = "day" if frequency == "daily" else frequency[:-2]
    if frequency not in ("hourly", "daily", "weekly", "monthly", "quarterly", "yearly"):
        raise ValueError(f"Unrecognized frequency {frequency}")

    start_times = _get_start_times()

    parsed_data = {}
    for segment in ("train", "test"):
        reader = pd.read_csv(_url(frequency, segment), chunksize=1000, nrows=max_time_series)
        df = pd.concat([x for x in reader], ignore_index=True)
        for _, row in df.iterrows():
            row_data = list(row)
            uid, data = row_data[0], row_data[1:]
            if uid not in parsed_data:
                parsed_data[uid] = {}
            parsed_data[uid][segment] = [x for x in data if not np.isnan(x)]

    results = {}
    for uid, data in parsed_data.items():
        n_train = len(data["train"])
        n_test = len(data["test"])
        global_time = time_range(start_times[uid], freq=uff_granularity, periods=n_train + n_test)
        train = TimeIndexedData(global_time[:n_train], data["train"], granularity=uff_granularity)
        test = TimeIndexedData(global_time[n_train:], data["test"], granularity=uff_granularity)
        results[uid] = (train, test)

    return results
