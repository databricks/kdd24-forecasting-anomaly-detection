from __future__ import annotations

import functools
import json
import random
from contextlib import contextmanager
from random import Random
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np
import pandas as pd

from .base import Estimator
from .tstypes import TimeIndex, TimeIndexedData, _get_custom_granularity, _infer_granularity, _round, _time_range

get_custom_granularity = _get_custom_granularity

time_range = _time_range

round = _round

infer_granularity = _infer_granularity


__global_rng: np.random.Generator = np.random.default_rng()


def global_rng() -> np.random.Generator:
    return __global_rng


def random_seed() -> int:
    return int(global_rng().integers(2**32))


def _coalesce(seed):
    """This function will coalesce a non-None seed.

    If the state of np.random has been seeded then the output will be random but deterministic
    within a process.
    """
    if seed is None:
        return random_seed()
    return seed


def load_estimator_from_json(json_data: Union[str, Dict[str, Any]]) -> Estimator:
    """Load an Estimator from a JSON string or dictionary

    Parameters
    ----------
    json_data: Union[str, Dict[str, Any]]
        A JSON string or dictionary containing the Estimator specification

    Returns
    -------
    Estimator
        An Estimator instance
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    return InitializationSpec.from_dict(json_data).create_instance()


@contextmanager
def deterministic_rng(valid_seed: int) -> None:
    """Reseeds all known random number generators and blocks common sources of nondeterminism

    This is a best-effort attempt to make all enclosed code deterministic. This context manager seeds `random` and
    `np.random` global states with `valid_seed`. This context manager also monkey-patches functions that can
    introduce non-determinism (random.seed, Random.seed, np.random.seed, np.random.default_rng). The patching is
    performed in such a way that the RNGs are still reseeded upon request, but any non-specified seed value is drawn
    from the previously-seeded np.random state rather than the OS (keeping the following operations deterministic).

    There are a few ways nondeterminism can still be introduced within this context manager.

    * External sources of randomness (e.g. asynchronous code) affecting the order of code execution.
    * Modules that use a library besides numpy or random for random number generation.
    * Modules that make direct use of bit generators like np.random.PCg64
    * Modules that make direct references to the random state attribute (from numpy.random import RandomState)
      rather than the module (import numpy; numpy.random.RandomState())
    * Any pre-existing RNG instances or new ones that don't use the patched constructors in this context manager.

    In general, a `random_state` or `seed` API should be preferred if it is offered by a class/function; this
    context manager should be used as a best-effort attempt to apply determinism to objects that don't offer
    such an API.

    Parameters
    ----------
    valid_seed: int
        The new global seed value

    Yields
    ------
    None

    Raises
    ------
    ValueError
        If valid_seed is None, as this would not guarantee determinism
    """
    if valid_seed is None:
        raise ValueError("Setting seed to None does not guarantee determinism")

    # Set global RNG
    global __global_rng
    old_rng_snapshot = __global_rng

    __global_rng = np.random.default_rng(valid_seed)

    # Take state snapshots
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    # Seed the global RNGs
    random.seed(valid_seed)
    np.random.seed(valid_seed)

    # Snapshot and patch known RNG callables
    random_seed = random.seed
    random.seed = lambda a=None, **kwargs: random_seed(_coalesce(a), **kwargs)

    Random_seed = Random.seed
    Random.seed = lambda self, a=None, **kwargs: Random_seed(self, _coalesce(a), **kwargs)

    np_seed = np.random.seed
    np.random.seed = lambda seed=None: np_seed(_coalesce(seed))

    default_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: default_rng(_coalesce(seed))

    try:
        yield
    finally:
        # Reset global seed
        __global_rng = old_rng_snapshot

        # Reinstate the original callables
        random.seed = random_seed
        Random.seed = Random_seed
        np.random.seed = np_seed
        np.random.default_rng = default_rng

        # Reset the original global states
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def make_deterministic(valid_seed: int) -> Callable:
    """Decorator to make wrapped function deterministic by reseeding all known random number generators.

    See `deterministic_rng` for details on how this is achieved.

    Parameters
    ----------
    valid_seed: int
        The new global seed value

    Returns
    -------
    Callable
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with deterministic_rng(valid_seed):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_estimator(name: str) -> Type[Estimator]:
    """Fetch an Estimator subclass by its classname

    Parameters
    ----------
    name: str
        The name of the desired estimator(s)

    Returns
    -------
    Type[Estimator]
        The estimator subclass matching the specified name

    Raises
    ------
    KeyError
        If the estimator is not found
    """
    if name == "ProphetForecaster":
        from kdd_integrated_anomaly_detection.uff_example.forecasters.prophet import ProphetForecaster

        return ProphetForecaster

    if name == "MiniProphetForecaster":
        from kdd_integrated_anomaly_detection.uff_example.forecasters.miniprophet import MiniProphetForecaster

        return MiniProphetForecaster

    if name == "ResidualsTransformer":
        from kdd_integrated_anomaly_detection.uff_example.transformers.residual import ResidualsTransformer

        return ResidualsTransformer
    raise KeyError(f"Estimator {name} not found")


def estimator_from_json(json_data: Union[str, Dict[str, Any]]) -> Estimator:
    """Load an Estimator from a JSON string or dictionary

    Parameters
    ----------
    json_data: Union[str, Dict[str, Any]]
        A JSON string or dictionary containing the Estimator specification

    Returns
    -------
    Estimator
        An Estimator instance
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    estimator_class = get_estimator(json_data["cls"])
    return estimator_class(**json_data["kwargs"])


def is_univariate(data: TimeIndexedData) -> bool:
    return len(data.column_names) == 1 and data.values.ndim < 3


def prophet_future_dataframe(
    data: Union[TimeIndex, TimeIndexedData], *, time_col: str = "ds", value_col: str = "y"
) -> pd.DataFrame:
    """Create a prophet-style "future" dataframe

    A future dataframe is a dataframe with "ds" as the time column and optional regressor columns.

    Parameters
    ----------
    data: Union[TimeIndex, TimeIndexedData]
        ds (the time index) and optional regressors
    time_col: str
        time column name, by default "ds"
    value_col: str
        value column name, by default "y"

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(data, TimeIndexedData):
        ds, covariates = data.time_index, data
    else:
        ds, covariates = data, None

    return prophet_style_dataframe(ds, covariates, time_col=time_col, value_col=value_col)


def prophet_style_dataframe(
    data: Union[TimeIndex, TimeIndexedData],
    covariates: Optional[TimeIndexedData] = None,
    *,
    time_col: str = "ds",
    value_col: str = "y",
) -> pd.DataFrame:
    """Create a Prophet-style dataframe with data and covariates

    A Prophet-style dataframe is a univariate time series with "ds" as the time column name and "y" as the observable
    column name (if provided). All of the columns from `covariates` are included as extra regressor columns.

    Parameters
    ----------
    data: Union[TimeIndex, TimeIndexedData]
        The observable columns. If a TimeIndex is provided then only the `time_col` column will be populated,
        not the `value_col` column. This is useful for generating future dataframes.
    covariates: Optional[TimeIndexedData], default None
        If applicalble, a set of extra regressors.
    time_col: str
        time column name, by default "ds"
    value_col: str
        value column name, by default "y"

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If `data` is multivariate, or the `covariate` time index does not match the one from `data`, or any regressors
        are named `time_col` or `value_col`.
    """
    obs_col, regressor_cols = {}, {}

    if isinstance(data, TimeIndexedData):
        if not is_univariate(data):
            raise ValueError("Expecting univariate time series data")
        obs_col = {value_col: data.values.flatten()}

    if covariates is not None:
        index = data if isinstance(data, TimeIndex) else data.time_index
        if not index.values_match(covariates.time_index):
            raise ValueError("Covariate time index does not match expected")
        if any(c in (time_col, value_col) for c in covariates.column_names):
            raise ValueError(f"Regressors cannot be named {time_col} or {value_col}")
        regressor_cols = {c: covariates[c].flatten() for c in covariates.column_names}

    time_col = {time_col: data.pd_timestamp_index()}
    return pd.DataFrame(dict(**obs_col, **time_col, **regressor_cols))
