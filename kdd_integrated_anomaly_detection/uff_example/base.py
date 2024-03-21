from __future__ import annotations

import pickle
from abc import ABC
from pathlib import Path
from typing import Optional, Union

import cloudpickle

from .tstypes import TimeIndex, TimeIndexedData, TimeIndexedOutput, TimeIndexedOutputWithInterval


class Estimator:
    """Estimator base class

    The base class for all objects that calculate an estimate from observed data as part of their operation.
    This could be a simple mean/standard deviation calculation, or an optimization procedure to fit the
    parameters of a model.
    """

    def fit(self, data: TimeIndexedData, covariates: Optional[TimeIndexedData] = None, **kwargs) -> Estimator:
        return self

    def save(self, path: str) -> None:
        """Persist this estimator at the location specified by `path`

        Parameters
        ----------
        path: str
            A path to the serialized Estimator's output destination.
        """
        with Path(path).open("wb") as f:
            cloudpickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> Estimator:
        """Load a previously saved Estimator instance from `path`

        Parameters
        ----------
        path: str
            A path to the serialized Estimator
        """
        with Path(path).open("rb") as f:
            return cloudpickle.load(f)


class Incremental(ABC):
    """Base class for all estimators with the ability to learn "online"

    Estimators with the Incremental mixin can perform a partial fit, or a parameter update based on a small number of
    samples. This method can be called multiple times throughout the Estimator instance's lifecycle.
    """

    def partial_fit(
        self, data: TimeIndexedData, covariates: Optional[TimeIndexedData] = None, **kwargs
    ) -> Estimator: ...


class Transformer(Estimator):
    """Base class providing the .transform() method

    Transformers use the `transform()` method to accept TimeIndexedData and output TimeIndexedOutput.
    This can be used for scoring, detection, feature extraction, compression/encoding, etc.

    The `out` attribute of .transform() is expected to preserve the same shapes (same index and column values)
    as the `data` object passed into .transform().  Additional outputs can be specified as separate attributes.
    """

    def transform(self, data: TimeIndexedData, **kwargs) -> TimeIndexedOutput: ...


class ReversibleTransformer(Transformer):
    """A Transformer subclass which also implements the `inverse_transform` method.

    Besides the additional method, ReversibleTransformers share all the same properties as Transformers.
    """

    def inverse_transform(self, data: TimeIndexedData, **kwargs) -> TimeIndexedOutput: ...


class Forecaster(Estimator):
    """Base class providing the .forecast() method

    Forecasters predict values of a given TimeIndexedData object. This mixin should almost always be used
    in conjunction with an Estimator. When used inside an Estimator, `forecast()` should predict the
    past/future values of the same TimeIndexedData that was used during `.fit()` / `.update()`

    The `out` attribute of .forecast() is expected to preserve the same shapes (same index and column values)
    as the `data` object passed into .transform().  Typically, these will be point estimates from the model.
    Additional outputs can be specified as separate attributes.
    """

    def forecast(
        self,
        data: Union[TimeIndex, TimeIndexedData],
        **kwargs,
    ) -> TimeIndexedOutput: ...


class ForecasterWithInterval(Forecaster):
    """A Forecaster object with an adjustable prediction interval width.

    All subclasses of this type must accept `prediction_interval_width` as an initialization parameter
    """

    @property
    def prediction_interval_width(self) -> float:
        raise NotImplementedError()

    def forecast(
        self,
        data: Union[TimeIndex, TimeIndexedData],
        **kwargs,
    ) -> TimeIndexedOutputWithInterval: ...
