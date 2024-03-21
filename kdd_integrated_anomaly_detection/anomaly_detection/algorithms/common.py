import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Type, Union

from ...uff_example.base import Transformer
from ...uff_example.tstypes import TimeIndexedData

logger = logging.getLogger(__name__)


class UnivariateOutlierDetectionAlgorithm(ABC):
    @abstractmethod
    def check_is_anomaly(self, timestamp: Union[datetime, Sequence[datetime]], value: Union[float, Sequence[float]]):
        pass

    @abstractmethod
    def train(self, data: TimeIndexedData, **kwargs):
        pass


class MultivariateOutlierDetectionAlgorithm(ABC):
    @abstractmethod
    def check_is_anomaly(
        self,
        timestamp: Union[datetime, Sequence[datetime]],
        value: Union[Dict[Hashable, float], Sequence[Dict[Hashable, float]]],
    ):
        pass

    @abstractmethod
    def train(self, data: TimeIndexedData, **kwargs):
        pass


class ApplyTransformersMixin:
    def __init__(self, transformers=Iterable[Transformer], **kwargs) -> None:
        self.transformers = transformers

    def apply_transforms(self, data: TimeIndexedData) -> TimeIndexedData:
        for t in self.transformers:
            data = t.transform(data).out
        return data

    def apply_fit_transforms(self, data: TimeIndexedData) -> TimeIndexedData:
        for t in self.transformers:
            t.fit(data)
            data = t.transform(data).out
        return data


class MultivariateOutlierDetectionAdapter(MultivariateOutlierDetectionAlgorithm):
    """
    This adapter class allows us to replicate multiple instances of a UnivariateOutlierDetectionAlgorithms
    across a multivariate TimeIndexedData object

    The advantage is that this class allows us to have a unified interface for multivariate and univariate
    outlier detection algorithms
    """

    def __init__(
        self,
        univariate_algo_class: Type[UnivariateOutlierDetectionAlgorithm],
        algo_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.univariate_algo_class = univariate_algo_class
        self.algo_arguments = algo_arguments or {}
        self.algo_instances = {}

    def train(self, data: TimeIndexedData, **kwargs) -> None:
        """Train multiple univariate models, one for each column in `data`
        If training fails, the offending algo instance is discarded

        Parameters
        ----------
        data : TimeIndexedData
            Training data, a multivariate time series (multiple columns expected)
        """
        if len(data.shape) > 2:
            raise ValueError(
                "expected 2-dimenional TimeIndexedData instance, general tensor-like TimeIndexedData "
                "instances are currently not supported"
            )

        for c in data.column_names:
            self.algo_instances[c] = self.univariate_algo_class(**deepcopy(self.algo_arguments))
            try:
                self.algo_instances[c].train(
                    TimeIndexedData.from_time_index(
                        index=data.time_index,
                        values=data[c],
                        column_names=c,
                    )
                )
            except Exception as e:
                logger.error(f"error training model for column {c}, discarding algorithm instance: {e}")
                del self.algo_instances[c]

    def check_is_anomaly(
        self,
        timestamp: Union[datetime, Sequence[datetime]],
        value: Union[Dict[Hashable, float], Sequence[Dict[Hashable, float]]],
    ) -> List[Dict[Hashable, bool]]:
        """Checks if the values from a multivariate time series contains anomalies, at a given timestamp

        Parameters
        ----------
        timestamp : Union[datetime, Sequence[datetime]]
            Timestamp(s) being evaluated
        value : Union[Dict[Hashable, float], Sequence[Dict[Hashable, float]]]
            Each item in the dict corresponds to an entity, with an associated numeric value

        Returns
        -------
        Dict[Hashable, bool]
            Dictionary with same keys as `value`, with a boolean value designated if it is an anomaly or not
        """
        ret = [{}]
        # if `value` is a Sequence, re-orient from records to 1 list of floats per column
        # if keys are missing in some records, None is assigned
        if isinstance(value, Sequence):
            ret = [{} for _ in value]
            columns_dict = {}
            for _i, _d in enumerate(value):
                for _k, _v in _d.items():
                    if _k not in columns_dict:
                        columns_dict[_k] = [None] * len(value)
                    columns_dict[_k][_i] = _v
            value = columns_dict

        for k, v in value.items():
            m = self.algo_instances.get(k)
            if m is None:
                logger.warn(f"trained model for column {k} not found, no anomalies will be labeled for this segment")
            else:
                for i, b in enumerate(m.check_is_anomaly(timestamp, v)):
                    ret[i][k] = b
        return ret
