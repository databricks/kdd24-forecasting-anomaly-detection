from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from statsmodels.robust.scale import Huber, mad

from .common import (
    ApplyTransformersMixin,
    UnivariateOutlierDetectionAlgorithm,
)
from ...uff_example.base import Forecaster, Transformer
from ...uff_example.tstypes import TimeIndexedData, TimeStamp


class UnivariateModelAnomalyDetection(UnivariateOutlierDetectionAlgorithm, ApplyTransformersMixin):
    """
    Simple anomaly detection for a univariate time series
    An anomaly is defined as a residual with distance N * Huber deviation, where N is defined using the
        parameter `tolerance`
    User can optionally pass in a sequence of transformers and/or a model class.  UnivariateModelAnomalyDetection
        will apply the transformers sequentially, then fit the model on the result.  If a model is fit, anomaly
        detection will be performed on the prediction errors.
    """

    def __init__(
        self,
        transformers: Optional[Iterable[Transformer]] = None,
        model: Optional[Forecaster] = None,
        tolerance: float = 5.0,
        direction: Optional[str] = None,
    ):
        super().__init__(transformers or [])
        self.model = model
        self.tolerance = tolerance
        possible_directions = ("up", "down")
        self.direction = (direction,) if direction in possible_directions else possible_directions

    def train(self, data: TimeIndexedData, **kwargs) -> None:
        """
        Passes an input series through a sequence of transformers and/or into the fitting routine
            of a model
        If transformers and a model are specified, all the transformations are applied, and the
            transformed time series is passed into model.fit()
        The Huber deviation is calculated for the resulting residual process
        """
        data = self.apply_fit_transforms(data)

        if self.model is not None:
            self.model.fit(data)
            data -= self.model.forecast(data=data.time_index).out

        series_values = data.to_pandas(time_col="ts", time_as_pd_timestamps=True).set_index("ts").dropna().values
        if len(series_values) < 10:
            raise ValueError(
                "Less than 10 non-null values after transformations, insufficient for estimating Huber deviations"
            )

        _, self._huber_estimator = Huber(maxiter=200)(
            series_values, mu=np.array(0), initscale=mad(series_values, center=0)
        )

    def compute_residual(
        self, timestamp: Union[TimeStamp, Sequence[TimeStamp]], val: Union[float, Sequence[float]]
    ) -> Sequence[float]:
        """
        Transforms a single observation and returns its residual value
        Transformers and Model are assumed to remove any nonstationarities, so residual here refers to the residual
            process, after nonstationary effects have been removed.  This can be, but is not necessarily, the errors
            for a given time series model.
        """
        series = TimeIndexedData(time_array=timestamp, values=val, column_names=["val"])
        series = self.apply_transforms(series)
        if self.model is not None:
            series -= self.model.forecast(data=series).out.set_column_names("val")
        return series.values

    def check_is_anomaly(
        self, timestamp: Union[TimeStamp, Sequence[TimeStamp]], val: Union[float, Sequence[float]]
    ) -> List[bool]:
        """
        Returns true if an observation is an anomaly based on a Huber distance criterion
        """
        resid = self.compute_residual(timestamp, val)
        res = np.array([False] * len(resid))
        if "down" in self.direction:
            res = res + (resid < -self.tolerance * self._huber_estimator)
        if "up" in self.direction:
            res = res + (resid > self.tolerance * self._huber_estimator)
        return list(res)
