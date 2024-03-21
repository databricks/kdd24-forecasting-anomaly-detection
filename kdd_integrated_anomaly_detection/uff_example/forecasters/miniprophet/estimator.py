from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from ...base import ForecasterWithInterval
from ...tstypes import ColumnSet, TimeIndex, TimeIndexedData, TimeIndexedOutputWithInterval, TimeStamp
from ...utils import is_univariate
from ._miniprophet_impl import MiniProphet

if TYPE_CHECKING:
    import datetime


class MiniProphetForecaster(ForecasterWithInterval):
    def __init__(
        self,
        weekly_order: Optional[int] = None,
        daily_order: Optional[int] = None,
        n_changepoints: Optional[int] = None,
        lambda_reg: float = 0.001,
        prediction_interval_width: float = 0.95,
        holidays: Optional[Collection[TimeStamp]] = None,
    ) -> None:
        """Initialize a MiniProphetForecaster

        Parameters
        ----------
        weekly_order: Optional[int], default None
            Fourier order of the weekly seasonality component. If unspecified, this will be determined dynamically
            based on attributes of the `fit()` data.
        daily_order: Optional[int], default None
            Fourier order of the daily seasonality component. If unspecified, this will be determined dynamically based
            on attributes of the `fit()` data.
        n_changepoints: Optional[int], default None
            The number of changepoints to be distributed throughout the training data. If unspecified, this will be
            determined dynamically based on attributes of the `fit()` data.
        lambda_reg: float, default 0.001
            Amount of regularization for the point estimate model parameters.
        prediction_interval_width: float, default 0.95
            Width of the uncertainty intervals provided for the forecast.
        holidays: Optional[TimeStamp], default None
            A list of dates to be considered as holidays by the model.
        """

        self._weekly_order = weekly_order
        self._daily_order = daily_order
        self._n_changepoints = n_changepoints
        self._lambda_reg = lambda_reg
        self._holidays: Set[datetime.date] = {pd.Timestamp(h).date() for h in (holidays or [])}

        # Models (initialized during fit)
        self._point_model: Optional[MiniProphet] = None
        self._high_model: Optional[MiniProphet] = None
        self._low_model: Optional[MiniProphet] = None

        # Fit attributes
        self._out_cols: Optional[ColumnSet] = None
        self._fit_shape: Optional[Tuple[int]] = None

        # Required ForecasterWithInterval
        self._prediction_interval_width = prediction_interval_width

    @property
    def prediction_interval_width(self) -> float:
        return self._prediction_interval_width

    def _instantiate_unfitted_models(self, data: TimeIndexedData) -> None:
        """Assign unfitted MiniProphet instances to the private model attributes.

        This method is intended to be called during `fit()`.

        Parameters
        ----------
        data: TimeIndexedData
            The data that is currently being fit.
        """
        data = data.dropna()
        n_samples = len(data)
        duration = data.last_timestamp() - data.first_timestamp()

        # Auto parameters
        # If unset, no more than n_samples // 30 parameters will be fit to each of these components.
        n_changepoints, daily_order, weekly_order = self._n_changepoints, self._daily_order, self._weekly_order

        if weekly_order is None:
            if duration < pd.Timedelta(days=14):
                weekly_order = 0
            else:
                weekly_order = np.clip(n_samples // 60, 1, 6)

        if daily_order is None:
            if duration < pd.Timedelta(days=2):
                daily_order = 0
            else:
                daily_order = np.clip(n_samples // 60, 1, 6)

        if n_changepoints is None:
            n_changepoints = min(n_samples // 30, 10)

        self._point_model = MiniProphet(
            weekly_order=weekly_order,
            daily_order=daily_order,
            n_changepoints=n_changepoints,
            lambda_reg=self._lambda_reg,
        )

        # Heuristics to keep extreme quantiles robust
        # - 2x less Fourier coefficients
        # - 10x regularization strength
        quantile_kw = {
            "n_changepoints": n_changepoints,
            "lambda_reg": self._lambda_reg * 10,
            "weekly_order": weekly_order // 2,
            "daily_order": daily_order // 2,
        }
        self._high_model = MiniProphet(quantile=0.5 + (self.prediction_interval_width / 2), **quantile_kw)
        self._low_model = MiniProphet(quantile=0.5 - (self.prediction_interval_width / 2), **quantile_kw)

    def fit(
        self,
        data: TimeIndexedData,
        covariates: Optional[TimeIndexedData] = None,
    ) -> MiniProphetForecaster:
        """Fit a MiniProphet model to the input time series

        Parameters
        ----------
        data: TimeIndexedData
            The univariate input observations
        covariates: Optional[TimeIndexedData], default None
            This argument is unused as MiniProphet does not yet support external regressors. It is kept here to
            maintain signature compatibility with other estimators.

        Returns
        -------
        MiniProphetForecaster
            A reference to `self`

        Raises
        ------
        ValueError
            If `data` is not univariate.
        """
        if not is_univariate(data):
            raise ValueError("MiniProphetForecaster only supports univariate data.")

        self._instantiate_unfitted_models(data)

        # Train point, upper, and lower estimates
        x = pd.Series(data.pd_timestamp_index())
        y = pd.Series(data.values.flatten())

        is_holiday = self.__get_is_holiday(data.time_index)
        self._point_model.learn(x, y, is_holiday)
        self._high_model.learn(x, y, is_holiday, self._point_model)
        self._low_model.learn(x, y, is_holiday, self._point_model)
        self._out_cols = data.column_tuples
        self._fit_shape = data.shape[1:]
        return self

    def forecast(
        self,
        data: Union[TimeIndex, TimeIndexedData],
        **kwargs: Any,
    ) -> TimeIndexedOutputWithInterval:
        """Use the fitted model to forecast future values

        Parameters
        ----------
        data: Union[TimeIndex, TimeIndexedData]
            The data to forecast.
        kwargs: Dict[str, Any]
            Unused

        Returns
        -------
        TimeIndexedOutputWithInterval
            The forecast result and underlying model.
        """
        index = data if isinstance(data, TimeIndex) else data.time_index
        x = pd.Series(data.pd_timestamp_index())
        is_holiday = self.__get_is_holiday(index)
        return TimeIndexedOutputWithInterval(
            out=self._format_result(index, self._point_model.forecast(x, is_holiday, **kwargs)),
            upper=self._format_result(index, self._high_model.forecast(x, is_holiday, **kwargs)),
            lower=self._format_result(index, self._low_model.forecast(x, is_holiday, **kwargs)),
            interval_width=self.prediction_interval_width,
        )

    def _format_result(self, index: TimeIndex, result: pd.Series) -> TimeIndexedData:
        return TimeIndexedData.from_time_index(
            index,
            result.to_numpy().reshape((len(index),) + self._fit_shape),
            column_names=self._out_cols,
        )

    def __get_is_holiday(self, index: TimeIndex) -> "npt.NDArray":
        if len(self._holidays) == 0:
            return None

        return np.array([t.date() in self._holidays for t in index.timestamp_values], dtype=np.bool_)
