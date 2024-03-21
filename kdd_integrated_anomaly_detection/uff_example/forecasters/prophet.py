from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .. import utils
from ..base import ForecasterWithInterval
from ..tstypes import ColumnSet, TimeIndex, TimeIndexedData, TimeIndexedOutputWithInterval, TimeStamp

PrimitiveType = Optional[Union[str, int, float, bool]]
ParamDict = Dict[str, PrimitiveType]

logger = logging.getLogger(__name__)


@dataclass
class ProphetOutput(TimeIndexedOutputWithInterval):
    """Output for Prophet models

    Attributes
    ----------
    out: TimeIndexedData
        Forecasted point estimates
    upper: TimeIndexedData
        The upper bound of the forecast prediction interval
    lower: TimeIndexedData
        The lower bound of the forecast prediction interval
    interval_width: float
        The width of the prediction interval
    components: TimeIndexedData
        Forecasted components and their prediction intervals
    """

    out: TimeIndexedData
    upper: TimeIndexedData
    lower: TimeIndexedData
    interval_width: float
    components: TimeIndexedData


class ProphetForecaster(ForecasterWithInterval):
    def __init__(
        self,
        growth: str = "linear",
        changepoints: Optional[List[TimeStamp]] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[str, bool, int] = "auto",
        weekly_seasonality: Union[str, bool, int] = "auto",
        daily_seasonality: Union[str, bool, int] = "auto",
        holidays: Optional[pd.DataFrame] = None,
        country_holidays: Optional[str] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        prediction_interval_width: float = 0.95,
        uncertainty_samples: int = 1000,
        cap: Optional[float] = None,
        floor: Optional[float] = None,
        custom_seasonalities: Optional[List[ParamDict]] = None,
        extra_regressors: Optional[List[ParamDict]] = None,
    ) -> None:
        """Initialize a prophet forecaster

        Parameters
        ----------
        growth: str
            'linear', 'logistic' or 'flat' trend.
        changepoints: List[TimeStamp]
            List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
        n_changepoints: int
            Number of potential changepoints to include. Not used if input `changepoints` is supplied.
            If `changepoints` is not supplied, then n_changepoints potential changepoints are selected
            uniformly from the first `changepoint_range` proportion of the history.
        changepoint_range: float
            Proportion of history in which trend changepoints will be estimated. Defaults to 0.8 for the
            first 80%. Not used if `changepoints` is specified.
        yearly_seasonality: Union[str, bool, int]
            Fit yearly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
        weekly_seasonality: Union[str, bool, int]
            Fit weekly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
        daily_seasonality: Union[str, bool, int]
            Fit daily seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
        holidays: Optional[pd.DataFrame], default None
            A dataframe with required columns "holiday" & "ds", and optional columns "lower_window", "upper_window"
            and "prior_scale". lower_window and upper_window specify a range of days around the date to be included
            as holidays. lower_window=-2 will include 2 days prior to the date as holidays. "prior_scale" specifies
            the prior scale for that holiday.
        country_holidays: Optional[str], default None
            If specified, then holidays from this country will automatically be included in the model in addition
            to any custom holidays specified using the `holidays` argument. Only a single country / country-code
            can be specified. A list of available countries can be found here
            https://github.com/dr-prodigy/python-holidays
        seasonality_mode: str
            'additive' (default) or 'multiplicative'.
        seasonality_prior_scale: float
            Parameter modulating the strength of the seasonality model. Larger values allow the model to fit larger
            seasonal fluctuations, smaller values dampen the seasonality. Can be specified for individual seasonalities
            using add_seasonality.
        holidays_prior_scale: float
            Parameter modulating the strength of the holiday components model, unless overridden in the holidays input.
        changepoint_prior_scale: float
            Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
        mcmc_samples: int
            If greater than 0, will do full Bayesian inference with the specified number of MCMC samples. If 0, will do
            MAP estimation.
        prediction_interval_width: float
            Width of the uncertainty intervals provided for the forecast. If mcmc_samples=0, this will be only the
            uncertainty in the trend using the MAP estimate of the extrapolated generative model. If
            mcmc.samples>0, this will be integrated over all model parameters, which will include uncertainty in
            seasonality.
        uncertainty_samples: int
            Number of simulated draws used to estimate uncertainty intervals. Settings this value to 0 or False will
            disable uncertainty estimation and speed up the calculation.
        cap: Optional[float], default None
            A maximum value for saturating forecasts. This value is ignored if growth != "logistic"
        floor: Optional[float], default None
            A minimum value for saturating forecasts. This value is ignored if growth != "logistic"
        custom_seasonlities: Optional[List[ParamDict]], default None
            A list of custom seasonalities with required keys "name", "period" and "fourier_order", and optional
            keys "prior_scale", "mode", and "condition_name"
        extra_regressors: Optional[List[ParamDict]], default None
            A list of dictionary with required key "name" and optional keys "prior_scale", "mode", and "standardize".
        """
        try:
            from fbprophet import Prophet
        except Exception:
            from prophet import Prophet

        # UFF specific parameter checks
        if growth == "logistic" and cap is None:
            raise ValueError("`cap` is required when growth='logistic'")

        for seasonality in custom_seasonalities or []:
            for req in ("name", "period", "fourier_order"):
                if req not in seasonality:
                    raise ValueError(f"Seasonality {seasonality} missing required field {req}")

        for regressor in extra_regressors or []:
            if "name" not in regressor:
                raise ValueError(f"Regressor {regressor} is missing 'name'")

        self.model = Prophet(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=prediction_interval_width,
            uncertainty_samples=uncertainty_samples,
        )

        self._prediction_interval_width = prediction_interval_width

        for seasonality in custom_seasonalities or []:
            self.model.add_seasonality(**seasonality)

        for regressor in extra_regressors or []:
            self.model.add_regressor(**regressor)

        if country_holidays:
            self.model.add_country_holidays(country_holidays)

        # These affect which hyperparameter subspace is in scope
        self.cap = cap
        self.floor = floor
        self.custom_seasonalities = custom_seasonalities
        self.extra_regressors = extra_regressors

        # These are updated during fit
        self.out_columns: Optional[ColumnSet] = None
        self.expects_covariates: bool = False

    @property
    def prediction_interval_width(self) -> float:
        return self._prediction_interval_width

    """
    BatchEstimator methods
    """

    def fit(self, data: TimeIndexedData, covariates: Optional[TimeIndexedData] = None, **kwargs) -> ProphetForecaster:
        """Fit the Prophet model to the input time series.

        Parameters
        ----------
        data: TimeIndexedData
            The time series to be modeled. Expected to have values.ndim == 1.
        covariates: Optional[TimeIndexedData], default None
            If applicable, the extra regressors that should be registered with the Prophet model.
        kwargs: Dict[str, Any]
            Passed directly to Prophet().fit()

        Returns
        -------
        ProphetForecaster
            self
        """
        if covariates is not None:
            for col in covariates.column_names:
                if col not in self.model.extra_regressors:
                    self.model.add_regressor(col)

        df = utils.prophet_style_dataframe(data, covariates)
        df = self._add_cap_and_floor(df)

        self.model.fit(df, **kwargs)
        self.out_columns = data.column_names
        self.expects_covariates = covariates is not None
        return self

    """
    Forecaster methods
    """

    def forecast(
        self,
        data: Union[TimeIndex, TimeIndexedData],
        **kwargs: Dict[str, Any],
    ) -> ProphetOutput:
        """Use the fitted model to forecast future values

        Parameters
        ----------
        data: Union[TimeIndex, TimeIndexedData]
            The data to forecast. If no covariates were used during `fit()` then only the time_index
            of `TimeIndexedData` is used.
        kwargs: Dict[str, Any]
            Passed directly to Prophet.predict()

        Returns
        -------
        ProphetOutput
            The forecast result and underlying model.
        """
        if not self.expects_covariates and isinstance(data, TimeIndexedData):
            data = data.time_index

        df = utils.prophet_future_dataframe(data)
        df = self._add_cap_and_floor(df)
        res = self.model.predict(df, **kwargs)
        return ProphetOutput(
            out=self._unpack_result(data, res["yhat"]),
            upper=self._unpack_result(data, res["yhat_upper"]),
            lower=self._unpack_result(data, res["yhat_lower"]),
            interval_width=self.prediction_interval_width,
            components=self._unpack_components(data, res),
        )

    def _unpack_result(self, domain: Union[TimeIndex, TimeIndexedData], values: Any) -> TimeIndexedData:
        return TimeIndexedData.from_time_index(
            domain if isinstance(domain, TimeIndex) else domain.time_index,
            values,
            column_names=self.out_columns,
        )

    def _unpack_components(
        self, domain: Union[TimeIndex, TimeIndexedData], raw_results: pd.DataFrame
    ) -> TimeIndexedData:
        components = raw_results.drop(["ds", "yhat", "yhat_lower", "yhat_upper"], axis=1)
        return TimeIndexedData.from_time_index(
            domain if isinstance(domain, TimeIndex) else domain.time_index,
            components,
            column_names=list(components.columns),
        )

    def _add_cap_and_floor(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cap is not None:
            df["cap"] = self.cap
        if self.floor is not None:
            df["floor"] = self.floor
        return df
