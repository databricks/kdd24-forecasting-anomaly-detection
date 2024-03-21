from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import huber


class MiniProphet:
    """Implements the key components of the Prophet forecasting procedure

    MiniProphet has the advantage of offering faster training and a 0-dependency simple codebase covering 90% of the
    common prediction cases solved by these librairies. The memory footprint is also reduced significantly (orders of
    magnitude on large datasets), as the implementation minimizes memory copies and does not hold references to the
    training dataframe, which prevents garbage collection. The model has two components:

    * trend = piecewise linear, with changepoints evenly distributed within the first 80% of training data
    * seasonality = Fourier decomposition using weekly and daily frequencies

    The default loss function is Huber, but one can also choose a quantile loss.
    A big emphasis has been put on making the model "parameter free" so it works well without any knob-tweaking.
    """

    def __init__(
        self,
        weekly_order: int = 6,
        daily_order: int = 6,
        n_changepoints: int = 2,
        lambda_reg: float = 0.01,
        quantile: Optional[float] = None,
    ):
        """Initialize the MiniProphet torch model

        Parameters
        ----------
        weekly_order: int, default 6
            Fourier order of the weekly seasonality component.
        daily_order: int, default 6
            Fourier order of the daily seasonality component.
        n_changepoints: int, default 2
            The number of changepoints to be distributed throughout the training data.
        lambda_reg: float, default 0.01
            Amount of regularization for the model parameters.
        quantile: Optional[float], default None
            If specified, the model will perform quantile regression.

        Raises
        ------
        ValueError
            If `quantile` <= 0 or `quantile >= 1`
        """
        self.weekly_order = weekly_order
        self.daily_order = daily_order
        self.n_changepoints = n_changepoints
        self.lambda_reg = lambda_reg
        self.quantile = quantile
        if quantile is None:
            self.loss = lambda p, y: huber(1.0, np.abs(p - y)).mean()
        elif quantile > 0 and quantile < 1:
            self.loss = lambda p, y: np.where(p > y, (p - y) * (1 - quantile), (y - p) * quantile).mean()
        else:
            raise ValueError("Invalid quantile value. Needs to be in ]0, 1[")
        # model parameters are initialized once we've seen the data
        self.Wtrend, self.Wseas_week, self.Wseas_day, self.Wholiday = None, None, None, None
        self.trend_slopes = None
        self.trend_offsets = None
        self.change_points = None

    def __transform_input(self, X: pd.Series):
        x_trend = (X - self.x_first).dt.total_seconds().values / 60
        x_trend = np.array(x_trend / self.tot_minutes, dtype=np.float64)

        x_seas_week = self._fourier_terms(
            X.apply(lambda t: t.dayofweek + (60 * t.hour + t.minute) / (60 * 24)), 7, self.weekly_order
        )
        x_seas_day = self._fourier_terms(X.apply(lambda t: t.hour + t.minute / 60), 24, self.daily_order)

        trend_inds = np.searchsorted(self.change_points, x_trend) - 1

        return x_trend, x_seas_week, x_seas_day, trend_inds

    def __predict_components(self, x_trend, x_seas_week, x_seas_day, trend_inds):
        y_trend = x_trend * self.trend_slopes[trend_inds] - self.trend_offsets[trend_inds] + self.Wtrend[0]
        y_seas_week = x_seas_week[:, :, 0] @ self.Wseas_week[:, 0] + x_seas_week[:, :, 1] @ self.Wseas_week[:, 1]
        y_seas_day = x_seas_day[:, :, 0] @ self.Wseas_day[:, 0] + x_seas_day[:, :, 1] @ self.Wseas_day[:, 1]
        return y_trend, y_seas_week, y_seas_day

    def forward(
        self, X: pd.Series, is_holiday: "npt.NDArray" = None, normalized: bool = True, trend_only: bool = False
    ):
        y_trend, y_seas_week, y_seas_day = self.__predict_components(*self.__transform_input(X))
        y = y_trend
        if not trend_only:
            y += y_seas_week + y_seas_day
        if self.Wholiday is not None and is_holiday is not None:
            y += self.Wholiday * is_holiday
        if not normalized:
            return y * self.yscale
        return y

    @staticmethod
    def _fourier_terms(s: pd.Series, period: int, order: int):
        res = (2 * np.pi / period) * np.array(s.values, dtype=np.float64)
        res = np.arange(1, order + 1)[:, None] * res
        return np.concatenate([np.cos(res), np.sin(res)]).T.reshape(len(s), order, 2)

    def __init_data(
        self, X: pd.Series, Y: pd.Series, is_holiday: "npt.NDArray" = None, seed_point_estimate: MiniProphet = None
    ):
        self.x_first = X.iloc[0]
        self.tot_minutes = (X.iloc[-1] - self.x_first).total_seconds() / 60
        Ytrain = np.array(Y.values, dtype=np.float64)
        self.yscale = np.percentile(np.abs(Ytrain), q=95.0)
        if np.abs(self.yscale) < 1e-6:
            self.yscale = 1.0
        Ytrain /= self.yscale

        if len(X) < 2 * self.n_changepoints:
            self.n_changepoints = 0
        self.Wtrend = np.zeros((self.n_changepoints + 2,), dtype=np.float64)
        self.Wseas_week = np.zeros((self.weekly_order, 2), dtype=np.float64)
        self.Wseas_day = np.zeros((self.daily_order, 2), dtype=np.float64)
        self.change_points = 0.8 * np.arange(self.n_changepoints + 1) / (self.n_changepoints + 1)

        if is_holiday is not None:
            holiday_indices = np.nonzero(is_holiday)
            is_holiday = is_holiday.astype(np.float64)
            self.Wholiday = Ytrain[holiday_indices].mean()

        # heuristic to set the changepoints slopes
        if self.quantile is None:
            self.Wtrend[0] = Ytrain.mean()
        else:
            self.Wtrend[0] = np.percentile(Ytrain, q=self.quantile)

        if seed_point_estimate is not None and self.quantile is not None:
            # init Fourier coeffs from seed values
            self.Wseas_week[:] = seed_point_estimate.Wseas_week[: self.weekly_order]
            self.Wseas_day[:] = seed_point_estimate.Wseas_day[: self.daily_order]

        return Ytrain, is_holiday

    def learn(self, X: pd.Series, Y: pd.Series, isHoliday=None, seed_point_estimate: MiniProphet = None):
        Ytrain, is_holiday = self.__init_data(X, Y, isHoliday, seed_point_estimate)

        # todo: maybe batch + accumulate based on scalability of matmul
        x_trend, x_seas_week, x_seas_day, trend_inds = self.__transform_input(X)
        n = len(X)

        x0 = np.concatenate([self.Wtrend.flatten(), self.Wseas_week.flatten(), self.Wseas_day.flatten()])
        if self.Wholiday is not None:
            x0 = np.append(x0, [self.Wholiday])

        def assign_from_flat(vec):
            offset = len(self.Wtrend)
            self.Wtrend = vec[:offset]
            self.trend_slopes = np.cumsum(self.Wtrend[1:])
            self.trend_offsets = np.cumsum(self.change_points * self.Wtrend[1:])
            self.Wseas_week = vec[offset : offset + self.weekly_order * 2].reshape((self.weekly_order, 2), order="F")
            self.Wseas_day = vec[
                offset + self.weekly_order * 2 : offset + self.weekly_order * 2 + self.daily_order * 2
            ].reshape((self.daily_order, 2), order="F")
            if self.Wholiday is not None:
                self.Wholiday = vec[-1]

        def callback(vec):
            """
            compute loss and gradient from flattened representation:
            trend_intercepts, trend_slopes, weekly_cos_terms, weekly_sin_terms
            daily_cos_terms, daily_sin_terms

            There are `n_changepoints` + 1 intercept and slope terms, `weekly_order` cos and sin terms
            for weekly seasonality `daily_order` cos and sin terms for daily seasonality.

            vec has length 2 * (`n_changepoints` + 1) + (2 * `weekly_order`) + (2 * `daily_order`)
            """
            assign_from_flat(vec)

            y_trend, y_seas_week, y_seas_day = self.__predict_components(x_trend, x_seas_week, x_seas_day, trend_inds)
            pred = y_trend + y_seas_week + y_seas_day
            if self.Wholiday is not None:
                pred += self.Wholiday * is_holiday

            loss = self.loss(pred, Ytrain)

            grad = np.zeros(x0.shape, dtype=np.float64)

            if self.quantile is None:
                g_loss = np.where(np.abs(pred - Ytrain) <= 1.0, pred - Ytrain, np.sign(pred - Ytrain))
            else:
                g_loss = np.where(pred > Ytrain, (1.0 - self.quantile), -self.quantile)

            # Compute d(loss) / d(trend parameters)
            grad[0] = g_loss.sum()
            for j in range(self.n_changepoints + 1):
                inds = np.argwhere(trend_inds >= j).squeeze()
                xm = np.take(x_trend, inds) - self.change_points[j]
                grad[j + 1] = np.dot(np.take(g_loss, inds), xm)

            # d(loss) / d(weekly fourier terms)
            weekly_terms_offset = self.Wtrend.shape[0]
            xw = (g_loss[:, None, None] * x_seas_week).sum(axis=0)
            grad[weekly_terms_offset : weekly_terms_offset + self.weekly_order] = xw[:, 0]
            grad[weekly_terms_offset + self.weekly_order : weekly_terms_offset + 2 * self.weekly_order] = xw[:, 1]

            # d(loss) / d(daily fourier terms)
            daily_terms_offset = weekly_terms_offset + 2 * self.weekly_order
            xd = (g_loss[:, None, None] * x_seas_day).sum(axis=0)
            grad[daily_terms_offset : daily_terms_offset + self.daily_order] = xd[:, 0]
            grad[daily_terms_offset + self.daily_order : daily_terms_offset + 2 * self.daily_order] = xd[:, 1]

            # d(loss) / d(isHoliday)
            if self.Wholiday is not None:
                grad[-1] = is_holiday.sum()

            grad /= n

            # regularization
            lt = self.lambda_reg / (self.n_changepoints + 1)
            w_delta_trend = vec[1 : self.n_changepoints + 2]
            grad[1 : self.n_changepoints + 2] += lt * w_delta_trend
            loss += 0.5 * lt * np.square(w_delta_trend).sum()

            if self.weekly_order > 0:
                lw = self.lambda_reg / self.weekly_order
                w_week = vec[weekly_terms_offset : weekly_terms_offset + self.weekly_order * 2]
                grad[weekly_terms_offset : weekly_terms_offset + self.weekly_order * 2] += lw * w_week
                loss += 0.5 * lw * np.square(w_week).sum()

            if self.daily_order > 0:
                ld = self.lambda_reg / self.daily_order
                w_day = vec[daily_terms_offset : daily_terms_offset + self.daily_order * 2]
                grad[daily_terms_offset : daily_terms_offset + self.daily_order * 2] += ld * w_day
                loss += 0.5 * ld * np.square(w_day).sum()

            return loss, grad

        res = minimize(
            callback,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 1000,  # control worst case runtime in case of pathological adversarial input
                "maxcor": 25,  # this is ok, because we don't have that many parameters
            },
        )
        assign_from_flat(res.x)

    def forecast(self, x: pd.Series, is_holiday: "npt.NDArray" = None, trend_only=False) -> pd.Series:
        return pd.Series(self.forward(x, is_holiday, normalized=False, trend_only=trend_only))
