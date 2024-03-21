from __future__ import annotations

import json
import operator
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from numbers import Number
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset, DateOffset

from .consts import EPOCH_TIME_NO_TZ, ONE_SECOND, ZERO_DELTA

_NonePlaceholder = object()


FloatTensor = "npt.NDArray[np.float64]"
TimeStamp = Union[datetime, date, pd.Timestamp, str, np.datetime64, int]
TimeUnit = Union[timedelta, pd.Timedelta, BaseOffset, str]

ColumnType = Union[str, int, float, None, TimeStamp]
ColumnPath = Union[ColumnType, Tuple[ColumnType]]
ColumnSet = Union[ColumnPath, List[ColumnPath]]


def _week_floor(ts: pd.Timestamp, weekday: int) -> pd.Timestamp:
    ds = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)
    if ds.weekday() != weekday:
        return ds - pd.offsets.Week(weekday=weekday)
    return ds


_quarter_start = [1, 4, 7, 10]
_quarter_end = [3, 6, 9, 12]
_transformers = {
    "M": lambda x: pd.Timestamp(year=x.year, month=x.month, day=1) + pd.offsets.MonthEnd(),
    "MS": lambda x: pd.Timestamp(year=x.year, month=x.month, day=1),
    "Y": lambda x: pd.Timestamp(year=x.year, month=12, day=31),
    "YS": lambda x: pd.Timestamp(year=x.year, month=1, day=1),
    "Q": lambda x: pd.Timestamp(year=x.year, month=_quarter_end[(x.month - 1) // 3], day=1) + pd.offsets.MonthEnd(),
    "QS": lambda x: pd.Timestamp(year=x.year, month=_quarter_start[(x.month - 1) // 3], day=1),
    "W-MON": lambda x: _week_floor(x, 0),
    "W-TUE": lambda x: _week_floor(x, 1),
    "W-WED": lambda x: _week_floor(x, 2),
    "W-THU": lambda x: _week_floor(x, 3),
    "W-FRI": lambda x: _week_floor(x, 4),
    "W-SAT": lambda x: _week_floor(x, 5),
    "W-SUN": lambda x: _week_floor(x, 6),
}
_transformers["A"] = _transformers["Y"]
_transformers["W"] = _transformers["W-SUN"]

_ONE_YEAR = DateOffset(years=1)
_ONE_MONTH = DateOffset(months=1)
_THREE_MONTHS = DateOffset(months=3)
_SIX_MONTHS = DateOffset(months=6)
_ONE_WEEK = DateOffset(days=7)
_ONE_DAY = DateOffset(days=1)


_SPECIAL_FREQUENCIES = {
    # Spark DATE_TRUNC() input
    "year": _ONE_YEAR,
    "month": _ONE_MONTH,
    "quarter": _THREE_MONTHS,
    "week": _ONE_WEEK,
    "day": _ONE_DAY,
    "hour": pd.Timedelta(hours=1),
    "minute": pd.Timedelta(minutes=1),
    "second": pd.Timedelta(seconds=1),
    # Custom frequencies
    "calendar_week": _ONE_WEEK,
    "calendar week": _ONE_WEEK,
    "calendar_month": _ONE_MONTH,
    "calendar month": _ONE_MONTH,
    "year_month": _ONE_MONTH,
    "calendar_quarter": _THREE_MONTHS,
    "calendar quarter": _THREE_MONTHS,
    "calendar_year": _ONE_YEAR,
    "calendar year": _ONE_YEAR,
    "fiscal_month": _ONE_MONTH,
    "fiscal month": _ONE_MONTH,
    "fiscal_quarter": _THREE_MONTHS,
    "fiscal quarter": _THREE_MONTHS,
    "fy_quarter": _THREE_MONTHS,
    "fiscal_year": _ONE_YEAR,
    "fiscal year": _ONE_YEAR,
    "fy_year": _ONE_YEAR,
    "fiscal_half_year": _SIX_MONTHS,
    "fiscal half year": _SIX_MONTHS,
}


def _get_custom_granularity(key: str) -> Union[DateOffset, pd.Timedelta, None]:
    """Return a mathematical granularity from a custom alias, or `None` if not found.

    Parameters
    ----------
    key: str
        The identifier of the custom granularity

    Returns
    -------
    Union[DateOffset, pd.Timedelta, None]
        The granularity, if found. Otherwise None.
    """
    key = key.lower()
    if key in _SPECIAL_FREQUENCIES:
        return _SPECIAL_FREQUENCIES[key]

    unit_regex = "|".join(_SPECIAL_FREQUENCIES)
    int_unit_pattern = f"^(\\d+) ({unit_regex})s?$"
    match = re.match(int_unit_pattern, key)
    if match is not None:
        return int(match.group(1)) * _SPECIAL_FREQUENCIES[match.group(2)]

    return None


def _time_range(
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    periods: Optional[int] = None,
    freq: Optional[TimeUnit] = None,
    **kwargs: Any,
) -> List[pd.Timestamp]:
    """A thin wrapper for pd.date_range that strips `.freq` from the result and returns a list

    Parameters
    ----------
    start: Optional[pd.Timestamp]
        The start of the time range
    end: Optional[pd.Timestamp]
        The end of the time range
    periods: Optional[int]
        The number of periods to generate. This can be used if either start or end are None
    freq: Optional[TimeUnit]
        The frequency of the output time range
    kwargs: Any
        Passed directly to pd.date_range

    Returns
    -------
    List[pd.Timestamp]
        The pd.date_range result as a list, with `.freq` attribute removed from the timestamps.
    """
    if isinstance(freq, str):
        freq = _get_custom_granularity(freq) or freq
    dr = pd.date_range(start=start, end=end, periods=periods, freq=freq, **kwargs)
    dr.freq = None
    return list(dr)


def _round(ts: pd.Timestamp, freq: TimeUnit) -> pd.Timestamp:
    """Round a time stamp to the target frequency

    This operation will .floor() the timestamp unless `freq` is a supported "end" frequency (e.g. month end). If an
    "end" frequency is provided, the timestamp will be rounded up.

    Parameters
    ----------
    ts: pd.Timestamp
        The input timestamp
    freq: TimeUnit
        The target frequency.

    Returns
    -------
    pd.Timestamp
        The new timestamp. This will be computed with ts.floor(freq) unless frequency is one of the supported special
        cases (year, quarter, month, and week start/end)
    """
    if freq in _transformers:
        return _transformers[freq](ts)

    return ts.floor(freq)


def _time_unit_to_dict(granularity: TimeUnit) -> Dict:
    """Return a JSON-compatible dictionary representing this granularity"""
    if type(granularity) == DateOffset:
        return {
            "type": "date_offset",
            "n": granularity.n,
            "normalize": granularity.normalize,
            "kwds": granularity.kwds,
        }
    elif isinstance(granularity, BaseOffset):
        try:
            if granularity == to_offset(granularity.freqstr):
                return {
                    "type": "offset",
                    "value": granularity.freqstr,
                }
        except ValueError:
            pass

        raise ValueError(f"No JSON serialization available for {granularity}")
    elif isinstance(granularity, (pd.Timedelta, timedelta)):
        return {
            "type": "delta",
            "value": str(granularity),
        }
    elif isinstance(granularity, str):
        return {"type": "string", "value": granularity}

    raise TypeError(f"Unexpected granularity type {type(granularity)}")


def _time_unit_from_dict(data: Dict) -> TimeUnit:
    """Return a TimeUnit from a JSON-compatible dictionary

    The dictionary is expected to match the output from _time_unit_to_dict
    """
    if data["type"] == "offset":
        return to_offset(data["value"])
    elif data["type"] == "date_offset":
        return DateOffset(data["n"], data["normalize"], **data["kwds"])
    elif data["type"] == "delta":
        return pd.Timedelta(data["value"])
    elif data["type"] == "string":
        return data["value"]

    raise ValueError(f"Unexpected dictionary type: {data}")


def _infer_granularity(
    ts_values: List[pd.Timestamp],
    strict: bool = True,
    sort_list: bool = True,
) -> TimeUnit:
    """Infer granularity given a list of timestamps. Arbitrary length (nanosecond - inf) granularities are supported.

    Parameters
    ----------
    ts_values: List[pd.Timestamp]
        A list of time stamps
    strict: bool, default True
        If true, the result will ensure that every point in `ts_values` falls on an some
        ts_values[0] + (n * granularity) where n is an integer. Otherwise, the granularity will be computed using the
        most recent observations
    sort_list: bool, default True
        If true, the function will sort ts_values; otherwise ts_values is assumed to be pre-sorted. Set this flag to
        false to speed up inference on large pre-sorted lists

    Returns
    -------
    TimeUnit
        An inferred granularity from the timestamp values.
    """
    if len(ts_values) == 0:
        return ONE_SECOND

    if len(ts_values) == 1:
        if ts_values[0].second > 0:
            return ONE_SECOND
        elif ts_values[0].minute > 0:
            return 60 * ONE_SECOND
        elif ts_values[0].hour > 0:
            return 3600 * ONE_SECOND
        else:
            return DateOffset(days=1)

    if sort_list:
        ts_values = sorted(ts_values)

    if not strict:
        most_recent = ts_values[-10:]
        votes = {}
        for i in reversed(range(len(most_recent) - 1)):
            granularity = _infer_granularity(most_recent[i : i + 2], strict=True, sort_list=False)
            votes[granularity] = votes.get(granularity, 0) + 1
        return sorted(votes, key=lambda k: -votes[k])[0]

    # Strict
    diffs = np.diff(ts_values)
    min_diff = np.min(diffs)

    if min_diff >= pd.Timedelta(hours=23):
        if min_diff >= pd.Timedelta(days=27):
            offsets_to_try = [
                (
                    pd.offsets.MonthEnd()
                    if all((t + DateOffset(days=1)).month != t.month for t in ts_values)
                    else DateOffset(months=1)
                ),
                DateOffset(days=1),
            ]
        else:
            offsets_to_try = [DateOffset(days=1)]

        for offset in offsets_to_try:
            t_idx = {t: i for i, t in enumerate(_time_range(ts_values[0], ts_values[-1], freq=offset))}
            relative_ts_values = []
            for t in ts_values:
                if t not in t_idx:
                    break
                relative_ts_values.append(t_idx[t])
            else:
                return np.gcd.reduce(np.diff(relative_ts_values)) * offset

    # If no relative frequency can be inferred, use the greatest common divisor of the observed differences
    return np.gcd.reduce(diffs)


@dataclass
class TimeIndexedOutput:
    """A base class containint the result of a forecast/transform operation.

    This is meant to be sub-classed by individual models which may want to include additional
    data in addition to the TimeIndexedData output. Because this is a dataclass, the first field
    will always be the `out` attribute representing the output from a previous step.
    """

    out: TimeIndexedData


@dataclass
class TimeIndexedOutputWithInterval(TimeIndexedOutput):
    """The result of a forecast/transform operation with error bars.

    This is meant to be sub-classed by individual estimators.
    """

    out: TimeIndexedData
    upper: TimeIndexedData
    lower: TimeIndexedData
    interval_width: float


class TimeIndexedData:
    """The main abstraction for time series data.

    Attributes
    ----------
    time_index: TimeIndex
        Represents the discrete points in time that correspond to the entries in `values`. A TimeIndex has an
        unambiguous time scale provided by `granularity`, and a parallel time scale expressed as integers (most
        commonly, unixtime) provided by `unixtime_t0` and `unixtime_unit`.
    values: npt.NDArray[np.float64]
        An NDArray of floats representing the time series. The second dimension of `values` is considered to be the
        column dimension and can be referenced by name with subscripting. If `values.ndim < 2` then it is treated as
        if it has 1 column. 3 and higher dimensions should be used sparingly e.g. when constructing sliding windows
        over multivariate data. Hierarchical data should typically be expressed as 2D data with hierarchical columns.
    column_names: Union[List[ColumnType], List[Tuple[ColumnType]]]
        A list of column names that can be used to access values. The 2nd dimension of values is considered to be the
        "column" dimension. Tuples of strings are treated like a multi-index and interpreted to represent grouped /
        hierarchical data.
    """

    __array_priority__ = 10000

    def __init__(
        self,
        time_array: Union[TimeStamp, Collection[TimeStamp]],
        values: "npt.ArrayLike",
        column_names: Optional[ColumnSet] = None,
        granularity: Optional[TimeUnit] = None,
        unixtime_t0: pd.Timestamp = EPOCH_TIME_NO_TZ,
        unixtime_unit: pd.Timedelta = ONE_SECOND,
    ) -> None:
        """Initialize a new TimeIndexedData object.

        Parameters
        ----------
        time_array: Union[TimeStamp, Collection[TimeStamp]]
            An interable containing one of the accepted TimeStamp types or integers. This is used to initialize the
            TimeIndex.
        values: npt.ArrayLike
            This is unpacked into an npt.NDArray[np.float64] and stored in the `values` attribute.
        column_names: Optional[ColumnSet], default None
            Optional column names. If not provided, "y1", "y2", ... "yn" are used to match the
            number of columns in `values`. The length of column_names must match the size of the second dimension of
            `values`. Tuples of strings are treated like a multi-index and interpreted to represent grouped/hierarchical
            data.
        granularity: Optional[TimeUnit]
            One of the accepted time units, used to initialize the TimeIndex with a frequency to
            generate future values.
        unixtime_t0: pd.Timestamp, default pd.Timestamp('1970-01-01 00:00:00')
            Time zero; used by the TimeIndex to provide unambiguous conversions between integers
            and timestamps.
        unixtime_unit: pd.Timedelta, default pd.Timedelta(seconds=1)
            The unit used by the TimeIndex to provide unambiguous conversions between integers and
            timestamps.

        Raises
        ------
        ValueError
            If time_array and values have different lengths
        ValueError
            If an unexpected number of column names are provided
        ValueError
            If the provided column names are not unique

        Examples
        --------
        >>> data = TimeIndexedData([0, 1], [4, 4])
        >>> data.int_time_index()
        [0, 1]
        >>> data.pd_timestamp_index()
        [Timestamp('1970-01-01 00:00:00', freq='S'), Timestamp('1970-01-01 00:00:01', freq='S')]
        >>> data.values
        array([4., 4.])

        Specify a time series granularity

        >>> t = ["2023-01-01", "2023-01-02"]
        >>> data = TimeIndexedData(t, [4, 4], granularity=pd.Timedelta(hours=12))  # Implies one missing value
        >>> data.pd_timestamp_index()
        [Timestamp('2023-01-01 00:00:00', freq='12H'), Timestamp('2023-01-02 00:00:00', freq='12H')]

        Granularity will be determine using the greatest common divisor of the timestamp differences.

        >>> t = ["2023-01-01", "2023-01-02", "2023-01-04"]
        >>> data = TimeIndexedData(t, [4, 4])
        >>> data.pd_timestamp_index()
        >>> data.granularity
        Timedelta('1 days 00:00:00')
        """
        time_index = TimeIndex(
            time_array,
            granularity,
            unixtime_t0,
            unixtime_unit,
        )
        values = np.atleast_1d(np.array(values, dtype=np.float64))

        if len(time_index) != len(values):
            raise ValueError("Time and value arrays must have equal lengths")

        exp_names = 1 if values.ndim < 2 else values.shape[1]
        if column_names is None:
            column_names = [f"y{i}" for i in range(exp_names)]

        self.time_index = time_index
        self.values = values
        self.set_column_names(column_names)

    def __array__(self, dtype=None) -> "npt.NDArray":
        return self.values.astype(dtype) if dtype else self.values

    def __len__(self):
        return len(self.time_index)

    def __eq__(self, other: TimeIndexedData) -> bool:
        return (
            type(self) is type(other)
            and self.time_index == other.time_index
            and np.array_equal(self.values, other.values)
            and self.column_names == other.column_names
        )

    def __repr__(self) -> str:
        if len(self) == 0:
            return "TimeIndexedData()"

        index = list(zip(self.int_time_index(), self.pd_timestamp_index()))
        values = self.values_at_least_2d
        df = pd.DataFrame(
            [[values[i][j] for j in range(values.shape[1])] for i in range(values.shape[0])],
            columns=self.column_names,
            index=index,
        )
        with pd.option_context("display.multi_sparse", False):
            return "\n" + str(df)

    def __getitem__(self, columns: ColumnSet) -> Union[FloatTensor, TimeIndexedData]:
        """Enable column selection with the subscript operator []. Internally this calls `TimeIndexedData.select()`

        If key references a column with associated numerical data (i.e. not a prefix of another column) then the data
        will be returned as a FloatTensor. Otherwise key will be treated as a prefix and a new TimeIndexedData object
        will be returned that contains all columns matching the prefix. The intention is to provide a pandas-like API
        where single columns return Series objects and multi-column selections return Dataframe slices.

        Parameters
        ----------
        columns: ColumnSet
            A set of columns. Tuples are be used to designate grouped data

        Returns
        -------
        Union[FloatTensor, TimeIndexedData]
            If `columns` is a scalar or a tuple, then a numpy array of the associated numerical data will be
            returned. If a list of columns is provided, then a a new TimeIndexedData object with the selected columns
            will be returned even if the list has len() == 1.
        """
        selection = self.select(columns)
        if isinstance(columns, list):
            return selection

        return selection.values.squeeze(axis=1) if self.ndim <= 2 else selection.values

    def __contains__(self, key: Union[ColumnType, Tuple[ColumnType]]) -> bool:
        """Support `column in data`

        Parameters
        ----------
        key: Union[ColumnType, Tuple[ColumnType]]
            A column name, or prefix if using hierarchical columns

        Returns
        -------
        bool
            True if key is a column or the prefix of an existing column.
        """
        tup_key = key if isinstance(key, tuple) else (key,)
        return tup_key in self._flat_index

    @staticmethod
    def _get_column_tuples(columns: ColumnSet) -> List[Tuple[ColumnType]]:
        """Refine user-provided types

        Parameters
        ----------
        columns: ColumnSet
            User-provided column names / tuples.

        Returns
        -------
        List[Tuple[ColumnType]]
            A list with length >= 1 containing tuple keys
        """
        if not isinstance(columns, list):
            columns = [columns]

        return [(c,) if not isinstance(c, tuple) else c for c in columns]

    def copy(self) -> TimeIndexedData:
        """Create a copy of this object

        Returns
        -------
        TimeIndexedData
            A copy of `self`
        """
        return type(self).from_time_index(
            deepcopy(self.time_index),
            np.copy(self.values),
            column_names=deepcopy(self._column_tuples),
        )

    @property
    def column_names(self) -> Union[List[ColumnType], List[Tuple[ColumnType]]]:
        return [c[0] for c in self._column_tuples]

    @property
    def column_tuples(self) -> List[Tuple[ColumnType]]:
        return deepcopy(self._column_tuples)

    @property
    def unixtime_t0(self):
        """Provides easier access to the TimeIndex field"""
        return self.time_index.unixtime_t0

    @property
    def unixtime_unit(self):
        """Provides easier access to the TimeIndex field"""
        return self.time_index.unixtime_unit

    @property
    def granularity(self):
        """Provides easier access to the TimeIndex field"""
        return self.time_index.granularity

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def values_at_least_2d(self) -> FloatTensor:
        return self.values.reshape(-1, 1) if self.values.ndim < 2 else self.values

    def reshape(self, new_shape: Tuple[int], column_names: Optional[ColumnSet] = None) -> TimeIndexedData:
        if column_names is None:
            column_names = self._column_tuples
        return type(self).from_time_index(
            self.time_index,
            self.values.reshape(new_shape),
            column_names=column_names,
        )

    def next_timestamp(self) -> pd.Timestamp:
        return self.time_index.next_timestamp()

    def first_timestamp(self) -> pd.Timestamp:
        return self.time_index.first_timestamp()

    def last_timestamp(self) -> pd.Timestamp:
        return self.time_index.last_timestamp()

    def contains_nan(self) -> bool:
        return np.isnan(self.values).any()

    def add_constant_group(self, name: ColumnType) -> TimeIndexedData:
        return self.set_column_names([(name,) + c for c in self._column_tuples])

    def set_column_names(
        self,
        columns: ColumnSet,
        in_place: bool = True,
    ) -> TimeIndexedData:
        """Performs an in-place update of the column names

        Parameters
        ----------
        columns: ColumnSet
            The new set of column names.
        in_place: bool, default True
            If True, column names will be updated in-place.

        Returns
        -------
        TimeIndexedData
            self, for method chaining
        """
        columns = self._get_column_tuples(columns)
        c_lens = [len(c) for c in columns]

        exp_names = 1 if self.values.ndim < 2 else self.values.shape[1]
        if len(columns) != exp_names:
            raise ValueError("Unexpected number of column names provided")
        if len(columns) != len(set(columns)):
            raise ValueError("Column names must be unique")
        if len(set(c_lens)) > 1:
            raise ValueError("Column tuples must have the same length.")

        if not in_place:
            return type(self).from_time_index(
                self.time_index,
                self.values,
                column_names=columns,
            )

        self._column_tuples = columns
        self._flat_index: Dict[Tuple[ColumnType], int] = {}
        self._group_index: Tuple[Dict[ColumnType, Set[int]]] = tuple({} for _ in range(c_lens[0]))

        for i, tup in enumerate(columns):
            self._flat_index[tup] = i
            for level, grp in enumerate(tup):
                if grp not in self._group_index[level]:
                    self._group_index[level][grp] = set()
                self._group_index[level][grp].add(i)

        return self

    def select(self, columns: ColumnSet) -> TimeIndexedData:
        """Select specific columns from TimeIndexedData

        Parameters
        ----------
        columns: ColumnSet
            A set of columns. Tuples are be used to designate grouped data

        Returns
        -------
        TimeIndexedData
            The matching columns of `self`
        """
        columns = self._get_column_tuples(columns)
        return type(self).from_time_index(
            self.time_index,
            self.values_at_least_2d[:, [self._flat_index[c] for c in columns], ...],
            column_names=columns,
        )

    def dropna(self, *, in_place: bool = False) -> TimeIndexedData:
        """Drops all rows containing one or more NaN values

        Parameters
        ----------
        in_place: bool, default False
            If True, overwrite self.time_index and self.values after dropping NaNs. Otherwise
            return a new object.

        Returns
        -------
        TimeIndexedData
            self if in_place=True, otherwise the new object
        """
        is_nan = np.isnan(self.values)
        if is_nan.ndim > 1:
            high_dims = tuple(range(1, is_nan.ndim))
            is_nan = is_nan.max(axis=high_dims)

        time_array = np.array(self.time_index.timestamp_values)[~is_nan]
        values = self.values[~is_nan]
        if in_place:
            self.time_index = TimeIndex(
                time_array,
                self.granularity,
                self.unixtime_t0,
                self.unixtime_unit,
            )
            self.values = values
            return self
        else:
            return TimeIndexedData(
                time_array=time_array,
                values=values,
                column_names=self._column_tuples,
                granularity=self.granularity,
                unixtime_t0=self.unixtime_t0,
                unixtime_unit=self.unixtime_unit,
            )

    def starts_before(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Checks if this object starts before another

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the first timestamp value in the current time index is less than
            the first timestamp value in the other time index.
        """
        return self.time_index.starts_before(other)

    def strictly_before(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Checks if this object ends before `other` begins

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the last timestamp value in the current time index is less than
            the first timestamp value in the other time index.
        """
        return self.time_index.strictly_before(other)

    def same_scale(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Checks self.time_index.same_scale(other)

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the TimeIndex objects share according to the logic in TimeIndex
        """
        return self.time_index.same_scale(other)

    def same_shape(self, other: TimeIndexedData) -> bool:
        """Checks if `other` has the same time axis, columns and shape as `self`

        Parameters
        ----------
        other: TimeIndexedData
            An object to compare

        Returns
        -------
        bool
            True if `other` has the same time axis, columns and shape as `self`
        """
        if isinstance(other, TimeIndexedData):
            return (
                self.time_index == other.time_index
                and self.shape == other.shape
                and self._column_tuples == other._column_tuples
            )
        return NotImplemented

    def time_slice(
        self,
        start: Union[int, pd.Timestamp],
        end: Optional[Union[int, pd.Timestamp]] = None,
        copy_data: bool = True,
    ) -> TimeIndexedData:
        """Return the entries in this TimeIndexedData matching start <= t < end

        If end is not provided, then the returned time series will have start <= t

        Parameters
        ----------
        start: Union[int, pd.Timestamp]
            The start of the interval (inclusive). If this entry is an integer then the unixtime
            scale will be used.
        end: Optional[Union[int, pd.Timestamp]], default None
            The end of the interval (exclusive). If this entry is an integer then the unixtime scale
            will be used.
        copy_data: bool, default True
            If true, deepcopy() is used to create copies of the underlying data for the new instance.
            This is to reduce the possibility of unintended side effects if the slice is manipulated
            later. If memory is a concern, copy_data can be set to False.

        Returns
        -------
        TimeIndexedData
            The data from this object matching the time interval
        """
        start_vals = self.pd_timestamp_index() if isinstance(start, pd.Timestamp) else self.int_time_index()
        start = np.searchsorted(start_vals, start)

        if end is not None:
            end_vals = self.pd_timestamp_index() if isinstance(end, pd.Timestamp) else self.int_time_index()
            end = np.searchsorted(end_vals, end)

        return self.slice(start, end, copy_data)

    def slice(self, start: int, end: Optional[int] = None, copy_data: bool = True) -> TimeIndexedData:
        """Create a new TimeIndexedData object from a slice of the current one.

        Parameters
        ----------
        start: int
            The start index of the slice
        end: Optional[int], default None
            The end index of the slice. If not provided, the slice will continue through the length of
            the TimeIndexedData.
        copy_data: bool, default True
            If true, deepcopy() is used to create copies of the underlying data for the new instance.
            This is to reduce the possibility of unintended side effects if the slice is manipulated
            later. If memory is a concern, copy_data can be set to False.

        Returns
        -------
        TimeIndexedData
            A new TimeIndexedData instsance
        """
        if end is None:
            end = len(self)
        opt_copy = deepcopy if copy_data else lambda x: x
        return type(self)(
            opt_copy(self.time_index.timestamp_values[start:end]),
            opt_copy(self.values[start:end]),
            opt_copy(self._column_tuples),
            opt_copy(self.granularity),
            opt_copy(self.unixtime_t0),
            opt_copy(self.unixtime_unit),
        )

    def mask(self, idx: "npt.NDArray") -> TimeIndexedData:
        """Apply a 1D mask to the TimeIndexedData

        Parameters
        ----------
        idx: npt.NDArray
            A mask array specifying which values should be kept. Must be squeezable to a 1D array with the same length
            as `self`

        Returns
        -------
        TimeIndexedData
            A new TimeIndexedData object with the filtered items from `self`

        Raises
        ------
        ValueError
            If `idx` is not squeezable to a 1D array with the same length as `self`
        """
        idx = np.atleast_1d(np.squeeze(idx))
        if idx.ndim > 1:
            raise ValueError("Mask must be squeezable to a 1D array")
        time_array = np.array(self.pd_timestamp_index())[idx]
        values = self.values[idx]
        return TimeIndexedData(
            time_array,
            values,
            column_names=self._column_tuples,
            granularity=self.granularity,
            unixtime_t0=self.unixtime_t0,
            unixtime_unit=self.unixtime_unit,
        )

    def int_time_index(self) -> List[int]:
        """Return a copy of the TimeIndex unixtime values.

        Returns
        -------
        List[int]
        """
        return self.time_index.int_time_index()

    def pd_timestamp_index(self) -> List[pd.Timestamp]:
        """Return a copy of the TimeIndex values as pandas timestamp objects.

        Returns
        -------
        List[pd.Timestamp]
        """
        return self.time_index.pd_timestamp_index()

    def future_time_index(self, n_steps: int, n_steps_past: int = 0) -> TimeIndex:
        """Call TimeIndex.future() and return the result.

        Parameters
        ----------
        n_steps: int
            The number of steps in the future to generate
        n_steps_past: int, default: 0
            The number of steps into the past to return. Steps in the past are empirical,
            so if the TimeIndexedData has some missing data according to `granularity`
            then the past data will reflect that missing data.

        Returns
        -------
        TimeIndex
            A new TimeIndex instance
        """
        return self.time_index.future(n_steps, n_steps_past)

    def make_dense(self, fill_value: "npt.ArrayLike" = np.nan, in_place: bool = False) -> TimeIndexedData:
        """Create a dense time series based on TimeIndexedData granularity.

        New rows in `values` will be created with the same shape as the current values, filled with
        `fill_value`. Within the bounds of data.time_index there will be no missing time stamps.
        If data.time_index is empty then this function is a no-op

        Parameters
        ----------
        fill_value: npt.ArrayLike, default np.nan
            The value that will be used to fill the entries in missing time stamps
        in_place: bool, default False
            If True, the result will overwrite the current values.

        Returns
        -------
        TimeIndexedData
            The result, or self for method chaining if `in_place=True`
        """
        from . import utils

        res = utils.make_dense(self, fill_value)
        if in_place:
            self.time_index = res.time_index
            self.values = res.values
            return self
        return res

    def fill_values(self, fill_value: float, in_place: bool = False) -> TimeIndexedData:
        """Fill any `np.nan` in `self.values` with `fill_value`.

        Parameters
        ----------
        fill_value: float
            The value that will be used to replace any np.nan
        in_place: bool, default False
            If True, the result will overwrite the current values.

        Returns
        -------
        TimeIndexedData
            The result, or self for method chaining if `in_place=True`
        """
        from . import utils

        res = utils.fill_values(self, fill_value)
        if in_place:
            self.values = res.values
            return self
        return res

    @classmethod
    def from_time_index(
        cls,
        index: TimeIndex,
        values: "npt.ArrayLike",
        column_names: Optional[ColumnSet] = None,
    ) -> TimeIndexedData:
        """Use a TimeIndex to create a new TimeIndexedData object

        Parameters
        ----------
        index: TimeIndex
            A valid TimeIndex object.
        values: npt.ArrayLike
            Will be cast to npt.NDArray[np.float64] and addressable in the output as .values
        column_names: Optional[Union[str, List[str]]] = None
            Optional column names which will be applied to the new TimeIndexedData

        Returns
        -------
        TimeIndexedData
            A new TimeIndexedData object
        """
        return cls(
            index.timestamp_values,
            values,
            column_names=column_names,
            granularity=index.granularity,
            unixtime_t0=index.unixtime_t0,
            unixtime_unit=index.unixtime_unit,
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        value_col: Union[ColumnType, List[ColumnType]],
        time_col: Optional[ColumnType] = None,
        group_by: Optional[Union[ColumnType, List[ColumnType]]] = None,
        granularity: Optional[TimeUnit] = None,
        unixtime_t0: pd.Timestamp = EPOCH_TIME_NO_TZ,
        unixtime_unit: pd.Timedelta = ONE_SECOND,
    ) -> TimeIndexedData:
        """Create a new TimeIndexedData instance from a pandas dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe
        value_col: Union[ColumnType, List[ColumnType]]
            The column(s) of the dataframe to interpret as `values` in the TimeIndexedData. These
            should be castable to np.float64
        time_col: Optional[ColumnType], default None
            The column of the dataframe containing the time variable. If not provided, then the
            index of the dataframe is used. The entries of `time_col` should be one of the
            acceptable `TimeStamp` types or integers.
        group_by: Optional[Union[ColumnType, List[ColumnType]]], default None
            Column(s) to group by. This will result in a "wide format" `TimeIndexedData` object with one row per
            timestamp and one column per group. Group by columns are treated as ordered and will be used to define
            a hierarchy if the resulting object is used for reconciliation. In the resulting hierarchy, the provided
            `value_col` will be the "top" followed by the groups.
        granularity: Optional[TimeUnit], default None
            The granularity of the time series. If not provided, TimeIndexedData will attempt to
            determine a reasonable granularity value given the input.
        unixtime_t0: pd.Timestamp, default pd.Timestamp('1970-01-01 00:00:00')
            Time zero; used by the TimeIndex to provide unambiguous conversions between integers
            and timestamps.
        unixtime_unit: pd.Timedelta, default pd.Timedelta(seconds=1)
            The unit used by the TimeIndex to provide unambiguous conversions between integers and
            timestamps.

        Returns
        -------
        TimeIndexedData
            A new TimeIndexedData instance

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"time": [1, 2, 3, 4, 5], "x1": [1, 0, 1, 0, 1], "x2": [4, 5, 6, 7, 8]})
        >>> data = TimeIndexedData.from_pandas(df, value_col=["x1", "x2"], time_col="time")
        >>> data.int_time_index()
        [1, 2, 3, 4, 5]
        >>> data.column_names
        ['x1', 'x2']
        >>> data.values
        array([[1., 4.],
               [0., 5.],
               [1., 6.],
               [0., 7.],
               [1., 8.]])

        The index is used if no time_col is provided

        >>> df = pd.DataFrame(
        ...     data={"x1": [1, 0, 1, 0, 1], "x2": [4, 5, 6, 7, 8]},
        ...     index=[3, 5, 6, 8, 9],
        ... )
        >>> data = TimeIndexedData.from_pandas(df, value_col=["x1", "x2"])
        >>> data.int_time_index()
        [3, 5, 6, 8, 9]
        """
        if not isinstance(value_col, list):
            value_col = [value_col]

        is1d = len(value_col) == 1

        if group_by is not None and not isinstance(group_by, list):
            group_by = [group_by]
        if time_col is not None:
            df = df.set_index(time_col)

        ordered_df = df.sort_index()  # Creates a copy

        for grp in group_by or []:
            if ordered_df[grp].dtype == "O":
                ordered_df[grp].fillna(_NonePlaceholder, inplace=True)

        if group_by:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                wide_df = pd.DataFrame(
                    {
                        cls._hierarchy_concat(c, grp): cls._check_unique_index(df[c])
                        for grp, df in ordered_df.groupby(group_by, dropna=False)
                        for c in value_col
                    }
                )
            value_col = wide_df.columns.to_list()
            is1d = False
        else:
            wide_df = ordered_df[value_col]

        return cls(
            [t for t in wide_df.index],
            values=wide_df[value_col[0] if is1d else value_col].to_numpy(dtype=np.float64),
            column_names=cls._resolve_none_placeholder(value_col),
            granularity=granularity,
            unixtime_t0=unixtime_t0,
            unixtime_unit=unixtime_unit,
        )

    @staticmethod
    def _resolve_none_placeholder(columns: List[ColumnPath]) -> List[ColumnPath]:
        res = []
        for path in columns:
            if path is _NonePlaceholder:
                res.append(None)
            elif isinstance(path, tuple):
                res.append(tuple(None if col is _NonePlaceholder else col for col in path))
            else:
                res.append(path)
        return res

    @staticmethod
    def _check_unique_index(values: pd.Series) -> pd.Series:
        if not values.index.is_unique:
            raise ValueError("Timestamp values must be unique")
        return values

    @staticmethod
    def _hierarchy_concat(*args: Union[ColumnType, Tuple[ColumnType]]) -> Tuple[ColumnType]:
        res = tuple()
        for arg in args:
            res += arg if isinstance(arg, tuple) else (arg,)
        return res

    def to_pandas(self, time_col: Optional[ColumnType] = "t", time_as_pd_timestamps: bool = False) -> pd.DataFrame:
        """Export TimeIndexedData as a pandas dataframe

        Parameters
        ----------
        time_col: Optional[ColumnType], default "t"
            The column name to give to the TimeIndex column. Must not match any of TimeIndexedData.column_tuples
            (strings will be compared against tuples of length 1 if hierarchical columns exist). If `None`, then the
            time information will be stored in the Dataframe index.
        time_as_pd_timestamps: bool, default False
            If True, export TimeIndex as a series of pd.Timestamp values. Otherwise, TimeIndex is exported as a series
            of integers unixtimes.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe

        Examples
        --------
        >>> data = TimeIndexedData([1, 2, 3], [4, 4, 5])
        >>> data.to_pandas()
           t   y0
        0  1  4.0
        1  2  4.0
        2  3  5.0
        """
        data = {c: list(self[ct]) for c, ct in zip(self.column_names, self._column_tuples)}
        t_index = self.pd_timestamp_index() if time_as_pd_timestamps else self.int_time_index()

        if time_col is not None:
            if time_col in self:
                raise ValueError("time_col overlaps with an existing column name")
            data[time_col] = t_index
            return pd.DataFrame(data)

        return pd.DataFrame(data, index=t_index)

    def to_json(self) -> Dict:
        """Serialize to a JSON compatible dictionary

        Returns
        -------
        Dict
        """
        return {
            "time_array": [str(t) for t in self.pd_timestamp_index()],
            "values": self.values.tolist(),
            "column_names": self._column_tuples,
            "unixtime_t0": str(self.unixtime_t0),
            "unixtime_unit": str(self.unixtime_unit),
            "granularity": _time_unit_to_dict(self.granularity),
        }

    @classmethod
    def from_json(cls, data: Union[str, dict]) -> TimeIndexedData:
        """Construct a new time series from a JSON string.

        The JSON string is expected to have an encoding compatible with the one specified by TimeIndexedData.to_json()

        Parameters
        ----------
        data: Union[str, dict]
            A JSON string or dictionary

        Returns
        -------
        TimeIndexedData
        """
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            [pd.Timestamp(t) for t in data["time_array"]],
            np.array(data["values"], dtype=np.float64),
            column_names=[tuple(ct) for ct in data["column_names"]],
            granularity=_time_unit_from_dict(data["granularity"]),
            unixtime_t0=pd.Timestamp(data["unixtime_t0"]),
            unixtime_unit=pd.Timedelta(data["unixtime_unit"]),
        )

    def missing_time_stamps(self) -> List[pd.Timestamp]:
        """Return a list of missing timestamps given granularity

        Returns
        -------
        List[pd.Timestamp]
            The missing timestamps
        """
        return self.time_index.missing_time_stamps()

    """
    Mathematical operations on TimeIndexedData.values
    """

    def _self_with_new_values(self, new_values: FloatTensor) -> TimeIndexedData:
        col_names = self._column_tuples if new_values.shape == self.values.shape else None
        return TimeIndexedData(
            self.pd_timestamp_index(),
            new_values,
            column_names=col_names,
            granularity=self.granularity,
            unixtime_t0=self.unixtime_t0,
            unixtime_unit=self.unixtime_unit,
        )

    def _values_math_override(
        self,
        other: Union[TimeIndexedData, "npt.NDArray"],
        op: Callable[["npt.NDArray", "npt.NDArray"], "npt.NDArray"],
        reverse_args: bool = False,
    ) -> TimeIndexedData:
        if isinstance(other, TimeIndexedData):
            if self.time_index != other.time_index:
                raise ValueError("Cannot operate on TimeIndexedData objects with different TimeIndex")
            if self._column_tuples != other._column_tuples:
                raise ValueError("Cannot operate on TimeIndexedData objects with different columns")
            other = other.values

        arg = (other, self.values) if reverse_args else (self.values, other)
        res = op(*arg)

        if len(res) != len(self.values):
            raise ValueError("Math operations must preserve `values` first dimension size")

        return self._self_with_new_values(res)

    def __abs__(self) -> TimeIndexedData:
        return self._self_with_new_values(abs(self.values))

    def __add__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.add)

    def __floordiv__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.floordiv)

    def __ge__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.ge)

    def __gt__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.gt)

    def __le__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.le)

    def __lt__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.lt)

    def __mod__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.mod)

    def __mul__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.mul)

    def __neg__(self) -> TimeIndexedData:
        return self._self_with_new_values(-self.values)

    def __pos__(self) -> TimeIndexedData:
        return self._self_with_new_values(+self.values)

    def __pow__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.pow)

    def __radd__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.add, reverse_args=True)

    def __rfloordiv__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.floordiv, reverse_args=True)

    def __rmul__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.mul, reverse_args=True)

    def __rpow__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.pow, reverse_args=True)

    def __rsub__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.sub, reverse_args=True)

    def __rtruediv__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.truediv, reverse_args=True)

    def __sub__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.sub)

    def __truediv__(self, other: Union[TimeIndexedData, "npt.NDArray"]) -> TimeIndexedData:
        return self._values_math_override(other, operator.truediv)


class TimeIndex:
    """A container of ordered time stamps, viewable as pd.Timestamp or integer values.

    Attributes
    ----------
    timestamp_values: List[pd.Timestamp]
        A strictly increasing collection of pd.Timestamp objects
    int_values: List[int]
        A strictly increasing collection of integers representing the unixtimes of the
        pd.Timestamp objects.
    granularity: TimeUnit
        One of the acceptable TimeUnit types. Can be a pandas frequency alias
        (https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases),
        a pandas DateOffset, or a timedelta/pd.Timedelta object.
    unixtime_t0: pd.Timestamp
        A point in time represented by "0" on the integer scale.
    unixtime_unit: pd.Timedelta
        A time delta representing the unixtime unit.
    """

    def __init__(
        self,
        values: Union[TimeStamp, Collection[TimeStamp]],
        granularity: Optional[TimeUnit] = None,
        unixtime_t0: pd.Timestamp = EPOCH_TIME_NO_TZ,
        unixtime_unit: pd.Timedelta = ONE_SECOND,
    ) -> None:
        """
        Create a new TimeIndex instance

        Parameters
        ----------
        values: Union[TimeStamp, Collection[TimeStamp]]
            A collection of TimeStamp or integer values. These should be strictly increasing
            after being cast to pd.Timestamp objects, and aligned with `granularity` if provided.
        granularity: Optional[TimeUnit], default None
            The granularity of the TimeIndex, used to generate future values. If not provided,
            TimeIndex will attempt to determine a suitable granularity given the `values` input.
        unixtime_t0: pd.Timestamp, default pd.Timestamp('1970-01-01 00:00:00')
            A point in time represented by "0" on the integer scale.
        unixtime_unit: pd.Timedelta, default pd.Timedelta(seconds=1)
            The unit of the integer unixtime scale.

        Raises
        ------
        ValueError
            If `values` contains duplicates. If `values` are not strictly increasing after casting to pd.Timestamp
            If `values` and `granularity` are incompatible. E.g. if granularity is "B" for business day, but `values`
            includes some days that are not business days.
        """
        unixtime_unit = pd.Timedelta(unixtime_unit)
        if self._is_timestamp_castable(values):
            values = [values]

        ts_values = [
            unixtime_t0 + (int(v) * unixtime_unit) if isinstance(v, Number) else pd.Timestamp(v) for v in values
        ]
        if len(ts_values) != len(set(ts_values)):
            raise ValueError("Timestamp values must be unique")
        if not self._is_strict_increasing(ts_values):
            raise ValueError("TimeIndex must be strictly increasing after integer cast and conversion to pd.Timestamp")

        int_values = [(t - unixtime_t0) // unixtime_unit for t in ts_values]
        if not self._is_strict_increasing(int_values):
            raise ValueError("TimeIndex unixtime must be strictly increasing given unixtime_unit")

        if granularity is None:
            granularity = _infer_granularity(ts_values, strict=True, sort_list=False)
        elif isinstance(granularity, str):
            granularity = _get_custom_granularity(granularity) or granularity

        if len(ts_values) > 0:
            msg = "Timestamp values and granularity are not compatible"
            if isinstance(granularity, (pd.Timedelta, timedelta)):
                if any((t - ts_values[0]) % granularity > ZERO_DELTA for t in ts_values[1:]):
                    raise ValueError(msg)
            else:
                valid_ts = set(_time_range(ts_values[0], ts_values[-1], freq=granularity))
                if any(t not in valid_ts for t in ts_values):
                    raise ValueError(msg)

        self.timestamp_values = ts_values
        self.int_values = int_values
        self.granularity = granularity
        self.unixtime_t0 = unixtime_t0
        self.unixtime_unit = unixtime_unit

        self._t_index = {ts: i for i, ts in enumerate(ts_values)}
        self._i_index = {ut: i for i, ut in enumerate(int_values)}

    @staticmethod
    def _is_strict_increasing(values: Collection) -> bool:
        return all(t2 > t1 for t1, t2 in zip(values, values[1:]))

    def __len__(self) -> int:
        return len(self.timestamp_values)

    def __eq__(self, other: TimeIndex) -> bool:
        return (
            type(self) is type(other)
            and self.unixtime_t0 == other.unixtime_t0
            and self.unixtime_unit == other.unixtime_unit
            and self.granularity == other.granularity
            and np.array_equal(self.timestamp_values, other.timestamp_values)
        )

    def __repr__(self) -> str:
        end = "]" if len(self) <= 3 else ", ...]"
        ints = "[" + ", ".join(map(str, self.int_values[:3]))
        ts = "[" + ", ".join(map(str, self.timestamp_values[:3]))
        return f"TimeIndex({ints}{end}, {ts}{end}, len={len(self)})"

    def isin(self, values: Union[Iterable, TimeIndex, TimeIndexedData]) -> "npt.NDArray[np.bool_]":
        """Returns a boolean array where this time index's value match the provided ones.

        If a TimeIndex is provided, then both the integer unixtime and the pd.Timestamp values will be considered when
        finding matches. If TimeIndexedData is provided, then the TimeIndex will be extracted. Otherwise, the match
        will be inferred based on if the value is an integer or a pd.Timestamp.

        Parameters
        ----------
        values: Union[Iterable, TimeIndex, TimeIndexedData]
            An iterable of values to check, or a TimeIndex / TimeIndexedData object to compare.

        Returns
        -------
        npt.NDArray[np.bool_]
            A 1D array of booleans with the same length as `self` indicating the matched values
        """
        if isinstance(values, TimeIndexedData):
            values = values.time_index

        keep = set()
        if isinstance(values, TimeIndex):
            for ut, ts in zip(values.int_values, values.timestamp_values):
                if ut in self._i_index and ts in self._t_index and self._t_index[ts] == self._i_index[ut]:
                    keep.add(self._i_index[ut])
        else:
            for v in values:
                if v in self._i_index:
                    keep.add(self._i_index[v])
                elif v in self._t_index:
                    keep.add(self._t_index[v])

        mask = np.zeros(len(self), dtype=np.bool_)
        mask[list(keep)] = True
        return mask

    def int_time_index(self) -> List[int]:
        """Return a copy of the TimeIndex unixtime values.

        Returns
        -------
        List[int]
        """
        return deepcopy(self.int_values)

    def pd_timestamp_index(self) -> List[pd.Timestamp]:
        """Return a copy of the TimeIndex values as pandas timestamp objects.

        Returns
        -------
        List[pd.Timestamp]
        """
        return deepcopy(self.timestamp_values)

    def starts_before(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Checks if this object starts before another

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the first timestamp value in the current time index is less than
            the first timestamp value in the other time index.
        """
        if isinstance(other, TimeIndexedData):
            other = other.time_index
        return len(self) > 0 and len(other) > 0 and self.timestamp_values[0] < other.timestamp_values[0]

    def strictly_before(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Checks if this object ends before `other` begins

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the last timestamp value in the current time index is less than
            the first timestamp value in the other time index.
        """
        if isinstance(other, TimeIndexedData):
            other = other.time_index
        return len(self) > 0 and len(other) > 0 and self.timestamp_values[-1] < other.timestamp_values[0]

    def same_scale(self, other: Union[TimeIndex, TimeIndexedData]) -> bool:
        """Check if two TimeIndex object have the same scale

        To share the same scale two TimeIndex objects must have the following properties:

        * `unixtime_t0`, `unixtime_unit`, and `granularity` are the same in both objects
        * And at least one of the following is true
          * The time indices have values in common
          * One time index is in the future of the other
          * One of the time indices is empty

        Parameters
        ----------
        other: Union[TimeIndex, TimeIndexedData]
            An object to compare

        Returns
        -------
        bool
            True if the TimeIndex objects share a scale given the logic above.

        Examples
        --------
        >>> a = TimeIndex([0, 1, 2])
        >>> b = TimeIndex([3])
        >>> a.same_scale(b)
        True

        Integer scales must also be the same

        >>> a = TimeIndex(["2022-01-01", "2022-01-02"], unixtime_t0=pd.Timestamp("2022-01-01"))
        >>> b = TimeIndex(["2022-01-01", "2022-01-02"], unixtime_t0=pd.Timestamp("1970-01-01"))
        >>> a.same_scale(b)
        False

        Values must be consistent even if granularity matches

        >>> a = TimeIndex(["2022-01-01 12:00:00", "2022-01-02 12:00:00"], granularity="D")
        >>> b = TimeIndex(["2022-01-01", "2022-01-02"], granularity="D")
        >>> a.same_scale(b)
        False

        Date ranges are projected to check if one object is in the future of the other

        >>> a = TimeIndex(["2023-01-19", "2023-01-20"], granularity="B")  # Business days
        >>> b = TimeIndex(["2023-01-03", "2023-01-04"], granularity="B")
        >>> a.same_scale(b)
        True
        """
        if isinstance(other, TimeIndexedData):
            other = other.time_index

        if (
            self.unixtime_t0 != other.unixtime_t0
            or self.unixtime_unit != other.unixtime_unit
            or self.granularity != other.granularity
        ):
            return False

        if len(self) == 0 or len(other) == 0:
            return True

        self_before_other = self.starts_before(other)
        a_values = (self if self_before_other else other).timestamp_values
        b_values = (other if self_before_other else self).timestamp_values

        if isinstance(self.granularity, (timedelta, pd.Timedelta)):
            return (b_values[0] - a_values[0]) % self.granularity == pd.Timedelta(0)

        if len(set(a_values).intersection(b_values)) > 0:
            return True

        a_idx = -1 if b_values[0] > a_values[-1] else 0
        rng = _time_range(a_values[a_idx], b_values[0], freq=self.granularity)
        return rng[-1] == b_values[0]

    def values_match(self, other: TimeIndex) -> bool:
        """Check if `timestamp_values` matches between two TimeIndex objects.

        Parameters
        ----------
        other: TimeIndex
            Another TimeIndex instance

        Returns
        -------
        bool
            True if self.timestamp_values matches other.timestamp_values. Note that
            the indices themselves may have different granularities and integer
            representations, but they represent an equivalent collection of
            points in time.
        """
        return np.array_equal(self.timestamp_values, other.timestamp_values)

    def next_timestamp(self) -> pd.Timestamp:
        return _time_range(self.timestamp_values[-1], periods=2, freq=self.granularity)[-1]

    def first_timestamp(self) -> pd.Timestamp:
        return self.timestamp_values[0]

    def last_timestamp(self) -> pd.Timestamp:
        return self.timestamp_values[-1]

    def future(self, n_steps: int, n_steps_past: int = 0) -> TimeIndex:
        """Return a new TimeIndex with the current granularity projected into the future.

        Parameters
        ----------
        n_steps: int
            The number of steps into the future to project. Should be semi-positive
        n_steps_past: int, default 0
            The number of steps into the past to return. Steps in the past are empirical,
            so if the TimeIndexedData has some missing data according to `granularity`
            then the past data will reflect that missing data.

        Returns
        -------
        TimeIndex
            A new TimeIndex object with future and optional past entries.

        Raises
        ------
        ValueError
            If there are no past time stamps (len(self) == 0)

        Examples
        --------
        >>> t = ["2023-01-05", "2023-01-06"]
        >>> ind = TimeIndex(t, granularity="B")  # Business days
        >>> ind.future(n_steps=1, n_steps_past=1).timestamp_values
        [Timestamp('2023-01-06 00:00:00', freq='B'), Timestamp('2023-01-09 00:00:00', freq='B')]
        """
        past_ts = self.timestamp_values[-n_steps_past:] if n_steps_past > 0 else []
        future_ts = []

        if n_steps > 0:
            if len(self.timestamp_values) == 0:
                raise ValueError("Cannot generate future from an empty time index")
            future_ts = _time_range(
                self.timestamp_values[-1],
                periods=n_steps + 1,
                freq=self.granularity,
            )[1:]

        return type(self)(
            past_ts + future_ts,
            self.granularity,
            self.unixtime_t0,
            self.unixtime_unit,
        )

    def slice(self, start: int, end: Optional[int] = None, copy_data: bool = True) -> TimeIndex:
        """Create a new TimeIndex object from a slice of the current one.

        Parameters
        ----------
        start: int
            The start index of the slice
        end: Optional[int], default None
            The end index of the slice. If not provided, the slice will continue through the length of
            the TimeIndex.
        copy_data: bool, default True
            If true, deepcopy() is used to create copies of the underlying data for the new instance.
            This is to reduce the possibility of unintended side effects if the slice is manipulated
            later. If memory is a concern, copy_data can be set to False.

        Returns
        -------
        TimeIndex
            A new TimeIndex instsance
        """
        if end is None:
            end = len(self)
        opt_copy = deepcopy if copy_data else lambda x: x
        return type(self)(
            opt_copy(self.timestamp_values[start:end]),
            opt_copy(self.granularity),
            opt_copy(self.unixtime_t0),
            opt_copy(self.unixtime_unit),
        )

    def missing_time_stamps(self) -> List[pd.Timestamp]:
        """Return a list of missing timestamps given granularity

        Returns
        -------
        List[pd.Timestamp]
            The missing timestamps
        """
        if len(self.timestamp_values) == 0:
            return []

        dense = _time_range(self.first_timestamp(), self.last_timestamp(), freq=self.granularity)
        return sorted(set(dense) - set(self.timestamp_values))

    @staticmethod
    def _is_timestamp_castable(value: Any) -> bool:
        try:
            pd.Timestamp(value)
            return True
        except TypeError:
            return False
