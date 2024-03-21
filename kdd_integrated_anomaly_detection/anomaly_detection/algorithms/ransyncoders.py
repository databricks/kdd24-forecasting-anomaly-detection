import logging
from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .common import MultivariateOutlierDetectionAlgorithm
from ...uff_example.transformers.ransyncoders.ransyncoders import RanSynCodersTransformer
from ...uff_example.tstypes import TimeIndexedData, TimeStamp

logger = logging.getLogger(__name__)


class RANSynCodersDetector(MultivariateOutlierDetectionAlgorithm, RanSynCodersTransformer):
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int = 3,
        encoding_depth: int = 2,
        latent_dim: int = 2,
        decoding_depth: int = 2,
        activation: str = "linear",
        output_activation: str = "linear",
        delta: float = 0.05,
        synchronize: bool = True,
        min_periods: int = 3,
        freq_init: Optional[List[float]] = None,
        max_freqs: int = 1,
        min_dist: int = 60,
        trainable_freq: bool = False,
        direction: Optional[str] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            encoding_depth=encoding_depth,
            latent_dim=latent_dim,
            decoding_depth=decoding_depth,
            activation=activation,
            output_activation=output_activation,
            delta=delta,
            synchronize=synchronize,
            force_synchronization=synchronize,
            min_periods=min_periods,
            freq_init=freq_init,
            max_freqs=max_freqs,
            min_dist=min_dist,
            trainable_freq=trainable_freq,
        )
        self.cached_data = None
        self.train_batch_size = 0
        possible_directions = ("up", "down")
        self.direction = (direction,) if direction in possible_directions else possible_directions

    def train(
        self,
        data: TimeIndexedData,
        epochs: int = 10,
        batch_size: int = 180,
        shuffle: bool = True,
        freq_warmup: int = 5,
        sin_warmup: int = 5,
        pos_amp: bool = True,
        **kwargs,
    ) -> None:
        self.fit(
            data=data,
            covariates=None,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            freq_warmup=freq_warmup,
            sin_warmup=sin_warmup,
            pos_amp=pos_amp,
        )
        self.train_batch_size = batch_size
        self.cached_data = (
            data.to_pandas(time_col="ts", time_as_pd_timestamps=True).set_index("ts").iloc[-self.train_batch_size :]
        )

    def check_is_anomaly(
        self, timestamp: Sequence[TimeStamp], value: Sequence[Dict[Hashable, float]]
    ) -> Sequence[Dict[Hashable, bool]]:
        new_obs = pd.DataFrame(value, index=timestamp)
        new_obs.index = pd.to_datetime(new_obs.index)
        # make sure columns match
        new_cols = set(new_obs.columns) - set(self.cached_data.columns)
        if new_cols:
            logger.warn(f"new columns passed to check_is_anomaly, will be ignored: {list(new_cols)}")
            # this will throw an error if `new_obs` does not contain the same columns as in `self.cached_data`
            new_obs = new_obs.loc[:, self.cached_data.columns]

        self.cached_data = pd.concat([self.cached_data, new_obs])
        td = TimeIndexedData.from_pandas(self.cached_data, value_col=list(self.cached_data.columns))
        out = self.transform(td)
        lb = np.mean(out.lower_estimates.values, axis=2)
        hb = np.mean(out.upper_estimates.values, axis=2)
        anomalies = (
            pd.DataFrame(
                ("down" in self.direction) * (td < lb) + ("up" in self.direction) * (td > hb),
                index=self.cached_data.index,
                columns=self.cached_data.columns,
            )
            .iloc[-len(new_obs) :]
            .to_dict(orient="records")
        )
        self.cached_data = self.cached_data.iloc[-self.train_batch_size :]
        return anomalies
