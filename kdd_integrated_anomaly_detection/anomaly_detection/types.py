from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

from metric_store_example import metric_store

from .algorithms.common import MultivariateOutlierDetectionAlgorithm, UnivariateOutlierDetectionAlgorithm

UnivariateDetectionModel = UnivariateOutlierDetectionAlgorithm

MultivariateDetectionModel = MultivariateOutlierDetectionAlgorithm

AnomalyDetectionModel = Union[UnivariateDetectionModel, MultivariateDetectionModel]


@dataclass
class MetricStoreMetricSpec:
    name: str
    time_granularity: str
    dimension_groupings: List[str]
    dimension_filters: List[str]

    def query(self, start: Union[str, datetime], end: Union[str, datetime]) -> str:
        # TODO: Construct a valid query from the anomaly detection config
        return metric_store.TO_BE_DETERMINED_MODULE.query(
            time_granularity=self.time_granularity,
            start=str(start),
            end=str(end),
            dimension_groupings=self.dimension_groupings,
            dimension_filters=self.dimension_filters,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> MetricStoreMetricSpec:
        return cls(**json.loads(data))


@dataclass
class AlgorithmSpec:
    cls: str
    args: field(default_factory=dict)


@dataclass
class TimeSpec:
    granularity: str
    start: Optional[str]
    end: Optional[str]
    retrain: Optional[str]


@dataclass
class DetectorConfig:
    name: str
    description: str
    budget_group: str
    git_owners: List[str]
    time_spec: TimeSpec
    metric_store_metrics: List[MetricStoreMetricSpec]
    algorithms: Optional[Dict[str, AlgorithmSpec]]

    @classmethod
    def from_dict(cls, data: Dict) -> DetectorConfig:
        time_spec = TimeSpec(**data["time_spec"])

        metric_store_metrics = [
            MetricStoreMetricSpec(
                name=msm["name"],
                time_granularity=time_spec.granularity,
                dimension_groupings=msm.get("dimension_groupings"),
                dimension_filters=msm.get("dimension_filters"),
            )
            for msm in data["metric_store_metrics"] or []
        ]

        algorithms = None
        if data.get("algorithms") is not None:
            algorithms = {name: AlgorithmSpec(args.pop("cls", name), args) for name, args in data["algorithms"].items()}

        return cls(
            data["name"],
            data["description"],
            data["budget_group"],
            data["git_owners"],
            time_spec=time_spec,
            metric_store_metrics=metric_store_metrics,
            m3_metrics=[],
            algorithms=algorithms,
        )
