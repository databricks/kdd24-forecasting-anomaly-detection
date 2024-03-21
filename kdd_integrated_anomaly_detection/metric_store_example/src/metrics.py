from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List

from sqlglot import expressions as exp
from sqlglot import parse_one
from src.helpers import CONFIG_DIR, DIALECT, METRIC_TIME_COLUMN, TIMESTAMP_COLUMN, load_config
from src.tables import BaseTable, DimensionTable, FactTable

if TYPE_CHECKING:
    from src.collection import Collection


class TimeGranularity(Enum):
    DAY = f"date_trunc('day', {TIMESTAMP_COLUMN})"
    MONTH = f"date_trunc('month', {TIMESTAMP_COLUMN})"
    YEAR = f"date_trunc('year', {TIMESTAMP_COLUMN})"


@dataclass
class QuerySpec:
    time_grain: TimeGranularity
    start_date: str
    end_date: str
    groupings: List[str] = field(default_factory=list)


@dataclass
class MetricConf:
    name: str
    definition: str


class Metric:
    config_folder = "metrics"

    def __init__(self, conf: dict):
        self.conf = MetricConf(**conf)

    def __repr__(self) -> str:
        return f"{self.conf.name}"

    def name(self):
        return self.conf.name

    @classmethod
    def collect(cls):
        collection = []
        assert cls.config_folder is not None, "config_folder must be defined"
        for file_path in CONFIG_DIR.joinpath(cls.config_folder).iterdir():
            conf = load_config(file_path)
            if isinstance(conf, list):
                for sub_conf in conf:
                    collection.append(cls(conf=sub_conf))
            else:
                collection.append(cls(conf=conf))
        return collection

    def _apply_time_grain(self, query: exp.Select, time_grain: TimeGranularity):
        query = query.select(f"{time_grain.value} as {METRIC_TIME_COLUMN}", dialect=DIALECT)
        query = query.group_by(METRIC_TIME_COLUMN)
        return query

    def _apply_grouping(self, query: exp.Select, groupings: List[str]):
        for grouping in groupings:
            query = query.select(grouping)
            query = query.group_by(grouping)
        return query

    def _required_columns(self):
        return [column.alias_or_name for column in parse_one(self.conf.definition, read=DIALECT).find_all(exp.Column)]

    def _build_subquery(self, tables: Dict[BaseTable, List[str]]):
        query = exp.Select()

        # join tables
        fact_table = [table for table in tables if isinstance(table, FactTable)].pop()
        dimension_tables = [table for table in tables if isinstance(table, DimensionTable)]
        query = query.from_(fact_table.name())
        for table in dimension_tables:
            query = query.join(
                table.name(),
                join_type="LEFT",
                on=f"{fact_table.name()}.{table.join_key()} = {table.name()}.{table.join_key()}",
            )

        # add columns
        query = query.select(exp.Column(this=TIMESTAMP_COLUMN, table=fact_table.name()))
        for table, columns in tables.items():
            for column in columns:
                query = query.select(exp.Column(this=column, table=table.name()))

        return query.subquery(alias="source")

    def _determine_source(self, q: QuerySpec, collection: "Collection"):
        required_tables_and_columns: Dict[BaseTable, List[str]] = {}
        for column in self._required_columns() + q.groupings:
            for table in collection.all_tables():
                if column in [column.name for column in table.columns()]:
                    if table not in required_tables_and_columns:
                        required_tables_and_columns[table] = []
                    required_tables_and_columns[table].append(column)
        assert len(required_tables_and_columns) > 0, f"Cannot determine source table for {self.name()}"
        if len(required_tables_and_columns) == 1:
            return list(required_tables_and_columns.keys())[0].name()
        else:
            assert (
                len([table for table in required_tables_and_columns if isinstance(table, FactTable)]) == 1
            ), "Must only have one fact table in the query"
            return self._build_subquery(required_tables_and_columns)

    def get_query(self, q: QuerySpec, collection: "Collection"):
        query = exp.Select()
        query = query.from_(self._determine_source(q, collection))
        query = query.where(f"{TIMESTAMP_COLUMN} between '{q.start_date}' and '{q.end_date}'")
        query = self._apply_time_grain(query, q.time_grain)
        query = query.select(exp.Alias(this=self.conf.definition, alias=self.conf.name))
        query = self._apply_grouping(query, q.groupings)
        return query.sql(pretty=True, identify=True, dialect=DIALECT)

    @staticmethod
    def get(metric_name: str, collection: "Collection"):
        return [metric for metric in collection.metrics if metric.name() == metric_name].pop()
