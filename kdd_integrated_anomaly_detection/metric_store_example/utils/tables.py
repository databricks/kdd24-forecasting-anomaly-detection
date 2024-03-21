from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List

from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType, TimestampType
from utils.helpers import CONFIG_DIR, DATA_DIR, TIMESTAMP_COLUMN, load_config


@dataclass
class TableConf:
    name: str
    type: str
    source: str
    columns: List[Dict]


class ColumnType(Enum):
    KEY = auto()
    DIMENSION = auto()
    MEASURE = auto()
    TIMESTAMP = auto()


@dataclass
class Column:
    name: str
    data_type: str

    def column_type(self):
        if self.name.endswith("_id"):
            return ColumnType.KEY
        elif self.name.startswith("dim_"):
            return ColumnType.DIMENSION
        elif self.name.startswith("m_"):
            return ColumnType.MEASURE
        elif self.name == TIMESTAMP_COLUMN:
            return ColumnType.TIMESTAMP

    def spark_type(self):
        if self.data_type == "int":
            return IntegerType()
        elif self.data_type == "float":
            return FloatType()
        elif self.data_type == "str":
            return StringType()
        elif self.data_type == "timestamp":
            return TimestampType()
        else:
            raise Exception(f"Unknown data type: {self.data_type}")

    def schema(self):
        return StructField(self.name, self.spark_type(), True)


class BaseTable:
    config_folder = None

    def __init__(self, conf: dict):
        self.conf = TableConf(**conf)

    def __repr__(self) -> str:
        return f"{self.conf.name}"

    def name(self):
        return self.conf.name

    def columns(self):
        return [Column(**conf) for conf in self.conf.columns]

    def schema(self):
        return StructType([column.schema() for column in self.columns()])

    def load_data(self, spark):
        df = spark.read.csv(str(DATA_DIR.joinpath(self.conf.source)), schema=self.schema(), header=False)
        df.createOrReplaceTempView(self.conf.name)

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


class FactTable(BaseTable):
    config_folder = "fact_tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            sum([1 for column in self.columns() if column.column_type() == ColumnType.TIMESTAMP]) == 1
        ), f"Fact table must have exactly one timestamp ('{TIMESTAMP_COLUMN}') column"


class DimensionTable(BaseTable):
    config_folder = "dimension_tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            sum([1 for column in self.columns() if column.column_type() == ColumnType.KEY]) == 1
        ), "Dimension table must have exactly one ID column"

    def join_key(self):
        return [column for column in self.columns() if column.column_type() == ColumnType.KEY][0].name
