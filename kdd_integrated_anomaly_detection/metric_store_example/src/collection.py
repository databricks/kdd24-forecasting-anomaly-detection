from typing import List

from pyspark.sql import SparkSession
from src.metrics import Metric
from src.tables import DimensionTable, FactTable


class Collection:
    def __init__(self):
        self.fact_tables: List[FactTable] = FactTable.collect()
        self.dimension_tables: List[DimensionTable] = DimensionTable.collect()
        self.metrics: List[Metric] = Metric.collect()

    @property
    def spark(self):
        return SparkSession.builder.appName("example").getOrCreate()

    def load_data(self, show: bool = False):
        for table in self.all_tables():
            table.load_data(self.spark)
            if show:
                self.spark.sql(f"select * from {table.name()}").show()
        if show:
            self.spark.sql("show tables").show()

    def all_tables(self):
        return self.fact_tables + self.dimension_tables

    def get_metric(self, name: str) -> Metric:
        return Metric.get(name, self)

    @property
    def sql(self):
        return self.spark.sql
