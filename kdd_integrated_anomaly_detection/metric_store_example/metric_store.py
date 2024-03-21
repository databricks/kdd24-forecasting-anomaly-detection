import textwrap

from pyspark.sql import SparkSession
from utils.metrics import Metric, QuerySpec, TimeGranularity
from utils.tables import DimensionTable, FactTable

fact_tables = FactTable.collect()
dimension_tables = DimensionTable.collect()
metrics = Metric.collect()

print(fact_tables)
print(dimension_tables)
print(metrics)

for metric in metrics:
    print(f"{metric.name()}:")
    spec = QuerySpec(
        time_grain=TimeGranularity.DAILY,
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    print(textwrap.indent(metric.get_query(spec, fact_tables, dimension_tables), prefix="> "))

for metric in metrics:
    print(f"{metric.name()}:")
    spec = QuerySpec(
        time_grain=TimeGranularity.MONTHLY,
        start_date="2020-01-01",
        end_date="2020-01-31",
        groupings=["dim_payment_type", "dim_customer_region"],
    )
    print(textwrap.indent(metric.get_query(spec, fact_tables, dimension_tables), prefix="> "))

spark = SparkSession.builder.appName("example").getOrCreate()
for table in fact_tables + dimension_tables:
    table.load_data(spark)

spark.sql("show tables").show()
for table in fact_tables + dimension_tables:
    spark.sql(f"select * from {table.name()}").show()

spec = QuerySpec(
    time_grain=TimeGranularity.MONTHLY,
    start_date="2023-01-01",
    end_date="2024-01-01",
)
metric = Metric.get("transaction_amount", metrics)
query = metric.get_query(spec, fact_tables, dimension_tables)
print(textwrap.indent(metric.get_query(spec, fact_tables, dimension_tables), prefix="> "))
spark.sql(query).show()
