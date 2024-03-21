import textwrap

from src.collection import Collection
from src.metrics import QuerySpec, TimeGranularity

collection = Collection()

for metric in collection.metrics:
    print(f"{metric.name()}:")
    spec = QuerySpec(
        time_grain=TimeGranularity.DAY,
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    print(textwrap.indent(metric.get_query(spec, collection), prefix="> "))
    print("\n")

for metric in collection.metrics:
    print(f"{metric.name()}:")
    spec = QuerySpec(
        time_grain=TimeGranularity.DAY,
        start_date="2020-01-01",
        end_date="2020-01-31",
        groupings=["dim_payment_type", "dim_customer_region"],
    )
    print(textwrap.indent(metric.get_query(spec, collection), prefix="> "))
    print("\n")
