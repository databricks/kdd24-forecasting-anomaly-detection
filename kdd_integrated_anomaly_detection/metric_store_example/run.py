import textwrap

import click
from src.collection import Collection
from src.helpers import METRIC_TIME_COLUMN
from src.metrics import QuerySpec, TimeGranularity


@click.command()
@click.option(
    "-d",
    "--dimension",
    "dimensions",
    multiple=True,
    type=str,
    help="Specify the dimensions to group by (supports multiple).",
)
@click.option(
    "-t",
    "--time_grain",
    "time_grain",
    type=str,
    default="MONTH",
    help="Specify time grain for the query (default: MONTH).",
)
@click.option(
    "-s",
    "--start",
    "start",
    type=str,
    default="2023-01-01",
    help="Start date for query (default: 2023-01-01).",
)
@click.option(
    "-e",
    "--end",
    "end",
    type=str,
    default="2024-01-01",
    help="End date for query (default: 2024-01-01).",
)
@click.option("--verbose", is_flag=True, help="Verbose mode")
@click.argument("metric_name", type=str, required=True)
def cli(metric_name: str, dimensions: list, time_grain: str, start: str, end: str, verbose: bool):
    """example: python run.py num_transactions -d dim_payment_type"""
    collection = Collection()
    collection.load_data(show=verbose)

    spec = QuerySpec(
        time_grain=TimeGranularity[time_grain],
        start_date=start,
        end_date=end,
        groupings=list(dimensions),
    )
    metric = collection.get_metric(metric_name)
    query = metric.get_query(spec, collection)
    print(textwrap.indent(metric.get_query(spec, collection), prefix="> "))
    collection.sql(query).sort(METRIC_TIME_COLUMN).show()


if __name__ == "__main__":
    cli()  # type: ignore
