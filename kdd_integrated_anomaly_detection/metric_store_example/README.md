# Metric Store Example

This is an example to demonstrate basic functionality of the Metric Store as a POC. This does not include code from the actual Databricks Metric Store but does leverage some of the same concepts as described in the paper.

Folder structure:

```sh
metric_store_example/
 └───configs/
 │   └───dimension_tables/       # Dimension Table definitions
 │   └───fact_tables/            # Fact Table definitions
 │   └───metrics/                # Metric definitions
 └───data/                       # generated raw source data for tables
 │   └───data_generation.py      # script to [re]generate raw source data
 └───src/                        # Metric Store example source code
 └───metric_query_examples.py    # script to generate Metric query examples
 └───run.py                      # cli script to query Metrics and return data
```

## Quering a Metric

You can utilize `run.py` to query metrics from Terminal.

```sh
Usage: run.py [OPTIONS] METRIC_NAME

  example: python run.py num_transactions -d dim_payment_type

Options:
  -d, --dimension TEXT   Specify the dimensions to group by (supports
                         multiple).
  -t, --time_grain TEXT  Specify time grain for the query (default: MONTH).
  -s, --start TEXT       Start date for query (default: 2023-01-01).
  -e, --end TEXT         End date for query (default: 2024-01-01).
  --verbose              Verbose mode
  --help                 Show this message and exit.
```

Example:

```sql
$ python run.py num_transactions
> SELECT
>   DATE_TRUNC('MONTH', `ts`) AS `time`,
>   COUNT(transaction_id) AS num_transactions
> FROM `fct_transactions`
> WHERE
>   `ts` BETWEEN '2023-01-01' AND '2024-01-01'
> GROUP BY
>   `time`
+-------------------+----------------+
|               time|num_transactions|
+-------------------+----------------+
|2023-01-01 00:00:00|              10|
|2023-02-01 00:00:00|              26|
|2023-03-01 00:00:00|              43|
|2023-04-01 00:00:00|              70|
|2023-05-01 00:00:00|              84|
|2023-06-01 00:00:00|              96|
|2023-07-01 00:00:00|             112|
|2023-08-01 00:00:00|             110|
|2023-09-01 00:00:00|             121|
|2023-10-01 00:00:00|             103|
|2023-11-01 00:00:00|             102|
|2023-12-01 00:00:00|             123|
+-------------------+----------------+
```

## Simple example of Metric query generation

You can see generated Metric queries for predefined example specifications by running:

```sh
python -m metric_query_examples
```

Example output:

```sh
num_transactions:
> SELECT
>   DATE_TRUNC('DAY', `ts`) AS `time`,
>   COUNT(transaction_id) AS num_transactions
> FROM `fct_transactions`
> WHERE
>   `ts` BETWEEN '2020-01-01' AND '2020-01-31'
> GROUP BY
>   `time`


transaction_amount:
> SELECT
>   DATE_TRUNC('DAY', `ts`) AS `time`,
>   SUM(m_total) AS transaction_amount
> FROM `fct_transactions`
> WHERE
>   `ts` BETWEEN '2020-01-01' AND '2020-01-31'
> GROUP BY
>   `time`


num_transactions sliced by dim_payment_type + dim_customer_region:
> SELECT
>   DATE_TRUNC('DAY', `ts`) AS `time`,
>   COUNT(transaction_id) AS num_transactions,
>   `dim_payment_type`,
>   `dim_customer_region`
> FROM (
>   SELECT
>     fct_transactions.ts,
>     fct_transactions.transaction_id,
>     fct_transactions.dim_payment_type,
>     dim_customers.dim_customer_region
>   FROM `fct_transactions`
>   LEFT JOIN `dim_customers`
>     ON `fct_transactions`.`customer_id` = `dim_customers`.`customer_id`
> ) AS `source`
> WHERE
>   `ts` BETWEEN '2020-01-01' AND '2020-01-31'
> GROUP BY
>   `time`,
>   `dim_payment_type`,
>   `dim_customer_region`


transaction_amount sliced by dim_payment_type + dim_customer_region:
> SELECT
>   DATE_TRUNC('DAY', `ts`) AS `time`,
>   SUM(m_total) AS transaction_amount,
>   `dim_payment_type`,
>   `dim_customer_region`
> FROM (
>   SELECT
>     fct_transactions.ts,
>     fct_transactions.m_total,
>     fct_transactions.dim_payment_type,
>     dim_customers.dim_customer_region
>   FROM `fct_transactions`
>   LEFT JOIN `dim_customers`
>     ON `fct_transactions`.`customer_id` = `dim_customers`.`customer_id`
> ) AS `source`
> WHERE
>   `ts` BETWEEN '2020-01-01' AND '2020-01-31'
> GROUP BY
>   `time`,
>   `dim_payment_type`,
>   `dim_customer_region`
```
