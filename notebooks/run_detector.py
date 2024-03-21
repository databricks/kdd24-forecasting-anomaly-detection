# Databricks notebook source

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import mlflow
import pandas as pd
from delta import DeltaTable
from mlflow.tracking.client import MlflowClient
from pyspark.sql import Row

from ..kdd_integrated_anomaly_detection.anomaly_detection.types import MetricStoreMetricSpec
from ..kdd_integrated_anomaly_detection.anomaly_detection.utils.mlflow import MLFlowAnomalyDetectorWrapper, MLFlowModelManager
from ..kdd_integrated_anomaly_detection.uff_example.tstypes import TimeIndexedData

db_utils.widgets.text("name", "", "Detector ID")
db_utils.widgets.text("deploymentMode", "dev", "Deployment mode")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect config and time context information

# COMMAND ----------

deployment_mode = dbutils.widgets.get("deploymentMode")
RAW_ANOMALIES_TBL = f"raw_anomalies_{deployment_mode}"

# COMMAND ----------

detector_cfg = get_detector_config(get_widget_value("name"))

# COMMAND ----------

current_time = pd.Timestamp.now()
end = current_time - pd.Timedelta(detector_cfg.time_spec.end or "0s")
start = None
# Attempt to cast time_spec.start to a timedelta
# If this fails, it is assumed time_spec.start is a timestamp
try:
    start = current_time - pd.Timedelta(detector_cfg.time_spec.start)
except ValueError:
    start = pd.Timestamp(detector_cfg.time_spec.start)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Inference

# COMMAND ----------

mlflow.set_experiment(f"/tmp/anomaly_detection_{deployment_mode}")

client = MlflowClient()
anomalies = []
# if retrain is not specified in the time_spec, we will default to retrain
retrain_flag = detector_cfg.time_spec.retrain is None

with mlflow.start_run(run_name=f"anomaly_detection_inference_{detector_cfg.name}:{deployment_mode}") as run:
    mlflow_mm = MLFlowModelManager(run, detector_cfg.name, deployment_mode=deployment_mode)
    model_version_infos = client.search_registered_models(
        f"tags.online = 'True' and tags.detector = '{detector_cfg.name}' and tags.deployment_mode = '{deployment_mode}'"
    )
    if len(model_version_infos) == 0:
        # Cold start
        retrain_flag = True
    for registered_model in model_version_infos:
        # if *any* registered model needs to be retrain, we will retrain *all* the models for the detector
        if not retrain_flag and (
            current_time - pd.Timestamp(registered_model.creation_timestamp, unit="ms")
            >= pd.Timedelta(detector_cfg.time_spec.retrain)
        ):
            retrain_flag = True
        model_version = registered_model.latest_versions[0]
        ms_spec = MetricStoreMetricSpec.from_json(model_version.tags["query"])
        last_evaluation_ts = pd.Timestamp(model_version.tags["last_evaluation_ts"])
        # query new observations
        try:
            new_obs = pd.pivot_table(
                spark.sql(ms_spec.query(last_evaluation_ts, end)).where(f"time > '{last_evaluation_ts}'").toPandas(),
                values=ms_spec.name,
                index="time",
                columns=ms_spec.dimension_groupings,
                fill_value=0,
            )
        except Exception as e:
            print("Error encountered, skip remaining inference and retrain")
            print(ms_spec)
            print(e)
            retrain_flag = True
        else:
            if len(new_obs) == 0:
                continue
            # load the model
            loaded_model = mlflow.pyfunc.load_model(model_version.source)
            # perform inference
            new_anomalies = loaded_model.predict(new_obs)
            for a in new_anomalies:
                a.update({"datasource": "metric_store", "query_json": model_version.tags["query"]})
            anomalies.extend(new_anomalies)
            # log the updated model to the run
            mlflow_mm.update_online_model(
                registered_model=registered_model,
                updated_model=loaded_model,
                tags={**model_version.tags, "last_evaluation_ts": str(new_obs.index[-1])},
            )
    shared_metadata = {
        "detection_run_timestamp": current_time.to_pydatetime(),
        "detector": detector_cfg.name,
        "run_id": run.info.run_id,
    }
    anomalies = [Row(**a, **shared_metadata) for a in anomalies]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log anomalies

# COMMAND ----------


def log_anomalies(anomalies_list):
    try:
        (
            DeltaTable.createIfNotExists(spark)
            .tableName(RAW_ANOMALIES_TBL)
            .addColumn(
                "detection_run_timestamp",
                "TIMESTAMP",
                comment="timestamp of the eval job run that generated this alert",
            )
            .addColumn("detector", "STRING", comment="detector ID which generated this alert")
            .addColumn("run_id", "STRING", comment="MLFlow run ID of the inference job which generated this alert")
            .addColumn("datasource", "STRING", comment="source of the data, e.g. metric_store")
            .addColumn("query_json", "STRING", comment="JSON string for querying the datasource")
            .addColumn("entity", "STRING", comment="specific aggregated entity the alert maps to, e.g. us-west-1")
            .addColumn("algorithm_name", "STRING", comment="algorithm which generated this alert")
            .addColumn("timestamp", "TIMESTAMP", comment="timestamp of this alert")
            .addColumn("value", "DOUBLE", comment="observed value for this entity, at this timestamp")
            .addColumn("metadata", "STRING", comment="JSON object with additional algorithm-specific metadata")
            .addColumn("tags", "MAP<STRING, STRING>", comment="string key-value pairs which can be used for filtering")
            .property("description", "preprocessed anomaly alert data")
            .execute()
        )
        spark.createDataFrame(anomalies_list).write.format("delta").mode("append").saveAsTable(RAW_ANOMALIES_TBL)
    except ValueError:
        print(anomalies_list)
        raise
    except TypeError:
        print(anomalies_list)
        raise
    return


# COMMAND ----------

if len(anomalies) > 0:
    log_anomalies(anomalies)
else:
    print("No anomalies detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform training, if necessary

# COMMAND ----------

# note that retrain_flag gets computed above, during the inference step
if retrain_flag:
    with mlflow.start_run(run_name=f"anomaly_detection_training_{detector_cfg.name}:{deployment_mode}") as run:
        mlflow_mm = MLFlowModelManager(run, detector_cfg.name, deployment_mode=deployment_mode)
        algorithms = detector_cfg.algorithms
        for algo_name, algo_spec in algorithms.items():
            mlflow.log_param(algo_name, algo_spec)
        mlflow.set_tag("detector_name", detector_cfg.name)
        mlflow.set_tag("deployment_mode", deployment_mode)
        for metric in detector_cfg.metric_store_metrics:
            train_ts = TimeIndexedData.from_pandas(
                spark.sql(metric.query(start, end))
                .toPandas()
                .dropna(subset=metric.dimension_groupings)
                .sort_values("time"),
                value_col=metric.name,
                time_col="time",
                group_by=metric.dimension_groupings,
            ).fill_values(0)
            if metric.dimension_groupings:
                # TimeIndexedData.from_pandas() introduces an additional level to the column names, which is the metric name
                # In our case the metric name is redundant, and leads to naming conflicts with pandas.pivot_table().columns
                # So, unless dimension_groupings is None or an empty list, we drop the metric name for consistency
                train_ts = train_ts.drop_group_level(level=0)
            print(f"training models for {metric.name}, aggregated at {metric.dimension_groupings}...")
            mlflow.log_dict(
                train_ts.to_json(), f"data/metric_store/{metric.name}/{'__'.join(metric.dimension_groupings or [])}/"
            )
            model = MLFlowAnomalyDetectorWrapper(algorithms)
            model.train(train_ts)
            mlflow_mm.register_trained_model(model=model, metric_store_spec=metric)
            mlflow_mm.register_new_online_model(train_ts, metric_store_spec=metric)

        # at the end of the run let's clean up the registered models that are no longer needed
        mlflow_mm.clean_up_registered_models()
        mlflow_mm.clean_up_runs(f"anomaly_detection_training_{detector_cfg.name}", 15)
