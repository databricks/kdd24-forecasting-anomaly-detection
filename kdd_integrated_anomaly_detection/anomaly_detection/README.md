# Real-Time Anomaly Detection Example

Folder structure:

```sh
anomaly_detection/
 └───algorithms/                 # Contains algorithm implementations
 │   └───ransyncoders.py         # Implementation of RanSynCoders
 │   └───univariate.py           # Implementation of a baseline detection algorithm
 └───utils/
 │   └───algorithms.py           # convenience utils to enable looking up algorithms by name
 │   └───mlflow.py               # abstractions for interacting with MLFlow
 └───types.py                    # type definitions
```

## Running anomaly detection

`run_detector.py` in the `../notebooks/` directory shows the steps for querying metrics from Metric Store, training the algorithms, storing the trained model artifacts in MLFlow, and retrieving the models and using them for inference
1. Querying metrics from Metric Store
```
metric = detector_cfg.metric_store_metrics

train_ts = TimeIndexedData.from_pandas(
    spark.sql(metric.query(start, end))
    .toPandas()
    .dropna(subset=metric.dimension_groupings)
    .sort_values("time"),
    value_col=metric.name,
    time_col="time",
    group_by=metric.dimension_groupings,
).fill_values(0)
```
2. Training the algorithms and storing them in MLFlow
```
algorithms = detector_cfg.algorithms

model = MLFlowAnomalyDetectorWrapper(algorithms)
model.train(train_ts)
mlflow_mm.register_trained_model(model=model, metric_store_spec=metric)
mlflow_mm.register_new_online_model(train_ts, metric_store_spec=metric)
```
3. Retrieving the models and using them for inference
```
client = MlflowClient()

model_version_infos = client.search_registered_models(
    f"tags.online = 'True' and tags.detector = '{detector_cfg.name}' and tags.deployment_mode = '{deployment_mode}'"
)
for registered_model in model_version_infos:
    model_version = registered_model.latest_versions[0]
    ms_spec = MetricStoreMetricSpec.from_json(model_version.tags["query"])
    last_evaluation_ts = pd.Timestamp(model_version.tags["last_evaluation_ts"]
    new_obs = pd.pivot_table(
        spark.sql(ms_spec.query(last_evaluation_ts, end)).where(f"time > '{last_evaluation_ts}'").toPandas(),
        values=ms_spec.name,
        index="time",
        columns=ms_spec.dimension_groupings,
        fill_value=0,
    )
    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_version.source)
    # perform inference
    new_anomalies = loaded_model.predict(new_obs)
    # update the online model
    mlflow_mm.update_online_model(
        registered_model=registered_model,
        updated_model=loaded_model,
        tags={**model_version.tags, "last_evaluation_ts": str(new_obs.index[-1])},
    )
```
