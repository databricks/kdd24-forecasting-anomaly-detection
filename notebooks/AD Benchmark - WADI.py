# Databricks notebook source
# MAGIC %run ./evaluation

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

wadi_normal = table("main.data_infra_service_health.wadi_14_days").toPandas().drop(["is_attack"], axis=1)
wadi_normal.shape

# COMMAND ----------

wadi_attack = table("main.data_infra_service_health.wadi_attackdata").toPandas()
wadi_attack.shape

# COMMAND ----------

normal, attack = min_max_scale(wadi_normal, wadi_attack)

# COMMAND ----------

delta = pd.Timedelta(1, "s")
normal_ts = uff.TimeIndexedData.from_pandas(
    normal, time_col="timestamp", value_col=list(normal.columns[:-1]), granularity=delta
)
attack_ts = uff.TimeIndexedData.from_pandas(
    attack, time_col="timestamp", value_col=list(attack.columns[:-2]), granularity=delta
)
label_ts = uff.TimeIndexedData.from_pandas(
    attack, time_col="timestamp", value_col=list(attack.columns[-1:]), granularity=delta
)

# COMMAND ----------

# MAGIC %md
# MAGIC Use the same hyperparameters from [Original paper](https://drive.google.com/file/d/11EEzSGOhRXE76239E8aXnfNjKdRUBJd6/view) (**A.5 Hyperparameters and settings**):
# MAGIC * Batch size: 180
# MAGIC * Pre-training epochs: 5
# MAGIC * Training epochs: 10
# MAGIC * Frequency components: 5
# MAGIC * Esimators (N) and bootstrap sample size: one-third of the input dimension, rounded to nearest 5
# MAGIC * Latent dimension: 0.5N-1
# MAGIC * Hidden layer activation: Relu
# MAGIC * Output layer activation: Sigmoid
# MAGIC * Learning rate (Adam optimizer): 0.001
# MAGIC
# MAGIC See also this [example notebook](https://github.com/eBay/RANSynCoders/blob/main/example.ipynb)

# COMMAND ----------

resample_delta = pd.Timedelta(1, unit="s")

# COMMAND ----------

# downsample
if resample_delta > pd.Timedelta(1, unit="s"):
    normal_ts_sampled = normal_ts.resample(resample_delta, reducer=np.nanmean)
    attack_ts_sampled = attack_ts.resample(resample_delta, reducer=np.nanmean)
    label_ts_sampled = label_ts.resample(resample_delta, reducer=np.nanmax)
else:
    normal_ts_sampled = normal_ts
    attack_ts_sampled = attack_ts
    label_ts_sampled = label_ts

# COMMAND ----------

print(tf.config.list_physical_devices("GPU"))
# tf.debugging.set_log_device_placement(True)

# COMMAND ----------

import random

import numpy as np

random.seed(42)
np.random.seed(42)
N = 5 * round((normal_ts_sampled.shape[1] / 3) / 5)
model = RanSynCodersTransformer(
    n_estimators=N,
    max_features=N,
    encoding_depth=1,
    latent_dim=int((N / 2) - 1),
    decoding_depth=2,
    activation="relu",
    output_activation="sigmoid",
    delta=0.05,
    synchronize=True,
    max_freqs=5,
).fit(normal_ts_sampled, epochs=10, batch_size=180, freq_warmup=5, sin_warmup=5)

output = model.transform(attack_ts_sampled)

# COMMAND ----------

inference = anomaly_inference(output)

# COMMAND ----------

with open("/Volumes/users/juanyan_li/ransynccoder_benchmark/wadi_resample_1s_inference_2.pkl", "wb") as f:
    import cloudpickle

    cloudpickle.dump(inference, f)

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameter Tuning
# MAGIC
# MAGIC **Not used**

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
    num_worker_nodes=ray.util.spark.MAX_NUM_WORKER_NODES,
    num_cpus_per_node=16,
)
# ray.util.spark.shutdown_ray_cluster()

# COMMAND ----------

import tensorflow as tf

# COMMAND ----------

train_ts_sampled, validate_ts_sampled = uff.temporal_split(normal_ts_sampled, 0.9)

# COMMAND ----------

N = 5 * round((normal_ts_sampled.shape[1] / 3) / 5)
delta = 0.05
fit_kwargs = dict(epochs=10, batch_size=180, freq_warmup=5, sin_warmup=5)

# COMMAND ----------

ray.shutdown()

for encoding_depth in (1, 2, 3, 4):
    for decoding_depth in (1, 2, 3, 4):
        for activation in ("relu", "linear"):
            for max_freqs in (1, 2, 3, 4):
                # TODO: Create ray tasks that fit and score the estimator
                model = RanSynCodersTransformer(
                    n_estimators=N,
                    max_features=N,
                    encoding_depth=encoding_depth,
                    latent_dim=int((N / 2) - 1),
                    decoding_depth=decoding_depth,
                    activation=activation,
                    output_activation="sigmoid",
                    delta=delta,
                    synchronize=True,
                    max_freqs=max_freqs,
                )
                model.fit(train_ts_sampled, **fit_kwargs)
                res = model.transform(validate_ts_sampled)
                loss = decoder_quantile_loss(res.result, y=validate_ts_sampled, n_estimators=N, delta=delta)

# COMMAND ----------

print(res.best.params.to_dict())

# COMMAND ----------

trials = []
for trial in res.trials:
    params = trial["spec"].to_dict()["kwargs"]
    trials.append((trial["loss"], trial["success"], params))
trial_df = pd.DataFrame(trials, columns=["loss", "success", "params"])
trial_df.display()

# COMMAND ----------

res.best.save("/Volumes/users/juanyan_li/ransynccoder_benchmark/swat_resample_1s.pkl")

# COMMAND ----------
best_model = RanSynCodersTransformer.load("/Volumes/users/juanyan_li/ransynccoder_benchmark/swat_resample_1s.pkl")
best_model.fit(normal_ts_sampled, **fit_kwargs)
output = best_model.transform(attack_ts_sampled)

inference = anomaly_inference(output)
auroc, ap, f1, _, _, _, _ = get_adjusted_composite_metrics(-inference, label_ts_sampled.values)
print(f"auroc={auroc:.2f},ap={ap:.2f},f1={f1:.2f}")

# COMMAND ----------

with open("/Volumes/users/juanyan_li/ransynccoder_benchmark/swat_resample_1s_inference_tuned.pkl", "wb") as f:
    import cloudpickle

    cloudpickle.dump(inference, f)

# COMMAND ----------

inference = anomaly_inference(output)
best_f1, best_threshold = best_f1_score(inference, label_ts_sampled.values)
print(f"Best: f1={best_f1:.2f};threshold={best_threshold:.2f}")
anomalies = np.where(inference > best_threshold, 1, 0)

# COMMAND ----------

visualize(output, label_ts_sampled, threshold=best_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bottom
