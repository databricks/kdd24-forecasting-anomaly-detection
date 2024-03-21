# Databricks notebook source
# MAGIC %pip install tadpak tqdm

# COMMAND ----------

import numpy as np
import pandas as pd
import pickle
from joblib import Parallel,delayed 
from tadpak import evaluate as pak
import sklearn

from kdd_integrated_anomaly_detection.anomaly_detection.evaluation import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Data

# COMMAND ----------

def load_inference(path: str) -> np.array:
    with open(path, "rb") as f:
        return pickle.load(f)

# COMMAND ----------

inference = {
    "dspot_swat": load_inference("../data/dspot_score_swat.pkl"),
    "dspot_wadi": load_inference("../data/dspot_score_wadi.pkl"),
    "interfusion_swat": load_inference("../data/interfusion_score_swat.pkl"),
    "interfusion_wadi": load_inference("../data/interfusion_score_wadi.pkl"),
    "ransynccoders_swat": load_inference("../data/ransynccoders_score_swat.pkl"),
    "ransynccoders_tuned_swat": load_inference("../data/ransynccoders_tuned_score_swat.pkl"),
    "ransynccoders_wadi": load_inference("../data/ransynccoders_score_wadi.pkl"),
}

# COMMAND ----------

for k, v in inference.items():
    print(f"{k}: {v.shape}")

# COMMAND ----------

with open("../data/attack_labels.pkl", "rb") as _f:
    labels = pickle.load(_f)

# COMMAND ----------

for k, v in labels.items():
    print(f"{k}: {v.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare Results

# COMMAND ----------

# non-rolling window size
tw_range = range(5, 21)

# PA%k window size
k_range = range(20, 120, 20)

# COMMAND ----------

def tranform_dspot_score(arr: np.array) -> np.array:
    """transform dspot score into (0, 1)"""

    # Calculate min and max for each feature (assuming features are columns)
    feature_min = np.min(arr, axis=0)
    feature_max = np.max(arr, axis=0)

    # Handle the case where the feature has zero range
    range_zero_mask = (feature_max - feature_min) == 0
    feature_range = np.where(range_zero_mask, 1, feature_max - feature_min)

    # Apply the min-max scaling transformation
    scaled_arr = (arr - feature_min) / feature_range

    return scaled_arr

def tranform_interfusion_score(arr: np.array) -> np.array:
    """transform interfusion score into (0, 1)"""

    # Calculate min and max for each feature (assuming features are columns)
    feature_min = np.min(arr, axis=0)
    feature_max = np.max(arr, axis=0)

    # Handle the case where the feature has zero range
    range_zero_mask = (feature_max - feature_min) == 0
    feature_range = np.where(range_zero_mask, 1, feature_max - feature_min)

    # Apply the min-max scaling transformation
    scaled_arr = (arr - feature_min) / feature_range

    return -scaled_arr + 1

# COMMAND ----------

def eval_one(name: str, scores: np.array, *, metric: str, tw: int=None, k: int=None):
    dataset_name = name.split("_")[-1]
    model_name = "_".join(name.split("_")[:-1])
    attack_labels = labels[dataset_name]
    attack_labels = attack_labels[-len(scores):]
    if model_name.startswith("interfusion"):
        scores = tranform_interfusion_score(scores)
    if model_name.startswith("dspot"):
        scores = tranform_dspot_score(scores)
    
    res = {
        "model": model_name,
        "dataset": dataset_name,
        "metric_name": metric,
    }

    if metric == "f1_pa":
        best_f1, _ = get_adjusted_f1(scores, attack_labels)
        res.update({
            "metric_value": best_f1,
            "metric_param": {}
        })
    elif metric == "f1_pak" and k:
        jitter_scale = 0.1
        # prevent computation error in tadpak
        noise = np.random.uniform(low=-jitter_scale, high=jitter_scale, size=scores.shape[0])
        scores = scores + noise * scores
        f1_pak = pak.evaluate(scores, attack_labels, k=k)
        res.update({
            "metric_value": f1_pak["best_f1_w_pa"],
            "metric_param": {"k": k}
        })
    return res

# COMMAND ----------

calls = []
for name, scores in inference.items():
    calls.append(delayed(eval_one)(name, scores, metric="f1_pa"))
    for k in k_range:
        calls.append(delayed(eval_one)(name, scores, metric="f1_pak", k=k))
res = Parallel(n_jobs=-1)(calls)

# COMMAND ----------

data = (
    pd.DataFrame(res, columns=["model", "dataset", "metric_name", "metric_value", "metric_param"])
    .sort_values(["dataset", "model", "metric_name"])
    .groupby(["dataset", "model", "metric_name"]).agg(
        metric_value = ("metric_value", "first"),
        metric_param = ("metric_param", "first"),
    )
)
data

# COMMAND ----------


