# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf

import uff
from uff.transformers import RanSynCodersTransformer, RanSynCodersOutput

import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------

def min_max_scale(normal: pd.DataFrame, attack: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    min_max_scaler = preprocessing.MinMaxScaler()
    
    normal_values = normal.iloc[:, 1:].values
    attack_values = attack.iloc[:, 1:-1].values
    
    normal_values_scaled = min_max_scaler.fit_transform(normal_values)
    attack_values_scaled = min_max_scaler.transform(attack_values)
    
    normal_scaled = pd.DataFrame(normal_values_scaled, columns=normal.columns[1:])
    attack_scaled = pd.DataFrame(attack_values_scaled, columns=attack.columns[1:-1])
    
    normal_scaled["timestamp"] = normal["timestamp"]
    attack_scaled["timestamp"] = attack["timestamp"]
    attack_scaled["is_attack"] = attack["is_attack"]
    return normal_scaled, attack_scaled


def best_f1_score(inference: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = inference.min()
    best_f1 = 0
    for threshold in np.arange(inference.min(), inference.max(), step=0.05):
        anomalies = np.where(inference > threshold, 1, 0)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, anomalies, average="binary")
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    return best_f1, best_threshold


def anomaly_inference(output: RanSynCodersOutput) -> np.ndarray:
    out = output.out.values
    reshaped = out.reshape(out.shape[0], out.shape[1], 1)
    result = np.where((reshaped < output.lower_estimates.values) | (reshaped > output.upper_estimates.values), 1, 0)
    inference = np.mean(np.mean(result, axis=2), axis=1)
    return inference


# Visualize model output now in original representation 
def visualize(output: RanSynCodersOutput, label_ts: uff.TimeIndexedData, *, threshold: float, window_size: int = -1):
    time_window = window_size
    plt.figure(figsize=(18,60))
    I = output.out.values.shape[1]
    time_index = output.out.int_time_index()
    anomalies = pd.Series(np.where(anomaly_inference(output) > threshold, 1, 0), index=time_index)
    labels = pd.Series(label_ts.values, index=time_index)
    for i in range(I):
        sample_i = pd.Series(output.out.values[:, i], index=time_index)[:time_window]
        upper_i = pd.Series(np.mean(output.upper_estimates.values[:,i,:], axis=1), index=time_index)[:time_window]
        lower_i = pd.Series(np.mean(output.lower_estimates.values[:,i,:], axis=1), index=time_index)[:time_window]
        plt.subplot(I,1,i+1)
        plt.plot(sample_i, "g", label="TS")
        plt.plot(sample_i.loc[(anomalies==1) & (labels==1)], "r.", label="TP")
        plt.plot(sample_i.loc[(anomalies==1) & (labels==0)], "k.", label="FP")
        plt.plot(sample_i.loc[(anomalies==0) & (labels==1)], "b.", label="FN")
        plt.fill_between(sample_i.index, lower_i, upper_i, color='0.8', label='CI')
        plt.legend(loc="center right")
        plt.grid()


def f1_loss(output: RanSynCodersOutput, *, y: uff.TimeIndexedData) -> float:
    inference = anomaly_inference(output)
    best_f1, _ = best_f1_score(inference, y.values)
    return -best_f1 # passed to fmin


def quantile_loss(q, y, f):
    import tensorflow.keras.backend as K
    e = y - f
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


def decoder_quantile_loss(output: RanSynCodersOutput, *, y: uff.TimeIndexedData, n_estimators: int, delta: float) -> float:
    """Quantile loss (hi + lo)"""
    hi_loss = quantile_loss(
        1-delta, 
        tf.tile(tf.expand_dims(y.values, axis=-1), (1, 1, n_estimators)),
        output.upper_estimates.values
    )

    lo_loss = quantile_loss(
        delta, 
        tf.tile(tf.expand_dims(y.values, axis=-1), (1, 1, n_estimators)),
        output.lower_estimates.values
    )

    return tf.reduce_mean(hi_loss + lo_loss)


# COMMAND ----------


