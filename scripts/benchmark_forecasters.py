# Databricks notebook source

import argparse
import base64
import json
import os
import pickle
import time
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kdd_integrated_anomaly_detection.uff_example.forecasters.miniprophet import MiniProphetForecaster
from kdd_integrated_anomaly_detection.uff_example.forecasters.prophet import ProphetForecaster


def smape(actual, prediction) -> float:
    # TODO: use what's in uff/evaluation/metrics
    return 2 * np.mean(np.ma.masked_invalid(np.abs(actual - prediction) / (np.abs(actual) + np.abs(prediction))))


def gen_plot_estimates(dataset_name, data_train, data_test, pred_train, pred_test) -> Tuple[plt.Figure, plt.Axes]:
    x_train = data_train.pd_timestamp_index()
    y_train = data_train.values
    x_test = data_test.pd_timestamp_index()
    y_test = data_test.values

    plt.clf()

    plt.plot(x_train, y_train, label="train", color="dodgerblue")
    plt.scatter(x_test, y_test, label="test", color="lightseagreen", s=6)

    plt.plot(x_train, pred_train.out.values, color="orangered")
    dup = np.clip(pred_train.upper.values - pred_train.out.values, a_min=0, a_max=None) + pred_train.out.values
    dlow = pred_train.out.values - np.clip(pred_train.out.values - pred_train.lower.values, a_min=0, a_max=None)
    plt.fill_between(x_train, dlow, dup, alpha=0.2, color="gold")

    plt.plot(x_test, pred_test.out.values, color="orangered", lw=1)
    dup = np.clip(pred_test.upper.values - pred_test.out.values, a_min=0, a_max=None) + pred_test.out.values
    dlow = pred_test.out.values - np.clip(pred_test.out.values - pred_test.lower.values, a_min=0, a_max=None)
    plt.fill_between(x_test, dlow, dup, alpha=0.2, color="gold")

    plt.legend()
    plt.title(f"{dataset_name} dataset")
    return plt.gcf(), plt.gca()


def eval_dataset(dataset_name, data_train, data_test, with_graph, estimator_instantiation):
    start_time = time.time()
    est = estimator_instantiation().fit(data_train)
    training_time = time.time() - start_time
    start_time = time.time()
    pred_train = est.forecast(data_train.time_index)
    pred_test = est.forecast(data_train.future_time_index(len(data_test.values)))
    predict_time = time.time() - start_time

    y_train = data_train.values
    y_test = data_test.values
    # TODO: use what's in uff/evaluation/metrics

    # 1) MAE
    mae_train = np.abs(y_train - pred_train.out.values).mean()
    mae_test = np.abs(y_test - pred_test.out.values).mean()

    # 2) SMAPE
    smape_train = smape(y_train, pred_train.out.values)
    smape_test = smape(y_test, pred_test.out.values)

    # 3) Average size of the predicted confidence (CI) interval
    ci_sz_train = np.clip(pred_train.upper.values - pred_train.lower.values, a_min=0, a_max=None).mean()
    ci_sz_test = np.clip(pred_test.upper.values - pred_test.lower.values, a_min=0, a_max=None).mean()

    # 4) Average fraction of ground truth data points falling inside the predicted CI
    coverage_train = (
        (y_train >= pred_train.lower.values).astype(int) + (y_train <= pred_train.upper.values).astype(int) == 2
    ).mean()
    coverage_test = (
        (y_test >= pred_test.lower.values).astype(int) + (y_test <= pred_test.upper.values).astype(int) == 2
    ).mean()

    # 5) Average fraction of predicted point estimates falling inside the predicted CI
    ci_ordered_train = (
        (pred_train.out.values >= pred_train.lower.values).astype(int)
        + (pred_train.out.values <= pred_train.upper.values).astype(int)
        == 2
    ).mean()
    ci_ordered_test = (
        (pred_test.out.values >= pred_test.lower.values).astype(int)
        + (pred_test.out.values <= pred_test.upper.values).astype(int)
        == 2
    ).mean()

    # 6) Average fraction of ground truth data points above the predicted higher quantile
    calib_high_train = (y_train > pred_train.upper.values).astype(int).mean()
    calib_low_train = (y_train < pred_train.lower.values).astype(int).mean()

    # 7) Average fraction of ground truth data points below the predicted lower quantile
    calib_high_test = (y_test > pred_test.upper.values).astype(int).mean()
    calib_low_test = (y_test < pred_test.lower.values).astype(int).mean()

    res = {
        "key": dataset_name,
        "training_time": training_time,
        "predict_time": predict_time,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "smape_train": smape_train,
        "smape_test": smape_test,
        "ci_sz_train": ci_sz_train,
        "ci_sz_test": ci_sz_test,
        "ci_cov_train": coverage_train,
        "ci_cov_test": coverage_test,
        "ci_ordered_train": ci_ordered_train,
        "ci_ordered_test": ci_ordered_test,
        "calib_h_train": calib_high_train,
        "calib_l_train": calib_low_train,
        "calib_h_test": calib_high_test,
        "calib_l_test": calib_low_test,
    }
    if with_graph:
        fig, _ = gen_plot_estimates(dataset_name, data_train, data_test, pred_train, pred_test)
        fname = f"tmp_{dataset_name}.png"
        fig.savefig(fname, format="png")
        content = base64.b64encode(open(fname, "rb").read())
        res["graph"] = content
        os.remove(fname)
    return res


def main(estimator, kwargs, datapath, output, max_datasets, with_graph, seed):
    np.random.seed(seed)
    data_dict = pickle.load(open(datapath, "rb"))
    n = len(data_dict)
    num_eval = n if max_datasets is None else min(n, max_datasets)
    print(f"Loaded {n} datasets from {datapath}. Running evaluation on {num_eval} of them.")
    all_datasets = list(data_dict.items())
    np.random.shuffle(all_datasets)
    all_datasets = all_datasets[:num_eval]
    print(f"{estimator} estimator will be instantiated with arguments: {kwargs}")
    EstClass = eval(estimator)
    tasks = [
        partial(
            eval_dataset,
            dataset_name,
            data_train,
            data_test,
            with_graph,
            lambda: EstClass(**json.loads(kwargs)),
        )
        for dataset_name, (data_train, data_test) in all_datasets
    ]

    results = pd.DataFrame([task() for task in tasks])
    print(results)
    if output is not None:
        results.to_parquet(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Benchmark an UFF estimator against a pickled dataset of timeseries.",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        required=False,
        default="MiniProphetForecaster",
        help="class name of the estimator to use (default to MiniProphetForecaster)",
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        required=True,
        help="python string of estimator kwargs. Examples: `{}` or for miniprophet: `{'weekly_order': 3}`",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Should point to the pickled result from uff.data.download_m4_dataset",
    )
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        required=False,
        help="Maximum number of timeseries to consider in the input dataset",
    )
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument(
        "--with_graph",
        type=int,
        default=0,
        required=False,
        help="Whether or not to generate a plot of the timeseries and predictions and b64 serialize it with the results",
    )
    args = parser.parse_args()
    assert args.with_graph in [0, 1]
    try:
        eval(args.estimator)
    except Exception:
        raise Exception(f"Unknown estimator `{args.estimator}`")
    main(args.estimator, args.kwargs, args.datapath, args.output, args.max_datasets, args.with_graph, args.seed)
