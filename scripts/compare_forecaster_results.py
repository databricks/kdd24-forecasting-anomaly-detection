import argparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wilcoxon


def main(model1_path, model2_path, output, seed):
    np.random.seed(seed)
    df1 = pd.read_parquet(model1_path)
    df2 = pd.read_parquet(model2_path)

    assert set(df1.key.values) == set(df2.key.values)
    assert df1.columns[0] == "key"
    metrics = df1.columns[1:]

    df = df2.set_index("key").join(df1.set_index("key"), lsuffix="_2", rsuffix="_1")

    results = []

    for m in metrics:
        if m == "graph":
            continue
        lvals = df[f"{m}_1"].values
        rvals = df[f"{m}_2"].values

        lmean = lvals.mean()
        rmean = rvals.mean()
        lqs = np.percentile(lvals, [10.0, 50.0, 90.0])
        rqs = np.percentile(rvals, [10.0, 50.0, 90.0])
        rel_diff_mean = 100 * np.ma.masked_invalid((rvals - lvals) / np.abs(lvals)).mean()
        wtest = wilcoxon(lvals, rvals)
        ktest = ks_2samp(lvals, rvals)

        results.append(
            {
                "metric": m,
                "lmean": lmean,
                "rmean": rmean,
                "rel_diff_pct": rel_diff_mean,
                "significant_w": int(wtest[1] < 0.05),
                "significant_k": int(ktest[1] < 0.05),
                "wilcoxon_pval": wtest[1],
                "ks_pval": ktest[1],
                "lp50": lqs[1],
                "rp50": rqs[1],
                "lp10": lqs[0],
                "rp10": rqs[0],
                "lp90": lqs[2],
                "rp90": rqs[2],
            }
        )

    results = pd.DataFrame(results)
    results.set_index("metric", inplace=True)
    # print(results.style.format(precision=3).to_string())
    print(results)
    if output is not None:
        results.to_parquet(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark", description="Compare two benchmarking results across all the datasets."
    )
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    args = parser.parse_args()
    main(args.model1, args.model2, args.output, args.seed)
