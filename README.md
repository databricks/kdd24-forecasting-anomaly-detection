# kdd24-forecasting-anomaly-detection

This is a companion demo repository for the framework as described in the paper "An Integrated Framework for Forecasting and Anomaly Detection in the Data Lakehouse".  If you are utilizing this work in any way, please cite the following paper as appropriate:
```
Martin Valdez-Vivas, Nicholas Stanisha, Benoit Rostykus, Juanyan Li, and Amit Pahwa. 2024.
An Integrated Framework for Forecasting and Anomaly Detection in the Data Lakehouse.
In Proceedings of KDD ‘24 (KDD’24). ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn (submitted)
```
*Disclaimer: this repository is intended for demonstration purposes only and is not a maintained OSS product.*

## Notebooks

The notebooks directory contains executable scripts to recompute forecasting & anomaly detection benchmarks, and also to demonstrate the execution of detectors and event generators as described in the paper.  The notebooks were run on Databricks using DBR version 13.3.  Notebooks are designed to be run in a Databricks workspace and may reference the global Spark session.
- `AD Benchmark - SWAT.py` and `AD Benchmark - WADI.py` demonstrate how we performed hyperparameter tuning in training a RanSynCoders using Ray.  The SWaT and WADI datasets are not released with this repo by agreement, users who wish to run these notebooks must obtain these datasets from [iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
- `run_benchmark.py` produces the results shown in Table 2 of the paper.  The dissimilarity scores we produced from the different algorithms are provided as pkl files under the `/data/` subdirectory.  We were not able to include an implementation of SPOT in this repo due to license restrictions, and omitted the implementation of Interfusion due to reasons as described in section 4.2 of the paper.
- `run_detector.py` shows the logic implemented in the Detector, demonstrating how the different services described in section 3 of the paper interface with each other.

## kdd_integrated_anomaly_detection

The nested directory contains library code required to run the notebooks. To get started run
```sh
python setup.py install
```
The directories under `/kdd_integrated_anomaly_detection/` contain seperate README files with more details of concepts for Metric Store (section 3.2), Unified Forecasting Framework (section 3.3), and Real-Time Anomaly Detection (section 3.4).  Note that the examples provided in this demo repo represent a select subset of the existing functionality for these codebases, which may be open sourced as separate projects in the future.
