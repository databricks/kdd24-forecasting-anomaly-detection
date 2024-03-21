import importlib
from typing import Any, Dict, Hashable, List, Optional, Tuple, Type

import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.pyfunc import PyFuncModel, PyFuncOutput, PythonModel
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.client import MlflowClient

from ..algorithms.common import (
    MultivariateOutlierDetectionAdapter,
    MultivariateOutlierDetectionAlgorithm,
    UnivariateOutlierDetectionAlgorithm,
)
from ..types import (
    AlgorithmSpec,
    AnomalyDetectionModel,
    MetricStoreMetricSpec,
)
from .algorithms import get_algorithm
from ...uff_example.tstypes import TimeIndexedData
from ...uff_example.utils import get_estimator


class MLFlowModelManager:
    def __init__(
        self,
        run: Run,
        detector_name: str,
        deployment_mode: str = "dev",
        online_model_version_retention: int = 5,
    ) -> None:
        self.run = run
        self.detector_name = detector_name
        self.deployment_mode = deployment_mode.lower()
        self.online_model_version_retention = online_model_version_retention
        self.new_registered_online_models = set()

    def register_trained_model(self, model: PythonModel, metric_store_spec: Optional[MetricStoreMetricSpec]) -> None:
        """Registers a trained PythonModel model produced during an anomaly detection training run
        If the model corresponds to a metric in Metric Store, `metric_name` and `dim_name` need to be specified

        Parameters
        ----------
        model : PythonModel
            A model object which contains the trained anomaly detection algorithm(s) for the target metric group
        metric_store_spec : Optional[MetricStoreMetricSpec]
            A Metric Store metric spec
        """
        model_uri, model_name, _ = self._construct_model_identifiers(metric_store_spec)
        mlflow.pyfunc.log_model(python_model=model, artifact_path=model_uri, registered_model_name=model_name)

    def register_new_online_model(
        self, training_data: TimeIndexedData, metric_store_spec: Optional[MetricStoreMetricSpec]
    ) -> None:
        """Registers a new online model during an anomaly detection training run
        If an online model with the same name already exists, the last evaluated timestamp is retrieved from the tags
            and the existing online model is deleted, before creating the new one
        If this is the first time creating the online model, the last evaluated timestamp is taken from the end of the
            time series used for training

        Parameters
        ----------
        training_data : TimeIndexedData
            Time series data used to train the model, used to extract the last evaluated timestamp if needed
        metric_store_spec : Optional[MetricStoreMetricSpec]
            A Metric Store metric spec
        """
        client = MlflowClient()
        model_uri, model_name, datasource = self._construct_model_identifiers(metric_store_spec)
        model_name += "_online"
        last_evaluation_ts = self.fetch_last_evaluation_ts(
            client=client, online_model_name=model_name, training_data=training_data
        )
        self.safe_delete_registered_model(client=client, model_name=model_name)
        client.create_registered_model(
            model_name,
            tags={
                "online": "True",
                "detector": self.detector_name,
                "creation_run_id": self.run.info.run_id,
                "deployment_mode": self.deployment_mode,
            },
        )
        model_src = RunsArtifactRepository.get_underlying_uri(f"runs:/{self.run.info.run_id}/{model_uri}")
        client.create_model_version(
            name=model_name,
            source=model_src,
            run_id=self.run.info.run_id,
            tags={
                "last_evaluation_ts": str(last_evaluation_ts),
                "datasource": datasource,
                "query": metric_store_spec.to_json(),
            },
        )
        self.new_registered_online_models.add(model_name)

    def update_online_model(
        self, registered_model: RegisteredModel, updated_model: PyFuncModel, tags: Dict[str, str]
    ) -> None:
        """Updates the online model during an anomaly detection inference run
        A new version is created for the online model, and the corresponding tags are added

        Parameters
        ----------
        registered_model : RegisteredModel
            MLFlow RegisteredModel object to which a new version will be added
        updated_model : PyFuncModel
            Updated model object.  PyFuncModel is the type of the return object for `mlflow.pyfunc.load_model`
        tags : Dict[str, str]
            Tags to be added onto the new model version
        """
        model_uri = f"runs:/{self.run.info.run_id}/{registered_model.name}"
        mlflow.pyfunc.log_model(
            # this is a hack, since `unwrap_python_model` method is still experimental and we can't log the PyFuncModel
            python_model=updated_model._model_impl.python_model,
            artifact_path=model_uri,
            registered_model_name=registered_model.name,
        )
        # set tags on the new model version
        self.set_tags_on_latest_version(registered_model.name, tags)
        self.clean_up_model_versions(registered_model.name)

    def fetch_last_evaluation_ts(
        self,
        client: MlflowClient,
        online_model_name: str,
        training_data: Optional[TimeIndexedData],
    ) -> pd.Timestamp:
        """Retrieves the last timestamp which was evaluated by an online model
        If an online model exists, this information will be in the tags
        If an online model does not exist, the last evaluated timestamp will be taken from the end of the
            time series used for training
        If an online model does not exist and a training time series is not provided, a TypeError is thrown

        Parameters
        ----------
        client : MlflowClient
            An instantiated MLFlow client object
        online_model_name : str
            Registered model name for the online model for which the last evaluated timestamp is requested
        training_data : Optional[TimeIndexedData]
            Optional time series data object, in case the online_model_name is not found

        Returns
        -------
        pd.Timestamp
            The last evaluated timestamp

        Raises
        ------
        TypeError
            If the `online_model_name` is not registered, and no `training_data` is available
        """
        try:
            latest_version = client.get_latest_versions(name=online_model_name)[0]
            last_evaluation_ts = pd.Timestamp(latest_version.tags["last_evaluation_ts"])
        except mlflow.exceptions.RestException:
            if training_data is None:
                raise TypeError(
                    f"No version found for model {online_model_name} and no training data provided, "
                    "unable to determine last evaluation timestamp"
                )
            last_evaluation_ts = training_data.time_index.pd_timestamp_index()[-1]
        return last_evaluation_ts

    def safe_delete_registered_model(self, client: MlflowClient, model_name: str) -> None:
        """Deletes a registered model if it exists, does nothing otherwise

        Parameters
        ----------
        client : MlflowClient
            An instantiated MLFlow client object
        model_name : str
            Registered model name of the model to be deleted
        """
        try:
            client.delete_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass

    def clean_up_registered_models(self) -> None:
        """Intended to be called exactly once after all the online models for a detector have been created in an
            anomaly detection training run
        Retrieves all registered models attributed to the detector, and deletes any registered models which were
            not created during the current training run
        """
        client = MlflowClient()
        model_version_infos = client.search_registered_models(
            "tags.online = 'True' and "
            f"tags.detector = '{self.detector_name}' and tags.deployment_mode = '{self.deployment_mode}'"
        )
        for online_model in model_version_infos:
            if online_model.name not in self.new_registered_online_models:
                client.delete_registered_model(online_model.name)

    def set_tags_on_latest_version(self, registered_model_name: str, tags: Dict[str, str]) -> None:
        """Retrieves the latest version of a registered model, and adds the provided tags

        Parameters
        ----------
        registered_model_name : str
            The registered model for which the latest version is to be retrieved
        tags : Dict[str, str]
            The tags to be added to the latest model version
        """
        client = MlflowClient()
        new_mv = client.get_latest_versions(name=registered_model_name)[0]
        for k, v in tags.items():
            client.set_model_version_tag(name=registered_model_name, version=new_mv.version, key=k, value=v)

    def clean_up_model_versions(self, registered_model_name: str) -> None:
        """If there are more than N model versions for a registered model, retain the N most recent model versions and
            delete the rest.  N is defined in `self.online_model_version_retention`.

        Parameters
        ----------
        registered_model_name : str
            The registered model for which we want to clean up the model versions
        """
        client = MlflowClient()
        latest_versions: List[ModelVersion] = client.get_latest_versions(name=registered_model_name)
        versions_to_delete = latest_versions[min(self.online_model_version_retention, len(latest_versions)) :]
        for v in versions_to_delete:
            mlflow.delete_model_version(registered_model_name, v.version)

    def clean_up_runs(self, run_name: str, num_runs_to_retain: int = 200) -> None:
        """If there are more than N runs for a given MLFlow run name, retain the N most recent runs and delete the
            rest.  N is defined by the optional argument `num_runs_to_retain`.

        Parameters
        ----------
        run_name : str
            Name of the run for which we want to truncate the number of runs
        num_runs_to_retain : int, optional
            Number of runs to retain, by default 200
        """
        sorted_run_ids = (
            mlflow.search_runs(filter_string=f"run_name = '{run_name}'")
            .sort_values("start_time", ascending=False)
            .run_id
        )
        run_ids_to_delete = sorted_run_ids[min(num_runs_to_retain, len(sorted_run_ids)) :]
        for run_id in run_ids_to_delete:
            mlflow.delete_run(run_id)

    def _construct_model_identifiers(self, metric_store_spec: Optional[MetricStoreMetricSpec]) -> Tuple[str, str, str]:
        datasource = ""
        if metric_store_spec:
            datasource = "metric_store"
            model_uri, model_name = self._construct_metric_store_model_identifiers(metric_store_spec)
        else:
            raise TypeError("currently only Metric Store is supported, must provide metric_store_spec")
        return (model_uri, model_name, datasource)

    def _construct_metric_store_model_identifiers(self, metric_store_spec: MetricStoreMetricSpec) -> Tuple[str, str]:
        dim_string = "__".join(metric_store_spec.dimension_groupings or [])
        model_uri = f"runs:/{self.run.info.run_id}/{metric_store_spec.name}/{dim_string}/model"
        model_name = f"{self.detector_name}__{metric_store_spec.name}__{dim_string}"
        if self.deployment_mode != "prod":
            model_name += f"__{self.deployment_mode}"
        return (model_uri, model_name)


class MLFlowAnomalyDetectorWrapper(PythonModel):
    def __init__(self, algorithm_specs: Dict[str, AlgorithmSpec]) -> None:
        self.algorithm_specs = algorithm_specs
        self.models = {}
        for name, spec in self.algorithm_specs.items():
            algo_cls, arg_dict = self.parse_algoritm_spec(spec)
            if issubclass(algo_cls, UnivariateOutlierDetectionAlgorithm):
                self.models[name] = MultivariateOutlierDetectionAdapter(algo_cls, arg_dict)
            elif issubclass(algo_cls, MultivariateOutlierDetectionAlgorithm):
                self.models[name] = algo_cls(**arg_dict)
            else:
                raise ValueError(f"algorithm class {spec.cls} not found or is currently not supported")

    def parse_algoritm_spec(self, spec: AlgorithmSpec) -> Tuple[Type[AnomalyDetectionModel], Dict[str, Any]]:
        return get_algorithm(spec.cls), {
            arg_name: self._resolve_arg(arg_value) for arg_name, arg_value in spec.args.items()
        }

    def _resolve_arg(self, arg_value: Any) -> Any:
        """
        Recursively instantiates classes expressed as a dict
        Currently this is limited to UFF Estimators and Transformers

        Example
            {SomeEstimator: {arg1: 1, arg2: 2, arg3: SomeTransformer}}
        will result in
            SomeEstimator(arg1=1, arg2=2, arg3=SomeTransformer())
        """
        if isinstance(arg_value, dict):
            if arg_value.get("cls") and arg_value.get("args"):
                module_name, class_name = arg_value.get("cls").rsplit(".", 1)
                return getattr(importlib.import_module(module_name), class_name)(**arg_value.get("args"))
            for cls_name, cls_args in arg_value.items():
                return get_estimator(cls_name)(**{k: self._resolve_arg(v) for k, v in cls_args.items()})
        elif isinstance(arg_value, list):
            return [self._resolve_arg(a) for a in arg_value]
        try:
            return get_estimator(arg_value)()
        except (KeyError, TypeError):
            return arg_value

    def train(self, training_data: TimeIndexedData) -> None:
        for _, model in self.models.items():
            model.train(training_data)

    def is_anomaly_info(
        self, model_name: str, model: MultivariateOutlierDetectionAlgorithm, model_input: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        is_anomaly_list: List[Dict[Hashable, bool]] = model.check_is_anomaly(
            list(model_input.index), model_input.to_dict(orient="records")
        )
        return [
            {
                "algorithm_name": model_name,
                "entity": str(k),
                "timestamp": pd.Timestamp(model_input.index[i]).to_pydatetime(),
                "value": float(model_input.iloc[i][k]),
                "metadata": str({}),
            }
            for i, is_anomaly in enumerate(is_anomaly_list)
            for k, v in is_anomaly.items()
            if v
        ]

    def predict(self, context, model_input: pd.DataFrame) -> PyFuncOutput:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("currently only pandas DataFrames are supported for anomaly detection inference")

        return [
            anomaly for name, model in self.models.items() for anomaly in self.is_anomaly_info(name, model, model_input)
        ]
