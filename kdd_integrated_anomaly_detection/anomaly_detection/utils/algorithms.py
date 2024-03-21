from __future__ import annotations

import inspect
import pkgutil
from importlib import import_module
from typing import List, Optional, Tuple, Type, Union, overload

import numpy as np

from ..algorithms.common import (
    UnivariateOutlierDetectionAlgorithm,
)

AnomalyDetectionAlgorithm = UnivariateOutlierDetectionAlgorithm


class _AlgorithmLookup:
    _instance: Optional[_AlgorithmLookup] = None

    def __new__(cls) -> _AlgorithmLookup:
        if cls._instance is None:
            cls._instance = object.__new__(cls)

            # Initialization
            algorithms = set()
            for subdir in ("algorithms",):
                for _, module_name, _ in pkgutil.walk_packages(
                    path=[str(anomaly_detection_root() / subdir)],
                    prefix=f"app.anomaly_detection.{subdir}.",
                ):
                    algorithms |= {
                        cls
                        for name, cls in _get_classes(module_name)
                        if not name.startswith("_") and "base" not in name.lower()
                    }

            cls._instance.all_algorithms = tuple(algorithms)
            cls._instance._algorithms_dict = {e.__name__: e for e in cls._instance.all_algorithms}

        return cls._instance

    def get(self, name: str) -> Type[AnomalyDetectionAlgorithm]:
        return self._algorithms_dict[name]


def all_algorithms(
    subclasses: Union[Type, Tuple[Type, ...]] = AnomalyDetectionAlgorithm
) -> Tuple[Type[AnomalyDetectionAlgorithm]]:
    """Return a tuple of all anomaly detection algorithms matching the optional subclasses
    Parameters
    ----------
    subclasses: Union[Type, Tuple[Type, ...]], default AnomalyDetectionAlgorithm
        The subclass(es) that should be used to filter the returned estimators. By default all non-ABC subclasses of
        `AnomalyDetectionAlgorithm` are returned.
    Returns
    -------
    Tuple[Type[AnomalyDetectionAlgorithm]]
        A tuple of AnomalyDetectionAlgorithm classes
    """
    return tuple(e for e in _AlgorithmLookup().all_algorithms if np.issubclass_(e, subclasses))


@overload
def get_algorithm(name: str) -> Type[AnomalyDetectionAlgorithm]:
    ...


@overload
def get_algorithm(name: List[str]) -> List[Type[AnomalyDetectionAlgorithm]]:
    ...


def get_algorithm(
    name: Union[str, List[str]]
) -> Union[Type[AnomalyDetectionAlgorithm], List[Type[AnomalyDetectionAlgorithm]]]:
    """Fetch an AnomalyDetectionAlgorithm subclass by its classname
    Parameters
    ----------
    name: Union[str, List[str]]
        The name(s) of the desired algorithm(s)
    Returns
    -------
    Union[Type[AnomalyDetectionAlgorithm], List[Type[AnomalyDetectionAlgorithm]]]
        The algorithm(s) subclass matching the specified name
    Raises
    ------
    KeyError
        If the algorithm is not found
    """
    singleton = _AlgorithmLookup()
    if isinstance(name, str):
        return singleton.get(name)
    else:
        return [singleton.get(n) for n in name]


def _get_classes(module_name: str) -> List[Tuple[str, Type]]:
    """Import a module by name and get all member classes, ignore abstract classes
    Parameters
    ----------
    module_name: str
        The name of the module to import
    Returns
    -------
    List[Tuple[str, Type]]
        A list of names (str) and class objects (type) discovered in the module
    """
    try:
        module = import_module(module_name)
        return inspect.getmembers(module, predicate=lambda m: inspect.isclass(m) and not inspect.isabstract(m))
    except ModuleNotFoundError:
        return []
