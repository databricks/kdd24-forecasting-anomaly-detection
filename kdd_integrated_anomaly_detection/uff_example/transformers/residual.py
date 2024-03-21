from __future__ import annotations

import io
import logging
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import cloudpickle

from .. import utils
from ..base import Forecaster, Transformer
from ..tstypes import TimeIndexedData, TimeIndexedOutput

PrimitiveType = Optional[Union[str, int, float, bool]]
ParamDict = Dict[str, PrimitiveType]

logger = logging.getLogger(__name__)


class ResidualsTransformer(Transformer):
    """Compute in-sample residuals from a Forecaster"""

    def __init__(
        self,
        forecaster: Union[Forecaster, Dict[str, Any]],
    ) -> None:
        """Initialize a ResidualsTransformer

        Parameters
        ----------
        forecaster: Union[Forecaster, Dict[str, Any]]
            A Forecaster instance or its equivalent JSON representation
        """
        self.forecaster = forecaster if isinstance(forecaster, Forecaster) else utils.estimator_from_json(forecaster)

    def transform(self, data: TimeIndexedData, **kwargs: Dict[str, Any]) -> TimeIndexedOutput:
        in_sample = deepcopy(self.forecaster).fit(data).forecast(data.time_index).out
        return TimeIndexedOutput(data - in_sample)

    def save(self, path: str) -> None:
        """Persist this estimator at the location specified by `path`

        Parameters
        ----------
        path: str
            A path to the serialized Estimator's output destination.
        """
        tempdir = TemporaryDirectory()
        prefix = Path(tempdir.name)

        # Save attributes to a temp directory
        json_data = {"forecaster_type": type(self.forecaster)}
        with (Path(prefix) / "json_data").open("wb") as f:
            cloudpickle.dump(json_data, f)
        self.forecaster.save(prefix / "forecaster")

        # Create an archive from the temp directory
        archive_dir = TemporaryDirectory()
        archive_name = str(Path(archive_dir.name) / "archive.zip")
        with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in prefix.rglob("*"):
                zip_file.write(entry, entry.relative_to(prefix))

        # Place the archive in the user provided path
        shutil.move(archive_name, path)

    @classmethod
    def load(cls, path: str) -> ResidualsTransformer:
        """Load a previously saved Estimator instance from `path`

        Parameters
        ----------
        path: str
            A path to the serialized Estimator
        """
        # Create a temporary directory to unpack the archive
        unpack_dir = TemporaryDirectory()
        unpack_path = Path(unpack_dir.name)

        # Unpack the archive
        with open(path, "rb") as f:
            buf = io.BytesIO(f.read())
            with zipfile.ZipFile(buf) as zipf:
                zipf.extractall(unpack_path)

        with (unpack_path / "json_data").open("rb") as f:
            json_data = cloudpickle.load(f)

        model_class = json_data["forecaster_type"]
        forecaster = model_class.load(unpack_path / "forecaster")
        return cls(forecaster)
