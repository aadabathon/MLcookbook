"""Versioned experiment files: JSON settings/results plus a Parquet data snapshot.

Files contain data, not executable Python or pickled models. Source paths are
informational; opening an experiment never rereads its original CSV/Parquet.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import to_jsonable

FORMAT = "mlcookbook-experiment"
FORMAT_VERSION = 1


def runtime_versions() -> dict[str, str]:
    import numpy
    import pyarrow
    import scipy
    import sklearn

    from mlcookbook import __version__

    # Module versions also work in the frozen app, without distribution metadata.
    return {"python": platform.python_version(), "mlcookbook": __version__, "pandas": pd.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__}


@dataclass(frozen=True)
class LoadedExperiment:
    dataset: Dataset
    recipe: str
    name: str
    state: dict
    source: dict
    versions: dict[str, str]
    fingerprint: str
    warnings: list[str]


def save_experiment(path: str | Path, dataset: Dataset, *, recipe: str, name: str,
                    state: dict, source: dict | None = None) -> str:
    """Atomically save one recipe, its UI state/report, and the complete dataset.

    Returns the SHA-256 fingerprint of the stored Parquet snapshot. An existing
    experiment is replaced only once the entire new archive has been written.
    """
    path = Path(path)
    snapshot = dataset.df.to_parquet(index=True, engine="pyarrow")
    fingerprint = hashlib.sha256(snapshot).hexdigest()
    manifest = {
        "format": FORMAT, "version": FORMAT_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe": recipe, "name": name, "state": state, "source": source or {},
        "roles": {col.name: col.var_type.value for col in dataset.schema().columns},
        "versions": runtime_versions(), "snapshot_sha256": fingerprint,
    }
    metadata = json.dumps(to_jsonable(manifest), ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("experiment.json", metadata)
                archive.writestr("data.parquet", snapshot)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return fingerprint


def load_experiment(path: str | Path) -> LoadedExperiment:
    """Read and verify an experiment without fitting anything or reading source paths."""
    try:
        with zipfile.ZipFile(path) as archive:
            if sorted(archive.namelist()) != ["data.parquet", "experiment.json"]:
                raise ValueError("Expected an experiment manifest and one dataset snapshot.")
            manifest = json.loads(archive.read("experiment.json"))
            if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
                raise ValueError("This is not an MLcookbook experiment.")
            if manifest.get("version") != FORMAT_VERSION:
                raise ValueError("This experiment uses an unsupported format version.")
            for key in ("roles", "state", "source", "versions"):
                if not isinstance(manifest.get(key), dict):
                    raise ValueError(f"Experiment {key} must be an object.")
            for key in ("name", "recipe", "snapshot_sha256"):
                if not isinstance(manifest.get(key), str) or not manifest[key]:
                    raise ValueError(f"Experiment {key} is missing or invalid.")
            snapshot = archive.read("data.parquet")
        fingerprint = hashlib.sha256(snapshot).hexdigest()
        if fingerprint != manifest["snapshot_sha256"]:
            raise ValueError("The dataset snapshot fingerprint does not match; the experiment may be damaged.")
        frame = pd.read_parquet(io.BytesIO(snapshot), engine="pyarrow")
        if set(manifest["roles"]) != set(frame.columns):
            raise ValueError("Experiment column roles do not match its dataset.")
        dataset = Dataset(frame, overrides=manifest["roles"])
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeError, OSError) as exc:
        raise ValueError(f"Could not open the experiment: {exc}") from exc
    current = runtime_versions()
    warnings = [f"Saved with {name} {saved}; installed version is {current.get(name, 'unavailable')}. "
                "Rerun results may differ."
                for name, saved in manifest["versions"].items() if saved != current.get(name)]
    report = manifest["state"].get("report", {})
    if isinstance(report, dict) and isinstance(report.get("provenance"), dict):
        run_versions = report["provenance"].get("versions", {})
        if isinstance(run_versions, dict) and run_versions != manifest["versions"]:
            warnings.extend(f"Displayed result used {name} {saved}; installed version is {current.get(name, 'unavailable')}."
                            for name, saved in run_versions.items() if saved != current.get(name))
    return LoadedExperiment(dataset, manifest["recipe"], manifest["name"], manifest["state"],
                            manifest["source"], manifest["versions"], fingerprint, warnings)
