"""Atomic .mlck Kitchen documents; v1 recipe .mlcb files remain independent."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.experiments import runtime_versions
from mlcookbook.kitchen import (
    PORTS,
    KitchenDocument,
    KitchenRecord,
    KitchenRun,
    graph_from_dict,
    graph_order,
)

FORMAT = "mlcookbook-kitchen"
VERSION = 1


def _snapshot_references(graph, run, snapshots):
    for node in graph.nodes:
        if node.kind == "dataset" and node.settings["snapshot"] and node.settings["snapshot"] not in snapshots:
            raise ValueError("A Kitchen node references a missing dataset snapshot.")
    for record in run.records.values():
        if record.payload.get("snapshot_id") not in snapshots:
            raise ValueError("A completed Kitchen result references a missing input snapshot.")


def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate Kitchen manifest field: {key}.")
        obj[key] = value
    return obj


def _invalid_number(value):
    raise ValueError(f"Invalid JSON number: {value}.")


def _presentation(graph, layout, selected):
    ids = {node.id for node in graph.nodes}
    if not isinstance(layout, dict) or not set(layout) <= ids:
        raise ValueError("Kitchen layout refers to missing nodes.")
    for point in layout.values():
        if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) or
                not math.isfinite(v) or abs(v) > 100_000 for v in point):
            raise ValueError("Kitchen positions must be two finite coordinates within the canvas.")
    if selected is not None and selected not in ids:
        raise ValueError("Kitchen selection refers to a missing node.")


def _run_from_dict(value, graph):
    if not isinstance(value, dict) or set(value) != {"records", "attempts", "execution_order", "warnings"}:
        raise ValueError("Invalid Kitchen run record.")
    ids = {node.id for node in graph.nodes}
    if not isinstance(value["records"], dict) or not set(value["records"]) <= ids:
        raise ValueError("Results refer to missing Kitchen nodes.")
    records = {}
    for key, raw in value["records"].items():
        record = KitchenRecord(**raw)
        if record.node_id != key or record.kind not in PORTS or not isinstance(record.payload, dict) \
                or not isinstance(record.settings, dict) or not isinstance(record.inputs, dict) \
                or not isinstance(record.versions, dict) or not isinstance(record.warnings, list) \
                or not all(isinstance(w, str) for w in record.warnings) \
                or not isinstance(record.completed_utc, str) or not isinstance(record.fingerprint, str) \
                or len(record.fingerprint) != 64 or any(c not in "0123456789abcdef" for c in record.fingerprint):
            raise ValueError(f"Invalid completed result for {key}.")
        records[key] = record
    attempts = value["attempts"]
    if not isinstance(attempts, dict) or not set(attempts) <= ids:
        raise ValueError("Attempts refer to missing Kitchen nodes.")
    for attempt in attempts.values():
        if not isinstance(attempt, dict) or set(attempt) != {"fingerprint", "state", "error"} \
                or attempt["state"] not in ("queued", "running", "completed", "failed", "blocked", "cancelled") \
                or not isinstance(attempt["fingerprint"], str) or not isinstance(attempt["error"], str):
            raise ValueError("Invalid Kitchen attempt state.")
    if not isinstance(value["execution_order"], list) or not all(key in ids for key in value["execution_order"]) \
            or not isinstance(value["warnings"], list) or not all(isinstance(w, str) for w in value["warnings"]):
        raise ValueError("Invalid Kitchen run metadata.")
    return KitchenRun(records, attempts, value["execution_order"], value["warnings"])


def save_kitchen(path: str | Path, document: KitchenDocument) -> None:
    """Save current edits and completed provenance, without running any nodes."""
    graph_order(document.graph, complete=False)
    _presentation(document.graph, document.layout, document.selected)
    _run_from_dict(document.run.to_dict(), document.graph)
    _snapshot_references(document.graph, document.run, document.snapshots)
    snapshots, blobs = {}, {}
    if set(document.snapshot_info) != set(document.snapshots):
        raise ValueError("Every Kitchen snapshot needs its name/source metadata.")
    for index, (key, dataset) in enumerate(sorted(document.snapshots.items())):
        member = f"data/{index:04d}.parquet"
        snapshot = dataset.df.to_parquet(index=True, engine="pyarrow")
        info = document.snapshot_info[key]
        if not isinstance(info, dict) or not isinstance(info.get("name"), str) or not isinstance(info.get("source"), dict):
            raise ValueError("Invalid Kitchen snapshot metadata.")
        snapshots[key] = {"member": member, "sha256": hashlib.sha256(snapshot).hexdigest(), "info": info,
                          "roles": {c.name: c.var_type.value for c in dataset.schema().columns}}
        blobs[member] = snapshot
    manifest = {"format": FORMAT, "version": VERSION, "graph": document.graph.to_dict(),
                "snapshots": snapshots, "run": document.run.to_dict(), "layout": document.layout,
                "selected": document.selected, "versions": runtime_versions(), "warnings": document.warnings}
    metadata = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
    path, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("kitchen.json", metadata)
                for member, snapshot in blobs.items():
                    archive.writestr(member, snapshot)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_kitchen(path: str | Path) -> KitchenDocument:
    """Read typed snapshots and recorded results only; never execute a graph."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or "kitchen.json" not in names:
                raise ValueError("Expected one Kitchen manifest without duplicate archive entries.")
            if archive.getinfo("kitchen.json").file_size > 64 * 1024**2:
                raise ValueError("Kitchen manifest exceeds 64 MiB.")
            manifest = json.loads(archive.read("kitchen.json"), object_pairs_hook=_object, parse_constant=_invalid_number)
            json.dumps(manifest, allow_nan=False)  # also reject overflowing JSON numbers such as 1e999
            if not isinstance(manifest, dict) or set(manifest) != {
                "format", "version", "graph", "snapshots", "run", "layout", "selected", "versions", "warnings"
            } or manifest["format"] != FORMAT or type(manifest["version"]) is not int or manifest["version"] != VERSION:
                raise ValueError("This is not a supported Kitchen document.")
            graph = graph_from_dict(manifest["graph"])
            _presentation(graph, manifest["layout"], manifest["selected"])
            run = _run_from_dict(manifest["run"], graph)
            if not isinstance(manifest["snapshots"], dict) or not isinstance(manifest["versions"], dict) \
                    or not isinstance(manifest["warnings"], list) or not all(isinstance(w, str) for w in manifest["warnings"]):
                raise ValueError("Invalid Kitchen snapshot/runtime metadata.")
            snapshots, info, members = {}, {}, {"kitchen.json"}
            for key, entry in manifest["snapshots"].items():
                member = entry["member"]
                if not isinstance(member, str) or not member.startswith("data/") or not member.endswith(".parquet") \
                        or "/" in member[5:] or ".." in member or member in members:
                    raise ValueError("Invalid Kitchen dataset entry.")
                members.add(member)
                snapshot = archive.read(member)
                if hashlib.sha256(snapshot).hexdigest() != entry["sha256"]:
                    raise ValueError("Kitchen dataset checksum mismatch.")
                frame = pd.read_parquet(io.BytesIO(snapshot), engine="pyarrow")
                if not isinstance(entry["roles"], dict) or set(entry["roles"]) != set(frame.columns):
                    raise ValueError("Kitchen roles do not match the stored dataset.")
                snapshots[key] = Dataset(frame, overrides=entry["roles"])
                info[key] = entry["info"]
                if not isinstance(info[key], dict) or not isinstance(info[key].get("name"), str) \
                        or not isinstance(info[key].get("source"), dict):
                    raise ValueError("Invalid Kitchen dataset provenance.")
            if set(names) != members:
                raise ValueError("Unexpected files in the Kitchen archive.")
            _snapshot_references(graph, run, snapshots)
        current = runtime_versions()
        warnings = list(manifest["warnings"])
        for package, saved in manifest["versions"].items():
            if current.get(package) != saved:
                warnings.append(f"Saved with {package} {saved}; installed {current.get(package, 'unavailable')}. Reruns may differ.")
        for record in run.records.values():
            for package, saved in record.versions.items():
                if current.get(package) != saved:
                    warnings.append(f"{record.node_id} completed with {package} {saved}; installed {current.get(package, 'unavailable')}.")
        return KitchenDocument(graph, snapshots, info, run, manifest["layout"], manifest["selected"], list(dict.fromkeys(warnings)))
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f"Could not open the Kitchen: {exc}") from exc
