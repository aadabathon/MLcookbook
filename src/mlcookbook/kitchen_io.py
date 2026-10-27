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
from mlcookbook.core.schema import VarType
from mlcookbook.experiments import runtime_versions
from mlcookbook.kitchen import (
    PORTS,
    KitchenDocument,
    KitchenRecord,
    KitchenRun,
    graph_from_dict,
    graph_order,
)
from mlcookbook.kitchen_artifacts import TableArtifact, VectorArtifact, artifact_roots

FORMAT = "mlcookbook-kitchen"
VERSION = 2
MAX_MANIFEST_BYTES = 64 * 1024**2


def _record_refs(run):
    references = set()
    for record in run.records.values():
        for field in ("outputs", "input_artifacts"):
            values = record.payload.get(field, {})
            if not isinstance(values, dict) or not all(isinstance(port, str) and _hex(key) for port, key in values.items()):
                raise ValueError("Invalid completed output artifact reference.")
            references.update(values.values())
        for value in record.inputs.values():
            if isinstance(value, dict) and value.get("artifact") is not None:
                references.add(value["artifact"])
    return references


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _retained_artifacts(document):
    pending, retained = list(_record_refs(document.run)), {}
    while pending:
        key = pending.pop()
        if key in retained:
            continue
        artifact = document.artifacts.get(key)
        if not isinstance(artifact, (TableArtifact, VectorArtifact)) or artifact.id != key:
            raise ValueError("A completed result is missing its complete typed artifact; accept the execution into the document before saving.")
        retained[key] = artifact
        pending.extend(artifact_roots(artifact) - retained.keys())
    return retained


def _dependencies(value):
    if not isinstance(value, list) or any(not isinstance(pair, list) or len(pair) != 2 or
            not _hex(pair[0]) or not isinstance(pair[1], str) for pair in value):
        raise ValueError("Invalid artifact source-column dependencies.")
    return frozenset(tuple(pair) for pair in value)


def _expressions(value):
    if not isinstance(value, list) or not all(_hex(key) for key in value):
        raise ValueError("Invalid artifact expression ancestry.")
    return frozenset(value)


def _write_artifacts(artifacts, blobs):
    entries = {}
    # Content sharing avoids duplicating a source frame also retained as a completed input.
    by_hash = {hashlib.sha256(blob).hexdigest(): name for name, blob in blobs.items()}
    for index, (key, artifact) in enumerate(sorted(artifacts.items())):
        table = isinstance(artifact, TableArtifact)
        frame = artifact.dataset.df if table else pd.DataFrame({"value": artifact.values.array}, index=artifact.index)
        data = frame.to_parquet(index=True, engine="pyarrow")
        checksum = hashlib.sha256(data).hexdigest()
        member = by_hash.get(checksum, f"objects/{index:04d}.parquet")
        blobs.setdefault(member, data)
        by_hash[checksum] = member
        rows = pd.DataFrame({"source": pd.Series([r[0] for r in artifact.rows], dtype="string"),
                             "position": pd.Series([r[1] for r in artifact.rows], dtype="int64")})
        row_data = rows.to_parquet(index=False, engine="pyarrow")
        row_hash = hashlib.sha256(row_data).hexdigest()
        row_member = by_hash.get(row_hash, f"lineage/{index:04d}.parquet")
        blobs.setdefault(row_member, row_data)
        by_hash[row_hash] = row_member
        entry = {"kind": "dataset" if table else "vector", "member": member, "sha256": checksum,
                 "lineage": row_member, "lineage_sha256": row_hash, "warnings": list(artifact.warnings),
                 "row_dependencies": [list(dep) for dep in sorted(artifact.row_dependencies)],
                 "row_expressions": sorted(artifact.row_expressions)}
        if table:
            entry.update(roles={c.name: c.var_type.value for c in artifact.dataset.schema().columns},
                         dependencies={name: [list(dep) for dep in sorted(deps)] for name, deps in artifact.dependencies.items()},
                         root_snapshot=artifact.root_snapshot, expressions=artifact.expressions,
                         ancestors={name: sorted(values) for name, values in artifact.ancestors.items()})
        else:
            entry.update(name=artifact.name, role=artifact.role, expression=artifact.expression,
                         ancestors=sorted(artifact.ancestors), dependencies=[list(dep) for dep in sorted(artifact.dependencies)])
        entries[key] = entry
    return entries


def _read_artifacts(entries, archive, members, snapshots):
    if not isinstance(entries, dict):
        raise ValueError("Kitchen artifacts must be an object.")
    artifacts = {}

    def read(member, checksum):
        if not isinstance(member, str) or not member.endswith(".parquet") or "/" in member.split("/", 1)[-1] \
                or member.split("/", 1)[0] not in ("data", "objects", "lineage") or ".." in member:
            raise ValueError("Invalid Kitchen artifact member.")
        raw = archive.read(member)
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("Kitchen artifact checksum mismatch.")
        members.add(member)
        return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")

    for key, entry in entries.items():
        if not _hex(key) or not isinstance(entry, dict):
            raise ValueError("Invalid Kitchen artifact identity.")
        frame = read(entry["member"], entry["sha256"])
        lineage = read(entry["lineage"], entry["lineage_sha256"])
        if list(lineage.columns) != ["source", "position"] or len(lineage) != len(frame):
            raise ValueError("Artifact row lineage length/schema mismatch.")
        rows = tuple(lineage.itertuples(index=False, name=None))
        if any(not _hex(source) or type(position) is not int or position < 0 for source, position in rows):
            raise ValueError("Artifact row lineage must contain source IDs and nonnegative positions.")
        if not isinstance(entry["warnings"], list) or not all(isinstance(w, str) for w in entry["warnings"]):
            raise ValueError("Invalid artifact warnings.")
        row_dependencies = _dependencies(entry["row_dependencies"])
        row_expressions = _expressions(entry["row_expressions"])
        if entry["kind"] == "dataset":
            roles, expressions, ancestry = entry["roles"], entry["expressions"], entry["ancestors"]
            if not isinstance(roles, dict) or not isinstance(expressions, dict) or not isinstance(ancestry, dict) \
                    or set(roles) != set(frame.columns) or set(expressions) != set(roles) or set(ancestry) != set(roles) \
                    or not all(_hex(value) for value in expressions.values()) or not isinstance(entry["dependencies"], dict) \
                    or set(entry["dependencies"]) != set(roles):
                raise ValueError("Artifact columns, roles and expression lineage disagree.")
            root_snapshot = entry["root_snapshot"]
            if root_snapshot is not None and root_snapshot not in snapshots:
                raise ValueError("An artifact refers to a missing source snapshot.")
            artifacts[key] = TableArtifact(key, Dataset(frame, overrides=roles), rows,
                {name: _dependencies(value) for name, value in entry["dependencies"].items()}, row_dependencies,
                tuple(entry["warnings"]), root_snapshot, expressions, {name: _expressions(value) for name, value in ancestry.items()}, row_expressions)
        elif entry["kind"] == "vector":
            if list(frame.columns) != ["value"] or not isinstance(entry["name"], str) or not _hex(entry["expression"]):
                raise ValueError("Invalid stored vector.")
            VarType(entry["role"])
            artifacts[key] = VectorArtifact(key, frame["value"].reset_index(drop=True), frame.index.copy(), entry["role"],
                entry["name"], rows, _dependencies(entry["dependencies"]), row_dependencies,
                tuple(entry["warnings"]), entry["expression"], _expressions(entry["ancestors"]), row_expressions)
        else:
            raise ValueError("Unknown artifact kind.")
    for artifact in artifacts.values():
        roots = artifact_roots(artifact)
        for root in roots:
            if root not in artifacts or not isinstance(artifacts[root], TableArtifact) or artifacts[root].root_snapshot is None:
                raise ValueError("An artifact is missing its original source rows.")
        for source, position in artifact.rows:
            if position >= len(artifacts[source].dataset.df):
                raise ValueError("Artifact source row position is out of bounds.")
        deps = artifact.dependencies.values() if isinstance(artifact, TableArtifact) else [artifact.dependencies]
        for values in [*deps, artifact.row_dependencies]:
            if any(column not in artifacts[root].dataset.df for root, column in values):
                raise ValueError("Artifact column lineage refers to a missing source column.")
        row_roots = {source for source, _ in artifact.rows}
        if len(row_roots) == 1:
            root = artifacts[next(iter(row_roots))]
            expected = root.dataset.df.index.take([position for _, position in artifact.rows])
            actual = artifact.dataset.df.index if isinstance(artifact, TableArtifact) else artifact.index
            if not actual.equals(expected):
                raise ValueError("Artifact index no longer agrees with its positional source lineage.")
    return artifacts


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
        if set(record.inputs) != set(PORTS[record.kind][0]):
            raise ValueError(f"Invalid completed input ports for {key}.")
        for provenance in record.inputs.values():
            if isinstance(provenance, str):
                if not _hex(provenance):
                    raise ValueError(f"Invalid legacy input fingerprint for {key}.")
            elif not isinstance(provenance, dict) or set(provenance) != {"node", "port", "fingerprint", "artifact"} \
                    or not isinstance(provenance["node"], str) or not provenance["node"] \
                    or not isinstance(provenance["port"], str) or not provenance["port"] \
                    or not _hex(provenance["fingerprint"]) \
                    or (provenance["artifact"] is not None and not _hex(provenance["artifact"])):
                raise ValueError(f"Invalid completed input provenance for {key}.")
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
    retained = _retained_artifacts(document)
    snapshots, blobs = {}, {}
    if set(document.snapshot_info) != set(document.snapshots):
        raise ValueError("Every Kitchen snapshot needs its name/source metadata.")
    needed = {n.settings["snapshot"] for n in document.graph.nodes if n.kind == "dataset" and n.settings["snapshot"]}
    needed.update(record.payload.get("snapshot_id") for record in document.run.records.values())
    needed.update(artifact.root_snapshot for artifact in retained.values() if isinstance(artifact, TableArtifact) and artifact.root_snapshot)
    needed.discard(None)
    for index, key in enumerate(sorted(needed)):
        dataset = document.snapshots[key]
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
                "selected": document.selected, "versions": runtime_versions(), "warnings": document.warnings,
                "artifacts": _write_artifacts(retained, blobs)}
    metadata = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(metadata) > MAX_MANIFEST_BYTES:
        raise ValueError("Kitchen manifest exceeds 64 MiB; the existing file was not replaced.")
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
            if archive.getinfo("kitchen.json").file_size > MAX_MANIFEST_BYTES:
                raise ValueError("Kitchen manifest exceeds 64 MiB.")
            manifest = json.loads(archive.read("kitchen.json"), object_pairs_hook=_object, parse_constant=_invalid_number)
            json.dumps(manifest, allow_nan=False)  # also reject overflowing JSON numbers such as 1e999
            expected = {
                "format", "version", "graph", "snapshots", "run", "layout", "selected", "versions", "warnings"
            }
            if isinstance(manifest, dict) and manifest.get("version") == 2:
                expected.add("artifacts")
            if not isinstance(manifest, dict) or set(manifest) != expected or manifest["format"] != FORMAT \
                    or type(manifest["version"]) is not int or manifest["version"] not in (1, VERSION):
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
            artifacts = _read_artifacts(manifest["artifacts"], archive, members, snapshots) if manifest["version"] == 2 else {}
            if manifest["version"] == 2 and not _record_refs(run) <= artifacts.keys():
                raise ValueError("Completed results refer to missing typed output artifacts.")
            if set(names) != members:
                raise ValueError("Unexpected files in the Kitchen archive.")
            _snapshot_references(graph, run, snapshots)
        current = runtime_versions()
        warnings = list(manifest["warnings"])
        if manifest["version"] == 1 and run.records:
            warnings.append("Foundational Kitchen restored with preview-only data outputs; rerun paths to materialize full exportable artifacts.")
        for package, saved in manifest["versions"].items():
            if current.get(package) != saved:
                warnings.append(f"Saved with {package} {saved}; installed {current.get(package, 'unavailable')}. Reruns may differ.")
        for record in run.records.values():
            for package, saved in record.versions.items():
                if current.get(package) != saved:
                    warnings.append(f"{record.node_id} completed with {package} {saved}; installed {current.get(package, 'unavailable')}.")
        return KitchenDocument(graph, snapshots, info, run, manifest["layout"], manifest["selected"], list(dict.fromkeys(warnings)), artifacts)
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f"Could not open the Kitchen: {exc}") from exc
