"""Typed runtime tables/vectors and row-local kernels for the Kitchen.

Rows are identified by an immutable source artifact and zero-based position.
Index labels remain display data, never alignment keys. No learned transforms.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.fingerprint import dataset_fingerprint
from mlcookbook.core.schema import VarType

RowKey = tuple[str, int]
Dependency = tuple[str, str]
MAPS = ("add", "multiply", "log", "exp", "sqrt", "abs", "square", "greater_than", "less_than", "equal")
COMBINATIONS = ("add", "subtract", "multiply", "divide", "greater_than", "less_than", "equal")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rows_digest(rows):
    hashed = hashlib.sha256()
    for root, position in rows:
        hashed.update(json.dumps([root, position], separators=(",", ":")).encode())
        hashed.update(b"\n")
    return hashed.hexdigest()


@dataclass(frozen=True)
class TableArtifact:
    id: str
    dataset: Dataset
    rows: tuple[RowKey, ...]
    dependencies: dict[str, frozenset[Dependency]]
    row_dependencies: frozenset[Dependency] = frozenset()
    warnings: tuple[str, ...] = ()
    root_snapshot: str | None = None
    expressions: dict[str, str] = field(default_factory=dict)
    ancestors: dict[str, frozenset[str]] = field(default_factory=dict)
    row_expressions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class VectorArtifact:
    id: str
    values: pd.Series
    index: pd.Index
    role: str
    name: str
    rows: tuple[RowKey, ...]
    dependencies: frozenset[Dependency]
    row_dependencies: frozenset[Dependency] = frozenset()
    warnings: tuple[str, ...] = ()
    expression: str = ""
    ancestors: frozenset[str] = frozenset()
    row_expressions: frozenset[str] = frozenset()


@dataclass
class PreparedKitchen:
    order: list[str]
    outputs: dict[tuple[str, str], TableArtifact | VectorArtifact] = field(default_factory=dict)
    roots: dict[str, TableArtifact] = field(default_factory=dict)
    analysis_notes: dict[str, list[str]] = field(default_factory=dict)
    cancelled: bool = False


def root_table(snapshot: str, data: Dataset) -> TableArtifact:
    # Keep completed inputs independent even for a Python caller mutating its Dataset later.
    copied = Dataset(data.df, overrides={c.name: c.var_type for c in data.schema().columns})
    key = digest({"snapshot": snapshot, "fingerprint": dataset_fingerprint(copied)})
    rows = tuple((key, i) for i in range(len(copied.df)))
    return TableArtifact(key, copied, rows, {c: frozenset({(key, c)}) for c in copied.df.columns}, root_snapshot=snapshot,
                         expressions={c: digest({"source": key, "column": c}) for c in copied.df.columns},
                         ancestors={c: frozenset() for c in copied.df.columns})


def table_artifact(data, rows, dependencies, row_dependencies=frozenset(), warnings=(), *, expressions, ancestors, row_expressions=frozenset()):
    key = digest({"kind": "dataset", "data": dataset_fingerprint(data), "rows": _rows_digest(rows),
                  "dependencies": {k: sorted(v) for k, v in dependencies.items()},
                  "row_dependencies": sorted(row_dependencies), "expressions": expressions,
                  "ancestors": {k: sorted(v) for k, v in ancestors.items()}, "row_expressions": sorted(row_expressions)})
    return TableArtifact(key, data, tuple(rows), dependencies, row_dependencies, tuple(dict.fromkeys(warnings)),
                         expressions=expressions, ancestors=ancestors, row_expressions=frozenset(row_expressions))


def vector_artifact(values, index, role, name, rows, dependencies, row_dependencies=frozenset(), warnings=(), *, expression, ancestors, row_expressions=frozenset()):
    values = values.reset_index(drop=True).copy()
    frame = pd.DataFrame({"value": values.array}, index=index)
    key = digest({"kind": "vector", "data": hashlib.sha256(frame.to_parquet(index=True, engine="pyarrow")).hexdigest(),
                  "role": role, "name": name, "rows": _rows_digest(rows), "dependencies": sorted(dependencies),
                  "row_dependencies": sorted(row_dependencies), "expression": expression,
                  "ancestors": sorted(ancestors), "row_expressions": sorted(row_expressions)})
    return VectorArtifact(key, values, index.copy(), role, name, tuple(rows), frozenset(dependencies),
                          frozenset(row_dependencies), tuple(dict.fromkeys(warnings)), expression,
                          frozenset(ancestors), frozenset(row_expressions))


def select_columns(table: TableArtifact, settings) -> TableArtifact:
    columns = settings["columns"]
    if not columns or not set(columns) <= set(table.dataset.df.columns):
        raise ValueError("Select existing columns, including any target needed downstream.")
    roles = {c.name: c.var_type.value for c in table.dataset.schema().columns if c.name in columns}
    roles.update(settings["roles"])
    data = Dataset(table.dataset.df.loc[:, columns], overrides=roles)
    return table_artifact(data, table.rows, {c: table.dependencies[c] for c in columns}, table.row_dependencies, table.warnings,
                          expressions={c: table.expressions[c] for c in columns}, ancestors={c: table.ancestors[c] for c in columns},
                          row_expressions=table.row_expressions)


def column_vectors(table: TableArtifact, column: str) -> dict[str, VectorArtifact]:
    if column not in table.dataset.df:
        raise ValueError("Choose an existing input column.")
    role = table.dataset.schema().var_type(column)
    values = table.dataset.df[column]
    if role == VarType.UNSUPPORTED:
        raise ValueError("Unsupported columns cannot become Kitchen vectors.")
    if role == VarType.TEXT and not values.dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError("Text roles require actual string values; no coercion is performed.")
    if role == VarType.NUMERIC:
        numeric_values(values, role.value)
    return {
        "vector": vector_artifact(values, table.dataset.df.index, role.value, column, table.rows,
                                  table.dependencies[column], table.row_dependencies, table.warnings,
                                  expression=table.expressions[column], ancestors=table.ancestors[column], row_expressions=table.row_expressions),
        "missing": vector_artifact(values.isna().astype("boolean"), table.dataset.df.index, "boolean", f"{column} missing",
                                   table.rows, table.dependencies[column], table.row_dependencies, table.warnings,
                                   expression=digest({"op": "missing", "input": table.expressions[column]}),
                                   ancestors=table.ancestors[column] | {table.expressions[column]}, row_expressions=table.row_expressions),
    }


def numeric_values(values, role):
    if role != "numeric" or not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError("Numeric kernels require a numeric role and numeric values; labels/strings are not converted.")
    if pd.api.types.is_integer_dtype(values.dtype) and (values.dropna().gt(2**53) | values.dropna().lt(-(2**53))).any():
        raise ValueError("Integer vector values exceed exact float64 precision (2**53).")
    array = values.to_numpy(dtype="float64", na_value=np.nan)
    if np.isinf(array).any():
        raise ValueError("Numeric vectors require finite values or missing entries, not infinity.")
    return array


def aligned(*artifacts):
    if any(artifact.rows != artifacts[0].rows for artifact in artifacts[1:]):
        raise ValueError("Row lineage differs: combine vectors from the same snapshot and ordered source positions. "
                         "Index labels are not alignment keys.")


def _math_result(result, missing, policy, description):
    invalid = ~np.isfinite(result) & ~missing
    count = int(invalid.sum())
    warnings = []
    if count:
        if policy == "error":
            raise ValueError(f"{description} produced {count} invalid/non-finite value(s). Choose explicit missing-and-warn handling or repair the inputs.")
        result = result.copy()
        result[invalid] = np.nan
        warnings.append(f"{description}: marked {count} invalid result(s) missing; row positions were retained.")
    return pd.Series(pd.array(result, dtype="Float64")), warnings


def map_vector(vector: VectorArtifact, settings) -> VectorArtifact:
    values, operation, scalar = numeric_values(vector.values, vector.role), settings["operation"], settings["value"]
    missing = np.isnan(values)
    warnings = list(vector.warnings)
    with np.errstate(all="ignore"):
        if operation == "add":
            result = values + scalar
        elif operation == "multiply":
            result = values * scalar
        elif operation == "log":
            result = np.log(values)
        elif operation == "exp":
            result = np.exp(values)
        elif operation == "sqrt":
            result = np.sqrt(values)
        elif operation == "abs":
            result = np.abs(values)
        elif operation == "square":
            result = np.square(values)
        elif operation == "greater_than":
            result = values > scalar
        elif operation == "less_than":
            result = values < scalar
        else:
            result = values == scalar
    if operation in ("greater_than", "less_than", "equal"):
        series = pd.Series(pd.array(result, dtype="boolean"))
        series[missing] = pd.NA
        role = "boolean"
    else:
        series, notes = _math_result(result, missing, settings["invalid"], operation)
        warnings.extend(notes)
        role = "numeric"
    return vector_artifact(series, vector.index, role, f"{operation}({vector.name})", vector.rows,
                           vector.dependencies, vector.row_dependencies, warnings,
                           expression=digest({"op": "map", "settings": settings, "input": vector.expression}),
                           ancestors=vector.ancestors | {vector.expression}, row_expressions=vector.row_expressions)


def combine_vectors(left: VectorArtifact, right: VectorArtifact, settings) -> VectorArtifact:
    aligned(left, right)
    a, b = numeric_values(left.values, left.role), numeric_values(right.values, right.role)
    operation = settings["operation"]
    missing = np.isnan(a) | np.isnan(b)
    warnings = list(dict.fromkeys([*left.warnings, *right.warnings]))
    with np.errstate(all="ignore"):
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b
        elif operation == "greater_than":
            result = a > b
        elif operation == "less_than":
            result = a < b
        else:
            result = a == b
    if operation in ("greater_than", "less_than", "equal"):
        series = pd.Series(pd.array(result, dtype="boolean"))
        series[missing] = pd.NA
        role = "boolean"
    else:
        series, notes = _math_result(result, missing, settings["invalid"], operation)
        warnings.extend(notes)
        role = "numeric"
    return vector_artifact(series, left.index, role, f"{left.name} {operation} {right.name}", left.rows,
                           left.dependencies | right.dependencies, left.row_dependencies | right.row_dependencies, warnings,
                           expression=digest({"op": "combine", "settings": settings, "left": left.expression, "right": right.expression}),
                           ancestors=left.ancestors | right.ancestors | {left.expression, right.expression},
                           row_expressions=left.row_expressions | right.row_expressions)


def build_dataset(vector: VectorArtifact, name: str) -> TableArtifact:
    if not name.strip():
        raise ValueError("Choose a nonempty output column name.")
    frame = pd.DataFrame({name: vector.values.array}, index=vector.index)
    return table_artifact(Dataset(frame, overrides={name: vector.role}), vector.rows,
                          {name: vector.dependencies}, vector.row_dependencies, vector.warnings,
                          expressions={name: vector.expression}, ancestors={name: vector.ancestors}, row_expressions=vector.row_expressions)


def add_column(table: TableArtifact, vector: VectorArtifact, name: str) -> TableArtifact:
    aligned(table, vector)
    if not name.strip():
        raise ValueError("Choose a nonempty output column name.")
    if name in table.dataset.df:
        raise ValueError("The new column name already exists; original columns are never overwritten.")
    frame = table.dataset.df.copy()
    frame[name] = vector.values.array  # positional, never an index-label join
    roles = {c.name: c.var_type.value for c in table.dataset.schema().columns}
    roles[name] = vector.role
    return table_artifact(Dataset(frame, overrides=roles), table.rows, table.dependencies | {name: vector.dependencies},
                          table.row_dependencies | vector.row_dependencies, [*table.warnings, *vector.warnings],
                          expressions=table.expressions | {name: vector.expression}, ancestors=table.ancestors | {name: vector.ancestors},
                          row_expressions=table.row_expressions | vector.row_expressions)


def filter_rows(table: TableArtifact, mask: VectorArtifact) -> TableArtifact:
    aligned(table, mask)
    if mask.role != "boolean" or not pd.api.types.is_bool_dtype(mask.values.dtype):
        raise ValueError("Row filtering requires a boolean vector, such as a threshold comparison.")
    keep = mask.values.fillna(False).to_numpy(dtype=bool)
    positions = np.flatnonzero(keep)
    roles = {c.name: c.var_type.value for c in table.dataset.schema().columns}
    warnings = [*table.warnings, *mask.warnings,
                f"Row filter retained {len(positions)} of {len(table.rows)} rows before splitting; missing mask values are excluded."]
    return table_artifact(Dataset(table.dataset.df.iloc[positions], overrides=roles),
                          tuple(table.rows[i] for i in positions), table.dependencies,
                          table.row_dependencies | mask.dependencies | mask.row_dependencies, warnings,
                          expressions=table.expressions, ancestors=table.ancestors,
                          row_expressions=table.row_expressions | mask.row_expressions | mask.ancestors | {mask.expression})


def validate_target_dependencies(table: TableArtifact, target: str, features: list[str]):
    if target not in table.dependencies:
        raise ValueError("The target column is missing.")
    target_inputs, target_expression = table.dependencies[target], table.expressions[target]
    leaked = [name for name in features if table.expressions.get(name) == target_expression or
              target_expression in table.ancestors.get(name, frozenset())]
    if leaked:
        raise ValueError("Feature(s) derived from the selected target would leak its values: " + ", ".join(leaked))
    if target_expression in table.row_expressions:
        raise ValueError("Row selection depends on the selected target; filter on independent inputs before model evaluation.")
    overlap = [name for name in features if table.dependencies.get(name, frozenset()) & target_inputs]
    return (["Target and predictor definitions share source columns (" + ", ".join(overlap) +
             "). Check mathematical coupling and availability before interpreting predictive performance."] if overlap else [])


def artifact_roots(artifact):
    dependencies = set(artifact.row_dependencies)
    if isinstance(artifact, TableArtifact):
        for values in artifact.dependencies.values():
            dependencies.update(values)
    else:
        dependencies.update(artifact.dependencies)
    roots = {root for root, _ in artifact.rows} | {root for root, _ in dependencies}
    if isinstance(artifact, TableArtifact) and artifact.root_snapshot is not None:
        roots.add(artifact.id)
    return roots
