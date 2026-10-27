"""Stable identity for a typed, ordered dataset snapshot."""
import hashlib
import json

from mlcookbook.core.dataset import Dataset


def dataset_fingerprint(dataset: Dataset) -> str:
    """Include values, row/index order, dtypes and statistical roles."""
    snapshot = dataset.df.to_parquet(index=True, engine="pyarrow")
    roles = {col.name: col.var_type.value for col in dataset.schema().columns}
    payload = {"parquet_sha256": hashlib.sha256(snapshot).hexdigest(), "roles": roles}
    # Keep the identity used by existing saved LLM workflows byte-for-byte.
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()
