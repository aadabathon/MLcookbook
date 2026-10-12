"""Feature selection shared by every workflow that takes a feature list."""

from __future__ import annotations

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.quality import column_issues
from mlcookbook.core.schema import VarType

FEATURE_TYPES = (VarType.NUMERIC, VarType.CATEGORICAL, VarType.BOOLEAN, VarType.DATETIME)


def resolve_features(
    dataset: Dataset, target: str | None, features: list[str] | None
) -> tuple[list[str], list[str]]:
    """Validate or pick the feature columns; return ``(features, warnings)``.

    With ``features=None`` every numeric/categorical/boolean/datetime column
    except the target is used and the rest are skipped with a warning. An
    explicit list must only name usable columns. Quality issues of the chosen
    features (constant, mostly missing, identifier-like, ...) become warnings.
    """
    schema = dataset.schema()
    warnings: list[str] = []
    if features is None:
        candidates = [c for c in schema.columns if c.name != target]
        features = [c.name for c in candidates if c.var_type in FEATURE_TYPES]
        skipped = [c.name for c in candidates if c.var_type not in FEATURE_TYPES]
        if skipped:
            warnings.append(f"Skipped columns with non-feature types (text/unsupported): {skipped}.")
    else:
        if target is not None and target in features:
            raise ValueError(f"Target {target!r} cannot also be a feature.")
        if len(set(features)) != len(features):
            raise ValueError(f"Features must be unique; got {features}.")
        for name in features:
            var_type = schema.var_type(name)  # raises KeyError for unknown columns
            if var_type not in FEATURE_TYPES:
                raise ValueError(f"Feature {name!r} has unsupported type {var_type.value}.")
    if not features:
        raise ValueError("No usable feature columns.")
    warnings.extend(issue.message for issue in column_issues(dataset.df, schema, list(features)))
    return list(features), warnings
