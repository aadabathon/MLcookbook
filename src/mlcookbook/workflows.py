"""Discover existing tabular workflows without fitting or changing a dataset.

Applicability is a preflight for the default recipe settings, not a claim that
an analysis answers a scientific question. The same feature and supervised
problem validators used by execution check requests here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from sklearn.model_selection import train_test_split

from mlcookbook.classification._problem import prepare_classification_problem
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.quality import column_issues
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.preprocessing.features import FEATURE_TYPES, resolve_features
from mlcookbook.regression._problem import prepare_regression_problem
from mlcookbook.statistics.scan import MAX_COLUMNS, MIN_OBSERVATIONS, SCAN_TYPES


@dataclass(frozen=True)
class WorkflowColumn(Result):
    name: str
    role: str
    dtype: str
    reason: str
    suggested: bool
    identifier_like: bool


@dataclass(frozen=True)
class WorkflowOption(Result):
    key: str
    label: str
    intent: str
    task: str
    available: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowDiscoveryResult(Result):
    columns: list[WorkflowColumn]
    workflows: list[WorkflowOption]
    context: dict
    warnings: list[str] = field(default_factory=list)


_METHODS = (
    ("linear-regression", "Linear regression · OLS / ridge / lasso", "regression", "regression"),
    ("decision-tree", "Decision tree regression", "regression", "regression"),
    ("random-forest", "Random forest regression", "regression", "regression"),
    ("svm", "Support vector regression", "regression", "regression"),
    ("neural-network", "Neural network regression", "regression", "regression"),
    ("model-comparison", "Compare ridge and random forest", "regression", "regression"),
    ("logistic-regression", "Logistic regression", "classification", "classification"),
    ("decision-tree", "Decision tree classification", "classification", "classification"),
    ("random-forest", "Random forest classification", "classification", "classification"),
    ("svm", "Support vector classification", "classification", "classification"),
    ("neural-network", "Neural network classification", "classification", "classification"),
    ("association-scan", "Association scan", "relationships", "unsupervised"),
    ("kmeans", "K-means clustering", "structure", "unsupervised"),
    ("pca", "Principal component analysis", "structure", "unsupervised"),
)


def discover_workflows(
    dataset: Dataset,
    *,
    target: str | None = None,
    features: list[str] | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
) -> WorkflowDiscoveryResult:
    """Describe existing recipe choices for an ordinary dataset snapshot.

    Identifier-like columns are excluded from the initial suggested feature
    list. An explicit *features* list can include them; their warning is kept.
    Group/time context is declared by the user, never guessed. Ordinary
    predictive and inferential recipes are blocked for that context until an
    appropriate evaluator exists; descriptive PCA/clustering remain available.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("discover_workflows expects a Dataset.")
    schema = dataset.schema()
    names = [column.name for column in schema.columns]
    for purpose, name in (("target", target), ("group", group_column), ("time", time_column)):
        if name is not None and name not in names:
            raise ValueError(f"Unknown {purpose} column {name!r}.")
    issues = column_issues(dataset.df, schema)
    identifiers = {issue.column for issue in issues if issue.kind == "likely_id"}
    unusable = {issue.column for issue in issues if issue.kind in ("all_missing", "constant")}
    columns = []
    for column in schema.columns:
        role = column.var_type
        if column.overridden:
            reason = f"Explicit source role: {role.value}; values are not converted."
        elif role is VarType.CATEGORICAL:
            reason = "Categorical storage or repeated string values; review whether levels are ordered."
        elif role is VarType.NUMERIC:
            reason = "Numeric storage; this may represent a measurement, class code or identifier."
        elif role is VarType.BOOLEAN:
            reason = "Boolean storage with two possible values."
        elif role is VarType.DATETIME:
            reason = "Datetime storage; ordering alone does not establish independent observations."
        elif role is VarType.TEXT:
            reason = "String values with high cardinality; ordinary recipes do not use raw text predictors."
        else:
            reason = "No supported statistical role was inferred; inspect the values before analysis."
        if column.name in identifiers:
            reason += " Looks like a row identifier; initially excluded. Include it deliberately if meaningful."
        if column.name in unusable:
            reason += " This column is constant or entirely missing."
        columns.append(WorkflowColumn(
            column.name, role.value, column.dtype, reason,
            role in FEATURE_TYPES and column.name not in identifiers | unusable,
            column.name in identifiers,
        ))
    if features is None:
        features = [column.name for column in columns if column.suggested
                    and column.name not in (target, group_column, time_column)]
    else:
        features = list(features)
    context = {"target": target, "features": features, "group_column": group_column, "time_column": time_column}
    by_intent: dict[str, tuple[bool, str, list[str]]] = {}
    for intent in ("regression", "classification", "relationships", "structure"):
        try:
            notes = _validate_intent(dataset, intent, target, features, group_column, time_column)
            by_intent[intent] = (True, "Choose settings in the recipe, then run explicitly.", notes)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            by_intent[intent] = (False, str(exc).strip("\""), [])
    options = []
    for key, label, intent, task in _METHODS:
        available, reason, notes = by_intent[intent]
        if available and key == "kmeans" and len(dataset.df) <= 3:
            available, reason = False, "Default k-means needs more than 3 rows; open PCA or add observations."
        options.append(WorkflowOption(key, label, intent, task, available, reason, list(notes)))
    warnings = []
    if group_column or time_column:
        warnings.append("Declared group/time context is not a general grouped or temporal evaluator.")
    return WorkflowDiscoveryResult(columns, options, context, warnings)


def _validate_intent(dataset, intent, target, features, group_column, time_column):
    if (group_column or time_column) and intent != "structure":
        raise ValueError(
            "Declared group/time context needs an appropriate grouped or temporal evaluator; "
            "ordinary row-wise prediction and association workflows are unavailable."
        )
    if not len(dataset.df):
        raise ValueError("Load a dataset with observations first.")
    if any(name in features for name in (group_column, time_column) if name is not None):
        raise ValueError("Exclude the declared group/time columns from analysis inputs.")
    if intent in ("regression", "classification"):
        if target is None:
            raise ValueError("Select a numeric target." if intent == "regression" else
                             "Select a categorical, boolean or discrete numeric target.")
        if intent == "regression":
            _, frame, notes = prepare_regression_problem(dataset, target, features, "Regression")
            train, test = train_test_split(frame, test_size=0.25, random_state=0)
            if len(train) < 4 or len(test) < 2:
                raise ValueError("Need at least 4 training and 2 test rows with a non-missing target.")
        else:
            _, frame, _, notes = prepare_classification_problem(dataset, target, features, None, "Classification")
            train_test_split(frame, test_size=0.25, random_state=0, stratify=frame[target])
        return notes
    if intent == "relationships":
        if len(features) < 2:
            raise ValueError("Select at least two numeric, categorical or boolean columns.")
        if len(features) > MAX_COLUMNS:
            raise ValueError(f"Association scan supports at most {MAX_COLUMNS} columns; select fewer inputs.")
        if len(set(features)) != len(features):
            raise ValueError("Selected columns must be unique.")
        schema = dataset.schema()
        for name in features:
            if schema.var_type(name) not in SCAN_TYPES:
                raise ValueError(f"Column {name!r} is {schema.var_type(name).value}; association scan needs numeric, categorical or boolean columns.")
        if len(dataset.df) < MIN_OBSERVATIONS:
            raise ValueError(f"Association scan needs at least {MIN_OBSERVATIONS} complete observations per pair.")
        varying = [name for name in features if dataset.df[name].nunique(dropna=True) > 1]
        if len(varying) < 2:
            raise ValueError("Select at least two columns with observed variation.")
        for x, y in combinations(varying, 2):
            complete = dataset.df[[x, y]].dropna()
            if len(complete) >= MIN_OBSERVATIONS and complete[x].nunique() > 1 and complete[y].nunique() > 1:
                break
        else:
            raise ValueError(f"No selected pair has {MIN_OBSERVATIONS} complete observations with variation in both columns.")
        return []
    _, notes = resolve_features(dataset, None, features)
    if len(dataset.df) < 2:
        raise ValueError("Structure analysis needs at least two observations.")
    if not any(dataset.df[name].nunique(dropna=True) > 1 for name in features):
        raise ValueError("Every selected feature is constant or entirely missing.")
    return notes
