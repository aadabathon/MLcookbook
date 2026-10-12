"""Principal component analysis workflow.

PCA rotates the (preprocessed) feature space onto orthogonal directions of
decreasing variance. Each component is a unit-length weighting of the encoded
features (its *loadings*); its eigenvalue is the variance along it, and the
explained-variance ratio is that variance's share of the total.

Variance depends on units, so PCA on unscaled features mostly rediscovers the
feature with the largest numbers. Standard scaling (the default) makes PCA
work on correlations instead. Like clustering, PCA has no target: it is fitted
on all rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.features import resolve_features
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor

VARIANCE_TARGET = 0.90
MAX_SCORES = 2  # per-row coordinates are returned for the first components only


@dataclass(frozen=True)
class PrincipalComponent(Result):
    index: int  # 1-based: PC1, PC2, ...
    eigenvalue: float  # variance along the component (preprocessed units)
    explained_variance_ratio: float
    cumulative_ratio: float
    loadings: dict[str, float]  # weight per encoded feature; squares sum to 1


@dataclass(frozen=True)
class PCAResult(Result):
    method: str
    features: list[str]
    encoded_features: list[str]
    n_rows: int
    n_components: int
    components: list[PrincipalComponent]
    components_for_90_percent: int | None  # fewest components reaching 90% of the variance, if any
    kaiser_components: int | None  # components with eigenvalue > 1 (meaningful only for standardized data)
    scores: list[list[float]]  # per row, coordinates on the first (up to) two components
    preprocessing: PreprocessingSummary
    warnings: list[str] = field(default_factory=list)


def pca(
    dataset: Dataset,
    features: list[str] | None = None,
    n_components: int | None = None,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
) -> PCAResult:
    """Run PCA on *features* (default: every numeric/categorical/boolean/datetime column).

    *n_components* defaults to every component (the smaller of rows and
    encoded features). The result lists each component's variance and loadings
    and the per-row scores on the first two components for plotting.
    """
    features, warnings = resolve_features(dataset, None, features)
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append(
            "PCA without scaling follows raw variance, so features measured in large units dominate "
            "the first components."
        )
    X = dataset.df[features]
    preprocessor = build_preprocessor(dataset, features, config=config)
    matrix = preprocessor.fit_transform(X)
    encoded = [str(n) for n in preprocessor.get_feature_names_out()]
    n_rows = len(X)
    available = min(n_rows, matrix.shape[1])
    if n_components is None:
        n_components = available
    if not 1 <= n_components <= available:
        raise ValueError(f"n_components must be between 1 and {available}; got {n_components}.")
    if np.ptp(matrix, axis=0).max(initial=0) == 0:
        raise ValueError("Every feature is constant after preprocessing; there is no variance to analyze.")
    if len(encoded) > len(features):
        warnings.append(
            "Categorical features enter PCA as one-hot columns; their loadings are per level, "
            "and binary columns do not fit PCA's linear-Gaussian picture well."
        )

    model = PCA(n_components=n_components, random_state=random_state).fit(matrix)
    ratios = model.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    components = [
        PrincipalComponent(
            index=i + 1,
            eigenvalue=float(model.explained_variance_[i]),
            explained_variance_ratio=float(ratios[i]),
            cumulative_ratio=float(cumulative[i]),
            loadings={name: float(w) for name, w in zip(encoded, model.components_[i])},
        )
        for i in range(n_components)
    ]
    reach = np.nonzero(cumulative >= VARIANCE_TARGET - 1e-12)[0]
    standardized = config.scaling == "standard"
    return PCAResult(
        method="pca",
        features=list(features),
        encoded_features=encoded,
        n_rows=n_rows,
        n_components=n_components,
        components=components,
        components_for_90_percent=int(reach[0]) + 1 if len(reach) else None,
        kaiser_components=int((model.explained_variance_ > 1).sum()) if standardized else None,
        scores=model.transform(matrix)[:, :MAX_SCORES].tolist(),
        preprocessing=describe_preprocessor(preprocessor, X, dataset, config),
        warnings=warnings,
    )
