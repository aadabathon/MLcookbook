"""K-means clustering workflow.

K-means places k centroids to minimize the within-cluster sum of squared
Euclidean distances (*inertia*). Two consequences shape this workflow:

* Distances mix every encoded feature, so features must be on comparable
  scales (standard scaling by default) or the largest-unit feature dominates.
* Inertia always falls as k grows, so it cannot choose k by itself. The result
  scans k and reports both inertia (look for the "elbow") and the silhouette
  score (how much closer points are to their own cluster than to the next one;
  -1..1, higher is better), plus the k with the best silhouette.

Clustering has no target and no held-out set: preprocessing is fitted on all
rows, which is the whole dataset being described, not a leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result
from mlcookbook.core.schema import VarType
from mlcookbook.preprocessing.builders import build_preprocessor, resolve_config
from mlcookbook.preprocessing.config import PreprocessingConfig
from mlcookbook.preprocessing.features import resolve_features
from mlcookbook.preprocessing.summary import PreprocessingSummary, describe_preprocessor

SILHOUETTE_SAMPLE = 2000  # silhouette is O(n^2); score a random sample beyond this
WEAK_SILHOUETTE = 0.25
N_INIT = 10


@dataclass(frozen=True)
class KScore(Result):
    k: int
    inertia: float
    silhouette: float


@dataclass(frozen=True)
class ClusterProfile(Result):
    cluster: int
    size: int
    share: float
    numeric_means: dict[str, float]  # per numeric feature, in original units
    top_levels: dict[str, str]  # per categorical/boolean feature: most common level in the cluster


@dataclass(frozen=True)
class KMeansResult(Result):
    method: str
    features: list[str]
    n_rows: int
    n_clusters: int
    inertia: float  # within-cluster sum of squared distances, in preprocessed units
    silhouette: float
    labels: list[int]  # cluster per row, in dataset order
    profiles: list[ClusterProfile]
    k_scan: list[KScore]
    best_k_by_silhouette: int | None
    projection: list[list[float]]  # first two principal components of the preprocessed rows, for plotting
    projection_variance: list[float]  # share of variance those two components explain
    preprocessing: PreprocessingSummary
    warnings: list[str] = field(default_factory=list)


def kmeans(
    dataset: Dataset,
    features: list[str] | None = None,
    n_clusters: int = 3,
    max_k: int = 10,
    random_state: int = 0,
    preprocessing: PreprocessingConfig | None = None,
) -> KMeansResult:
    """Cluster rows into *n_clusters* groups with k-means.

    If *features* is None, every numeric/categorical/boolean/datetime column is
    used. The result profiles each cluster in original units, scans k from 2
    to *max_k* (inertia and silhouette), and projects rows onto two principal
    components so the clusters can be drawn.
    """
    features, warnings = resolve_features(dataset, None, features)
    n_rows = len(dataset.df)
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be at least 2; got {n_clusters}.")
    if n_rows <= n_clusters:
        raise ValueError(f"Need more rows ({n_rows}) than clusters ({n_clusters}).")
    config = resolve_config(preprocessing, scale=True)
    if config.scaling == "none":
        warnings.append(
            "K-means uses Euclidean distance; without scaling, features measured in large units dominate it."
        )
    X = dataset.df[features]
    preprocessor = build_preprocessor(dataset, features, config=config)
    matrix = preprocessor.fit_transform(X)
    if np.ptp(matrix, axis=0).max(initial=0) == 0:
        raise ValueError("Every feature is constant after preprocessing; there is nothing to cluster.")

    model = KMeans(n_clusters=n_clusters, n_init=N_INIT, random_state=random_state).fit(matrix)
    labels = model.labels_
    distinct = len(np.unique(labels))
    if distinct < n_clusters:
        warnings.append(f"Only {distinct} distinct clusters were found; the data has too few distinct rows.")
    silhouette = _silhouette(matrix, labels, random_state)
    if silhouette < WEAK_SILHOUETTE:
        warnings.append(
            f"Weak cluster structure (silhouette {silhouette:.2f} < {WEAK_SILHOUETTE}): "
            "k-means always returns clusters, but these may not reflect real groups."
        )

    k_scan = []
    for k in range(2, min(max_k, n_rows - 1) + 1):
        if k == n_clusters:
            k_scan.append(KScore(k, float(model.inertia_), silhouette))
            continue
        fitted = KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit(matrix)
        k_scan.append(KScore(k, float(fitted.inertia_), _silhouette(matrix, fitted.labels_, random_state)))
    scored = [s for s in k_scan if s.silhouette == s.silhouette]  # drop NaN
    best_k = max(scored, key=lambda s: s.silhouette).k if scored else None

    components = min(2, matrix.shape[1], n_rows)
    pca = PCA(n_components=components, random_state=random_state).fit(matrix)
    projected = pca.transform(matrix)
    if components == 1:
        projected = np.column_stack([projected, np.zeros(n_rows)])

    return KMeansResult(
        method="kmeans",
        features=list(features),
        n_rows=n_rows,
        n_clusters=n_clusters,
        inertia=float(model.inertia_),
        silhouette=silhouette,
        labels=[int(v) for v in labels],
        profiles=_profiles(dataset, features, labels, n_clusters),
        k_scan=k_scan,
        best_k_by_silhouette=best_k,
        projection=projected.tolist(),
        projection_variance=[float(v) for v in pca.explained_variance_ratio_],
        preprocessing=describe_preprocessor(preprocessor, X, dataset, config),
        warnings=warnings,
    )


def _silhouette(matrix: np.ndarray, labels: np.ndarray, random_state: int) -> float:
    if not 2 <= len(np.unique(labels)) <= len(labels) - 1:
        return float("nan")
    sample = SILHOUETTE_SAMPLE if len(labels) > SILHOUETTE_SAMPLE else None
    return float(silhouette_score(matrix, labels, sample_size=sample, random_state=random_state))


def _profiles(dataset: Dataset, features: list[str], labels: np.ndarray, n_clusters: int) -> list[ClusterProfile]:
    schema = dataset.schema()
    frame = dataset.df[features]
    numeric = [f for f in features if schema.var_type(f) is VarType.NUMERIC]
    categorical = [f for f in features if schema.var_type(f) in (VarType.CATEGORICAL, VarType.BOOLEAN)]
    profiles = []
    for cluster in range(n_clusters):
        rows = frame[labels == cluster]
        top_levels = {}
        for name in categorical:
            counts = rows[name].value_counts()
            if len(counts):
                top_levels[name] = str(counts.index[0])
        profiles.append(ClusterProfile(
            cluster=cluster,
            size=len(rows),
            share=len(rows) / len(frame),
            numeric_means={name: float(rows[name].mean()) for name in numeric},
            top_levels=top_levels,
        ))
    return profiles
