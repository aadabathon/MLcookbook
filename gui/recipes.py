"""Recipe definitions — the data that drives the front-page grid.

Adding a recipe to the app = appending one ``Recipe`` here. Icons are unicode
glyphs (Greek/math symbols keep the scientific feel without an asset pipeline).

``status`` is the label shown on the card's pill; ``variant`` picks its style:
    "ready"   — filled violet pill (implemented in the mlcookbook core)
    "planned" — outlined violet pill (on the roadmap)
    "soon"    — dim outlined pill (placeholder)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    title: str
    description: str
    icon: str = "∴"  # ∴
    status: str = "Planned"
    variant: str = "planned"
    key: str = ""  # routing key; app.py maps it to a screen factory. "" = no screen yet.


RECIPES: list[Recipe] = [
    Recipe(
        title="Linear Regression",
        description="Least squares, ridge, or lasso, with residual checks and cross-validated error.",
        icon="β",  # β
        status="Ready",
        variant="ready",
        key="linear-regression",
    ),
    Recipe(
        title="Logistic Regression",
        description="Binary or multinomial classification via the sigmoid / softmax, with probability-aware metrics.",
        icon="σ",  # σ
        status="Ready",
        variant="ready",
        key="logistic-regression",
    ),
    Recipe(
        title="Decision Trees",
        description="Interpretable recursive splits for classes or numbers, drawn as the fitted tree.",
        icon="τ",  # τ
        status="Ready",
        variant="ready",
        key="decision-tree",
    ),
    Recipe(
        title="Random Forests",
        description="Bootstrap-aggregated tree ensembles for classification or regression, with importances.",
        icon="⁂",  # ⁂
        status="Ready",
        variant="ready",
        key="random-forest",
    ),
    Recipe(
        title="K-Means",
        description="Centroid-based clustering with cluster profiles, silhouette, and an elbow scan.",
        icon="∴",  # ∴
        status="Ready",
        variant="ready",
        key="kmeans",
    ),
    Recipe(
        title="PCA",
        description="Principal component analysis: variance-preserving dimensionality reduction.",
        icon="λ",  # λ
        status="Ready",
        variant="ready",
        key="pca",
    ),
    Recipe(
        title="Association Scan",
        description="Every column pair tested by type: Spearman, chi-square, or Kruskal-Wallis, FDR-adjusted.",
        icon="χ",  # χ
        status="Ready",
        variant="ready",
        key="association-scan",
    ),
    Recipe(
        title="Support Vector Machines",
        description="Maximum-margin classifiers and regressors with kernels for nonlinear boundaries.",
        icon="∥",  # ∥
        status="Ready",
        variant="ready",
        key="svm",
    ),
    Recipe(
        title="Neural Networks",
        description="Multilayer perceptrons with loss and validation curves, for classes or numbers.",
        icon="ψ",  # ψ
        status="Ready",
        variant="ready",
        key="neural-network",
    ),
]
