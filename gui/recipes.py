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
        description="Ordinary least squares with leakage-safe preprocessing and held-out evaluation.",
        icon="β",  # β
        status="Ready",
        variant="ready",
        key="linear-regression",
    ),
    Recipe(
        title="Logistic Regression",
        description="Binary classification via the sigmoid, with probability-aware metrics.",
        icon="σ",  # σ
    ),
    Recipe(
        title="Decision Trees",
        description="Interpretable recursive splits for regression and classification.",
        icon="τ",  # τ
    ),
    Recipe(
        title="Random Forests",
        description="Bootstrap-aggregated tree ensembles with feature importances.",
        icon="⁂",  # ⁂
    ),
    Recipe(
        title="Support Vector Machines",
        description="Maximum-margin classifiers with kernel tricks for nonlinear boundaries.",
        icon="∥",  # ∥
    ),
    Recipe(
        title="PCA",
        description="Principal component analysis: variance-preserving dimensionality reduction.",
        icon="λ",  # λ
    ),
    Recipe(
        title="K-Means",
        description="Centroid-based clustering with silhouette and inertia diagnostics.",
        icon="∴",  # ∴
    ),
    Recipe(
        title="Neural Networks",
        description="Multilayer perceptrons for when linear just isn't enough.",
        icon="ψ",  # ψ
    ),
    Recipe(
        title="Coming Soon",
        description="A growing collection — the next recipe is always simmering.",
        icon="…",  # …
        status="Coming Soon",
        variant="soon",
    ),
]
