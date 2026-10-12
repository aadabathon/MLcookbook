"""K-means clustering and PCA."""
import json

import numpy as np
import pandas as pd
import pytest

from mlcookbook import Dataset, PreprocessingConfig, kmeans, pca


@pytest.fixture
def blobs_df() -> pd.DataFrame:
    """Three tight, well-separated blobs in (height, weight), plus a categorical tag."""
    rng = np.random.default_rng(9)
    centres = [(150, 50), (170, 70), (190, 90)]
    rows = []
    for i, (h, w) in enumerate(centres):
        for _ in range(60):
            rows.append({"height": h + rng.normal(0, 2), "weight": w + rng.normal(0, 2), "tag": "abc"[i]})
    return pd.DataFrame(rows)


def test_kmeans_recovers_blobs(blobs_df):
    result = kmeans(Dataset(blobs_df), features=["height", "weight"], n_clusters=3)
    assert result.n_rows == len(blobs_df) == len(result.labels)
    assert sorted(p.size for p in result.profiles) == [60, 60, 60]
    assert result.silhouette > 0.7
    assert result.best_k_by_silhouette == 3
    means = sorted(p.numeric_means["height"] for p in result.profiles)
    assert means == pytest.approx([150, 170, 190], abs=1.5)  # profiles are in original units
    assert sum(p.share for p in result.profiles) == pytest.approx(1)
    json.dumps(result.to_dict(), allow_nan=False)


def test_kmeans_scan_inertia_falls_with_k(blobs_df):
    result = kmeans(Dataset(blobs_df), features=["height", "weight"], n_clusters=3, max_k=6)
    assert [s.k for s in result.k_scan] == [2, 3, 4, 5, 6]
    inertias = [s.inertia for s in result.k_scan]
    assert all(a >= b for a, b in zip(inertias, inertias[1:]))
    assert next(s for s in result.k_scan if s.k == 3).inertia == pytest.approx(result.inertia)


def test_kmeans_profiles_categoricals_and_projects_to_2d(blobs_df):
    result = kmeans(Dataset(blobs_df), n_clusters=3)
    assert result.features == ["height", "weight", "tag"]
    assert {p.top_levels["tag"] for p in result.profiles} == {"a", "b", "c"}
    assert np.array(result.projection).shape == (len(blobs_df), 2)
    assert 0 < sum(result.projection_variance) <= 1


def test_kmeans_warns_on_weak_structure_and_unscaled_features():
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({f"x{i}": rng.normal(size=200) for i in range(6)})  # no groups at all
    assert any("Weak cluster structure" in w for w in kmeans(Dataset(noise), n_clusters=4).warnings)
    noise["x0"] *= 1000
    result = kmeans(Dataset(noise), n_clusters=4, preprocessing=PreprocessingConfig(scaling="none"))
    assert any("large units dominate" in w for w in result.warnings)


def test_kmeans_validation(blobs_df):
    with pytest.raises(ValueError, match="n_clusters"):
        kmeans(Dataset(blobs_df), n_clusters=1)
    with pytest.raises(ValueError, match="more rows"):
        kmeans(Dataset(blobs_df.head(3)), n_clusters=3)
    with pytest.raises(ValueError, match="constant"):
        kmeans(Dataset(pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10})), n_clusters=2)


def test_pca_finds_the_shared_direction():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=500)
    frame = pd.DataFrame({
        "a": latent + rng.normal(0, 0.1, 500),
        "b": 2 * latent + rng.normal(0, 0.2, 500),
        "c": rng.normal(size=500),
    })
    result = pca(Dataset(frame))
    pc1, pc2, pc3 = result.components
    assert result.n_components == 3 and pc1.index == 1
    assert pc1.explained_variance_ratio > 0.6
    assert abs(pc1.loadings["a"]) == pytest.approx(abs(pc1.loadings["b"]), abs=0.05)  # standardized
    assert abs(pc1.loadings["c"]) < 0.2
    assert pc3.cumulative_ratio == pytest.approx(1)
    assert sum(w ** 2 for w in pc1.loadings.values()) == pytest.approx(1)
    assert result.components_for_90_percent == 2
    assert pc1.eigenvalue > 1.9  # a and b share one direction worth about two variables
    assert result.kaiser_components == sum(c.eigenvalue > 1 for c in result.components)
    assert np.array(result.scores).shape == (500, 2)
    json.dumps(result.to_dict(), allow_nan=False)


def test_pca_without_scaling_follows_units():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"grams": rng.normal(0, 1000, 300), "kilos": rng.normal(0, 1, 300)})
    result = pca(Dataset(frame), preprocessing=PreprocessingConfig(scaling="none"))
    assert abs(result.components[0].loadings["grams"]) > 0.99
    assert result.kaiser_components is None
    assert any("large units" in w for w in result.warnings)


def test_pca_component_count_and_categoricals(mixed_df):
    result = pca(Dataset(mixed_df), features=["age", "income", "city"], n_components=2)
    assert result.n_components == 2 and len(result.encoded_features) == 5
    assert any("one-hot" in w for w in result.warnings)
    with pytest.raises(ValueError, match="n_components"):
        pca(Dataset(mixed_df), features=["age", "income"], n_components=3)
