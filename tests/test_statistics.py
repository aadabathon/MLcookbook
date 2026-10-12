"""Correlation intervals, chi-square, ANOVA/Kruskal-Wallis, and the association scan."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mlcookbook import Dataset, anova, association_scan, chi_square, correlation


def test_correlation_interval_contains_estimate_and_narrows_with_n():
    rng = np.random.default_rng(6)
    x = rng.normal(size=1000)
    frame = pd.DataFrame({"x": x, "y": 0.5 * x + rng.normal(size=1000)})
    small = correlation(Dataset(frame.head(30)), "x", "y")
    large = correlation(Dataset(frame), "x", "y")
    for r in (small, large):
        assert r.ci_low < r.statistic < r.ci_high and r.confidence_level == 0.95
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)
    expected = stats.pearsonr(frame.x, frame.y).confidence_interval(0.95)
    assert (large.ci_low, large.ci_high) == pytest.approx((expected.low, expected.high), abs=1e-6)
    json.dumps(large.to_dict(), allow_nan=False)


def test_spearman_interval_and_tiny_samples():
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8], "b": [2.0, 1, 4, 3, 6, 5, 8, 7]})
    result = correlation(Dataset(frame), "a", "b", method="spearman", confidence_level=0.9)
    assert result.ci_low < result.statistic < result.ci_high
    tiny = correlation(Dataset(frame.head(3)), "a", "b")
    assert np.isnan(tiny.ci_low) and tiny.to_dict()["ci_low"] is None


def test_pearson_warns_about_skewed_data():
    rng = np.random.default_rng(1)
    x = rng.exponential(size=300)
    result = correlation(Dataset(pd.DataFrame({"x": x, "y": x + rng.normal(size=300)})), "x", "y")
    assert any("'x' departs from normality" in w for w in result.warnings)
    assert not correlation(Dataset(pd.DataFrame({"x": x, "y": x})), "x", "y", method="spearman").warnings


def test_chi_square_detects_dependence():
    rng = np.random.default_rng(3)
    smoker = rng.choice(["yes", "no"], 400)
    cough = np.where(rng.random(400) < np.where(smoker == "yes", 0.7, 0.2), "cough", "none")
    result = chi_square(Dataset(pd.DataFrame({"smoker": smoker, "cough": cough})), "smoker", "cough")
    assert result.p_value < 1e-10 and 0.3 < result.cramers_v < 0.7
    assert result.dof == 1 and result.n_obs == 400
    assert result.fisher_p_value is not None and result.fisher_p_value < 1e-10
    assert np.array(result.observed).sum() == 400
    assert result.x_levels == ["no", "yes"]
    json.dumps(result.to_dict(), allow_nan=False)


def test_chi_square_independent_and_low_expected_counts():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"a": rng.choice(list("xyz"), 600), "b": rng.choice(list("pq"), 600)})
    result = chi_square(Dataset(frame), "a", "b")
    assert result.p_value > 0.01 and result.cramers_v < 0.15 and result.fisher_p_value is None
    sparse = pd.DataFrame({"a": list("xxxyyz"), "b": list("pqpqpq")})
    assert any("unreliable" in w for w in chi_square(Dataset(sparse), "a", "b").warnings)


def test_chi_square_validation(mixed_df):
    with pytest.raises(ValueError, match="categorical or boolean"):
        chi_square(Dataset(mixed_df), "city", "income")
    with pytest.raises(ValueError, match="two different"):
        chi_square(Dataset(mixed_df), "city", "city")
    with pytest.raises(ValueError, match="two observed levels"):
        chi_square(Dataset(mixed_df), "city", "constant")


def test_anova_detects_group_differences_and_agrees_with_scipy():
    rng = np.random.default_rng(7)
    group = np.repeat(["a", "b", "c"], 50)
    value = rng.normal(0, 1, 150) + pd.Series(group).map({"a": 0, "b": 0.5, "c": 1.5}).to_numpy()
    result = anova(Dataset(pd.DataFrame({"value": value, "group": group})), "value", "group")
    samples = [value[group == g] for g in "abc"]
    assert result.f_statistic == pytest.approx(stats.f_oneway(*samples).statistic)
    assert result.p_value < 1e-6 and result.kruskal_p_value < 1e-6
    assert 0.2 < result.eta_squared < 0.6
    assert [g.level for g in result.groups] == ["a", "b", "c"] and result.n_obs == 150
    assert result.recommended_test == "anova" and result.normality_ok
    json.dumps(result.to_dict(), allow_nan=False)


def test_anova_recommends_kruskal_when_assumptions_fail():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "value": np.concatenate([rng.exponential(1, 20), rng.normal(0, 10, 20)]),
        "group": ["skewed"] * 20 + ["wide"] * 20,
    })
    frame.loc[0, "group"] = "lonely"
    result = anova(Dataset(frame), "value", "group")
    assert result.recommended_test == "kruskal"
    assert any("Levene" in w for w in result.warnings)
    assert any("lonely" in w for w in result.warnings)


def test_anova_validation(mixed_df):
    with pytest.raises(ValueError, match="numeric value"):
        anova(Dataset(mixed_df), "city", "is_member")
    with pytest.raises(ValueError, match="group column"):
        anova(Dataset(mixed_df), "income", "age")


def test_association_scan_picks_tests_by_type_and_ranks_by_effect():
    rng = np.random.default_rng(5)
    n = 300
    size = rng.normal(size=n)
    frame = pd.DataFrame({
        "size": size,
        "price": 3 * size + rng.normal(0, 0.5, n),
        "noise": rng.normal(size=n),
        "tier": np.where(size > 0, "large", "small"),
        "colour": rng.choice(["red", "blue"], n),
        "notes": [f"free text {i}" for i in range(n)],
    })
    result = association_scan(Dataset(frame))
    assert result.columns == ["size", "price", "noise", "tier", "colour"]
    assert result.n_pairs == 10
    assert any("notes" in w for w in result.warnings)
    tests = {(p.x, p.y): p.test for p in result.pairs}
    assert tests[("size", "price")] == "spearman"
    assert tests[("tier", "colour")] == "chi_square"
    assert tests[("size", "tier")] == "kruskal"
    top = result.pairs[0]
    assert {top.x, top.y} == {"size", "price"} and top.effect_size > 0.9
    effects = [p.effect_size for p in result.pairs]
    assert effects == sorted(effects, reverse=True)
    assert all(p.p_adjusted >= p.p_value for p in result.pairs)
    json.dumps(result.to_dict(), allow_nan=False)


def test_association_scan_validation(mixed_df):
    with pytest.raises(ValueError, match="at least two"):
        association_scan(Dataset(mixed_df), ["age"])
    with pytest.raises(ValueError, match="type"):
        association_scan(Dataset(mixed_df), ["age", "notes"])
    result = association_scan(Dataset(mixed_df), ["age", "income", "constant"])
    assert result.n_pairs == 1 and any("constant" in w for w in result.warnings)
