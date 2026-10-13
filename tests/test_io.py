import json

import pandas as pd
import pytest

from mlcookbook import decision_tree, linear_regression, logistic_regression, random_forest
from mlcookbook.io import load_dataset


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("algorithm,fixture", [(linear_regression, "linear_df"), (logistic_regression, "logistic_df"), (decision_tree, "logistic_df"), (random_forest, "logistic_df")])
def test_import_and_analyze(tmp_path, request, suffix, algorithm, fixture):
    frame = request.getfixturevalue(fixture)
    path = tmp_path / f"données.{suffix}"
    if suffix == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    data = load_dataset(path)
    result = algorithm(data, target="y", features=["x1", "group"])
    assert result.features == ["x1", "group"]
    assert result.n_train + result.n_test == len(frame)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("content,message", [
    ("", "header"), ("a,a\n1,2\n", "duplicate"), ("a,\n1,2", "non-empty"),
    ("a,b\n", "no data"), ("a,b\n1,2,3", "more fields"),
])
def test_bad_csv(tmp_path, content, message):
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_dataset(path)


def test_csv_encoding_delimiter_bom_and_quoted_fields(tmp_path):
    path = tmp_path / "data.CSV"
    path.write_text('city;value\n"Montréal;QC";3\nParis;\n', encoding="cp1252")
    data = load_dataset(path, delimiter=";", encoding="cp1252")
    assert data.df.city.iloc[0] == "Montréal;QC"
    assert data.df.value.isna().sum() == 1
    path.write_text("city,value\nParis,3", encoding="utf-8-sig")
    assert list(load_dataset(path).df.columns) == ["city", "value"]


def test_parquet_types_and_nulls(tmp_path):
    path = tmp_path / "data.parquet"
    frame = pd.DataFrame({"category": pd.Categorical(["a", "b", None]), "flag": pd.Series([True, False, None], dtype="boolean"), "when": pd.date_range("2026-01-01", periods=3)})
    frame.to_parquet(path)
    data = load_dataset(path)
    pd.testing.assert_frame_equal(data.df, frame)
    assert data.schema().var_type("category").value == "categorical"
    assert data.schema().var_type("when").value == "datetime"


def test_reject_nested_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    pd.DataFrame({"nested": [[1, 2], [3]]}).to_parquet(path)
    with pytest.raises(ValueError, match="nested"):
        load_dataset(path)


def test_invalid_file_and_extension(tmp_path):
    with pytest.raises(ValueError, match="Choose"):
        load_dataset(tmp_path / "data.json")
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")
    path = tmp_path / "broken.parquet"
    path.write_text("not parquet")
    with pytest.raises(ValueError):
        load_dataset(path)

@pytest.mark.parametrize("algorithm", [linear_regression, logistic_regression, decision_tree, random_forest])
def test_nullable_parquet_features(tmp_path, logistic_df, algorithm):
    frame = logistic_df.copy()
    frame["group"] = frame["group"].astype("string")
    frame.loc[:8, "group"] = pd.NA
    frame["flag"] = pd.Series([True, False] * 200, dtype="boolean")
    frame.loc[:5, "flag"] = pd.NA
    if algorithm is linear_regression:
        frame["y"] = frame["x1"] * 3
    path = tmp_path / "nullable.parquet"
    frame.to_parquet(path)
    result = algorithm(load_dataset(path), target="y", features=["x1", "group", "flag"])
    assert result.n_test == 100
    json.dumps(result.to_dict(), allow_nan=False)


def test_leading_zero_codes_stay_text(tmp_path):
    from mlcookbook.io import load_dataset_with_report

    path = tmp_path / "zips.csv"
    path.write_text("zip,amount\n01234,1.5\n98765,2.0\n00501,0.5\n", encoding="utf-8")
    data, report = load_dataset_with_report(path)
    assert data.df["zip"].tolist() == ["01234", "98765", "00501"]
    assert data.df["amount"].dtype == float  # "0.5" is not a leading-zero code
    assert report.kept_as_text == ["zip"]
    assert load_dataset(path, keep_leading_zeros=False).df["zip"].tolist() == [1234, 98765, 501]


def test_missing_markers_csv_and_parquet(tmp_path):
    from mlcookbook.io import load_dataset_with_report

    csv_path = tmp_path / "survey.csv"
    csv_path.write_text("score,answer\n3,yes\n-999,?\n5,no\n", encoding="utf-8")
    data, report = load_dataset_with_report(csv_path, missing_markers=["-999", "?"])
    assert data.df["score"].isna().tolist() == [False, True, False]
    assert data.df["answer"].isna().tolist() == [False, True, False]
    assert report.missing_markers == ["-999", "?"]

    parquet_path = tmp_path / "survey.parquet"
    pd.DataFrame({"score": [3, -999, 5], "answer": ["yes", "?", "no"]}).to_parquet(parquet_path)
    data, report = load_dataset_with_report(parquet_path, missing_markers=["-999", "?"])
    assert data.df["score"].isna().sum() == 1 and data.df["answer"].isna().sum() == 1
    assert len(report.notes) == 2


def test_iso_dates_are_parsed_and_ambiguous_dates_are_not(tmp_path):
    from mlcookbook.io import load_dataset_with_report

    path = tmp_path / "events.csv"
    path.write_text(
        "day,stamp,us_style,mixed_zone\n"
        "2024-01-31,2024-01-31T10:00:00+02:00,01/31/2024,2024-01-31T10:00:00Z\n"
        "2024-02-29,2024-02-29T23:30:00Z,02/29/2024,2024-02-29T10:00:00\n"
        ",2024-03-01T00:00:00-05:00,03/01/2024,\n",
        encoding="utf-8",
    )
    data, report = load_dataset_with_report(path)
    assert report.parsed_dates == ["day", "stamp"]
    assert data.schema().var_type("day").value == "datetime"
    assert data.df["day"].isna().tolist() == [False, False, True]
    assert str(data.df["stamp"].dt.tz) == "UTC"
    assert data.df["stamp"].iloc[0] == pd.Timestamp("2024-01-31T08:00:00Z")
    assert data.df["us_style"].dtype == object and data.df["mixed_zone"].dtype == object
    assert load_dataset(path, parse_dates=False).df["day"].dtype == object
    json.dumps(report.to_dict())


def test_imported_dates_feed_a_workflow(tmp_path, linear_df):
    frame = linear_df.assign(when=pd.date_range("2024-01-01", periods=len(linear_df)).strftime("%Y-%m-%d"))
    path = tmp_path / "dated.csv"
    frame.to_csv(path, index=False)
    result = linear_regression(load_dataset(path), target="y")
    assert "when" in result.features
    assert "when_days" in result.coefficients
