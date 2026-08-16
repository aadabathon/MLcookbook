import json

import pandas as pd
import pytest

from mlcookbook import Dataset, VarType


def test_schema_inference_covers_all_types(mixed_df):
    schema = Dataset(mixed_df).schema()
    assert schema.var_type("age") is VarType.NUMERIC
    assert schema.var_type("income") is VarType.NUMERIC
    assert schema.var_type("city") is VarType.CATEGORICAL
    assert schema.var_type("is_member") is VarType.BOOLEAN
    assert schema.var_type("signup") is VarType.DATETIME
    assert schema.var_type("notes") is VarType.TEXT
    assert schema.var_type("constant") is VarType.CATEGORICAL


def test_unsupported_type_for_nested_objects():
    df = pd.DataFrame({"blob": [[1, 2], [3], [4, 5]]})
    assert Dataset(df).schema().var_type("blob") is VarType.UNSUPPORTED


def test_overrides_at_init_and_via_set_type(mixed_df):
    data = Dataset(mixed_df, overrides={"age": "categorical"})
    assert data.schema().var_type("age") is VarType.CATEGORICAL
    assert data.schema()["age"].overridden

    data.set_type("age", VarType.NUMERIC)
    assert data.schema().var_type("age") is VarType.NUMERIC

    with pytest.raises(KeyError):
        data.set_type("nope", "numeric")


def test_describe_metadata(mixed_df):
    summary = Dataset(mixed_df).describe()
    assert summary.n_rows == 50
    assert summary.n_columns == 7
    assert summary.constant_columns == ["constant"]

    by_name = {c.name: c for c in summary.columns}
    assert by_name["age"].missing_count == 5
    assert by_name["age"].missing_fraction == pytest.approx(0.1)
    assert by_name["city"].unique_count == 3
    assert by_name["constant"].is_constant


def test_describe_is_json_serializable(mixed_df):
    payload = Dataset(mixed_df).describe().to_dict()
    json.dumps(payload)  # must not raise
    assert payload["columns"][0]["var_type"] == "numeric"


def test_dataset_rejects_non_dataframe_and_duplicate_columns():
    with pytest.raises(TypeError):
        Dataset([1, 2, 3])
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        Dataset(df)


def test_dataset_copies_input(mixed_df):
    data = Dataset(mixed_df)
    mixed_df.loc[0, "income"] = -1.0
    assert data.df.loc[0, "income"] != -1.0
