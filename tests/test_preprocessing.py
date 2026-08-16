import pytest

from mlcookbook import Dataset
from mlcookbook.preprocessing import build_preprocessor


def test_transforms_numeric_and_categorical(mixed_df):
    data = Dataset(mixed_df)
    pre = build_preprocessor(data, ["age", "income", "city", "is_member"])
    X = pre.fit_transform(data.df)
    # 2 scaled numeric + 3 city categories + 2 booleans = 7 output features
    assert X.shape == (len(mixed_df), 7)
    names = list(pre.get_feature_names_out())
    assert "age" in names
    assert any(n.startswith("city_") for n in names)


def test_rejects_unusable_columns(mixed_df):
    with pytest.raises(ValueError, match="not supported"):
        build_preprocessor(Dataset(mixed_df), ["age", "signup"])
