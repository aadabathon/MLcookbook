"""Public data connectors are bounded, typed, explicit and testable offline."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from mlcookbook import gathering


def fake_source(monkeypatch, response):
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return deepcopy(response(url) if callable(response) else response)

    monkeypatch.setattr(gathering, "fetch_json", fetch)
    return calls


def event(identifier="source-001", magnitude=4.8):
    return {"type": "Feature", "id": identifier,
            "properties": {"time": 1704067200000, "mag": magnitude, "magType": "mw",
                           "place": "Near a test location", "status": "reviewed"},
            "geometry": {"type": "Point", "coordinates": [-120.0, 35.0, 12.5]}}


def events(*items):
    return {"type": "FeatureCollection", "features": list(items)}


def product(code, **nutrients):
    return {"code": code, "product_name": "Céréales", "brands": "Example", "categories_tags": ["en:cereals"],
            "ingredients_text": "Oats, sugar", "nutriments": nutrients}


def test_catalog_has_five_independent_specs_without_network(monkeypatch):
    monkeypatch.setattr(gathering, "fetch_json", lambda *a, **kw: pytest.fail("Catalog must be offline"))
    specs = gathering.source_specs()
    assert [source["key"] for source in specs] == ["usgs", "food", "exoplanets", "census", "weather"]
    for spec in specs:
        keys = [column["key"] for column in spec["columns"]]
        assert len(keys) == len(set(keys))
        assert set(spec["default_columns"]) <= set(keys)
        assert 0 < spec["default_limit"] <= spec["max_limit"] <= 5000
        assert spec["url"].startswith("https://")
    census_key = next(field for field in specs[3]["fields"] if field["key"] == "api_key")
    assert census_key["secret"] and census_key["required"]
    specs[0]["fields"][0]["default"] = "invalid"
    assert gathering.source_specs()[0]["fields"][0]["default"] != "invalid"


def test_usgs_preserves_ids_units_dates_roles_and_truncation(monkeypatch):
    calls = fake_source(monkeypatch, events(event(), event("source-002")))
    query = {"start_date": "2024-01-01", "end_date": "2024-01-02", "region": "california", "limit": 1,
             "columns": ["event_id", "time", "magnitude", "depth_km"]}
    original = deepcopy(query)
    gathered = gathering.fetch_dataset("usgs", query)
    assert query == original
    assert gathered.dataset.df.event_id.tolist() == ["source-001"]
    assert gathered.dataset.df.time.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert gathered.dataset.schema().var_type("event_id").value == "unsupported"
    assert gathered.dataset.schema().var_type("magnitude").value == "numeric"
    assert gathered.result.units["depth_km"] == "km" and gathered.result.truncated
    assert gathered.result.n_rows == 1 and gathered.result.n_columns == 4
    assert any("not a random sample" in warning for warning in gathered.result.warnings)
    parameters = parse_qs(urlsplit(calls[0][0]).query)
    assert parameters["limit"] == ["2"] and parameters["orderby"] == ["time-asc"]
    assert parameters["minlatitude"] == ["32"]
    assert parameters["endtime"] == ["2024-01-02T23:59:59.999Z"]
    metadata = json.loads(json.dumps(gathered.result.to_dict(), allow_nan=False))
    assert "dataset" not in metadata and "df" not in metadata
    assert datetime.fromisoformat(metadata["retrieved_at"]).utcoffset() == UTC.utcoffset(None)
    with pytest.raises(FrozenInstanceError):
        gathered.result.n_rows = 9


def test_duplicate_events_warn_and_malformed_numeric_values_are_explicit_missing(monkeypatch):
    fake_source(monkeypatch, events(event(magnitude="unknown"), event(magnitude="unknown")))
    gathered = gathering.fetch_dataset("usgs", {"limit": 5, "columns": ["magnitude", "event_id"]})
    assert len(gathered.dataset.df) == 1 and gathered.dataset.df.magnitude.isna().all()
    assert any("duplicate" in warning for warning in gathered.result.warnings)
    assert any("malformed" in warning for warning in gathered.result.warnings)


@pytest.mark.parametrize("response", [None, events(), {"features": []}, events({"id": "x"}),
                                     events(dict(event(), geometry={"type": "Point", "coordinates": [1, 2]}))])
def test_empty_or_malformed_event_responses_do_not_return_partial_data(monkeypatch, response):
    fake_source(monkeypatch, response)
    with pytest.raises(ValueError):
        gathering.fetch_dataset("usgs", {})


@pytest.mark.parametrize("source,query", [
    ("unknown", {}), ("usgs", {"limit": True}), ("usgs", {"limit": 5001}),
    ("usgs", {"columns": []}), ("usgs", {"columns": ["magnitude", "magnitude"]}),
    ("usgs", {"columns": ["no-such-column"]}), ("usgs", {"url": "https://example.com"}),
    ("usgs", {"region": "unbounded"}), ("usgs", {"min_magnitude": float("inf")}),
    ("usgs", {"start_date": "2024-02-02", "end_date": "2024-01-01"}),
    ("usgs", {"start_date": "2024-1-1"}), ("food", {"category": "arbitrary free text"}),
    ("food", {"limit": 501}), ("exoplanets", {"discovery_method": "' or 1=1"}),
    ("exoplanets", {"complete_only": "true"}), ("census", {}), ("census", {"api_key": "invalid"}),
    ("weather", {"columns": ["date"]}), ("weather", {"latitude": 100}),
    ("weather", {"start_date": "1939-01-01"}),
])
def test_invalid_queries_are_rejected_before_contacting_source(monkeypatch, source, query):
    monkeypatch.setattr(gathering, "fetch_json", lambda *a, **kw: pytest.fail("Invalid query made a request"))
    with pytest.raises((ValueError, TypeError)):
        gathering.fetch_dataset(source, query)


def test_food_pages_deduplicate_and_keep_barcode_text(monkeypatch):
    monkeypatch.setattr(gathering, "_food_pause", lambda *args: None)

    def response(url):
        page = parse_qs(urlsplit(url).query)["page"][0]
        return {"count": 4, "products": ([product("00001", sugars_100g=12), product("00002")]
                                          if page == "1" else [product("00002"), product("00003", sugars_100g="bad")])}

    calls = fake_source(monkeypatch, response)
    gathered = gathering.fetch_dataset("food", {"category": "en:breakfast-cereals", "country": "en:united-states",
                                                "limit": 3, "columns": ["barcode", "product_name", "sugars_100g"]})
    assert gathered.dataset.df.barcode.tolist() == ["00001", "00002", "00003"]
    assert gathered.dataset.df.product_name.tolist() == ["Céréales"] * 3
    assert gathered.dataset.df.sugars_100g.isna().tolist() == [False, True, True]
    assert len(calls) == 2 and gathered.result.truncated
    assert any("duplicate" in warning for warning in gathered.result.warnings)
    assert gathered.result.units["sugars_100g"] == "g / 100 g"
    assert parse_qs(urlsplit(calls[0][0]).query)["countries_tags"] == ["en:united-states"]


def test_exoplanets_selects_composite_rows_with_checked_adql(monkeypatch):
    calls = fake_source(monkeypatch, [{"pl_name": "Planet A", "pl_bmasse": 4.0, "pl_rade": 2.0},
                                      {"pl_name": "Planet B", "pl_bmasse": None, "pl_rade": 1.0}])
    gathered = gathering.fetch_dataset("exoplanets", {"discovery_method": "Transit", "complete_only": True,
                                                     "limit": 1, "columns": ["pl_bmasse", "pl_rade"]})
    adql = parse_qs(urlsplit(calls[0][0]).query)["query"][0]
    assert "from pscomppars" in adql and "select top 2" in adql and "order by pl_name" in adql
    assert "discoverymethod = 'Transit'" in adql and "pl_rade is not null" in adql
    assert gathered.dataset.df.columns.tolist() == ["pl_bmasse", "pl_rade"]
    assert gathered.dataset.df.pl_bmasse.iloc[0] == 4
    assert any("inferred" in warning for warning in gathered.result.warnings)


@pytest.mark.parametrize("response", [[{"pl_name": "a"}, {"pl_name": "a"}], [{"pl_name": "a"}], {"error": "bad query"}])
def test_exoplanet_schema_changes_fail_without_fabricating_columns(monkeypatch, response):
    fake_source(monkeypatch, response)
    with pytest.raises(ValueError):
        gathering.fetch_dataset("exoplanets", {"columns": ["pl_name", "pl_rade"]})


def test_census_joins_fields_by_header_preserves_geography_and_strips_secret(monkeypatch):
    key = "abcdef1234567890" * 3
    calls = fake_source(monkeypatch, [["county", "B19013_001E", "NAME", "state", "B01003_001E"],
                                      ["003", "-666666666", "Area B", "01", "2000"],
                                      ["001", "50000", "Area A", "01", "1000"]])
    gathered = gathering.fetch_dataset("census", {"api_key": key, "state": "01", "limit": 5,
                                                 "columns": ["geoid", "population", "median_household_income"]})
    frame = gathered.dataset.df
    assert frame.geoid.tolist() == ["01001", "01003"]
    assert frame.population.tolist() == [1000, 2000]
    assert frame.median_household_income.isna().tolist() == [False, True]
    assert key in calls[0][0] and key not in json.dumps(gathered.result.to_dict())
    assert "key=" not in gathered.result.request_urls[0]
    assert any("sentinel" in warning for warning in gathered.result.warnings)
    assert gathered.dataset.schema().var_type("geoid").value == "unsupported"


def weather_response():
    return {"daily_units": {"time": "iso8601", "temperature_2m_mean": "°C", "precipitation_sum": "mm"},
            "daily": {"time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                      "temperature_2m_mean": [1.5, None, 3.5], "precipitation_sum": [0, 2, 0]}}


def test_weather_keeps_utc_days_checked_units_and_bounds_actual_request(monkeypatch):
    calls = fake_source(monkeypatch, weather_response())
    gathered = gathering.fetch_dataset("weather", {"start_date": "2024-01-01", "end_date": "2024-01-20", "limit": 2,
                                                  "columns": ["date", "temperature_2m_mean", "precipitation_sum"]})
    assert len(gathered.dataset.df) == 2 and gathered.result.truncated
    assert gathered.dataset.df.date.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert gathered.dataset.df.temperature_2m_mean.isna().tolist() == [False, True]
    parameters = parse_qs(urlsplit(calls[0][0]).query)
    assert parameters["end_date"] == ["2024-01-03"] and parameters["models"] == ["era5"]
    assert gathered.result.units["temperature_2m_mean"] == "°C"


@pytest.mark.parametrize("change", ["units", "length", "order", "incomplete"])
def test_weather_rejects_misaligned_or_changed_units(monkeypatch, change):
    response = weather_response()
    if change == "units":
        response["daily_units"]["temperature_2m_mean"] = "°F"
    elif change == "length":
        response["daily"]["temperature_2m_mean"].pop()
    elif change == "order":
        response["daily"]["time"].reverse()
    else:
        for values in response["daily"].values():
            values.pop()
    fake_source(monkeypatch, response)
    with pytest.raises(ValueError):
        gathering.fetch_dataset("weather", {"start_date": "2024-01-01", "end_date": "2024-01-03", "limit": 5,
                                            "columns": ["date", "temperature_2m_mean", "precipitation_sum"]})


def test_cancel_before_and_after_response_never_publishes_partial_result(monkeypatch):
    calls = fake_source(monkeypatch, events(event()))
    with pytest.raises(InterruptedError):
        gathering.fetch_dataset("usgs", {}, cancelled=lambda: True)
    assert not calls
    stopped = False

    def response(url):
        nonlocal stopped
        stopped = True
        return events(event())

    fake_source(monkeypatch, response)
    with pytest.raises(InterruptedError):
        gathering.fetch_dataset("usgs", {}, cancelled=lambda: stopped)


def test_food_rate_wait_is_cancellable(monkeypatch):
    monkeypatch.setattr(gathering, "_FOOD_NEXT_REQUEST", gathering.monotonic() + 20)
    with pytest.raises(InterruptedError):
        gathering._food_pause(lambda: True, gathering.monotonic() + 30)
