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


def test_historical_earthquake_timestamps_work_before_unix_epoch_on_windows(monkeypatch):
    historical = event("historical-1900")
    historical["properties"]["time"] = -2208988800000
    fake_source(monkeypatch, events(historical))
    gathered = gathering.fetch_dataset("usgs", {"start_date": "1900-01-01", "end_date": "1900-01-02",
                                               "columns": ["event_id", "time"]})
    assert gathered.dataset.df.time.iloc[0] == pd.Timestamp("1900-01-01", tz="UTC")
    assert gathered.dataset.df.event_id.iloc[0] == "historical-1900"


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
    assert "from pscomppars" in adql and "select top 5001" in adql and "order by pl_name" in adql
    assert "discoverymethod = 'Transit'" in adql and "is not null" not in adql
    assert gathered.dataset.df.columns.tolist() == ["pl_bmasse", "pl_rade"]
    assert gathered.dataset.df.pl_bmasse.iloc[0] == 4
    assert any("inferred" in warning for warning in gathered.result.warnings)
    assert not gathered.result.truncated
    assert gathered.result.query["complete_only"] is True
    assert gathered.result.request_urls == [calls[0][0]]
    json.dumps(gathered.result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("method,complete,limit_clause,constraints", [
    ("all", False, "select top 3", []),
    ("all", True, "select top 3", ["pl_bmasse is not null", "pl_rade is not null"]),
    ("Radial Velocity", False, "select top 3", ["discoverymethod = 'Radial Velocity'"]),
    ("Radial Velocity", True, "select top 5001", ["discoverymethod = 'Radial Velocity'"]),
])
def test_exoplanets_filter_matrix_uses_bounded_simple_public_queries(monkeypatch, method, complete, limit_clause, constraints):
    calls = fake_source(monkeypatch, [{"pl_name": "Planet A", "pl_bmasse": 4.0, "pl_rade": 2.0}])
    result = gathering.fetch_dataset("exoplanets", {"discovery_method": method, "complete_only": complete,
                                                    "limit": 2, "columns": ["pl_bmasse", "pl_rade"]})
    adql = parse_qs(urlsplit(calls[0][0]).query)["query"][0]
    assert limit_clause in adql
    for constraint in constraints:
        assert constraint in adql
    if method != "all":
        assert "is not null" not in adql
    assert len(calls) == 1 and len(result.dataset.df) == 1


def test_exoplanets_local_completeness_precedes_limit_and_internal_fields_stay_internal(monkeypatch):
    rows = [{"pl_name": "A", "pl_bmasse": None, "pl_rade": 1.0},
            {"pl_name": "B", "pl_bmasse": 2.0, "pl_rade": None},
            {"pl_name": "C", "pl_bmasse": 3.0, "pl_rade": 1.0},
            {"pl_name": "D", "pl_bmasse": 4.0, "pl_rade": 2.0}]
    calls = fake_source(monkeypatch, rows)
    result = gathering.fetch_dataset("exoplanets", {"discovery_method": "Radial Velocity", "complete_only": True,
                                                    "limit": 1, "columns": ["pl_name"]})
    assert result.dataset.df.columns.tolist() == ["pl_name"]
    assert result.dataset.df.pl_name.tolist() == ["C"]
    assert result.result.truncated
    assert result.result.roles == {"pl_name": "unsupported"}
    assert result.result.units == {"pl_name": ""}
    adql = parse_qs(urlsplit(calls[0][0]).query)["query"][0]
    assert "pl_name,pl_bmasse,pl_rade" in adql
    assert any("after retrieving" in warning for warning in result.result.warnings)


@pytest.mark.parametrize("complete_rows,limit,fails", [(0, 1, True), (1, 2, True), (1, 1, False)])
def test_exoplanets_local_candidate_cap_never_silently_underfills(monkeypatch, complete_rows, limit, fails):
    rows = [{"pl_name": f"Planet {index:05}", "pl_bmasse": 1.0 if index < complete_rows else None, "pl_rade": 1.0}
            for index in range(5001)]
    fake_source(monkeypatch, rows)
    query = {"discovery_method": "Transit", "complete_only": True, "limit": limit, "columns": ["pl_name"]}
    if fails:
        with pytest.raises(ValueError, match="bounded query scanned 5,001"):
            gathering.fetch_dataset("exoplanets", query)
    else:
        result = gathering.fetch_dataset("exoplanets", query)
        assert len(result.dataset.df) == limit and result.result.truncated
        assert any("additional matching planets may exist" in warning for warning in result.result.warnings)


def test_exoplanets_validates_even_excluded_candidates_before_local_filter(monkeypatch):
    rows = [{"pl_name": "A", "pl_bmasse": 1.0, "pl_rade": 1.0},
            {"pl_name": "A", "pl_bmasse": None, "pl_rade": None}]
    fake_source(monkeypatch, rows)
    with pytest.raises(ValueError, match="duplicate planet names"):
        gathering.fetch_dataset("exoplanets", {"discovery_method": "Transit", "complete_only": True,
                                                "limit": 1, "columns": ["pl_name"]})


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
