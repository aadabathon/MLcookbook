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


def test_catalog_and_presets_are_independent_and_available_offline(monkeypatch):
    monkeypatch.setattr(gathering, "fetch_json", lambda *a, **kw: pytest.fail("Catalog must be offline"))
    specs = gathering.source_specs()
    assert [source["key"] for source in specs] == [
        "usgs", "food", "exoplanets", "census", "weather", "world_bank", "nyc_trees",
    ]
    assert [source["key"] for source in specs if "preset" in source] == ["usgs", "exoplanets", "world_bank", "nyc_trees"]
    for spec in specs:
        keys = [column["key"] for column in spec["columns"]]
        assert len(keys) == len(set(keys))
        assert set(spec["default_columns"]) <= set(keys)
        assert 0 < spec["default_limit"] <= spec["max_limit"] <= 5000
        assert spec["url"].startswith("https://")
        if "preset" in spec:
            _, query = gathering._validate(spec["key"], spec["preset"]["query"])
            assert "api_key" not in query
    census_key = next(field for field in specs[3]["fields"] if field["key"] == "api_key")
    assert census_key["secret"] and census_key["required"]
    specs[0]["fields"][0]["default"] = "invalid"
    assert gathering.source_specs()[0]["fields"][0]["default"] != "invalid"


def _bank_page(rows):
    return [{"page": 1, "pages": 1, "total": len(rows)}, rows]


def _bank_economies():
    return [{"id": code, "iso2Code": short, "name": name, "region": {"id": region}}
            for code, short, name, region in [("USA", "US", "United States", "NAC"),
                                              ("WLD", "1W", "World", "NA"),
                                              ("CAN", "CA", "Canada", "NAC"),
                                              ("BRA", "BR", "Brazil", "LCN")]]


def _observation(code, indicator, value, *, year="2022", short=""):
    return {"countryiso3code": code, "country": {"id": short}, "indicator": {"id": indicator},
            "date": year, "value": value}


def test_world_bank_joins_by_economy_and_excludes_aggregates_before_capping(monkeypatch):
    observations = [
        _observation("USA", "SP.DYN.LE00.IN", 77),
        _observation("", "SP.DYN.LE00.IN", 72, short="1W"),
        _observation("CAN", "NY.GDP.PCAP.CD", None),
        _observation("BRA", "NY.GDP.PCAP.CD", 9500),
        _observation("CAN", "SP.DYN.LE00.IN", 82),
    ]
    calls = fake_source(monkeypatch, lambda url: _bank_page(observations if "/indicator/" in url else _bank_economies()))
    columns = ["gdp_per_capita", "economy_code", "year", "life_expectancy"]
    result = gathering.fetch_dataset("world_bank", {"limit": 2, "columns": columns})
    frame = result.dataset.df
    assert list(frame) == columns
    assert frame.economy_code.tolist() == ["BRA", "CAN"]
    assert frame.year.tolist() == [2022, 2022]
    assert frame.gdp_per_capita.iloc[0] == 9500 and pd.isna(frame.gdp_per_capita.iloc[1])
    assert pd.isna(frame.life_expectancy.iloc[0]) and frame.life_expectancy.iloc[1] == 82
    assert result.result.roles["economy_code"] == "unsupported"
    assert result.result.units["gdp_per_capita"] == "current USD"
    assert result.result.truncated and any("not a random sample" in warning for warning in result.result.warnings)
    assert len(calls) == 2 and all(urlsplit(url).hostname == "api.worldbank.org" for url, _ in calls)
    indicator_url = urlsplit(calls[1][0])
    assert indicator_url.path.endswith("/NY.GDP.PCAP.CD;SP.DYN.LE00.IN")
    assert parse_qs(indicator_url.query) == {"date": ["2022"], "source": ["2"], "format": ["json"], "per_page": ["1000"]}
    assert json.loads(json.dumps(result.result.to_dict()))["query"]["columns"] == columns


@pytest.mark.parametrize(("items", "pages", "error"), [
    ([_observation("CAN", "SP.DYN.LE00.IN", 82)] * 2, 1, "duplicate"),
    ([_observation("CAN", "SP.DYN.LE00.IN", 82, year="2021")], 1, "outside"),
    ([_observation("CAN", "SP.DYN.LE00.IN", 82)], 2, "incomplete"),
])
def test_world_bank_rejects_inconsistent_observations(monkeypatch, items, pages, error):
    response = _bank_page(items)
    response[0]["pages"] = pages
    fake_source(monkeypatch, lambda url: response if "/indicator/" in url else _bank_page(_bank_economies()))
    with pytest.raises(ValueError, match=error):
        gathering.fetch_dataset("world_bank", {})


def test_nyc_trees_uses_bounded_living_tree_query_and_retains_roles(monkeypatch):
    call = fake_source(monkeypatch, [
        {"tree_id": "0001", "block_id": "00006", "boroname": "Queens", "status": "Alive",
         "spc_common": "red maple", "tree_dbh": "12", "health": "Good"},
        {"tree_id": 2, "boroname": "Queens", "status": "Alive", "tree_dbh": "broken"},
        {"tree_id": "10", "boroname": "Queens", "status": "Alive"},
    ])
    result = gathering.fetch_dataset("nyc_trees", {"limit": 2})
    frame = result.dataset.df
    assert frame.tree_id.tolist() == ["0001", "2"] and frame.block_id.iloc[0] == "00006"
    assert frame.diameter_in.iloc[0] == 12 and pd.isna(frame.diameter_in.iloc[1])
    assert frame.health.iloc[0] == "Good" and pd.isna(frame.health.iloc[1])
    assert result.result.roles["species"] == "categorical" and result.result.roles["block_id"] == "unsupported"
    assert result.result.units["diameter_in"] == "in" and result.result.truncated
    assert any("malformed/non-finite" in warning for warning in result.result.warnings)
    request = urlsplit(call[0][0])
    assert request.hostname == "data.cityofnewyork.us" and request.path == "/resource/uvpi-gqnh.json"
    assert parse_qs(request.query) == {
        "$select": ["tree_id,boroname,status,block_id,spc_common,tree_dbh,health"],
        "$where": ["status = 'Alive' AND boroname = 'Queens'"], "$order": ["tree_id ASC"], "$limit": ["3"],
    }


@pytest.mark.parametrize("change", [{"boroname": "Bronx"}, {"status": "Dead"}, {"tree_id": "01"}])
def test_nyc_trees_rejects_duplicates_and_out_of_filter_rows(monkeypatch, change):
    rows = [{"tree_id": "1", "boroname": "Queens", "status": "Alive"},
            {"tree_id": "2", "boroname": "Queens", "status": "Alive", **change}]
    fake_source(monkeypatch, rows)
    with pytest.raises(ValueError, match="out-of-filter"):
        gathering.fetch_dataset("nyc_trees", {})


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
