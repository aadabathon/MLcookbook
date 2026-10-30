"""Explicit, bounded public-data queries for the desktop Data playground.

The five connectors produce typed Dataset snapshots and a separate JSON-safe
provenance record. Importing this submodule performs no network work. No source
is queried until fetch_dataset is explicitly called.
"""

from __future__ import annotations

import math
import re
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from time import monotonic, sleep
from urllib.parse import urlencode

import pandas as pd

from mlcookbook._data_http import check_cancelled, fetch_json, safe_url
from mlcookbook.core.dataset import Dataset
from mlcookbook.core.results import Result

_STATES = [
    ("Alabama", "01"), ("Alaska", "02"), ("Arizona", "04"), ("Arkansas", "05"), ("California", "06"),
    ("Colorado", "08"), ("Connecticut", "09"), ("Delaware", "10"), ("District of Columbia", "11"),
    ("Florida", "12"), ("Georgia", "13"), ("Hawaii", "15"), ("Idaho", "16"), ("Illinois", "17"),
    ("Indiana", "18"), ("Iowa", "19"), ("Kansas", "20"), ("Kentucky", "21"), ("Louisiana", "22"),
    ("Maine", "23"), ("Maryland", "24"), ("Massachusetts", "25"), ("Michigan", "26"), ("Minnesota", "27"),
    ("Mississippi", "28"), ("Missouri", "29"), ("Montana", "30"), ("Nebraska", "31"), ("Nevada", "32"),
    ("New Hampshire", "33"), ("New Jersey", "34"), ("New Mexico", "35"), ("New York", "36"),
    ("North Carolina", "37"), ("North Dakota", "38"), ("Ohio", "39"), ("Oklahoma", "40"), ("Oregon", "41"),
    ("Pennsylvania", "42"), ("Rhode Island", "44"), ("South Carolina", "45"), ("South Dakota", "46"),
    ("Tennessee", "47"), ("Texas", "48"), ("Utah", "49"), ("Vermont", "50"), ("Virginia", "51"),
    ("Washington", "53"), ("West Virginia", "54"), ("Wisconsin", "55"), ("Wyoming", "56"),
    ("Puerto Rico", "72"),
]
_REGIONS = {
    "world": {},
    "contiguous_us": {"minlatitude": 24, "maxlatitude": 50, "minlongitude": -125, "maxlongitude": -66},
    "california": {"minlatitude": 32, "maxlatitude": 42, "minlongitude": -125, "maxlongitude": -114},
    "japan": {"minlatitude": 24, "maxlatitude": 46, "minlongitude": 123, "maxlongitude": 146},
}
_CENSUS_VARIABLES = {
    "population": "B01003_001E", "median_household_income": "B19013_001E", "median_age": "B01002_001E",
    "housing_units": "B25001_001E", "median_home_value": "B25077_001E", "median_gross_rent": "B25064_001E",
    "poverty_universe": "B17001_001E", "below_poverty": "B17001_002E",
}
_FOOD_NUTRIENTS = {
    "energy_kcal_100g": "energy-kcal_100g", "fat_100g": "fat_100g", "saturated_fat_100g": "saturated-fat_100g",
    "carbohydrates_100g": "carbohydrates_100g", "sugars_100g": "sugars_100g", "fiber_100g": "fiber_100g",
    "proteins_100g": "proteins_100g", "salt_100g": "salt_100g",
}
_FOOD_LOCK = threading.Lock()
_FOOD_NEXT_REQUEST = 0.0
_FOOD_INTERVAL = 6.1  # Official search budget: ten requests per minute per IP.


def _column(key, label, unit="", role="numeric"):
    return {"key": key, "label": label, "unit": unit, "role": role}


def source_specs() -> list[dict]:
    """Return plain field/column descriptions for source-specific query forms."""
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=364)
    dates = [
        {"key": "start_date", "label": "Start date", "kind": "date", "default": start.isoformat()},
        {"key": "end_date", "label": "End date", "kind": "date", "default": end.isoformat()},
    ]
    sources = [
        {
            "key": "usgs", "title": "USGS earthquakes",
            "description": "Recorded seismic events, with physical measurements and event metadata.",
            "url": "https://earthquake.usgs.gov/fdsnws/event/1/",
            "fields": dates + [
                {"key": "region", "label": "Region", "kind": "choice", "default": "world",
                 "options": [("Worldwide", "world"), ("Contiguous United States", "contiguous_us"),
                             ("California region", "california"), ("Japan region", "japan")]},
                {"key": "min_magnitude", "label": "Minimum magnitude", "kind": "float", "default": 4.5,
                 "min": -2.0, "max": 10.0, "decimals": 1},
            ],
            "columns": [
                _column("event_id", "Event ID", role="unsupported"),
                _column("time", "Event time", "UTC", "datetime"),
                _column("latitude", "Latitude", "degrees north"), _column("longitude", "Longitude", "degrees east"),
                _column("depth_km", "Depth", "km"), _column("magnitude", "Magnitude"),
                _column("magnitude_type", "Magnitude type", role="categorical"),
                _column("place", "Place description", role="text"),
                _column("status", "Review status", role="categorical"),
            ],
            "default_columns": ["event_id", "time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type", "place"],
        },
        {
            "key": "food", "title": "Open Food Facts",
            "description": "Packaged products, nutrients per 100 g, and ingredient text. At most 500 products per fetch.",
            "url": "https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/",
            "fields": [
                {"key": "category", "label": "Category tag", "kind": "text", "default": "en:breakfast-cereals",
                 "help": "An Open Food Facts taxonomy tag, for example en:breakfast-cereals or en:yogurts."},
                {"key": "country", "label": "Country tag (optional)", "kind": "text", "default": "",
                 "help": "For example en:united-states. Leave blank for all countries."},
            ],
            "columns": [
                _column("barcode", "Barcode", role="unsupported"), _column("product_name", "Product name", role="text"),
                _column("brands", "Brands", role="categorical"), _column("categories", "Category tags", role="text"),
                _column("ingredients_text", "Ingredients", role="text"),
                _column("energy_kcal_100g", "Energy", "kcal / 100 g"),
                *[_column(key, label, "g / 100 g") for key, label in [
                    ("fat_100g", "Fat"), ("saturated_fat_100g", "Saturated fat"), ("carbohydrates_100g", "Carbohydrates"),
                    ("sugars_100g", "Sugars"), ("fiber_100g", "Fiber"), ("proteins_100g", "Protein"), ("salt_100g", "Salt"),
                ]],
                _column("nutrition_grade", "Nutri-Score grade", role="categorical"),
            ],
            "default_columns": ["barcode", "product_name", "brands", "ingredients_text", "energy_kcal_100g", "fat_100g",
                                "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g"],
            "max_limit": 500,
        },
        {
            "key": "exoplanets", "title": "NASA exoplanets",
            "description": "One row per confirmed planet from the composite parameters table.",
            "url": "https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html",
            "fields": [
                {"key": "discovery_method", "label": "Discovery method", "kind": "choice", "default": "all",
                 "options": [("All methods", "all"), ("Transit", "Transit"), ("Radial velocity", "Radial Velocity"),
                             ("Imaging", "Imaging"), ("Microlensing", "Microlensing")]},
                {"key": "complete_only", "label": "Require mass and radius", "kind": "bool", "default": False},
            ],
            "columns": [
                _column("pl_name", "Planet name", role="unsupported"), _column("hostname", "Host star", role="unsupported"),
                _column("discoverymethod", "Discovery method", role="categorical"), _column("disc_year", "Discovery year", "year"),
                _column("pl_bmasse", "Planet mass", "Earth masses"), _column("pl_rade", "Planet radius", "Earth radii"),
                _column("pl_orbper", "Orbital period", "days"), _column("pl_orbsmax", "Orbit semi-major axis", "AU"),
                _column("st_teff", "Stellar temperature", "K"), _column("st_mass", "Stellar mass", "solar masses"),
                _column("st_rad", "Stellar radius", "solar radii"), _column("sy_dist", "Distance", "pc"),
            ],
            "default_columns": ["pl_name", "hostname", "discoverymethod", "pl_bmasse", "pl_rade", "pl_orbper", "st_teff", "st_mass"],
        },
        {
            "key": "census", "title": "Census ACS",
            "description": "ACS five-year geographic estimates. A free Census API key is required.",
            "url": "https://www.census.gov/data/developers/guidance/api-user-guide.html",
            "fields": [
                {"key": "year", "label": "ACS vintage", "kind": "choice", "default": 2024,
                 "options": [(str(year), year) for year in range(2024, 2018, -1)]},
                {"key": "state", "label": "State", "kind": "choice", "default": "55", "options": _STATES},
                {"key": "geography", "label": "Geography", "kind": "choice", "default": "county",
                 "options": [("Counties", "county"), ("Census tracts", "tract")]},
                {"key": "api_key", "label": "Census API key", "kind": "text", "default": "", "secret": True, "required": True,
                 "help": "Required for all Census queries. Get a free key at https://api.census.gov/data/key_signup.html. "
                         "The key is used only for this session and excluded from saved provenance."},
            ],
            "columns": [
                _column("geoid", "Geographic ID", role="unsupported"), _column("name", "Area name", role="text"),
                _column("state", "State FIPS", role="unsupported"), _column("county", "County FIPS", role="unsupported"),
                _column("population", "Population", "people"),
                _column("median_household_income", "Median household income", "USD (vintage year)"),
                _column("median_age", "Median age", "years"), _column("housing_units", "Housing units", "units"),
                _column("median_home_value", "Median home value", "USD (vintage year)"),
                _column("median_gross_rent", "Median gross rent", "USD / month (vintage year)"),
                _column("poverty_universe", "Poverty status population", "people"),
                _column("below_poverty", "Below poverty level", "people"),
            ],
            "default_columns": ["geoid", "name", "population", "median_household_income", "median_age", "housing_units",
                                "poverty_universe", "below_poverty"],
        },
        {
            "key": "weather", "title": "Open-Meteo weather",
            "description": "Daily historical weather estimates at a location, using the ERA5 reanalysis model.",
            "url": "https://open-meteo.com/en/docs/historical-weather-api",
            "fields": [
                {"key": "latitude", "label": "Latitude", "kind": "float", "default": 43.0731, "min": -90.0, "max": 90.0, "decimals": 4},
                {"key": "longitude", "label": "Longitude", "kind": "float", "default": -89.4012, "min": -180.0, "max": 180.0, "decimals": 4},
                *dates,
            ],
            "columns": [
                _column("date", "Date", "UTC", "datetime"),
                _column("temperature_2m_mean", "Mean temperature", "°C"),
                _column("temperature_2m_max", "Maximum temperature", "°C"),
                _column("temperature_2m_min", "Minimum temperature", "°C"),
                _column("precipitation_sum", "Precipitation", "mm"), _column("rain_sum", "Rain", "mm"),
                _column("snowfall_sum", "Snowfall", "cm"), _column("wind_speed_10m_max", "Maximum wind speed", "km/h"),
                _column("shortwave_radiation_sum", "Solar radiation", "MJ/m²"),
            ],
            "default_columns": ["date", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                                "precipitation_sum", "wind_speed_10m_max"],
        },
    ]
    for spec in sources:
        spec.setdefault("max_limit", 5000)
        spec["default_limit"] = 500
    return deepcopy(sources)


@dataclass(frozen=True)
class GatheringResult(Result):
    source: str
    title: str
    source_url: str
    query: dict
    retrieved_at: str
    n_rows: int
    n_columns: int
    units: dict[str, str]
    roles: dict[str, str]
    request_urls: list[str]
    truncated: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GatheredDataset:
    dataset: Dataset
    result: GatheringResult


def _validate(source, query):
    specs = {spec["key"]: spec for spec in source_specs()}
    if source not in specs:
        raise ValueError("Choose one of the five supported data sources.")
    if not isinstance(query, dict):
        raise TypeError("A data query must be a dictionary.")
    spec = specs[source]
    allowed = {field["key"] for field in spec["fields"]} | {"columns", "limit"}
    if set(query) - allowed:
        raise ValueError("The data query contains unsupported settings.")
    values = {field["key"]: deepcopy(query.get(field["key"], field["default"])) for field in spec["fields"]}
    for field_spec in spec["fields"]:
        key, kind, value = field_spec["key"], field_spec["kind"], values[field_spec["key"]]
        if kind == "choice":
            options = [option[1] for option in field_spec["options"]]
            if isinstance(value, bool) or value not in options:
                raise ValueError(f"Choose a supported {field_spec['label'].lower()}.")
        elif kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{field_spec['label']} must be true or false.")
        elif kind in ("int", "float"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_spec['label']} must be a finite number.")
            if not field_spec["min"] <= value <= field_spec["max"]:
                raise ValueError(f"{field_spec['label']} is outside the supported range.")
        elif kind == "date":
            try:
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError:
                raise ValueError(f"{field_spec['label']} must be an ISO date (YYYY-MM-DD).") from None
        elif not isinstance(value, str) or len(value) > 250:
            raise ValueError(f"{field_spec['label']} must be at most 250 characters.")
    limit = query.get("limit", spec["default_limit"])
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= spec["max_limit"]:
        raise ValueError(f"Choose a row limit from 1 to {spec['max_limit']} for this source.")
    columns = query.get("columns", spec["default_columns"])
    available = {column["key"] for column in spec["columns"]}
    if (not isinstance(columns, list) or not columns or any(not isinstance(column, str) or column not in available for column in columns)
            or len(columns) != len(set(columns))):
        raise ValueError("Select at least one supported column, without duplicates.")
    values.update(limit=limit, columns=list(columns))
    if "start_date" in values:
        start, end = date.fromisoformat(values["start_date"]), date.fromisoformat(values["end_date"])
        minimum = date(1940 if source == "weather" else 1900, 1, 1)
        latest = date.today() - timedelta(days=1) if source == "weather" else date.today()
        if not minimum <= start <= end <= latest:
            raise ValueError(f"Choose ordered dates from {minimum.isoformat()} through {latest.isoformat()}.")
    if source == "food":
        for key in ("category", "country"):
            values[key] = values[key].strip()
            if not values[key] and key == "country":
                continue
            if not re.fullmatch(r"[a-z]{2}:[a-z0-9]+(?:-[a-z0-9]+)*", values[key]):
                raise ValueError(f"Use one taxonomy tag for {key}, for example en:breakfast-cereals or en:united-states.")
    if source == "census":
        if not re.fullmatch(r"[A-Za-z0-9]{20,128}", values["api_key"]):
            raise ValueError("Enter a Census API key. Get a free key at https://api.census.gov/data/key_signup.html.")
    if source == "weather" and all(column == "date" for column in columns):
        raise ValueError("Select at least one weather measurement alongside the optional date column.")
    return spec, values


def _url(base, parameters):
    return base + "?" + urlencode(parameters)


def _objects(value, label):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{label} returned an invalid records table.")
    return value


def _get(url, urls, cancelled, deadline):
    check_cancelled(cancelled, deadline)
    urls.append(safe_url(url))
    return fetch_json(url, cancelled=cancelled, timeout=max(0.01, min(30.0, deadline - monotonic())))


def _usgs(query, urls, warnings, cancelled, progress, deadline):
    parameters = dict(format="geojson", starttime=query["start_date"] + "T00:00:00Z",
                      endtime=query["end_date"] + "T23:59:59.999Z", minmagnitude=query["min_magnitude"],
                      limit=query["limit"] + 1, orderby="time-asc", eventtype="earthquake", **_REGIONS[query["region"]])
    raw = _get(_url("https://earthquake.usgs.gov/fdsnws/event/1/query", parameters), urls, cancelled, deadline)
    if raw is None:
        return [], False
    if not isinstance(raw, dict) or raw.get("type") != "FeatureCollection":
        raise ValueError("USGS returned an invalid event collection.")
    events = _objects(raw.get("features"), "USGS")
    if len(events) > query["limit"] + 1:
        raise ValueError("USGS returned more rows than requested.")
    rows, seen = [], set()
    for event in events:
        check_cancelled(cancelled, deadline)
        identity = event.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("USGS returned an event without its source ID.")
        if identity in seen:
            warnings.append("Skipped a duplicate USGS event ID.")
            continue
        seen.add(identity)
        properties, geometry = event.get("properties"), event.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            raise ValueError("USGS returned malformed event properties or coordinates.")
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) != 3:
            raise ValueError("USGS returned malformed event coordinates.")
        stamp = properties.get("time")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("USGS returned an invalid event time.")
        try:
            event_time = (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=stamp)).isoformat()
        except (ValueError, OSError, OverflowError):
            raise ValueError("USGS returned an event time outside the supported range.") from None
        rows.append({"event_id": identity, "time": event_time, "longitude": coordinates[0], "latitude": coordinates[1],
                     "depth_km": coordinates[2], "magnitude": properties.get("mag"), "magnitude_type": properties.get("magType"),
                     "place": properties.get("place"), "status": properties.get("status")})
    warnings.append("Catalog detection thresholds vary by place and time; nearby events can be dependent. This is exploratory event data, not an earthquake forecast.")
    return rows[:query["limit"]], len(rows) > query["limit"]


def _food_pause(cancelled, deadline):
    global _FOOD_NEXT_REQUEST
    # Across explicit fetches too, not only between the pages of one fetch.
    while True:
        check_cancelled(cancelled, deadline)
        with _FOOD_LOCK:
            wait = _FOOD_NEXT_REQUEST - monotonic()
            if wait <= 0:
                _FOOD_NEXT_REQUEST = monotonic() + _FOOD_INTERVAL
                return
        sleep(min(0.05, wait))


def _food(query, urls, warnings, cancelled, progress, deadline):
    rows, seen, repeated = [], set(), 0
    total = None
    page_size = min(100, query["limit"])
    for page in range(1, 7):
        _food_pause(cancelled, deadline)
        parameters = {"categories_tags": query["category"], "fields": "code,product_name,brands,categories_tags,ingredients_text,nutriments,nutrition_grades",
                      "page_size": page_size, "page": page}
        if query["country"]:
            parameters["countries_tags"] = query["country"]
        if progress is not None:
            progress(f"Fetching products · page {page} · {len(rows)} collected")
        raw = _get(_url("https://world.openfoodfacts.org/api/v2/search", parameters), urls, cancelled, deadline)
        if not isinstance(raw, dict):
            raise ValueError("Open Food Facts returned an invalid search response.")
        products = _objects(raw.get("products"), "Open Food Facts")
        count = raw.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Open Food Facts returned an invalid product count.")
        total = count
        if len(products) > page_size:
            raise ValueError("Open Food Facts returned more products than requested.")
        for product in products:
            identity = product.get("code")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Open Food Facts returned a product without a text barcode.")
            if identity in seen:
                repeated += 1
                continue
            seen.add(identity)
            nutrients = product.get("nutriments", {})
            categories = product.get("categories_tags", [])
            if not isinstance(nutrients, dict) or not isinstance(categories, list) or any(not isinstance(item, str) for item in categories):
                raise ValueError("Open Food Facts returned malformed nutrients or category tags.")
            rows.append({"barcode": identity, "product_name": product.get("product_name"), "brands": product.get("brands"),
                         "categories": "; ".join(categories), "ingredients_text": product.get("ingredients_text"),
                         "nutrition_grade": product.get("nutrition_grades"),
                         **{column: nutrients.get(api_column) for column, api_column in _FOOD_NUTRIENTS.items()}})
        if len(rows) >= query["limit"] or not products or page * page_size >= total:
            break
    if repeated:
        warnings.append(f"Skipped {repeated} duplicate product barcode(s) across search pages.")
    warnings.append("Products are crowd-sourced and can be incomplete or duplicated as variants. Nutrients are per 100 g as sold; Nutri-Score is derived from nutrient rules.")
    warnings.append("Open Food Facts: ODbL database / Database Contents License; retain source attribution. Search is capped at 500 products; use the provider's bulk download for larger collections.")
    return rows[:query["limit"]], total is not None and total > len(rows)


def _exoplanets(query, urls, warnings, cancelled, progress, deadline):
    # The public archive's gateway rejects the valid combination of an equality
    # constraint and both IS NOT NULL constraints. Retrieve ordinary method
    # candidates once, then apply the same completeness predicate locally.
    local_complete = query["discovery_method"] != "all" and query["complete_only"]
    columns = list(dict.fromkeys(["pl_name", *query["columns"],
                                  *(["pl_bmasse", "pl_rade"] if local_complete else [])]))
    candidate_limit = 5001 if local_complete else query["limit"] + 1
    clauses = []
    if query["discovery_method"] != "all":
        # Values came from a fixed allowlist; never interpolate free-form ADQL.
        clauses.append("discoverymethod = '" + query["discovery_method"] + "'")
    if query["complete_only"] and not local_complete:
        clauses += ["pl_bmasse is not null", "pl_rade is not null"]
    adql = f"select top {candidate_limit} {','.join(columns)} from pscomppars"
    if clauses:
        adql += " where " + " and ".join(clauses)
    adql += " order by pl_name"
    raw = _get(_url("https://exoplanetarchive.ipac.caltech.edu/TAP/sync", {"query": adql, "format": "json"}), urls, cancelled, deadline)
    rows = _objects(raw, "NASA Exoplanet Archive")
    if len(rows) > candidate_limit:
        raise ValueError("NASA returned more planets than requested.")
    seen = set()
    for row in rows:
        check_cancelled(cancelled, deadline)
        if not isinstance(row.get("pl_name"), str) or not row["pl_name"] or row["pl_name"] in seen:
            raise ValueError("NASA returned missing or duplicate planet names in its one-row-per-planet table.")
        seen.add(row["pl_name"])
        if any(column not in row for column in columns):
            raise ValueError("NASA response omitted a requested measurement.")
    candidate_capped = local_complete and len(rows) == candidate_limit
    if local_complete:
        rows = [row for row in rows if row["pl_bmasse"] is not None and row["pl_rade"] is not None]
        if candidate_capped and len(rows) < query["limit"]:
            raise ValueError("NASA's bounded query scanned 5,001 discovery-method candidates without filling "
                             "the requested complete rows. Reduce Maximum rows, or turn off Require mass and radius; "
                             "the previous preview is preserved.")
        warnings.append("Required mass and radius were checked after retrieving discovery-method candidates in planet-name order.")
        if candidate_capped:
            warnings.append("NASA candidate scan reached 5,001 planets; additional matching planets may exist beyond this ordered snapshot.")
    warnings.append("Discovery methods select different planet populations. Composite values may combine references or inferred measurements; planets sharing a host are related observations.")
    warnings.append("Cite the NASA Exoplanet Archive when using these data: https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html.")
    return rows[:query["limit"]], len(rows) > query["limit"] or candidate_capped


def _census(query, urls, warnings, cancelled, progress, deadline):
    wanted = [key for key in query["columns"] if key in _CENSUS_VARIABLES]
    api_columns = ["NAME", *[_CENSUS_VARIABLES[key] for key in wanted]]
    parameters = {"get": ",".join(api_columns), "for": query["geography"] + ":*", "in": "state:" + query["state"], "key": query["api_key"]}
    raw = _get(_url(f"https://api.census.gov/data/{query['year']}/acs/acs5", parameters), urls, cancelled, deadline)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        raise ValueError("Census returned an invalid table; check the API key and query.")
    header = raw[0]
    required = api_columns + ["state", "county"] + (["tract"] if query["geography"] == "tract" else [])
    if any(not isinstance(name, str) for name in header) or len(header) != len(set(header)) or any(name not in header for name in required):
        raise ValueError("Census response omitted required geography or estimate fields.")
    # One state's tracts remain bounded by the shared 12 MiB transport limit.
    rows, sentinels, seen = [], 0, set()
    for values in raw[1:]:
        check_cancelled(cancelled, deadline)
        if not isinstance(values, list) or len(values) != len(header):
            raise ValueError("Census returned a malformed data row.")
        item = dict(zip(header, values))
        for key, size in [("state", 2), ("county", 3), *([("tract", 6)] if query["geography"] == "tract" else [])]:
            if not isinstance(item[key], str) or not re.fullmatch(r"\d{" + str(size) + "}", item[key]):
                raise ValueError("Census returned a malformed geographic identifier.")
        geoid = item["state"] + item["county"] + item.get("tract", "")
        if geoid in seen:
            raise ValueError("Census returned duplicate geographic identifiers.")
        seen.add(geoid)
        row = {"geoid": geoid, "name": item["NAME"], "state": item["state"], "county": item["county"]}
        for key in wanted:
            value = item[_CENSUS_VARIABLES[key]]
            # Negative ACS estimate codes mean unavailable/suppressed, not negative people or money.
            if isinstance(value, str) and re.fullmatch(r"-\d{9}", value):
                value, sentinels = None, sentinels + 1
            row[key] = value
        rows.append(row)
    rows.sort(key=lambda row: row["geoid"])
    if sentinels:
        warnings.append(f"Converted {sentinels} documented negative ACS estimate sentinel(s) to missing values.")
    warnings.append("ACS five-year estimates describe areas, not individual people. Margins of error are not included in this curated import; overlapping vintages share survey years.")
    return rows[:query["limit"]], len(rows) > query["limit"]


def _weather(query, urls, warnings, cancelled, progress, deadline):
    measurements = [column for column in query["columns"] if column != "date"]
    start, end = date.fromisoformat(query["start_date"]), date.fromisoformat(query["end_date"])
    # At most one extra day establishes truncation without downloading an unbounded date range.
    effective_end = min(end, start + timedelta(days=query["limit"]))
    parameters = {"latitude": query["latitude"], "longitude": query["longitude"], "start_date": start.isoformat(),
                  "end_date": effective_end.isoformat(), "daily": ",".join(measurements), "timezone": "UTC",
                  "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm", "models": "era5"}
    raw = _get(_url("https://archive-api.open-meteo.com/v1/archive", parameters), urls, cancelled, deadline)
    if not isinstance(raw, dict) or not isinstance(raw.get("daily"), dict) or not isinstance(raw.get("daily_units"), dict):
        raise ValueError("Open-Meteo returned an invalid daily weather response.")
    daily, units = raw["daily"], raw["daily_units"]
    stamps = daily.get("time")
    if not isinstance(stamps, list) or len(stamps) > query["limit"] + 1:
        raise ValueError("Open-Meteo returned an invalid daily date list.")
    expected_units = {column["key"]: column["unit"] for column in source_specs()[-1]["columns"]}
    for column in measurements:
        if not isinstance(daily.get(column), list) or len(daily[column]) != len(stamps):
            raise ValueError("Open-Meteo returned weather arrays with inconsistent lengths.")
        if units.get(column) != expected_units[column]:
            raise ValueError(f"Open-Meteo returned an unexpected unit for {column}.")
    rows = []
    for position, stamp in enumerate(stamps):
        if stamp != (start + timedelta(days=position)).isoformat():
            raise ValueError("Open-Meteo returned missing, duplicate or unordered daily dates.")
        rows.append({"date": stamp, **{column: daily[column][position] for column in measurements}})
    if len(stamps) != (effective_end - start).days + 1:
        raise ValueError("Open-Meteo returned an incomplete daily date range.")
    warnings.append("ERA5 reanalysis supplies gridded historical estimates, not direct observations at the exact coordinate. Daily boundaries use UTC; time-series forecasting needs chronological evaluation.")
    warnings.append("Weather data attribution: Open-Meteo and ECMWF ERA5 (CC BY 4.0).")
    return rows[:query["limit"]], (end - start).days + 1 > query["limit"]


def _typed_frame(rows, spec, columns, warnings):
    frame = pd.DataFrame(index=range(len(rows)))
    metadata = {column["key"]: column for column in spec["columns"]}
    for key in columns:
        values = [row.get(key) for row in rows]
        role = metadata[key]["role"]
        if role == "numeric":
            converted, invalid = [], 0
            for value in values:
                if value is None or value == "":
                    converted.append(None)
                    continue
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    converted.append(None)
                    invalid += 1
                    continue
                try:
                    parsed = float(value)
                except (ValueError, OverflowError):
                    parsed = math.nan
                if not math.isfinite(parsed):
                    parsed, invalid = None, invalid + 1
                converted.append(parsed)
            frame[key] = pd.array(converted, dtype="Float64")
            if invalid:
                warnings.append(f"{key}: treated {invalid} malformed/non-finite numeric value(s) as missing; source values were not valid measurements.")
        elif role == "datetime":
            if any(value is not None and not isinstance(value, str) for value in values):
                raise ValueError(f"Data source returned an invalid datetime field: {key}.")
            try:
                frame[key] = pd.to_datetime(values, format="ISO8601", utc=True, errors="raise")
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f"Data source returned an invalid datetime field: {key}.") from None
        else:
            if any(value is not None and not isinstance(value, str) for value in values):
                raise ValueError(f"Data source returned a non-text value in {key}; the response schema may have changed.")
            frame[key] = pd.array(values, dtype="string")
    return frame


def fetch_dataset(source: str, query: dict, *, cancelled=None, progress=None) -> GatheredDataset:
    """Explicitly fetch a bounded typed snapshot; cancellation never returns partial data.

    ``columns`` selects curated output columns; ``limit`` bounds accepted rows.
    Source-specific fields/defaults are described by ``source_specs``. Census
    requires a transient ``api_key``, which is never retained in result metadata.
    ``progress`` receives plain text; ``cancelled`` is polled while waiting/reading.
    """
    spec, values = _validate(source, query)
    check_cancelled(cancelled)
    urls, warnings = [], []
    if progress is not None:
        progress(f"Fetching {spec['title']}…")
    functions = {"usgs": _usgs, "food": _food, "exoplanets": _exoplanets, "census": _census, "weather": _weather}
    rows, truncated = functions[source](values, urls, warnings, cancelled, progress, monotonic() + 120.0)
    check_cancelled(cancelled)
    if not rows:
        raise ValueError("No data rows matched the query. Adjust the source filters and fetch again.")
    if truncated:
        warnings.append(f"Row limit reached: kept {len(rows)} rows in the source's documented order; this is not a random sample.")
    frame = _typed_frame(rows, spec, values["columns"], warnings)
    if source in ("usgs", "exoplanets", "census"):
        order = {"usgs": "event time ascending", "exoplanets": "planet name ascending", "census": "geographic ID ascending"}[source]
        warnings.append(f"Retrieval order: {order}.")
    if source == "food":
        warnings.append("Retrieval order: provider default popularity ranking; this favors frequently scanned products. The live catalogue can change between pages.")
    metadata = {column["key"]: column for column in spec["columns"]}
    roles = {column: metadata[column]["role"] for column in values["columns"]}
    retained_query = {key: value for key, value in values.items() if key != "api_key"}
    result = GatheringResult(
        source=source, title=spec["title"], source_url=spec["url"], query=retained_query,
        retrieved_at=datetime.now(UTC).isoformat(), n_rows=len(frame), n_columns=len(frame.columns),
        units={column: metadata[column]["unit"] for column in frame.columns}, roles=roles,
        request_urls=urls, truncated=truncated, warnings=list(dict.fromkeys(warnings)),
    )
    return GatheredDataset(Dataset(frame, overrides=roles), result)
