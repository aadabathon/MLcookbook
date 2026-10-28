# Data playground

Open **Data playground** in the sidebar to gather a small real dataset. Nothing
is downloaded merely by opening the page or changing sources.

The review build for this addition is
`dist/data-playground-preview/MLcookbook/MLcookbook.exe`. Open that executable to
try the new sidebar area; the existing installed shortcut still opens its
earlier accepted bundle.

1. Choose a source and set its query filters.
2. Expand **Columns** to choose measurements, identifiers or text. Units and
   source roles are shown beside the choices.
3. Set **Maximum rows**, then **Fetch preview**. Work happens in a cancellable
   background job; the rest of the app locks while the request runs.
4. Inspect the first 20 rows, **Warnings** and **Source details**.
5. Choose **Use dataset** to copy the complete fetched table into active data.
   Open **Data** to inspect its roles and quality checks, or choose a recipe.

Fetching preserves active data and recipe work. Explicitly using a new dataset
replaces ordinary active data and invalidates its recipe/example sessions, just
like a successful file import. Existing Kitchen source snapshots and independent
Movement Study data remain independent. Capture active data explicitly inside
Kitchen to use it in a graph.

Changing filters or columns marks the preview stale and disables Use. Restore
the fetched settings or fetch again. Failed, empty and cancelled requests keep
the last completed preview; they never install a partial response. Per-source
unrun settings and the last preview survive navigation in memory. They are not
a workspace file format.

## Sources

| Source | Query | Returned data | Interpretation |
|---|---|---|---|
| [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) | Dates, worldwide/contiguous-US/California/Japan bounding region, minimum magnitude | Seismic event ID/time, coordinates, depth, magnitude/type, place and review status | Detection thresholds vary. Nearby events can be dependent; these exploratory records do not establish earthquake prediction. |
| [Open Food Facts](https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/) | One category tag, optional country tag | Barcode, names/brands/categories, ingredients, nutrients per 100 g, optional nutrition grade | Crowd-sourced completeness varies; variants can be similar. Nutri-Score is derived from nutrient rules. |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html) | Discovery method, optional requirement for mass and radius | One confirmed planet per `pscomppars` row, with selected planetary and stellar measurements | Discovery selection, inferred/composite values and shared host stars affect interpretation. |
| [Census ACS](https://www.census.gov/data/developers/guidance/api-user-guide.html) | 2019–2024 ACS five-year vintage, state, counties or tracts | Geographic identifiers, population, median age/income/home value/rent, housing and poverty counts | Areas are the observations. Margins of error are outside this curated import; overlapping vintages share years. |
| [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) | Latitude/longitude, start/end dates | Daily ERA5 temperature, precipitation, wind and optional other measurements; UTC boundaries | These are gridded historical estimates. Forecast claims require chronological evaluation beyond ordinary random-split recipes. |

The maximum is 5,000 rows for most sources and 500 products for Food Facts.
Truncation is explicit and preserves a documented order, rather than taking a
random sample: event time, planet name, geographic ID, provider product popularity
ranking or date. Product popularity favors frequently scanned products.
Food search pages are rate-spaced and can take longer than the other sources.
Live source availability can change; source errors remain recoverable.

Food tags use taxonomy syntax, such as `en:breakfast-cereals`, `en:yogurts` or
`en:united-states`. Search uses the provider's supported structured-search
endpoint; free-text search and bulk crawling are outside this interface. Retain
Open Food Facts attribution and its database/content licensing when reusing
outputs. NASA and weather attribution is included in the fetched warnings.

Census currently requires a [free API key](https://api.census.gov/data/key_signup.html)
for every query. Enter it into the password field before Fetch. Navigation
clears it; it is excluded from saved controls, query identity, provenance and
request URLs. A completed Census preview can still be used after the key has
been cleared. The other four sources need no key for these queries; Open-Meteo's
public service is for non-commercial use under its published limits.

## Reproducible snapshots

Source details retain the source link, exact non-secret query, UTC retrieval
time, column units/roles, request URLs, truncation/interpretation warnings and a
fingerprint of the typed data and roles. Numeric source values are explicitly
converted to nullable numeric arrays; malformed/non-finite values become
missing with warnings. Census unavailable-estimate codes become missing with
warnings; leading-zero barcodes and geographic IDs remain text identifiers.

After accepting data, open a recipe and **Save experiment** to keep its complete
typed snapshot, provenance, settings and any completed report in the existing
`.mlcb` format. Reopening restores the embedded values offline. A new Playground
fetch never silently refreshes a saved experiment or a captured Kitchen source.

The Python interface is deliberately submodule-only:

```python
from mlcookbook.gathering import fetch_dataset, source_specs

sources = source_specs()  # Plain metadata; no request.
fetched = fetch_dataset("usgs", {
    "start_date": "2024-01-01", "end_date": "2024-01-31",
    "region": "world", "min_magnitude": 4.5,
    "columns": ["event_id", "time", "magnitude", "depth_km"], "limit": 200,
})
dataset = fetched.dataset
provenance = fetched.result.to_dict()
```

`fetch_dataset` accepts optional `cancelled` and `progress` callbacks. It returns
complete data and a frozen JSON-safe result separately. Standard-library HTTPS
permits only the five source hosts, bounds response sizes/deadlines, and does not
follow redirects. No arbitrary URLs, new dependencies, scheduling or automatic
analysis are added.
