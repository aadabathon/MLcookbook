# Dataset-first workbench

Start on **Data** with your own table. CSV, Parquet and read-only SQLite query
results become independent in-memory snapshots. The schema and selected inputs
determine the existing analyses available in **Analyze this dataset**.

## Open a SQLite dataset

1. Choose **Open SQLite…**, select an existing database and inspect its tables
   and views. Selecting an object supplies a quoted SELECT query.
2. Edit the query if needed, then choose **Preview query**. Preview shows at most
   20 rows and explicitly says when more rows exist. It does not establish the
   types or total size of all later rows.
3. Choose **Import dataset** to execute the query again and accept its complete
   result. The row cap is a maximum, not a request to truncate: an oversized
   result fails. Use WHERE or an explicit LIMIT to choose a subset. Use ORDER BY
   when order must be reproducible across separate queries.

Queries cannot modify the selected database, create temporary tables, attach
other databases or load extensions. Imports have row, byte and time bounds and
can be cancelled. The byte bound covers retained query output; it is not a
process-memory limit on SQLite's evaluation of a query. Failed/cancelled work
retains the active dataset and any completed preview. Editing the query requires
a matching new preview.

SQLite integers and nullable numbers retain their values; strings retain leading
zeros. Unambiguous ISO dates are recognized. BLOBs, incompatible mixed storage
types and duplicate result-column names require an explicit SQL conversion or
alias. A SQL declaration such as BOOLEAN does not by itself establish a
statistical role. Import conversion/order notes remain with source provenance.

The saved recipe experiment includes the accepted rows, roles and SQL source
description. Reopening never executes the query or requires the database.
PostgreSQL, SQL Server and MySQL do not yet have connectors.

## Choose an analysis from the data

Choose a task, target where applicable, inputs and method. The method list uses
the existing recipes: regression and regression comparison, classification,
association scan, clustering and PCA. Availability checks run in the background
without fitting. Reasons explain missing choices or unmet requirements.

Identifier-like, constant and entirely missing columns start excluded. Expand
**Input columns** to inspect the role explanations and deliberately include a
meaningful identifier if appropriate. Source roles on Data remain read-only;
recipe-local role overrides do not change the ordinary source snapshot.

**Evaluation context** can explicitly declare a grouping or ordering column.
Ordinary row-wise prediction and association workflows are blocked for that
context until an appropriate evaluator exists. Descriptive structure workflows
remain available with context columns excluded. These declarations do not
automatically detect dependence or implement general grouped/temporal evaluation.
Movement Study retains its separate fixed-group evaluator and sealed holdouts.

**Open workflow** carries your dataset, target and predictors into the selected
recipe. Adjust technical settings and explicitly run. The recipe catalog and
independent example sessions remain available.

## Inspect activity and reproduce a run

The header's **Activity** button opens a bounded operation history that survives
screen navigation. It shows input identity, times, outcomes, warnings and failure
details. A rotating diagnostic file is kept under the app's local-data logs
directory; its location is available in the Activity view. Results and requests
are not dumped into the log. Analytical functions still return warnings as data.

A completed **Linear regression** report offers **Python for this run**. The
source uses its recorded request, even after unrun controls have changed.
**Export Python and data…** creates a new folder with:

- `run.py`: editable calls to the public core API;
- `data.parquet`: the full typed input snapshot, including index;
- `run.json`: recorded settings, roles, runtime versions and data checksum.

Run `run.py` in a compatible project environment. It verifies the snapshot and
prints the structured result as JSON. It needs no original SQLite/CSV source.
The first supported code reproduction is linear regression (OLS/ridge/lasso);
other recipes retain their existing Python APIs.

## Program with the same data and core

**Programming** contains a script editor, a collapsible Jupyter QtConsole and an
embedded JupyterLab notebook view. They start only when requested.

The Windows desktop bundle includes the Programming interface. Code and
notebooks run in a separately installed Python 3.12+ environment containing the
**same ML Cookbook version as the app** (1.1.0 for this release), pandas/pyarrow
and the programming extra. Selecting an interpreter does not install packages.

From a matching ML Cookbook source directory (containing `pyproject.toml`), create
or use a real environment and install the optional runtime:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[programming]"
```

Select that environment's `.venv\Scripts\python.exe` in **Programming**. The
workspace shows its interpreter and package versions and rejects a different
ML Cookbook version. Keep the source directory available when using this
editable installation. The frozen executable is not a general Python interpreter;
it can suggest an existing `.venv` beside its parent directories, while a portable
copy elsewhere requires selecting the environment explicitly. Running the app
from source also needs the `gui` extra; building it uses the separate build
environment described in ADAM.md.

Choose a **Folder** for scripts, notebooks and copied data, then **Start Python**.
Scripts and console share one separate kernel. **Run file** runs the editor's
current contents, including unsaved edits; **Run selection** runs selected text.
Open/save use ordinary UTF-8 `.py` files. Opening a file never executes it.
Interrupt and Restart apply to the console kernel; restart clears its variables.
On Windows, Interrupt can wait for a blocking native call to return. Use Restart
if that call does not respond. A stopped kernel exposes Restart and records the
failure in Activity.

**Send active dataset** explicitly copies ordinary active data into that kernel:

```python
df.head()                 # your editable DataFrame copy
data.schema()             # typed Dataset for core operations
from mlcookbook import linear_regression
result = linear_regression(data, target="response", features=["measurement"])
result.to_dict()
```

Changes to `df` do not mutate active data. To return a DataFrame, choose its
variable name and **Return variable**, inspect the published table, then choose
**Use dataset**. Notebook code uses `workbench.publish(df)` followed by **Preview
notebook table** and explicit acceptance. That action names the last notebook
opened through the app; opening other tabs inside JupyterLab does not change its
source. The Script tab's **Preview console table** uses the console's separate
snapshot. Accepted data is a new ordinary
snapshot; it follows existing recipe-session invalidation rules.

You can also create a new DataFrame without sending an input dataset. The
console starts with `pd` and `workbench`; publish the new frame through the same
preview and acceptance path.

## Use real notebooks

After starting Python, create or open an `.ipynb` inside the selected programming
folder. A local authenticated JupyterLab server provides its existing editor,
Markdown/code cells, execution and save behavior inside the app.

Notebook kernels are independent of the script/console kernel. A new notebook
contains an unexecuted input cell for the explicitly sent snapshot. Run that cell
to obtain `df`, `data` and `workbench`. Later sends do not silently edit existing
notebooks: use a new notebook/input cell for the desired snapshot. Save notebooks
using JupyterLab before leaving; the app reminds you because notebook unsaved
state is managed by JupyterLab. Opening/restoring never runs cells.

Each programming session has its own `.mlcookbook-runtime/<session-id>/`
directory under the selected folder. Copied input files remain there after the
kernel stops so saved notebooks can still load their recorded snapshots. Keep
that directory with notebooks that refer to those snapshots. Concurrent sessions
in the same folder have independent connection files and data bridges.
New notebooks record their original bridge in plain metadata, so their published
tables can be previewed after closing and reopening the app in the same folder.
Opening never runs or rewrites cells. Older notebooks without that metadata can
still be edited/run, but need a new ML Cookbook notebook to return a table.
Neither `.py` nor `.ipynb` saves live Python memory. Existing `.mlcb` recipe,
`.mlck` Kitchen and `.mlcl` research archives retain their separate contracts.

Code runs with the selected environment's normal user permissions. The separate
process provides interrupt/restart and failure recovery; it is not a sandbox.
Programming receives ordinary active data only. Example sessions, Kitchen
captures, LLM source runs and sealed study holdouts are never transferred merely
by visiting their pages.
