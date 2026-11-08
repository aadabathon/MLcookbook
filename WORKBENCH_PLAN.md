# Dataset-first workbench plan

Planned on 2026-11-03 and implemented after Adam's explicit approval. This
document retains the delivery criteria; [WORKBENCH.md](WORKBENCH.md) describes
the resulting interface and supported behavior.

As of 2026-11-08, all five stages are implemented and verified: read-only SQLite intake,
schema-driven handoffs, Activity and linear-regression code reproduction, a
selected-environment kernel, and the script/console/real-notebook workspace.
The final source suite passed 1,077 tests with one skip on 2026-11-08. Fresh
Windows candidate checks passed on 2026-11-08, including offline SQLite
reopening/Python reproduction, genuine notebooks, cross-session snapshot
returns, owned-process cleanup and real local inference. Delivery evidence is
in [ADAM.md](ADAM.md). Archive validation and publication are release-distribution
checks; the five-stage implementation is complete. The installed shortcut
bundle is preserved while the candidate is delivered.

## Purpose

Make ML Cookbook a place to learn and implement ML concepts, apply them to real
tabular data, and compose inspectable workflows. Loading data is the main entry.
The dataset's schema and profile expose applicable operations; the user chooses
the question, inputs and execution. Python remains a first-class way to use the
same core operations.

The first success case is deliberately concrete:

> Open a SQLite database -> preview a query -> accept its dataset -> choose a
> numeric target -> configure and run regression -> inspect errors and warnings
> -> save -> reopen with the database unavailable.

This establishes the data foundation shared by the programming workspace.
Existing recipes, Kitchen and independent research workspaces remain available.

## Delivery sequence

### 1. One dataset intake path

- CSV, Parquet and SQLite all produce the existing typed dataset snapshot and
  plain source/import descriptors. Keep the implementation small.
- SQLite offers tables/views and an explicit query, with read-only connections,
  preview, cancellation and explicit acceptance. A preview is distinct from the
  accepted row set; any row limit or sampling is disclosed.
- Preserve identifiers, nulls and timestamps, and report conversions. Duplicate
  result names from joins need explicit aliases. Do not infer statistical meaning
  merely from a database column's declared type.
- Retain query/source provenance with the saved rows. Reopening never reruns SQL;
  refreshing from a source is an explicit new import.
- SQLite uses the standard library. PostgreSQL, SQL Server and MySQL are later
  connector choices requiring a driver/dependency decision.

Done when equivalent CSV/Parquet/SQLite fixtures yield equivalent values and
roles, failed imports preserve current work, and an imported query result can be
saved through an existing recipe and reopened without the database.

### 2. Schema-driven workflow entry

- Show applicable existing workflows beside the imported dataset, expressed as
  useful tasks with exact method names available. Keep the recipe catalog.
- Distinguish a workflow awaiting choices from one with unmet requirements.
  Give specific reasons; discovery performs no fitting or automatic execution.
- Carry selected dataset, target and predictors into the destination recipe.
  Begin with the complete regression journey above, then extend the same pattern
  to existing classification, association and structure workflows.
- Separate storage dtype, statistical role and workflow-specific use. Target,
  predictor and exclusion choices stay local to the workflow. Data continues to
  show read-only source roles.
- First improve explicit identifier exclusion and explanations of inferred
  roles. Add units, ordered categories, group and time metadata only alongside
  code that consumes and validates their meaning. When group/time context is
  explicitly declared, explain and block incompatible row-wise workflows until
  an appropriate evaluator exists. Automatic detection and new general grouped
  or temporal evaluators are outside this milestone.
- Core applicability and execution validation share requirements; the GUI renders
  plain availability descriptors rather than duplicating statistical rules.

Done when a user can import data, understand why a workflow is available, select
its inputs and reach a ready recipe without re-entering those choices. Include
invalid, missing, small, explicitly grouped and ambiguous-type cases in validation.

### 3. Make completed work inspectable

- Add a collapsible Activity view for imports and runs: operation, input identity,
  start/end, outcome, warnings and useful failure details. Keep routine progress
  concise. Retain a bounded history across screen navigation and a bounded local
  diagnostic log for failures, with tracebacks available in details.
- Record activity in the coordinator/runtime boundary. Analyses continue returning
  data and warnings without printing or logging. Avoid logging dataset contents
  or credentials.
- For the supported regression journey, offer an exact Python reproduction using
  a portable dataset snapshot and the effective settings. A readable API call
  connects the interface to code Adam can inspect and change.
- Code reproduction must include role overrides, preprocessing, split/seed and
  CV settings; it must reproduce the actual run rather than current unrun edits.

Done when a failure remains understandable after navigation, and the exported
Python reproduces the recorded regression result in the declared compatible
environment without the original database.

### 4. Prove the programming foundation

The feasibility work established these requirements for the programming UI:

- Launch a separate Python kernel on demand using an explicitly selected real
  Python environment. Show the interpreter and relevant package versions.
- Initially use the existing development environment; do not treat the frozen
  MLcookbook executable as a general-purpose Python interpreter. Check that the
  selected environment contains a compatible core package.
- Send an explicit copy of ordinary active data, roles and provenance to the
  kernel. Let code inspect and change its own copy. Returning a dataset requires
  explicit acceptance through ordinary validation and source replacement rules.
- Test completion, output, tracebacks, input prompts, interrupt/restart, kernel
  failure and shutdown. A separate process is a lifecycle boundary, not a sandbox.
- Test an embedded QtConsole and a minimal real notebook hosted by a local Jupyter
  server in Qt WebEngine. Compare native and frozen launches, data transfer,
  save/reopen, process cleanup and distribution requirements.

Done when the frozen app can run code in the declared environment, survive a
kernel failure, exchange a typed dataset and close without orphan processes.
The notebook test must actually edit/run/save an .ipynb file, not just display a
web page. Record findings before choosing the supported interface.

### 5. Deliver a focused programming workspace

The implemented workspace combines an optional native Jupyter QtConsole, a
small .py editor and a real embedded JupyterLab notebook view. The editor runs
selections/current contents in the console kernel. Notebook kernels are separate;
they load explicitly copied snapshots rather than sharing live variables.
Console and Activity are collapsible. No programming runtime starts during
ordinary app startup.

The selected existing Python environment owns the kernels and local notebook
server; the frozen app is not an interpreter. Per-session directories under
`.mlcookbook-runtime` separate connections and data bridges, while durable
snapshots remain available to saved notebooks. Notebook edits and saves use
JupyterLab. Interrupt/restart, kernel failure, unsaved-file prompts, process
cleanup and Windows packaging are part of the integration.

Keep .py and .ipynb as ordinary editable files. Console history is not a notebook;
neither file reopening nor experiment restoration executes code. The initial
programming workspace does not make arbitrary Python a Kitchen node.

## Boundaries and verification

- Preserve UI-free core, plain GUI data/signals and app.py as the sole bridge.
- Keep training-only preprocessing/CV and explicit held-out evaluation intact.
- Ordinary data, examples, Kitchen captures, LLM source sessions and sealed
  Movement Study holdouts retain their existing isolation. Programming export
  starts with ordinary data only; switching pages does not expose other contexts.
- Preserve strict .mlcb v1 validation and offline reopening. Use existing source
  metadata where sufficient; design explicit migrations before adding persistent
  analysis settings. No pickled estimators or live kernel-state archives.
- Retain lazy startup, native Qt cleanup and the single application entry point.
- Each implemented milestone gets focused core/desktop checks, native visual
  review and a separate verified Windows preview before installed delivery.
- The optional `programming` extra declares QtConsole, ipykernel, jupyter-client
  and JupyterLab. The selected environment must also contain a compatible core
  installation. Exact verified environment versions are recorded with delivery
  evidence; selecting an environment does not install or download packages.
- Keep the current uncommitted Research timeline work separate. No checkpoint
  commit or installed-bundle replacement is part of this plan.

## Engineering assessment

Relative scope judgments, not calendar estimates:

| Capability | Scope | Main work |
| --- | --- | --- |
| Activity/error view | Small to medium | Useful event boundaries, retention and failure details |
| SQLite import | Medium | Types, preview/import consistency, cancellation and provenance |
| Schema-driven workflow entry | Medium | Shared requirements, scoped choices and correct handoffs |
| Embedded QtConsole | Medium | Kernel/environment lifecycle and explicit dataset exchange |
| Small script editor | Medium after kernel foundation | Files, selections, execution and unsaved edits |
| Full embedded notebook | Large | Jupyter server, notebook files, browser, recovery and packaging |

The shared environment/data boundary is the main programming investment. A
notebook surface alone would not connect programming to the analytical app.

## Technical references

- [Python SQLite](https://docs.python.org/3.12/library/sqlite3.html)
- [pandas SQL query input](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html)
- [QtConsole embedding and process model](https://qtconsole.readthedocs.io/en/stable/#embedding-the-qtconsole-in-a-qt-application)
- [QtConsole supports PySide6](https://qtconsole.readthedocs.io/en/stable/installation.html)
- [Jupyter kernel specifications](https://jupyter-client.readthedocs.io/en/stable/kernels.html#kernel-specs)
- [JupyterLab Desktop architecture](https://github.com/jupyterlab/jupyterlab-desktop/blob/master/user-guide.md#how-jupyterlab-desktop-works)
- [Qt WebEngine view](https://doc.qt.io/qtforpython-6/PySide6/QtWebEngineWidgets/QWebEngineView.html)
- [PyInstaller executable identity](https://pyinstaller.org/en/stable/runtime-information.html#using-sys-executable-and-sys-argv-0)
