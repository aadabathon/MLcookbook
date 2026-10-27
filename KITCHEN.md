# The Kitchen

The Kitchen is ML Cookbook's native graph workspace. Its first milestone connects **Dataset snapshot → Select columns → Model comparison → Report**. Selection is optional; one snapshot can feed several comparison branches. Each comparison uses the existing ridge/forest workflow with shared training folds and a separate test holdout.

## Build a workflow

1. Import your CSV or Parquet file in **Data**, then open **Kitchen** in the sidebar.
2. Choose **Dataset snapshot** in the operation selector and click **Add**. With the node selected, click **Use active dataset**. This captures an independent copy. Importing another active file later does not replace it.
3. Add **Select columns**, then connect the dataset's output circle to its input circle. Either drag between circles, click output then input, or use **Connect…** for keyboard access. Check the columns to retain, including the target. Role overrides change statistical interpretation; values and row order stay intact.
4. Add **Model comparison** and connect the selected dataset. Choose a numeric target, check predictor features, and set the ridge penalty, forest settings, split/seed and training CV folds. **Preprocessing & validation…** edits unfitted configuration; fitting still happens separately inside training folds.
5. Add **Report** and connect the comparison's result output. Click **Run workflow**, or select an intermediate node and **Run to selected** to execute only that node's ancestors and itself. Every explicit run repeats the requested operations. Selecting an output does not rerun anything.
6. Select a completed node and open **Output**. Dataset nodes show column roles and a preview; comparison/report nodes show the existing plots, metrics and preprocessing details. **Run details** records original settings, input fingerprint, timestamp and software versions. The warning button stays accessible above the inspector.

Drag nodes to move them, drag empty canvas space to pan, and use **Ctrl+wheel** to zoom. **Fit graph** frames the drawing. With the canvas focused, Left/Right selects nodes and Delete removes the selected node or connection. **Undo/Redo** (Ctrl+Z/Ctrl+Y) covers node, connection, settings and position edits. The last 50 edits remain available while navigating around the app.

## Keep and reopen work

**Save Kitchen…** writes a `.mlck` document containing the graph, current edits/layout/selection, typed Parquet snapshots, completed node results and their original provenance. **Open Kitchen…** restores these without fitting, reading source paths or contacting inference servers. Opening a Kitchen replaces that Kitchen document; ordinary data and recipe/Movement Study sessions remain independent. Existing single-recipe `.mlcb` experiments keep their original format and header actions.

Changed nodes and their descendants show **Stale** while retaining their last completed output. Reverting a change can make that output current again. Failed or cancelled runs preserve completed results; they are labelled with the settings/input fingerprint that produced them. Cancellation stops between operations. A classical model already fitting finishes before cancellation prevents the next node.

The initial editor supports 100 nodes/200 connections, 1–1000 forest trees, depth 1–100 or unlimited, 2–20 CV folds, a test fraction from .05 to .50 in .01 steps, and a nonnegative signed-32-bit seed. Ridge alpha is .0001–10000 with four decimal places. API documents with settings the controls cannot reproduce exactly are rejected on GUI open, rather than silently clamped. The UI-free core accepts its documented broader numeric configuration.

## Python use

Kitchen APIs are deliberately imported from their submodules; root recipe exports are unchanged.

```python
from mlcookbook import Dataset
from mlcookbook.kitchen import (
    KitchenGraph, KitchenNode, KitchenEdge, default_settings, run_graph,
)

# data is an existing Dataset; the graph refers to this explicit snapshot key.
settings = default_settings("comparison") | {"target": "response", "features": ["input"]}
graph = KitchenGraph(
    nodes=[KitchenNode("data", "dataset", {"snapshot": "my-data"}),
           KitchenNode("compare", "comparison", settings), KitchenNode("report", "report", {})],
    edges=[KitchenEdge("data", "data", "compare", "data"),
           KitchenEdge("compare", "result", "report", "result")],
)
execution = run_graph(graph, {"my-data": data})
result = execution.run.to_dict()  # JSON-safe, no estimators or Dataset objects
```

`KitchenDocument`, `save_kitchen` and `load_kitchen` provide explicit persistence; see `mlcookbook.kitchen_io`. Runtime datasets are separate from JSON result records. Structural/type/configuration/data applicability validation precedes fitting, including cycles, missing inputs/columns and incompatible ports. Independent model branches do not constitute a shared-fold model-selection experiment; that guarantee belongs to each comparison node.

This milestone supplies four operations. Vector kernels, further analysis nodes, LLM generation/derivation and reusable fitted models are not supplied by the first milestone. Arbitrary Python execution is outside the graph format.
