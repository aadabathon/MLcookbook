"""Presentation and controls for local language-model generation."""

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "workflow", "label": "Workflow", "kind": "choice",
     "options": [("Free text generation", "generate"), ("Extract fields", "extract"),
                 ("Summarize rows", "summarize"), ("Rewrite rows", "rewrite"), ("Classify rows", "classify")]},
    {"key": "backend", "label": "Inference engine", "kind": "choice",
     "options": [("PyTorch / Transformers", "pytorch"), ("Ollama", "ollama"), ("llama.cpp server", "llamacpp")]},
    {"key": "ollama_url", "label": "Ollama address", "kind": "text", "value": "http://127.0.0.1:11434",
     "max_length": 2048, "help": "An existing local Ollama server. Only loopback HTTP addresses are accepted."},
    {"key": "ollama_model", "label": "Installed model", "kind": "model", "value": "", "max_length": 512,
     "help": "Refresh to list installed local models. Models are never downloaded by this app."},
    {"key": "llamacpp_url", "label": "llama.cpp address", "kind": "text", "value": "http://127.0.0.1:8080",
     "max_length": 2048, "help": "An existing single-model llama-server. Only loopback HTTP addresses are accepted."},
    {"key": "llamacpp_model", "label": "Loaded model", "kind": "model", "value": "", "max_length": 512,
     "help": "Refresh to inspect the server's loaded model. Loading/downloading models is managed outside this app."},
    {"key": "gguf_path", "label": "GGUF file (optional)", "kind": "file", "value": "", "max_length": 4096,
     "help": "Optional local file to hash during a run. Does not load weights or change the server model. "
             "The hash identifies this file only; its binding to server-loaded weights is unverified."},
    {"key": "model_path", "label": "Model folder", "kind": "path", "value": "", "max_length": 4096,
     "help": "A local Hugging Face folder with config.json, tokenizer files, and safetensors weights."},
    {"key": "input_mode", "label": "Input", "kind": "choice",
     "options": [("Typed prompt", "prompt"), ("Dataset rows", "rows")]},
    {"key": "prompt", "label": "Prompt", "kind": "multiline", "max_length": 16000,
     "value": "Explain why preprocessing should be fitted only on training data.",
     "help": "For dataset rows, this instruction is prepended to each selected row's text."},
    {"key": "template", "label": "Prompt template", "kind": "multiline", "max_length": 16000,
     "value": "Extract the topic and sentiment from this text:\n{{text}}",
     "help": "Bind selected input columns by name: {{text}}. Only these columns enter the prompt."},
    {"key": "field_spec", "label": "Output fields / roles", "kind": "multiline", "max_length": 8000,
     "value": "topic: text\nsentiment: categorical: positive | neutral | negative",
     "help": "One name: role per line. Roles: text, categorical, numeric, boolean. "
             "Categorical labels follow another colon, separated by |. All fields allow null (unknown)."},
    {"key": "references", "label": "Reference labels (optional)", "kind": "multiline", "max_length": 8000,
     "value": "", "help": "For evaluation only: output_field=source_column, one per line. "
                           "Reference columns must be excluded from prompt inputs."},
    {"key": "prompt_format", "label": "Prompt format", "kind": "choice",
     "options": [("Auto (use chat template if present)", "auto"), ("Plain text", "plain"), ("Chat template", "chat")]},
    {"key": "max_new_tokens", "label": "Output token limit", "kind": "int", "min": 1, "max": 512, "value": 128},
    {"key": "max_input_tokens", "label": "Input token limit", "kind": "int", "min": 1, "max": 8192, "value": 512},
    {"key": "context_tokens", "label": "Context budget", "kind": "int", "min": 256, "max": 32768, "value": 4096,
     "help": "Requested Ollama context allocation, including input and output. The server manages input truncation."},
    {"key": "thinking", "label": "Thinking", "kind": "choice",
     "options": [("Off", "off"), ("On", "on"), ("Model default", "default")],
     "help": "On/Off require boolean thinking support. Choose Model default for models with thinking levels."},
    {"key": "temperature", "label": "Temperature", "kind": "float", "min": 0.0, "max": 2.0, "value": 0.0,
     "decimals": 2, "step": 0.1, "help": "Zero chooses the most likely next token; positive values sample."},
    {"key": "top_p", "label": "Top-p", "kind": "float", "min": 0.01, "max": 1.0, "value": 0.9,
     "decimals": 2, "step": 0.05, "help": "Sampling probability mass. Used only when temperature is above zero."},
    {"key": "random_state", "label": "Sampling seed", "kind": "int", "min": 0, "max": 2147483647, "value": 0},
    {"key": "device", "label": "Device", "kind": "choice", "options": [("CPU", "cpu"), ("NVIDIA CUDA", "cuda")]},
    {"key": "max_rows", "label": "Dataset row limit", "kind": "int", "min": 1, "max": 10, "value": 1},
    {"key": "max_seconds", "label": "Time limit (s)", "kind": "float", "min": 1.0, "max": 120.0,
     "value": 90.0, "decimals": 0, "step": 5.0,
     "help": "Server engines: per-row request deadline, including loading. PyTorch: decoding only, checked between steps."},
]

LLAMACPP_DEFAULTS = {"llamacpp_url": "http://127.0.0.1:8080", "llamacpp_model": "", "gguf_path": ""}
WORKFLOW_DEFAULTS = {"workflow": "generate", "template": "Extract the topic and sentiment from this text:\n{{text}}",
                     "field_spec": "topic: text\nsentiment: categorical: positive | neutral | negative", "references": ""}
NEW_DEFAULTS = {"backend": "pytorch", "ollama_url": "http://127.0.0.1:11434", "ollama_model": "",
                "context_tokens": 4096, "thinking": "off", **LLAMACPP_DEFAULTS, **WORKFLOW_DEFAULTS}


class LLMScreen(ReportScreen):
    auto_run_demo = False  # Model weights are chosen explicitly, never downloaded on opening a card.
    connection_requested = Signal(str, str, str)
    workflow_requested = Signal(str, object)

    def __init__(self, engine, report, parent=None):
        super().__init__("Local LLMs", engine, report, parent, task="unsupervised", params=PARAMS,
                         min_features=0, feature_roles=("text", "categorical"),
                         cross_validation=False, preprocessing=False, settings_title="Model & generation settings")
        self.run_button.setText("Generate text")
        self.workflow_tools = QWidget()
        tools = QVBoxLayout(self.workflow_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        self.preset_button = QPushButton("Use task template with selected columns")
        self.preset_button.clicked.connect(self._apply_preset)
        row, _ = self.param_form.getWidgetPosition(self.param_widgets["workflow"])
        self.param_form.insertRow(row + 1, self.preset_button)
        self.workflow_review = QPushButton("Review rows / retry / use as dataset…")
        self.workflow_review.clicked.connect(self._review_workflow)
        self.workflow_compare = QPushButton("Compare saved runs")
        self.workflow_compare.clicked.connect(self._compare_workflows)
        self.workflow_export = QPushButton("Export current run…")
        self.workflow_export.clicked.connect(self._export_workflow)
        for button in (self.workflow_review, self.workflow_compare, self.workflow_export):
            tools.addWidget(button)
        self.result_layout.insertWidget(2, self.workflow_tools)
        self.connection_box = QWidget()
        connection_layout = QVBoxLayout(self.connection_box)
        connection_layout.setContentsMargins(0, 0, 0, 0)
        self.connection_button = QPushButton("Test / refresh models")
        self.connection_button.clicked.connect(self._request_connection)
        connection_layout.addWidget(self.connection_button)
        self.connection_status = QLabel("Start Ollama, then refresh to choose an installed local model.")
        self.connection_status.setWordWrap(True)
        self.connection_status.setTextFormat(Qt.TextFormat.PlainText)
        connection_layout.addWidget(self.connection_status)
        row, _ = self.param_form.getWidgetPosition(self.param_widgets["gguf_path"])
        self.param_form.insertRow(row + 1, self.connection_box)
        self.param_widgets["ollama_model"].activated.connect(self._request_connection)
        self.param_widgets["llamacpp_model"].activated.connect(self._request_connection)
        for key in ("ollama_url", "llamacpp_url"):
            self.param_widgets[key].textChanged.connect(self._reset_connection)
        self.param_widgets["backend"].currentIndexChanged.connect(self._reset_connection)
        self.preview_box = QWidget()
        preview_layout = QVBoxLayout(self.preview_box)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Live preview · not a completed result")
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(180)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.preview_text)
        self.result_layout.insertWidget(2, self.preview_box)
        self.preview_box.hide()
        self.sections["settings"].setChecked(True)
        self._update_input_mode()

    def _request_connection(self, *_):
        backend = self.param_widgets["backend"].currentData()
        if backend in ("ollama", "llamacpp"):
            self.connection_requested.emit(backend, self.param_widgets[backend + "_url"].text(),
                                           self.param_widgets[backend + "_model"].currentText())

    def _reset_connection(self, *_):
        self._llamacpp_status = None
        self.connection_status.setText("Refresh to check this server and its local model.")
        self.connection_status.setMinimumHeight(0)
        self._update_run()

    def show_llamacpp_connection(self, result):
        combo = self.param_widgets["llamacpp_model"]
        selected = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([model["name"] for model in result["models"]])
        combo.setEditText(selected or (result["models"][0]["name"] if result["models"] else ""))
        combo.blockSignals(False)
        self._llamacpp_status = result["status"]
        info = result["models"][0]["metadata"].get("meta", {}) if result["models"] else {}
        detail = " · ".join(f"{label}: {info[key]}" for label, key in
                            (("Parameters", "n_params"), ("GGUF type", "ftype"), ("Context", "n_ctx")) if info.get(key) is not None)
        self.connection_status.setText(f"llama.cpp · {result['status']} · {result['version'] or 'version unavailable'}\n" + detail +
                                       "\nContext, device and thinking are set by the server.\n" +
                                       ("\n".join(result["warnings"]) if result["status"] == "loading" else ""))
        self.connection_status.setMinimumHeight(self.connection_status.heightForWidth(350))
        self._configuration_changed()

    def show_connection(self, result):
        combo = self.param_widgets["ollama_model"]
        selected = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([model["name"] for model in result["models"]])
        combo.setEditText(selected or (result["models"][0]["name"] if result["models"] else ""))
        combo.blockSignals(False)
        model = result.get("selected")
        detail = ""
        if model:
            info = model["details"]
            detail = " · ".join(str(info[key]) for key in ("family", "parameter_size", "quantization_level") if info.get(key))
            if model["size_bytes"] is not None:
                detail += f" · {model['size_bytes'] / 1_000_000_000:.2f} GB"
            contexts = [value for key, value in model["metadata"].get("model_info", {}).items()
                        if key.endswith(".context_length") and type(value) is int]
            if contexts:
                detail += f"\nModel context: {max(contexts):,} tokens; request budget is set below."
        self.connection_status.setText(f"Ollama {result['version'] or 'version unavailable'} · "
                                       f"{len(result['models'])} local model(s)\n{detail}\n" + "\n".join(result["warnings"]))
        self.connection_status.setMinimumHeight(self.connection_status.heightForWidth(350))
        self._configuration_changed()

    def show_progress(self, event):
        row = event["row_position"]
        batch = f"Row {event['output_index']} / {event['total_rows']} · " if "total_rows" in event else ""
        self.preview_label.setText("Live preview · " + batch + (f"source row {row}" if row is not None else "typed prompt") +
                                   " · not a completed result")
        text = event["text"]
        if event["thinking"]:
            text = "Thinking:\n" + event["thinking"] + "\n\nAnswer:\n" + text
        self.preview_text.setPlainText(text)
        self.preview_text.verticalScrollBar().setValue(self.preview_text.verticalScrollBar().maximum())
        self.preview_box.show()

    def set_busy(self, busy):
        super().set_busy(busy)
        self.workflow_tools.setEnabled(not busy)
        self.preview_box.hide()
        self.preview_text.clear()

    def _update_input_mode(self):
        if not hasattr(self, "param_widgets") or "input_mode" not in self.param_widgets:
            return
        structured = self.param_widgets["workflow"].currentData() != "generate"
        rows = structured or self.param_widgets["input_mode"].currentData() == "rows"
        for widget in (self.columns, self.columns_label, self.columns_help, self.load_button, self.preview_button,
                       self.checks_button, self.badge, self.sections["import"].parentWidget()):
            widget.setVisible(rows)
        for widget in self.param_rows["max_rows"]:
            widget.setVisible(rows)
        for key in ("template", "field_spec", "references"):
            for widget in self.param_rows[key]:
                widget.setVisible(structured)
        for key in ("prompt", "input_mode"):
            for widget in self.param_rows[key]:
                widget.setVisible(not structured)
        if hasattr(self, "preset_button"):
            self.preset_button.setVisible(structured)
            self.run_button.setText("Run text workflow" if structured else "Generate text")
        backend = self.param_widgets["backend"].currentData()
        for key in ("model_path", "device", "max_input_tokens"):
            for widget in self.param_rows[key]:
                widget.setVisible(backend == "pytorch")
        for key in ("ollama_url", "ollama_model", "context_tokens", "thinking"):
            for widget in self.param_rows[key]:
                widget.setVisible(backend == "ollama")
        for key in LLAMACPP_DEFAULTS:
            for widget in self.param_rows[key]:
                widget.setVisible(backend == "llamacpp")
        for widget in self.param_rows["prompt_format"]:
                widget.setVisible(backend != "llamacpp" and not structured)
        if hasattr(self, "connection_box"):
            self.connection_box.setVisible(backend in ("ollama", "llamacpp"))
        self.columns_help.setText("Select only the text columns named in {{bindings}}. Reference labels stay unchecked. "
                                  "The row limit counts source positions, including blank rows." if structured else
                                  "Select exactly one text or categorical column containing strings. Blank rows are skipped.")
        self._update_workflow_actions()

    def _configuration_changed(self, *_):
        super()._configuration_changed()
        self._update_input_mode()

    def restore_session(self, state):
        super().restore_session(state)
        self._update_input_mode()

    def prepare_session(self, state):
        state = deepcopy(state)
        old_keys = {spec["key"] for spec in PARAMS} - NEW_DEFAULTS.keys()
        current_keys = {spec["key"] for spec in PARAMS}
        for values in (state.get("params"), state.get("request", {}).get("params"),
                       (state.get("report_request") or {}).get("params")):
            if isinstance(values, dict) and set(values) == old_keys:
                values.update(NEW_DEFAULTS)
            elif isinstance(values, dict) and set(values) == current_keys - LLAMACPP_DEFAULTS.keys() - WORKFLOW_DEFAULTS.keys():
                values.update(LLAMACPP_DEFAULTS | WORKFLOW_DEFAULTS)
            elif isinstance(values, dict) and set(values) == current_keys - LLAMACPP_DEFAULTS.keys():
                values.update(LLAMACPP_DEFAULTS)
            elif isinstance(values, dict) and set(values) == current_keys - WORKFLOW_DEFAULTS.keys():
                values.update(WORKFLOW_DEFAULTS)
        return state

    def _problem(self):
        request = self.request()
        params = request["params"]
        if params["workflow"] != "generate":
            if params["backend"] == "pytorch":
                return "Choose Ollama or llama.cpp for structured data workflows."
            if not request["features"]:
                return "Select the input text columns and bind them in the prompt template."
            if not params["template"].strip() or not params["field_spec"].strip():
                return "Write a prompt template and define the output fields."
        if params["backend"] == "llamacpp":
            if getattr(self, "_llamacpp_status", None) == "loading":
                return "llama.cpp is loading its model. Refresh when loading finishes."
            if not params["llamacpp_model"].strip():
                return "Refresh the loaded llama.cpp model and choose it."
        if params["backend"] == "ollama" and not params["ollama_model"].strip():
            return "Refresh the installed Ollama models and choose one."
        if params["backend"] == "pytorch" and not params["model_path"].strip():
            return "Choose a local model folder, then generate text."
        if len(params["prompt"]) > 16000:
            return "Shorten the prompt to at most 16000 characters."
        if params["workflow"] != "generate":
            return None
        if params["input_mode"] == "prompt":
            return None if params["prompt"].strip() else "Write a prompt before generating."
        if len(request["features"]) != 1:
            return "Select exactly one input text column for dataset rows."
        if request["overrides"][request["features"][0]] not in ("text", "categorical"):
            return "The input column needs a text or categorical role."
        return None

    def _update_result_state(self):
        super()._update_result_state()
        self._update_workflow_actions()
        if hasattr(self, "result_state") and not self._report:
            self.result_state.setText("Choose a local model and write a prompt to begin")

    def _update_run(self):
        super()._update_run()
        self.run_button.setToolTip(self._problem() or "Generate with the chosen local model and current prompt settings.")

    def set_status(self, message):
        super().set_status(message.replace("Run Analysis", "Generate text").replace("Analysis complete.", "Generation complete."))

    def _build_report(self, report):
        if report:
            return super()._build_report(report)
        content = QWidget()
        layout = QVBoxLayout(content)
        for message in (
            "Choose a local language model to begin",
            "1. Choose an engine. For Ollama or llama.cpp, start the local server and click Test / refresh models. "
            "For PyTorch, choose a Hugging Face safetensors model folder.",
            "2. Write a prompt, or choose Dataset rows to prepend an instruction to a text column.",
            "3. Generate text. Your output, prompt, settings, model fingerprint, and runtime versions can be saved as an experiment.",
            "For data workflows, choose Extract fields, Summarize, Rewrite or Classify. Select input columns, apply a task template, "
            "and define output roles. Ollama and llama.cpp request structured JSON; review the validated rows before using generated columns.",
            "PyTorch supports GPT-2, Llama (including SmolLM2), Qwen2, and Qwen3. Ollama uses its installed local models, "
            "including quantized weights. A model is not bundled or downloaded automatically.",
            "Server engines manage their own device, context, thinking, and quantization. llama.cpp uses the server's chat template. "
            "For PyTorch, start with a small model on CPU; "
            "CUDA requires a compatible PyTorch installation. "
            "No training or evaluation scores are produced by this recipe.",
        ):
            label = QLabel(message)
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _apply_preset(self):
        columns = self.request()["features"]
        text = "\n".join("{{" + column + "}}" for column in columns) or "{{text}}"
        task = self.param_widgets["workflow"].currentData()
        instruction, fields = {
            "extract": ("Extract the topic and sentiment from this text:", WORKFLOW_DEFAULTS["field_spec"]),
            "summarize": ("Summarize this text in one sentence:", "summary: text"),
            "rewrite": ("Rewrite this text clearly, preserving its meaning:", "rewritten: text"),
            "classify": ("Classify the sentiment of this text:", "sentiment: categorical: positive | neutral | negative"),
        }.get(task, ("", ""))
        self.param_widgets["template"].setPlainText(instruction + "\n" + text)
        self.param_widgets["field_spec"].setPlainText(fields)

    def _update_workflow_actions(self):
        if not hasattr(self, "workflow_tools"):
            return
        run = self._report.get("workflow_run")
        self.workflow_tools.setVisible(bool(run))
        self.workflow_compare.setEnabled(len(self._report.get("workflow_history", [])) >= 2)

    def _export_workflow(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export LLM workflow", "llm-workflow.json", "JSON (*.json)")
        if path:
            self.workflow_requested.emit("export", path)

    def _compare_workflows(self):
        runs = self._report.get("workflow_history", [])
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose runs to compare")
        dialog.resize(900, 420)
        layout = QVBoxLayout(dialog)
        note = QLabel("Select two or more saved runs with the same source rows, output fields and reference bindings. "
                      "Prompt and model choices may differ; validation is not semantic accuracy.")
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(len(runs), 4)
        table.setHorizontalHeaderLabels(["RUN", "MODEL", "UTC", "TEMPLATE"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for index, run in enumerate(runs):
            for column, value in enumerate((run["run_id"][:8], run["config"]["model"], run["created_utc"], run["config"]["template"])):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(index, column, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        button = QPushButton("Compare selected runs")
        button.setEnabled(False)
        table.itemSelectionChanged.connect(lambda: button.setEnabled(len({cell.row() for cell in table.selectedIndexes()}) >= 2))
        def compare():
            ids = [runs[index]["run_id"] for index in sorted({cell.row() for cell in table.selectedIndexes()})]
            dialog.accept()
            self.workflow_requested.emit("compare", ids)
        button.clicked.connect(compare)
        layout.addWidget(button)
        dialog.exec()

    def _review_workflow(self):
        run = self._report.get("workflow_run")
        if not run:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Review generated fields")
        dialog.resize(1000, 650)
        layout = QVBoxLayout(dialog)
        note = QLabel("Zero-based source rows · null = unknown · invalid, skipped and unprocessed values remain missing. "
                      "Schema validity does not prove factual correctness. Review source text before accepting.")
        note.setWordWrap(True)
        layout.addWidget(note)
        fields = run["config"]["fields"]
        names = [field["name"] for field in fields]
        inputs = run["config"]["bindings"]
        table = QTableWidget(len(run["rows"]), 3 + len(inputs) + len(names))
        table.setHorizontalHeaderLabels(["SOURCE ROW", "OUTCOME", *inputs, *[f"{f['name']} ({f['role']})" for f in fields], "DETAILS"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for index, row in enumerate(run["rows"]):
            values = [str(row["position"]), row["status"], *[str(row["inputs"].get(key) or "") for key in inputs],
                      *[str(row["values"][key]) if row["values"] and row["values"][key] is not None else "null" for key in names], "\n".join(row["errors"])]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(index, column, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        raw = QPlainTextEdit()
        raw.setReadOnly(True)
        raw.setPlaceholderText("Select a row to inspect its exact prompt, raw response and validation errors.")
        raw.setMaximumHeight(180)
        def selected():
            selected_rows = sorted({index.row() for index in table.selectedIndexes()})
            if selected_rows:
                row = run["rows"][selected_rows[0]]
                raw.setPlainText("PROMPT\n" + (row["prompt"] or "[not sent]") + "\n\nRAW OUTPUT\n" + row["raw_text"] +
                                 "\n\n" + "\n".join(row["errors"]))
        table.itemSelectionChanged.connect(selected)
        table.selectRow(0)
        layout.addWidget(raw)
        buttons = QHBoxLayout()
        stale = self.request() != self._report_request
        for title, action, payload in (
            ("Retry selected rows", "retry", None),
            ("Retry failed rows", "retry", [row["position"] for row in run["rows"] if row["status"] in ("error", "invalid", "cancelled")]),
            ("Use as derived dataset", "derive", None),
        ):
            button = QPushButton(title)
            button.setEnabled(not stale and (action != "derive" or any(row["status"] == "valid" for row in run["rows"]))
                              and (payload is None or bool(payload)))
            def execute(*_, action=action, payload=payload):
                values = payload
                if action == "retry" and values is None:
                    values = [run["rows"][index]["position"] for index in sorted({cell.row() for cell in table.selectedIndexes()})]
                dialog.accept()
                self.workflow_requested.emit(action, values)
            button.clicked.connect(execute)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        if stale:
            layout.addWidget(QLabel("Settings changed. Run the full workflow before retrying or accepting fields."))
        dialog.exec()
