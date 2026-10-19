"""Presentation and controls for local PyTorch language-model generation."""

from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from gui.screens.report_screen import ReportScreen

PARAMS = [
    {"key": "model_path", "label": "Model folder", "kind": "path", "value": "", "max_length": 4096,
     "help": "A local Hugging Face folder with config.json, tokenizer files, and safetensors weights."},
    {"key": "input_mode", "label": "Input", "kind": "choice",
     "options": [("Typed prompt", "prompt"), ("Dataset rows", "rows")]},
    {"key": "prompt", "label": "Prompt", "kind": "multiline", "max_length": 16000,
     "value": "Explain why preprocessing should be fitted only on training data.",
     "help": "For dataset rows, this instruction is prepended to each selected row's text."},
    {"key": "prompt_format", "label": "Prompt format", "kind": "choice",
     "options": [("Auto (use chat template if present)", "auto"), ("Plain text", "plain"), ("Chat template", "chat")]},
    {"key": "max_new_tokens", "label": "Output token limit", "kind": "int", "min": 1, "max": 512, "value": 128},
    {"key": "max_input_tokens", "label": "Input token limit", "kind": "int", "min": 1, "max": 8192, "value": 512},
    {"key": "temperature", "label": "Temperature", "kind": "float", "min": 0.0, "max": 2.0, "value": 0.0,
     "decimals": 2, "step": 0.1, "help": "Zero chooses the most likely next token; positive values sample."},
    {"key": "top_p", "label": "Top-p", "kind": "float", "min": 0.01, "max": 1.0, "value": 0.9,
     "decimals": 2, "step": 0.05, "help": "Sampling probability mass. Used only when temperature is above zero."},
    {"key": "random_state", "label": "Sampling seed", "kind": "int", "min": 0, "max": 2147483647, "value": 0},
    {"key": "device", "label": "Device", "kind": "choice", "options": [("CPU", "cpu"), ("NVIDIA CUDA", "cuda")]},
    {"key": "max_rows", "label": "Dataset row limit", "kind": "int", "min": 1, "max": 10, "value": 1},
    {"key": "max_seconds", "label": "Time limit (s)", "kind": "float", "min": 1.0, "max": 120.0,
     "value": 30.0, "decimals": 0, "step": 5.0,
     "help": "Checked between decoding steps; does not limit model loading or a single forward pass."},
]


class LLMScreen(ReportScreen):
    auto_run_demo = False  # Model weights are chosen explicitly, never downloaded on opening a card.

    def __init__(self, engine, report, parent=None):
        super().__init__("Local LLM · PyTorch", engine, report, parent, task="unsupervised", params=PARAMS,
                         min_features=0, feature_roles=("text", "categorical"),
                         cross_validation=False, preprocessing=False, settings_title="Model & generation settings")
        self.run_button.setText("Generate text")
        self.sections["settings"].setChecked(True)
        self._update_input_mode()

    def _update_input_mode(self):
        if not hasattr(self, "param_widgets") or "input_mode" not in self.param_widgets:
            return
        rows = self.param_widgets["input_mode"].currentData() == "rows"
        for widget in (self.columns, self.columns_label, self.columns_help, self.load_button, self.preview_button,
                       self.checks_button, self.badge, self.sections["import"].parentWidget()):
            widget.setVisible(rows)
        for widget in self.param_rows["max_rows"]:
            widget.setVisible(rows)
        self.columns_help.setText("Select exactly one text or categorical column containing strings. Blank rows are skipped.")

    def _configuration_changed(self, *_):
        super()._configuration_changed()
        self._update_input_mode()

    def restore_session(self, state):
        super().restore_session(state)
        self._update_input_mode()

    def _problem(self):
        request = self.request()
        params = request["params"]
        if not params["model_path"].strip():
            return "Choose a local model folder, then generate text."
        if len(params["prompt"]) > 16000:
            return "Shorten the prompt to at most 16000 characters."
        if params["input_mode"] == "prompt":
            return None if params["prompt"].strip() else "Write a prompt before generating."
        if len(request["features"]) != 1:
            return "Select exactly one input text column for dataset rows."
        if request["overrides"][request["features"][0]] not in ("text", "categorical"):
            return "The input column needs a text or categorical role."
        return None

    def _update_result_state(self):
        super()._update_result_state()
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
            "1. Choose a Hugging Face model folder containing config.json, tokenizer files, and safetensors weights.",
            "2. Write a prompt, or choose Dataset rows to prepend an instruction to a text column.",
            "3. Generate text. Your output, prompt, settings, model fingerprint, and runtime versions can be saved as an experiment.",
            "Supported model families: GPT-2, Llama (including SmolLM2), Qwen2, and Qwen3. "
            "Generation uses your local files. A model is not bundled or downloaded automatically.",
            "Start with a small model on CPU. CUDA requires a compatible PyTorch installation. "
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
