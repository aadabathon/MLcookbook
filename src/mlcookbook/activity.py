"""Bounded coordinator activity; independent of analyses and scientific imports."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _redact(value: str) -> str:
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = re.sub(r"(?im)(authorization\s*:\s*)[^\r\n]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)([\"']?(?:token|api[_-]?key|password)[\"']?\s*:\s*)[\"'][^\"']*[\"']", r"\1'[redacted]'", value)
    value = re.sub(r"(?i)(token|api[_-]?key|password|authorization)(\s*[:=]\s*)[^\s&,;]+", r"\1\2[redacted]", value)
    value = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", value)
    return value


class ActivityHistory:
    """Keep plain operation records and an optional rotating diagnostic file."""

    def __init__(self, directory: str | Path | None = None, *, limit: int = 100):
        self.limit = limit
        self.records: list[dict] = []
        self.log_path = Path(directory) / "activity.log" if directory is not None else None
        self.log_error = ""
        self._logger = logging.Logger("mlcookbook.activity." + uuid.uuid4().hex)
        self._logger.propagate = False
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(self.log_path, maxBytes=512 * 1024, backupCount=2, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
            except OSError as exc:
                self.log_error = str(exc)

    def begin(self, operation: str, *, input_name: str = "", input_identity: str = "") -> str:
        record = {"id": uuid.uuid4().hex, "operation": operation, "input": input_name,
                  "input_identity": input_identity, "started_utc": datetime.now(UTC).isoformat(),
                  "finished_utc": "", "elapsed_seconds": None, "outcome": "Running",
                  "warnings": [], "error": "", "details": "", "_started": time.perf_counter()}
        self.records.append(record)
        self.records[:] = self.records[-self.limit:]
        return record["id"]

    def finish(self, identifier: str, *, outcome: str = "Completed", warnings=(), error: str = "", details: str = ""):
        record = next((item for item in self.records if item["id"] == identifier), None)
        if record is None:
            return
        record.update(finished_utc=datetime.now(UTC).isoformat(), outcome=outcome,
                      elapsed_seconds=round(time.perf_counter() - record.pop("_started", time.perf_counter()), 6),
                      warnings=[_redact(str(warning))[:2000] for warning in warnings],
                      error=_redact(error)[:4000], details=_redact(details)[-24000:])
        # Dataframes, result payloads and requests never enter this diagnostic sink.
        if self._logger.handlers:
            self._logger.warning(json.dumps(record, ensure_ascii=False, allow_nan=False))

    def view(self) -> list[dict]:
        return [{key: value for key, value in item.items() if not key.startswith("_")} for item in self.records]

    def close(self):
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
