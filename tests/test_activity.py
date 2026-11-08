"""Coordinator history is bounded and excludes request/data payloads."""
import json

from mlcookbook.activity import ActivityHistory


def test_activity_retains_failures_and_redacts_tokens(tmp_path):
    history = ActivityHistory(tmp_path, limit=2)
    old = history.begin("Old import")
    history.finish(old)
    failed = history.begin("SQL import", input_name="measurements", input_identity="sha256")
    history.finish(failed, outcome="Failed", error="api_key=secret-value password=hidden",
                   details="\x1b[31mTraceback\x1b[0m\nValueError: token:secret-token\nAuthorization: Bearer bearer-secret\n{'api_key': 'json-secret'}",
                   warnings=["A recorded warning"])
    current = history.begin("Regression")
    history.finish(current)
    records = history.view()
    assert len(records) == 2 and records[0]["outcome"] == "Failed"
    assert records[0]["input_identity"] == "sha256"
    assert records[0]["elapsed_seconds"] >= 0
    json.dumps(records, allow_nan=False)
    history.close()
    log = (tmp_path / "activity.log").read_text(encoding="utf-8")
    assert "secret-value" not in log and "secret-token" not in log and "hidden" not in log
    assert "bearer-secret" not in log and "json-secret" not in log
    assert "\\u001b" not in log
    assert "Traceback" in log and "A recorded warning" in log
    assert "_started" not in log
