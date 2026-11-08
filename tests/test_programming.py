"""Explicit programming snapshots and optional real local Python runtimes."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from mlcookbook import Dataset
from mlcookbook.programming import DatasetBridge, ProgrammingSession, inspect_environment


@pytest.fixture
def bridge(tmp_path):
    identifier = "a" * 32
    frame = pd.DataFrame({"id": pd.Series(["001", "002"], dtype="string"),
                          "value": pd.Series([1, None], dtype="Int64")})
    frame.index = [7, 7]
    payload = tmp_path / f"{identifier}.parquet"
    frame.to_parquet(payload, index=True)
    (tmp_path / f"{identifier}.json").write_text(json.dumps({
        "roles": {"id": "categorical", "value": "numeric"}, "source": {"kind": "test"},
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return DatasetBridge(tmp_path, identifier)


def test_bridge_preserves_nullable_types_duplicate_index_and_roles(bridge):
    dataset, source = bridge.load()
    assert dataset.df.index.tolist() == [7, 7]
    assert dataset.df.id.tolist() == ["001", "002"]
    assert str(dataset.df.value.dtype) == "Int64"
    assert dataset.schema().var_type("id").value == "categorical"
    assert source == {"kind": "test"}
    changed = dataset.df.copy()
    changed["extra"] = pd.array([True, None], dtype="boolean")
    result = bridge.publish(changed)
    assert result["rows"] == 2
    manifest = json.loads((bridge.directory / "published.json").read_text(encoding="utf-8"))
    assert manifest["roles"]["id"] == "categorical"
    assert manifest["roles"]["extra"] == "boolean"
    assert "extra" not in bridge.load()[0].df
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("frame, error", [
    (pd.DataFrame(), "at least one"),
    (pd.DataFrame([[1, 2]], columns=["x", "x"]), "unique"),
    (pd.DataFrame({"nested": [[1, 2]]}), "nested"),
    (pd.DataFrame({"": [1]}), "non-empty"),
])
def test_invalid_publish_preserves_completed_return(bridge, frame, error):
    dataset, _ = bridge.load()
    bridge.publish(dataset.df)
    before = (bridge.directory / "published.json").read_bytes()
    with pytest.raises(ValueError, match=error):
        bridge.publish(frame)
    assert (bridge.directory / "published.json").read_bytes() == before


def test_bridge_checks_input_checksum(bridge):
    (bridge.directory / f"{bridge.snapshot_id}.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        bridge.load()


def test_programming_module_import_is_lazy():
    code = "import sys; import mlcookbook.programming; print([m for m in ('pandas','numpy','PySide6','qtconsole','jupyter_client','ipykernel') if m in sys.modules])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_environment_missing_interpreter_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="interpreter"):
        inspect_environment(tmp_path / "missing.exe")


def test_empty_session_inputs_are_rejected_before_workspace_writes(tmp_path):
    with pytest.raises(ValueError, match="interpreter"):
        ProgrammingSession("", tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()
    with pytest.raises(ValueError, match="folder"):
        ProgrammingSession(sys.executable, "")


def test_server_log_is_bounded_and_redacts_authentication(tmp_path):
    session = ProgrammingSession(sys.executable, tmp_path)
    session._server_log = io.StringIO()
    session._capture_server_log(io.StringIO("x" * 300000 + "\nhttp://127.0.0.1:8888/?token=" + session._token + "\n"))
    content = session._server_log.getvalue()
    assert len(content) <= 262144
    assert session._token not in content
    assert "[token]" in content
    session._server_log = None


def test_sessions_in_one_folder_keep_independent_bridges_and_cleanup(tmp_path):
    left = ProgrammingSession(sys.executable, tmp_path)
    right = ProgrammingSession(sys.executable, tmp_path)
    assert left.runtime_directory != right.runtime_directory
    assert left.bridge_directory != right.bridge_directory
    DatasetBridge(left.bridge_directory, left.snapshot_id).publish(pd.DataFrame({"left": [1]}))
    DatasetBridge(right.bridge_directory, right.snapshot_id).publish(pd.DataFrame({"right": [2]}))
    for session in (left, right):
        (session.runtime_directory / "server").mkdir()
        (session.runtime_directory / "server" / "connection.json").write_text("{}", encoding="utf-8")
    left.close()
    assert not (left.runtime_directory / "server" / "connection.json").exists()
    assert (right.runtime_directory / "server" / "connection.json").exists()
    assert right.read_published_dataset()[0].df.right.tolist() == [2]
    # Saved notebook bootstrap paths remain valid after a session closes.
    assert (left.bridge_directory / f"{left.snapshot_id}.json").is_file()
    right.close()


def test_unexecuted_notebook_uses_ordinary_file_and_refuses_overwrite(tmp_path):
    session = ProgrammingSession(sys.executable, tmp_path)
    path = session.create_notebook()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"])
    assert session.manager is None and session.server is None
    with pytest.raises(ValueError, match="already exists"):
        session.create_notebook()
    with pytest.raises(ValueError, match="filename"):
        session.create_notebook("../bad.ipynb")


def test_saved_notebook_returns_its_original_snapshot_after_new_session(tmp_path, monkeypatch):
    original = Dataset(pd.DataFrame({"id": ["001", "002"], "value": [2., 3.]}, index=[7, 7]))
    prior = ProgrammingSession(sys.executable, tmp_path)
    monkeypatch.setattr(prior, "execute", lambda code: {})
    prior.send_dataset(original, source={"kind": "fixture", "name": "original notebook input"})
    notebook = prior.create_notebook("saved.ipynb")
    before = notebook.read_bytes()
    prior.close()

    current = ProgrammingSession(sys.executable, tmp_path)
    monkeypatch.setattr(current, "execute", lambda *args, **kwargs: pytest.fail("Reading notebook metadata must not execute cells"))
    DatasetBridge(current.bridge_directory, current.snapshot_id).publish(pd.DataFrame({"console": [99]}))
    with pytest.raises(ValueError, match="Publish a table first"):
        current.read_notebook_published_dataset(notebook)
    bridge = DatasetBridge(prior.bridge_directory, prior.snapshot_id)
    changed = bridge.load()[0].df.copy()
    changed["doubled"] = changed.value * 2
    bridge.publish(changed)
    returned, source = current.read_notebook_published_dataset(notebook)
    assert returned.df.doubled.tolist() == [4., 6.]
    assert returned.df.index.tolist() == [7, 7]
    assert source["snapshot_id"] == prior.snapshot_id != current.snapshot_id
    assert source["input_fingerprint"] == prior._source["fingerprint"]
    assert source["input_source"] == prior._source["source"]
    assert source["notebook"] == str(notebook)
    assert current.read_published_dataset()[0].df.console.tolist() == [99]
    assert "doubled" not in original.df
    assert notebook.read_bytes() == before
    current.close()


@pytest.mark.parametrize("change, error", [
    (lambda metadata, root: metadata.clear(), "no supported dataset bridge metadata"),
    (lambda metadata, root: metadata.update(version=2), "no supported dataset bridge metadata"),
    (lambda metadata, root: metadata.update(bridge_directory=str(root.parent)), "outside"),
    (lambda metadata, root: metadata.update(snapshot_id="wrong"), "snapshot identity"),
])
def test_notebook_return_rejects_missing_or_foreign_bridge_metadata(tmp_path, change, error):
    session = ProgrammingSession(sys.executable, tmp_path)
    notebook = session.create_notebook()
    document = json.loads(notebook.read_text(encoding="utf-8"))
    change(document["metadata"]["mlcookbook"], tmp_path)
    notebook.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        session.read_notebook_published_dataset(notebook)
    session.close()


def test_return_descriptor_rejects_foreign_snapshot_and_path(bridge):
    session = ProgrammingSession(sys.executable, bridge.directory)
    session.bridge_directory = bridge.directory
    session.snapshot_id = bridge.snapshot_id
    session._source = {"fingerprint": "original", "source": {}}
    dataset, _ = bridge.load()
    bridge.publish(dataset.df)
    manifest = bridge.directory / "published.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["snapshot_id"] = "b" * 32
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="different input"):
        session.read_published_dataset()
    metadata["snapshot_id"] = bridge.snapshot_id
    metadata["file"] = "../outside.parquet"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="payload name"):
        session.read_published_dataset()


def _has_programming():
    return all(importlib.util.find_spec(name) is not None for name in ("jupyter_client", "jupyterlab", "ipykernel"))


@pytest.mark.skipif(not _has_programming(), reason="optional programming extra")
def test_real_kernel_transfer_errors_restart_and_cleanup(tmp_path):
    session = ProgrammingSession(sys.executable, tmp_path)
    try:
        info = session.start_kernel()
        assert info["kernel_alive"]
        assert Path(info["environment"]["python"]).resolve() == Path(sys.executable).resolve()
        dataset = Dataset(pd.DataFrame({"id": ["001", "002"], "x": pd.array([1, None], dtype="Int64")}),
                          overrides={"id": "categorical"})
        session.send_dataset(dataset, source={"kind": "fixture"})
        session.execute("df['twice'] = df.x * 2")
        session.publish_variable()
        returned, source = session.read_published_dataset()
        assert returned.df.twice.tolist()[0] == 2
        assert dataset.df.columns.tolist() == ["id", "x"]
        assert source["input_source"] == {"kind": "fixture"}
        with pytest.raises(RuntimeError, match="ValueError: deliberate"):
            session.execute("raise ValueError('deliberate')")
        assert "42" in session.execute("print(6 * 7)")["output"]
        old = json.loads(Path(info["connection_file"]).read_text(encoding="utf-8"))["key"]
        restarted = session.restart_kernel()
        assert restarted["kernel_alive"]
        assert old != json.loads(Path(restarted["connection_file"]).read_text(encoding="utf-8"))["key"]
        with pytest.raises(RuntimeError, match="NameError"):
            session.execute("print(df)")
    finally:
        session.close()
    assert session.manager is None


@pytest.mark.skipif(not _has_programming(), reason="optional programming extra")
def test_real_kernel_can_create_and_publish_without_imported_dataset(tmp_path):
    session = ProgrammingSession(sys.executable, tmp_path)
    try:
        session.start_kernel()
        session.execute("df = pd.DataFrame({'created': [1, 2]})")
        session.publish_variable("df")
        returned, source = session.read_published_dataset()
        assert returned.df.created.tolist() == [1, 2]
        assert source["input_fingerprint"] is None
        assert source["input_source"] == {"kind": "programming"}
        notebook = json.loads(session.create_notebook().read_text(encoding="utf-8"))
        assert "workbench = DatasetBridge" in "".join(notebook["cells"][0]["source"])
        assert "workbench.load" not in "".join(notebook["cells"][0]["source"])
    finally:
        session.close()


@pytest.mark.skipif(not _has_programming(), reason="optional programming extra")
def test_real_sessions_share_folder_without_sharing_connection_or_return(tmp_path):
    left = ProgrammingSession(sys.executable, tmp_path)
    right = ProgrammingSession(sys.executable, tmp_path)
    try:
        left_info = left.start_kernel()
        right_info = right.start_kernel()
        assert left_info["connection_file"] != right_info["connection_file"]
        left.execute("df = pd.DataFrame({'left': [1]})")
        right.execute("df = pd.DataFrame({'right': [2]})")
        left.publish_variable()
        right.publish_variable()
        left.close()
        assert "still alive" in right.execute("print('still alive')")["output"]
        assert right.read_published_dataset()[0].df.right.tolist() == [2]
    finally:
        left.close()
        right.close()


@pytest.mark.skipif(not _has_programming(), reason="optional programming extra")
def test_real_kernel_completion_interrupt_stdin_and_crash_recovery(tmp_path):
    from queue import Empty

    session = ProgrammingSession(sys.executable, tmp_path)
    client = None
    try:
        session.start_kernel()
        client = session.manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=20)
        completion_id = client.complete("str.upp", len("str.upp"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            reply = client.get_shell_msg(timeout=10)
            if reply["parent_header"].get("msg_id") == completion_id:
                assert any("upper" in match for match in reply["content"]["matches"])
                break
        else:
            pytest.fail("Completion did not return.")
        input_id = client.execute("entered = input('Value: ')", allow_stdin=True)
        prompt = client.get_stdin_msg(timeout=10)
        assert prompt["parent_header"]["msg_id"] == input_id
        assert prompt["content"]["prompt"] == "Value: "
        client.input("answered")
        while True:
            reply = client.get_shell_msg(timeout=10)
            if reply["parent_header"].get("msg_id") == input_id:
                assert reply["content"]["status"] == "ok"
                break
        assert "answered" in session.execute("print(entered)")["output"]
        request_id = client.execute("while True:\n    pass")
        while True:
            message = client.get_iopub_msg(timeout=10)
            if message.get("parent_header", {}).get("msg_id") == request_id and message["msg_type"] == "status" and message["content"]["execution_state"] == "busy":
                break
        # busy precedes entry to IPython's signal-handling execution context.
        time.sleep(0.1)
        session.interrupt_kernel()
        while True:
            reply = client.get_shell_msg(timeout=10)
            if reply["parent_header"].get("msg_id") == request_id:
                assert reply["content"]["status"] == "error"
                assert reply["content"]["ename"] == "KeyboardInterrupt"
                break
        assert "42" in session.execute("print(42)")["output"]
        # Native blocking calls may ignore Windows' simulated SIGINT. Restart
        # must still terminate that owned kernel and recover without waiting
        # for the operation itself to finish.
        blocked_id = client.execute("import time; time.sleep(60)")
        while True:
            message = client.get_iopub_msg(timeout=10)
            if message.get("parent_header", {}).get("msg_id") == blocked_id and message["msg_type"] == "status" and message["content"]["execution_state"] == "busy":
                break
        client.stop_channels()
        client = None
        assert session.restart_kernel()["kernel_alive"]
        assert "after blocked call" in session.execute("print('after blocked call')")["output"]
        client = session.manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=20)
        client.execute("import os; os._exit(9)")
        deadline = time.monotonic() + 10
        while session.manager.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not session.manager.is_alive()
        client.stop_channels()
        client = None
        assert session.restart_kernel()["kernel_alive"]
        assert "recovered" in session.execute("print('recovered')")["output"]
    except Empty:
        pytest.fail("The kernel did not reply before the bounded deadline.")
    finally:
        if client is not None:
            client.stop_channels()
        session.close()


@pytest.mark.skipif(not _has_programming(), reason="optional programming extra")
def test_real_notebook_server_edits_runs_saves_and_closes(tmp_path):
    import nbformat
    from jupyter_client import BlockingKernelClient

    session = ProgrammingSession(sys.executable, tmp_path)
    kernel_client = None
    try:
        session.start_kernel()
        session.send_dataset(Dataset(pd.DataFrame({"x": [1.0, 2.0]})))
        notebook_path = session.create_notebook()
        server = session.start_notebook_server()
        assert server["url"].startswith("http://127.0.0.1:")
        assert "/lab/tree/experiment.ipynb?token=" in session.notebook_url(notebook_path)
        # The same local server used by the embedded view opens a real notebook
        # and owns an independent kernel using our explicit interpreter spec.
        from urllib.request import Request, urlopen
        request = Request(f"http://127.0.0.1:{session.server_port}/api/sessions",
                          data=json.dumps({"path": notebook_path.name, "type": "notebook", "name": "",
                                           "kernel": {"name": "mlcookbook"}}).encode(),
                          headers={"Authorization": "token " + session._token, "Content-Type": "application/json"},
                          method="POST")
        with urlopen(request, timeout=30) as response:
            remote = json.loads(response.read())
        connection = session.runtime_directory / "server" / ("kernel-" + remote["kernel"]["id"] + ".json")
        kernel_client = BlockingKernelClient(connection_file=str(connection))
        kernel_client.load_connection_file()
        kernel_client.start_channels()
        kernel_client.wait_for_ready(timeout=30)
        notebook = nbformat.read(notebook_path, as_version=4)
        notebook.cells.append(nbformat.v4.new_code_cell("df['notebook'] = df.x + 10\nworkbench.publish(df)"))
        for cell in notebook.cells:
            reply = kernel_client.execute_interactive(cell.source, timeout=30)
            assert reply["content"]["status"] == "ok"
        # Saving goes through the real contents API, as the Jupyter frontend does.
        save = Request(f"http://127.0.0.1:{session.server_port}/api/contents/{notebook_path.name}",
                       data=json.dumps({"type": "notebook", "format": "json", "content": notebook}).encode(),
                       headers={"Authorization": "token " + session._token, "Content-Type": "application/json"},
                       method="PUT")
        with urlopen(save, timeout=10) as response:
            assert response.status == 200
        assert "workbench.publish" in nbformat.read(notebook_path, as_version=4).cells[-1].source
        returned, _ = session.read_published_dataset()
        assert returned.df.notebook.tolist() == [11.0, 12.0]
    finally:
        if kernel_client is not None:
            kernel_client.stop_channels()
        session.close()
    assert session.server is None and session.manager is None
