"""Optional local Python workspace, with explicit typed dataset exchange.

Importing this module starts no processes and imports neither Qt nor Jupyter.
The desktop coordinator owns a ProgrammingSession; user code runs in independent
Python processes with the normal permissions of the selected environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SNAPSHOT = re.compile(r"[a-f0-9]{32}")
_MODULES = ("pandas", "pyarrow", "mlcookbook", "ipykernel", "jupyter_client", "jupyterlab")
_LAUNCH_LOCK = threading.Lock()


def _external_environment() -> dict:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        environment.pop(name, None)
    if getattr(sys, "frozen", False):
        bundle = str(Path(sys._MEIPASS).resolve()).casefold()
        environment["PATH"] = os.pathsep.join(
            entry for entry in environment.get("PATH", "").split(os.pathsep)
            if entry and not str(Path(entry).resolve()).casefold().startswith(bundle)
        )
        if "LD_LIBRARY_PATH_ORIG" in environment:
            environment["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH_ORIG"]
        else:
            environment.pop("LD_LIBRARY_PATH", None)
    return environment


@contextmanager
def _external_dll_search():
    """Restore system DLL lookup while spawning an external interpreter."""
    with _LAUNCH_LOCK:
        frozen_windows = os.name == "nt" and getattr(sys, "frozen", False)
        if frozen_windows:
            import ctypes
            ctypes.windll.kernel32.SetDllDirectoryW(None)
        try:
            yield
        finally:
            if frozen_windows:
                ctypes.windll.kernel32.SetDllDirectoryW(str(sys._MEIPASS))


def default_python() -> str:
    """Suggest an existing real interpreter, including beside a frozen bundle."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    executable = Path(sys.executable).resolve()
    for directory in executable.parents:
        candidate = directory / ".venv" / "Scripts" / "python.exe"
        if candidate.is_file():
            return str(candidate)
    return ""


def inspect_environment(python: str | Path, *, timeout: float = 20) -> dict:
    """Inspect an explicitly selected interpreter without importing its models."""
    executable = Path(python).expanduser().resolve()
    if not executable.is_file():
        raise ValueError("Select a Python interpreter from an environment with the programming extra installed.")
    if getattr(sys, "frozen", False) and executable == Path(sys.executable).resolve():
        raise ValueError("Select python.exe from a real Python environment, rather than the frozen application.")
    code = (
        "import importlib.metadata as m, importlib.util as u, json, sys; "
        f"names={_MODULES!r}; "
        "print(json.dumps({'python':sys.executable,'version':sys.version.split()[0],"
        "'packages':{n: (m.version(n.replace('_','-')) if u.find_spec(n) else None) for n in names}}))"
    )
    with _external_dll_search():
        process = subprocess.Popen([str(executable), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=_external_environment(), creationflags=_CREATE_NO_WINDOW)
    try:
        output, error = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise ValueError("Unable to inspect the selected Python environment: " + error.strip()[-1500:])
    try:
        info = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("The selected interpreter returned invalid environment information.") from exc
    missing = [name for name, version in info["packages"].items() if version is None]
    if missing:
        raise ValueError("The selected environment needs mlcookbook[programming]; missing: " + ", ".join(missing))
    from importlib.metadata import version
    if info["packages"]["mlcookbook"] != version("mlcookbook"):
        raise ValueError("The selected environment has a different ML Cookbook version.")
    return info


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_frame(frame) -> None:
    import pandas as pd
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Publish a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The returned dataset needs at least one row and one column.")
    if any(not isinstance(name, str) or not name.strip() for name in frame.columns):
        raise ValueError("Every returned column needs a non-empty text name.")
    if frame.columns.duplicated().any():
        raise ValueError("Returned column names must be unique.")
    for name in frame.columns:
        if not frame[name].map(pd.api.types.is_scalar).all():
            raise ValueError(f"Column {name!r} contains nested values; flatten them before publishing.")


class DatasetBridge:
    """An explicit immutable input snapshot and atomic output for user Python.

    Notebook bootstrap cells construct this helper. Publishing does not replace
    the app's active dataset: the app must preview and explicitly accept it.
    """

    def __init__(self, directory: str | Path, snapshot_id: str):
        if not _SNAPSHOT.fullmatch(snapshot_id):
            raise ValueError("Invalid programming snapshot identity.")
        self.directory = Path(directory).resolve()
        self.snapshot_id = snapshot_id

    def load(self):
        """Return a separate Dataset and its source descriptor."""
        import pandas as pd

        from mlcookbook.core.dataset import Dataset
        metadata = json.loads((self.directory / f"{self.snapshot_id}.json").read_text(encoding="utf-8"))
        if metadata.get("empty_origin"):
            raise ValueError("No dataset was sent for this snapshot; create or load a DataFrame in Python.")
        payload = self.directory / f"{self.snapshot_id}.parquet"
        if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("The programming input snapshot checksum does not match.")
        frame = pd.read_parquet(payload, engine="pyarrow")
        return Dataset(frame, overrides=metadata["roles"]), metadata["source"]

    def publish(self, frame, *, roles: dict[str, str] | None = None) -> dict:
        """Publish a table for explicit preview/acceptance in the application."""
        from mlcookbook.core.dataset import Dataset
        _validate_frame(frame)
        # Role names and referenced columns are checked before any output changes.
        original = json.loads((self.directory / f"{self.snapshot_id}.json").read_text(encoding="utf-8"))
        inherited = {name: role for name, role in original["roles"].items() if name in frame.columns}
        inherited.update(roles or {})
        candidate = Dataset(frame, overrides=inherited)
        identifier = uuid.uuid4().hex
        payload = self.directory / f"published-{identifier}.parquet"
        temporary = payload.with_suffix(".tmp")
        try:
            candidate.df.to_parquet(temporary, index=True, engine="pyarrow")
            os.replace(temporary, payload)
            metadata = {
                "format": "mlcookbook-programming-return", "version": 1,
                "snapshot_id": self.snapshot_id, "file": payload.name,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "roles": {column.name: column.var_type.value for column in candidate.schema().columns},
                "rows": len(frame), "columns": len(frame.columns),
            }
            _atomic_json(self.directory / "published.json", metadata)
        finally:
            temporary.unlink(missing_ok=True)
        return {"rows": len(frame), "columns": len(frame.columns), "snapshot_id": self.snapshot_id}


class ProgrammingSession:
    """Own one console kernel and an optional local notebook server."""

    def __init__(self, python: str | Path, workspace: str | Path):
        if not str(python).strip() or not Path(python).expanduser().is_file():
            raise ValueError("Select an existing Python interpreter.")
        if not str(workspace).strip():
            raise ValueError("Select a programming folder.")
        self.python = str(Path(python).expanduser().resolve())
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.runtime_directory = self.workspace / ".mlcookbook-runtime" / uuid.uuid4().hex
        self.runtime_directory.mkdir(parents=True, exist_ok=True)
        self.bridge_directory = self.runtime_directory / "datasets"
        self.bridge_directory.mkdir(exist_ok=True)
        self.environment: dict = {}
        self.manager = None
        self.server: subprocess.Popen | None = None
        self.server_port: int | None = None
        self._server_log = None
        self._server_reader = None
        self._token = secrets.token_urlsafe(32)
        self.snapshot_id: str | None = uuid.uuid4().hex
        self._source: dict = {"format": "mlcookbook-programming-input", "version": 1,
                              "roles": {}, "source": {"kind": "programming"},
                              "fingerprint": None, "empty_origin": True}
        _atomic_json(self.bridge_directory / f"{self.snapshot_id}.json", self._source)

    def start_kernel(self) -> dict:
        """Start a separate IPython process, only on an explicit user action."""
        if self.manager is not None and self.manager.is_alive():
            return self.runtime_info()
        self.environment = inspect_environment(self.python)
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        kernel_directory = self.runtime_directory / "console-kernels" / "mlcookbook"
        kernel_directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(kernel_directory / "kernel.json", {
            "argv": [self.python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ML Cookbook Python", "language": "python",
        })
        specifications = KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)])
        manager = KernelManager(kernel_name="mlcookbook", kernel_spec_manager=specifications,
                                connection_file=str(self.runtime_directory / "console.json"))
        try:
            with _external_dll_search():
                manager.start_kernel(cwd=str(self.workspace), env=_external_environment(),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=_CREATE_NO_WINDOW)
            client = manager.blocking_client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=30)
            finally:
                client.stop_channels()
        except BaseException:
            manager.shutdown_kernel(now=True)
            raise
        self.manager = manager
        self.execute(self.bootstrap_code(load_data=False))
        return self.runtime_info()

    def runtime_info(self) -> dict:
        alive = self.manager is not None and self.manager.is_alive()
        return {"connection_file": str(self.manager.connection_file) if alive else "",
                "environment": self.environment, "workspace": str(self.workspace),
                "snapshot_id": self.snapshot_id, "kernel_alive": bool(alive)}

    def execute(self, code: str, *, timeout: float = 60) -> dict:
        """Execute a coordinator bridge request; user editing uses QtConsole."""
        if self.manager is None or not self.manager.is_alive():
            raise ValueError("Start or restart the Python kernel first.")
        client = self.manager.blocking_client()
        client.start_channels()
        output: list[str] = []
        try:
            client.wait_for_ready(timeout=min(timeout, 10))
            message_id = client.execute(code, allow_stdin=False, store_history=False)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    message = client.get_iopub_msg(timeout=min(1, max(0.01, deadline - time.monotonic())))
                except Exception as exc:
                    from queue import Empty
                    if isinstance(exc, Empty):
                        if not self.manager.is_alive():
                            raise RuntimeError("The Python kernel stopped; restart it to continue.") from exc
                        continue
                    raise
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                content = message["content"]
                if message["msg_type"] == "stream":
                    output.append(content["text"])
                elif message["msg_type"] == "error":
                    raise RuntimeError(content.get("ename", "Python error") + ": " + content.get("evalue", ""))
                elif message["msg_type"] == "status" and content.get("execution_state") == "idle":
                    return {"output": "".join(output)}
            raise TimeoutError("The Python request timed out; interrupt or restart the kernel before retrying.")
        finally:
            client.stop_channels()

    def interrupt_kernel(self) -> None:
        if self.manager is not None and self.manager.is_alive():
            self.manager.interrupt_kernel()

    def restart_kernel(self) -> dict:
        self.stop_kernel()
        return self.start_kernel()

    def stop_kernel(self) -> None:
        if self.manager is not None:
            try:
                self.manager.shutdown_kernel(now=True)
            finally:
                self.manager.cleanup_resources()
                self.manager = None

    def send_dataset(self, dataset, *, source: dict | None = None) -> dict:
        """Capture and load an explicit ordinary-data copy into the console."""
        from mlcookbook.core.fingerprint import dataset_fingerprint
        _validate_frame(dataset.df)
        identifier = uuid.uuid4().hex
        payload = self.bridge_directory / f"{identifier}.parquet"
        dataset.df.to_parquet(payload, index=True, engine="pyarrow")
        metadata = {
            "format": "mlcookbook-programming-input", "version": 1,
            "roles": {column.name: column.var_type.value for column in dataset.schema().columns},
            "source": source or {}, "fingerprint": dataset_fingerprint(dataset),
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }
        _atomic_json(self.bridge_directory / f"{identifier}.json", metadata)
        code = self.bootstrap_code(identifier)
        self.execute(code)
        self.snapshot_id = identifier
        self._source = metadata
        return {"snapshot_id": identifier, "rows": len(dataset.df), "columns": len(dataset.df.columns),
                "fingerprint": metadata["fingerprint"]}

    def bootstrap_code(self, snapshot_id: str | None = None, *, load_data: bool = True) -> str:
        identifier = snapshot_id or self.snapshot_id
        metadata = json.loads((self.bridge_directory / f"{identifier}.json").read_text(encoding="utf-8"))
        code = (
            "import pandas as pd\n"
            "from mlcookbook.programming import DatasetBridge\n"
            f"workbench = DatasetBridge({str(self.bridge_directory)!r}, {identifier!r})\n"
        )
        if load_data and not metadata.get("empty_origin"):
            code += "data, source = workbench.load()\ndf = data.df.copy()\n"
        return code

    def publish_variable(self, name: str = "df") -> dict:
        if not name.isidentifier() or name.startswith("__"):
            raise ValueError("Choose a DataFrame variable name, such as df.")
        self.execute(f"workbench.publish({name})")
        dataset, source = self.read_published_dataset()
        return {"rows": len(dataset.df), "columns": len(dataset.df.columns), "source": source}

    def read_published_dataset(self):
        return self._read_published_dataset(self.bridge_directory, self.snapshot_id, self._source)

    def read_notebook_published_dataset(self, path: str | Path):
        """Read the opened notebook's recorded bridge without executing cells."""
        notebook = Path(path).resolve()
        if not notebook.is_relative_to(self.workspace) or notebook.suffix.lower() != ".ipynb":
            raise ValueError("Choose a notebook inside the selected programming folder.")
        document = json.loads(notebook.read_text(encoding="utf-8"))
        notebook_metadata = document.get("metadata") if isinstance(document, dict) else None
        metadata = notebook_metadata.get("mlcookbook") if isinstance(notebook_metadata, dict) else None
        if (not isinstance(metadata, dict) or metadata.get("format") != "mlcookbook-programming-notebook"
                or type(metadata.get("version")) is not int or metadata["version"] != 1):
            raise ValueError("This notebook has no supported dataset bridge metadata. Create a new notebook in ML Cookbook to return tables.")
        bridge_path = metadata.get("bridge_directory")
        if not isinstance(bridge_path, str) or not Path(bridge_path).is_absolute():
            raise ValueError("The notebook has an invalid dataset bridge directory.")
        directory = Path(bridge_path).resolve()
        identifier = metadata.get("snapshot_id", "")
        if not isinstance(identifier, str) or not _SNAPSHOT.fullmatch(identifier):
            raise ValueError("The notebook has an invalid dataset snapshot identity.")
        if not directory.is_relative_to(self.workspace):
            raise ValueError("The notebook dataset bridge is outside the selected programming folder.")
        parts = directory.relative_to(self.workspace).parts
        if len(parts) != 3 or parts[0] != ".mlcookbook-runtime" or not _SNAPSHOT.fullmatch(parts[1]) or parts[2] != "datasets":
            raise ValueError("The notebook has an invalid dataset bridge directory.")
        original = json.loads((directory / f"{identifier}.json").read_text(encoding="utf-8"))
        if original.get("format") != "mlcookbook-programming-input" or original.get("version") != 1:
            raise ValueError("The notebook input snapshot metadata is unsupported.")
        dataset, source = self._read_published_dataset(directory, identifier, original)
        source["notebook"] = str(notebook)
        return dataset, source

    def _read_published_dataset(self, directory: Path, snapshot_id: str, original: dict):
        import pandas as pd

        from mlcookbook.core.dataset import Dataset
        manifest = directory / "published.json"
        if not manifest.is_file():
            raise ValueError("Publish a table first: workbench.publish(df), or use Return variable from the console.")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata.get("format") != "mlcookbook-programming-return" or metadata.get("version") != 1
                or metadata.get("snapshot_id") != snapshot_id):
            raise ValueError("The returned table belongs to a different input snapshot; send and publish the current data.")
        filename = metadata.get("file", "")
        if not re.fullmatch(r"published-[a-f0-9]{32}\.parquet", filename):
            raise ValueError("The returned dataset descriptor has an invalid payload name.")
        payload = directory / filename
        if hashlib.sha256(payload.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("The returned dataset checksum does not match.")
        frame = pd.read_parquet(payload, engine="pyarrow")
        _validate_frame(frame)
        dataset = Dataset(frame, overrides=metadata["roles"])
        source = {"kind": "programming", "input_fingerprint": original.get("fingerprint"),
                  "input_source": original.get("source", {}), "snapshot_id": snapshot_id,
                  "python": self.python, "versions": self.environment.get("packages", {}),
                  "sha256": metadata["sha256"]}
        return dataset, source

    def create_notebook(self, name: str = "experiment.ipynb") -> Path:
        """Create a normal unexecuted notebook; never overwrite an existing file."""
        filename = Path(name).name
        if filename != name or not filename.lower().endswith(".ipynb"):
            raise ValueError("Choose a notebook filename ending in .ipynb.")
        path = self.workspace / filename
        if path.exists():
            raise ValueError("That notebook already exists; open it or choose another name.")
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "mlcookbook", "display_name": "ML Cookbook Python", "language": "python"},
                         "mlcookbook": {"format": "mlcookbook-programming-notebook", "version": 1,
                                        "bridge_directory": str(self.bridge_directory), "snapshot_id": self.snapshot_id}},
            "cells": [
                {"cell_type": "code", "id": uuid.uuid4().hex, "metadata": {}, "execution_count": None,
                 "outputs": [], "source": self.bootstrap_code().splitlines(keepends=True)},
                {"cell_type": "code", "id": uuid.uuid4().hex, "metadata": {}, "execution_count": None,
                 "outputs": [], "source": ["df.head()\n"] if not self._source.get("empty_origin") else ["# Create or load a DataFrame, for example:\n", "# df = pd.DataFrame({'x': [1, 2, 3]})\n"]},
                {"cell_type": "code", "id": uuid.uuid4().hex, "metadata": {}, "execution_count": None,
                 "outputs": [], "source": ["# Return an edited table explicitly:\n", "# workbench.publish(df)\n"]},
            ],
        }
        _atomic_json(path, notebook)
        return path

    def start_notebook_server(self, *, timeout: float = 45) -> dict:
        """Launch authenticated loopback JupyterLab in the selected environment."""
        if self.server is not None and self.server.poll() is None:
            return {"url": self.notebook_url(), "workspace": str(self.workspace), "pid": self.server.pid}
        if not self.environment:
            self.environment = inspect_environment(self.python)
        kernel_dir = self.runtime_directory / "jupyter" / "kernels" / "mlcookbook"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(kernel_dir / "kernel.json", {
            "argv": [self.python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ML Cookbook Python", "language": "python",
        })
        server_runtime = self.runtime_directory / "server"
        server_runtime.mkdir(exist_ok=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.server_port = probe.getsockname()[1]
        environment = _external_environment()
        environment["JUPYTER_TOKEN"] = self._token
        environment["JUPYTER_PATH"] = str(self.runtime_directory / "jupyter")
        environment["JUPYTER_RUNTIME_DIR"] = str(server_runtime)
        self._server_log = (self.runtime_directory / "notebook-server.log").open("w+", encoding="utf-8")
        arguments = [self.python, "-m", "jupyterlab", "--no-browser", "--ServerApp.ip=127.0.0.1",
                     f"--ServerApp.port={self.server_port}", "--ServerApp.port_retries=0",
                     f"--ServerApp.root_dir={self.workspace}", "--ServerApp.allow_remote_access=False",
                     "--ServerApp.password_required=False"]
        try:
            with _external_dll_search():
                self.server = subprocess.Popen(arguments, cwd=self.workspace, env=environment,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding="utf-8", errors="replace",
                                               creationflags=_CREATE_NO_WINDOW)
            self._server_reader = threading.Thread(target=self._capture_server_log, args=(self.server.stdout,), daemon=True)
            self._server_reader.start()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.server.poll() is not None:
                    self._server_log.flush()
                    details = (self.runtime_directory / "notebook-server.log").read_text(encoding="utf-8")[-2500:]
                    raise RuntimeError("JupyterLab failed to start: " + details.replace(self._token, "[token]"))
                try:
                    self._server_request("api", timeout=0.5)
                    return {"url": self.notebook_url(), "workspace": str(self.workspace), "pid": self.server.pid}
                except (OSError, ValueError):
                    time.sleep(0.1)
            raise TimeoutError("JupyterLab did not become ready before its startup deadline.")
        except BaseException:
            self.stop_notebook_server()
            raise

    def _capture_server_log(self, stream) -> None:
        """Drain server output without persisting authentication URLs or secrets."""
        retained = ""
        try:
            for line in stream:
                line = line.replace(self._token, "[token]")
                line = re.sub(r"([?&]token=)[^\s&]+", r"\1[token]", line)
                retained = (retained + line)[-262144:]
                self._server_log.seek(0)
                self._server_log.write(retained)
                self._server_log.truncate()
                self._server_log.flush()
        finally:
            stream.close()

    def _server_request(self, path: str, *, method: str = "GET", timeout: float = 3):
        request = Request(f"http://127.0.0.1:{self.server_port}/{path}",
                          headers={"Authorization": "token " + self._token}, method=method)
        with urlopen(request, timeout=timeout) as response:
            return response.read()

    def notebook_url(self, path: str | Path | None = None) -> str:
        if self.server_port is None:
            raise ValueError("Start the notebook server first.")
        relative = ""
        if path is not None:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(self.workspace) or resolved.suffix.lower() != ".ipynb":
                raise ValueError("Open a notebook inside the selected programming folder.")
            relative = "/tree/" + quote(resolved.relative_to(self.workspace).as_posix(), safe="/")
        return f"http://127.0.0.1:{self.server_port}/lab{relative}?" + urlencode({"token": self._token})

    def stop_notebook_server(self) -> None:
        process = self.server
        if process is not None:
            if process.poll() is None:
                try:
                    self._server_request("api/shutdown", method="POST", timeout=2)
                    process.wait(timeout=8)
                except (OSError, subprocess.TimeoutExpired):
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, creationflags=_CREATE_NO_WINDOW, timeout=10)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            self.server = None
        if self._server_reader is not None:
            self._server_reader.join(timeout=3)
            self._server_reader = None
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None
        server_runtime = self.runtime_directory / "server"
        if server_runtime.is_dir():
            for connection in server_runtime.glob("*.json"):
                connection.unlink(missing_ok=True)
            for redirect in server_runtime.glob("jpserver-*-open.html"):
                redirect.unlink(missing_ok=True)

    def close(self) -> None:
        try:
            self.stop_notebook_server()
        finally:
            self.stop_kernel()
