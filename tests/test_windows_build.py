"""Exercise preview isolation without packaging the app or importing ML runtimes."""

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="The packaging script is Windows-specific.")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def build_workspace(tmp_path):
    workspace = tmp_path / "build workspace"
    workspace.mkdir()
    shutil.copy2(ROOT / "build_windows.ps1", workspace / "build_windows.ps1")
    venv.EnvBuilder(with_pip=False).create(workspace / ".venv")
    for name in ("torch", "transformers"):
        (workspace / f"{name}.py").write_text("__version__ = 'test-runtime'\n", encoding="utf-8")
    package = workspace / "PyInstaller"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def option(name):
    return Path(args[args.index(name) + 1])
Path('build-invocation.json').write_text(json.dumps({
    'args': args, 'path': os.environ['PATH'],
    'cache': os.environ['PYINSTALLER_CONFIG_DIR'], 'cwd': os.getcwd(),
    'pythonpath': os.environ['PYTHONPATH']
}), encoding='utf-8')
dist, work, spec = option('--distpath'), option('--workpath'), option('--specpath')
bundle = dist / 'MLcookbook'
(bundle / '_internal').mkdir(parents=True, exist_ok=True)
(bundle / 'MLcookbook.exe').write_bytes(b'new executable')
(bundle / '_internal' / 'runtime.dll').write_bytes(b'new runtime')
work.mkdir(parents=True, exist_ok=True)
spec.mkdir(parents=True, exist_ok=True)
(spec / 'MLcookbook.spec').write_text('generated test spec', encoding='utf-8')
""",
        encoding="utf-8",
    )
    original = workspace / "dist" / "MLcookbook"
    (original / "_internal").mkdir(parents=True)
    (original / "MLcookbook.exe").write_bytes(b"existing executable")
    (original / "_internal" / "runtime.dll").write_bytes(b"existing runtime")
    helper = workspace / "tools" / "llama_research"
    (helper / "build").mkdir(parents=True)
    (helper / "vendor").mkdir()
    (helper / "build" / "llama-graph-capture.exe").write_bytes(b"capture helper fixture")
    (helper / "vendor" / "LICENSE").write_text("upstream license fixture", encoding="utf-8")
    return workspace


def _powershell():
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable.")
    return executable


def _quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def _run_build(workspace, *, preview=False, running_path=None):
    script = workspace / "build_windows.ps1"
    # Isolate tests from applications the user may have open on this computer.
    inventory = "@()" if running_path is None else f"[pscustomobject]@{{Path={_quoted(running_path)}}}"
    command = (
        "function Get-Process { param($Name, $ErrorAction) " + inventory + " }; "
        "$beforePath=$env:PATH; $beforeCache=$env:PYINSTALLER_CONFIG_DIR; $beforePythonPath=$env:PYTHONPATH; "
        f"& {_quoted(script)}" + (" -Preview" if preview else "") + "; "
        "if ($env:PATH -ne $beforePath -or $env:PYINSTALLER_CONFIG_DIR -ne $beforeCache -or $env:PYTHONPATH -ne $beforePythonPath) "
        "{ throw 'Build environment was not restored.' }"
    )
    environment = os.environ.copy()
    environment["PATH"] += os.pathsep + str(workspace / "foreign-runtime")
    environment["PYINSTALLER_CONFIG_DIR"] = str(workspace / "original-cache")
    environment["PYTHONPATH"] = str(workspace / "other-checkout")
    return subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
                          cwd=workspace, env=environment, capture_output=True, text=True, timeout=60)


def test_preview_build_preserves_the_original_bundle(build_workspace):
    workspace = build_workspace
    original = workspace / "dist" / "MLcookbook"
    result = _run_build(workspace, preview=True, running_path=original / "MLcookbook.exe")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (original / "MLcookbook.exe").read_bytes() == b"existing executable"
    assert (original / "_internal" / "runtime.dll").read_bytes() == b"existing runtime"
    preview = workspace / "dist" / "ui-preview" / "MLcookbook"
    assert (preview / "MLcookbook.exe").read_bytes() == b"new executable"
    assert (preview / "_internal" / "runtime.dll").read_bytes() == b"new runtime"
    assert (workspace / "build" / "ui-preview" / "MLcookbook.spec").is_file()
    invocation = json.loads((workspace / "build-invocation.json").read_text(encoding="utf-8"))
    assert Path(invocation["cache"]) == workspace / "build" / "ui-preview" / "pyinstaller-cache"
    assert "foreign-runtime" not in invocation["path"]
    assert invocation["pythonpath"].split(os.pathsep) == [str(workspace / "src"), str(workspace)]
    assert Path(invocation["path"].split(os.pathsep)[0]) == workspace / ".venv" / "Scripts"
    assert "torch" not in [invocation["args"][index + 1] for index, value in enumerate(invocation["args"])
                           if value == "--exclude-module"]
    assert Path(invocation["args"][-1]) == workspace / "app.py"
    assert invocation["args"][invocation["args"].index("--collect-submodules") + 1] == "mlcookbook"
    assert Path(invocation["args"][invocation["args"].index("--icon") + 1]) == workspace / "gui" / "assets" / "mlcookbook.ico"
    assets = invocation["args"][invocation["args"].index("--add-data") + 1]
    assert Path(assets.split(";")[0]) == workspace / "gui" / "assets"
    binary = invocation["args"][invocation["args"].index("--add-binary") + 1]
    assert binary.split(";")[1] == "tools/llama_research"
    assert Path(binary.split(";")[0]) == workspace / "tools" / "llama_research" / "build" / "llama-graph-capture.exe"
    data = [invocation["args"][index + 1] for index, value in enumerate(invocation["args"]) if value == "--add-data"]
    assert any(Path(value.split(";")[0]).name == "LICENSE" for value in data)
    assert "was left untouched" in result.stdout


def test_default_build_keeps_its_existing_output_location(build_workspace):
    workspace = build_workspace
    result = _run_build(workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "dist" / "MLcookbook" / "MLcookbook.exe").read_bytes() == b"new executable"
    assert not (workspace / "dist" / "ui-preview").exists()
    assert (workspace / "MLcookbook.spec").is_file()
    invocation = json.loads((workspace / "build-invocation.json").read_text(encoding="utf-8"))
    assert Path(invocation["cache"]) == workspace / "build" / "pyinstaller-cache"


def test_preview_build_requires_closing_only_its_own_running_output(build_workspace):
    workspace = build_workspace
    preview = workspace / "dist" / "ui-preview" / "MLcookbook" / "MLcookbook.exe"
    result = _run_build(workspace, preview=True, running_path=preview)
    assert result.returncode != 0
    assert "Close the UI preview" in result.stderr
    assert not (workspace / "build-invocation.json").exists()


def test_build_requires_the_capture_component_before_packaging(build_workspace):
    helper = build_workspace / "tools" / "llama_research" / "build" / "llama-graph-capture.exe"
    helper.unlink()
    result = _run_build(build_workspace, preview=True)
    assert result.returncode != 0
    assert "Build the graph capture helper first" in result.stderr
    assert not (build_workspace / "build-invocation.json").exists()


def test_preview_build_rejects_a_redirected_output_directory(build_workspace, tmp_path):
    workspace = build_workspace
    redirected = tmp_path / "redirected output"
    redirected.mkdir()
    sentinel = redirected / "keep.txt"
    sentinel.write_bytes(b"unchanged")
    junction = workspace / "dist" / "ui-preview"
    command = f"New-Item -ItemType Junction -Path {_quoted(junction)} -Target {_quoted(redirected)} | Out-Null"
    created = subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
                             capture_output=True, text=True, timeout=20)
    assert created.returncode == 0, created.stdout + created.stderr
    try:
        result = _run_build(workspace, preview=True)
        assert result.returncode != 0
        assert "reparse point" in result.stderr
        assert not (workspace / "build-invocation.json").exists()
        assert sentinel.read_bytes() == b"unchanged"
    finally:
        # Removing this specific junction does not recurse into its target.
        os.rmdir(junction)
