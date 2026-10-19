# Build the portable app with the optional CPU LLM runtime installed in .venv.
$ErrorActionPreference = 'Stop'
$buildRoot = $PSScriptRoot
$buildPython = Join-Path $buildRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $buildPython)) {
    throw 'Create .venv and install .[dev,gui,llm] plus PyInstaller first; see ADAM.md.'
}
foreach ($relativeTarget in @('build', 'dist', 'build/MLcookbook', 'build/pyinstaller-cache', 'dist/MLcookbook')) {
    $buildTarget = [IO.Path]::GetFullPath((Join-Path $buildRoot $relativeTarget))
    if (-not $buildTarget.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Build target escapes the repository: $buildTarget"
    }
    if ((Test-Path -LiteralPath $buildTarget) -and ((Get-Item -LiteralPath $buildTarget).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Build target is a reparse point: $buildTarget"
    }
}
if (Get-Process MLcookbook -ErrorAction SilentlyContinue) {
    throw 'Close MLcookbook before rebuilding.'
}
& $buildPython -c "import torch, transformers, PyInstaller; print('Build runtime:', torch.__version__, transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'The LLM build runtime is unavailable.' }
$savedBuildPath = $env:PATH
$savedBuildCache = $env:PYINSTALLER_CONFIG_DIR
Push-Location -LiteralPath $buildRoot
try {
    $buildPythonDir = Split-Path -Parent $buildPython
    $basePythonDir = (& $buildPython -c 'import sys; print(sys.base_prefix)').Trim()
    $env:PATH = "$buildPythonDir;$basePythonDir;$env:SystemRoot/System32;$env:SystemRoot"
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $buildRoot 'build/pyinstaller-cache'
    & $buildPython -m PyInstaller --clean --noconfirm --windowed --name MLcookbook `
        --icon gui/assets/mlcookbook.ico --add-data 'gui/assets;gui/assets' --paths . --paths src `
        --collect-all pyarrow --collect-data transformers `
        --copy-metadata torch --copy-metadata transformers --copy-metadata tokenizers `
        --copy-metadata safetensors --copy-metadata huggingface-hub `
        --hidden-import transformers.models.gpt2.modeling_gpt2 `
        --hidden-import transformers.models.llama.modeling_llama `
        --hidden-import transformers.models.qwen2.modeling_qwen2 `
        --hidden-import transformers.models.qwen3.modeling_qwen3 `
        --hidden-import transformers.models.gpt2.tokenization_gpt2_fast `
        --hidden-import transformers.models.llama.tokenization_llama_fast `
        --hidden-import transformers.models.qwen2.tokenization_qwen2_fast `
        --exclude-module torchvision --exclude-module torchaudio --exclude-module tensorflow --exclude-module keras app.py
    if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; inspect the build output.' }
} finally {
    $env:PATH = $savedBuildPath
    $env:PYINSTALLER_CONFIG_DIR = $savedBuildCache
    Pop-Location
}
Write-Output 'Built dist/MLcookbook/MLcookbook.exe. Launch it and test a local model before sharing the build.'
