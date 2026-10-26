# Build the portable app with the optional CPU LLM runtime installed in .venv.
param([switch]$Preview)

$ErrorActionPreference = 'Stop'
$buildRoot = $PSScriptRoot
$buildPython = Join-Path $buildRoot '.venv/Scripts/python.exe'
$relativeDist = if ($Preview) { 'dist/ui-preview' } else { 'dist' }
$relativeWork = if ($Preview) { 'build/ui-preview' } else { 'build' }
$relativeCache = if ($Preview) { 'build/ui-preview/pyinstaller-cache' } else { 'build/pyinstaller-cache' }
$buildDist = Join-Path $buildRoot $relativeDist
$buildWork = Join-Path $buildRoot $relativeWork
$buildSpec = if ($Preview) { $buildWork } else { $buildRoot }
$builtExe = [IO.Path]::GetFullPath((Join-Path $buildDist 'MLcookbook/MLcookbook.exe'))
$buildIcon = Join-Path $buildRoot 'gui/assets/mlcookbook.ico'
$buildAssets = Join-Path $buildRoot 'gui/assets'
$buildSource = Join-Path $buildRoot 'src'
$buildEntry = Join-Path $buildRoot 'app.py'
if (-not (Test-Path -LiteralPath $buildPython)) {
    throw 'Create .venv and install .[dev,gui,llm] plus PyInstaller first; see ADAM.md.'
}
$relativeTargets = @('build', 'dist', $relativeWork, $relativeDist, $relativeCache,
                     "$relativeWork/MLcookbook", "$relativeDist/MLcookbook") | Select-Object -Unique
foreach ($relativeTarget in $relativeTargets) {
    $buildTarget = [IO.Path]::GetFullPath((Join-Path $buildRoot $relativeTarget))
    if (-not $buildTarget.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Build target escapes the repository: $buildTarget"
    }
    if ((Test-Path -LiteralPath $buildTarget) -and ((Get-Item -LiteralPath $buildTarget).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Build target is a reparse point: $buildTarget"
    }
}
$runningApps = @(Get-Process MLcookbook -ErrorAction SilentlyContinue)
if ($Preview) {
    if ($runningApps | Where-Object { $_.Path -eq $builtExe }) {
        throw 'Close the UI preview before rebuilding it.'
    }
} elseif ($runningApps.Count) {
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
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $buildRoot $relativeCache
    & $buildPython -m PyInstaller --clean --noconfirm --windowed --name MLcookbook `
        --distpath $buildDist --workpath $buildWork --specpath $buildSpec `
        --icon $buildIcon --add-data "$buildAssets;gui/assets" --paths $buildRoot --paths $buildSource `
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
        --exclude-module torchvision --exclude-module torchaudio --exclude-module tensorflow --exclude-module keras $buildEntry
    if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; inspect the build output.' }
} finally {
    $env:PATH = $savedBuildPath
    $env:PYINSTALLER_CONFIG_DIR = $savedBuildCache
    Pop-Location
}
Write-Output "Built $builtExe. Launch it and test a local model before sharing the build."
if ($Preview) { Write-Output 'The existing dist/MLcookbook bundle was left untouched.' }
