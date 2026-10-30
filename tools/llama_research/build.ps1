param([string]$VcVars = "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat")
$ErrorActionPreference = "Stop"
$captureRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath $VcVars)) { throw "Select an installed Visual Studio x64 vcvars64.bat using -VcVars." }
& "$captureRoot/build.cmd" $VcVars
if ($LASTEXITCODE -ne 0) { throw "Graph helper build failed." }
