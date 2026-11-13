<#
.SYNOPSIS
  Create .venv and install pinned dependencies (Windows / PowerShell).
.EXAMPLE
  .\scripts\setup.ps1            # runtime + dev
  .\scripts\setup.ps1 -RuntimeOnly
#>
param(
    [string]$Python = "python",
    [switch]$RuntimeOnly,
    [switch]$Recreate
)
# pip writes warnings to stderr; rely on exit codes instead of terminating on stderr.\n$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$venv = Join-Path $root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"

if ($Recreate -and (Test-Path $venv)) { Remove-Item -Recurse -Force $venv }
if (-not (Test-Path $venvPy)) {
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (is '$Python' Python 3.13?)" }
}
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
$req = if ($RuntimeOnly) { "requirements.txt" } else { "requirements-dev.txt" }
& $venvPy -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"

