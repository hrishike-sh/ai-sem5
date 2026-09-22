param([string]$RuntimeHome = (Join-Path $PSScriptRoot '.ai-runtime'))
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv and reopen PowerShell.' }
$RuntimeHome = [System.IO.Path]::GetFullPath($RuntimeHome)
if (Test-Path -LiteralPath (Join-Path $RuntimeHome 'venv')) { throw 'A runtime already exists here. Choose a new RuntimeHome or use the existing environment.' }
uv venv --python 3.11 (Join-Path $RuntimeHome 'venv')
if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
$aiPython = Join-Path $RuntimeHome 'venv\Scripts\python.exe'
uv pip install --python $aiPython 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
uv pip install --python $aiPython -r requirements-ai.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
@{ home = $RuntimeHome } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ai-runtime.json') -Encoding utf8
Write-Host 'Dependencies installed. Supply the required model weights before detection; see README.md.'
