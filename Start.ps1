$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime = if ($env:TRAFFIC_AI_HOME) { $env:TRAFFIC_AI_HOME } elseif (Test-Path -LiteralPath 'ai-runtime.json') { (Get-Content -Raw -LiteralPath 'ai-runtime.json' | ConvertFrom-Json).home } else { Join-Path $PSScriptRoot '.ai-runtime' }
$aiPython = Join-Path $runtime 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $aiPython)) { throw 'Run Setup-AI.ps1 first, or configure TRAFFIC_AI_HOME.' }
Write-Host 'Open http://127.0.0.1:8521. Press Ctrl+C to stop this AI server.'
& $aiPython web_server.py --port 8521
