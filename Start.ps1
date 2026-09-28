$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimeConfig = if (Test-Path -LiteralPath 'ai-runtime.json') { Get-Content -Raw -LiteralPath 'ai-runtime.json' | ConvertFrom-Json } else { $null }
$runtime = if ($env:TRAFFIC_AI_HOME) { $env:TRAFFIC_AI_HOME } elseif ($runtimeConfig.home) { $runtimeConfig.home } else { Join-Path $PSScriptRoot '.ai-runtime' }
$aiPython = if ($runtimeConfig.python) { $runtimeConfig.python } else { Join-Path $runtime 'venv\Scripts\python.exe' }
$env:PYTHONDONTWRITEBYTECODE = '1'
$localPackages = Join-Path $runtime 'site-packages'
if (Test-Path -LiteralPath $localPackages -PathType Container) {
    $env:PYTHONPATH = if ($env:PYTHONPATH) { $localPackages + [IO.Path]::PathSeparator + $env:PYTHONPATH } else { $localPackages }
}
if (-not (Test-Path -LiteralPath $aiPython)) { throw 'Run Setup-AI.ps1 first, or configure TRAFFIC_AI_HOME.' }
Write-Host 'Open http://127.0.0.1:8521. Press Ctrl+C to stop this AI server.'
& $aiPython web_server.py --port 8521
