# Start or resume paper trading. Set $env:KALBOY_WEBHOOK_URL first for Discord alerts.
param([Parameter(Mandatory)][string]$Config, [Parameter(Mandatory)][string]$Label)
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$py = if (Test-Path venv\Scripts\python.exe) { 'venv\Scripts\python.exe' } else { 'python' }
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $py -m kalboy paper --config $Config --label $Label 2>&1 | ForEach-Object {
    $_; Add-Content -Path "logs\paper_$Label.log" -Value $_ -Encoding UTF8 }
