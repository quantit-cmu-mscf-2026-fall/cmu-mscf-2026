# One autopilot tick, as the scheduled task runs it (no prompt, no person).
#
#     powershell.exe -NoProfile -ExecutionPolicy Bypass -File deploy\autopilot\run-tick.ps1 `
#         -Checkout <checkout> -Python <venv>\Scripts\python.exe -Config <autopilot.toml> -Log <file>
#
# Puts the checkout first on PYTHONPATH with UTF-8 on, appends the tick's output to -Log and
# exits with the tick's code (0 ok, 2 blocked/stopped). The tick itself writes
# <run>\ALERT.json / .md on a STOP and <run>\autopilot\ops\status.json.
param(
    [Parameter(Mandatory = $true)][string]$Checkout,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Log
)
$ErrorActionPreference = "Stop"
Set-Location $Checkout
$env:PYTHONPATH = $Checkout
$env:PYTHONUTF8 = "1"
$logDir = Split-Path -Parent $Log
if ($logDir -and -not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$stamp = (Get-Date).ToString("s")
Add-Content -Path $Log -Encoding utf8 -Value "[$stamp] tick --config $Config"
$ErrorActionPreference = "Continue"
& $Python -m capstone.pipeline.autopilot tick --config $Config *>> $Log
$code = $LASTEXITCODE
Add-Content -Path $Log -Encoding utf8 -Value "[$((Get-Date).ToString('s'))] exit $code"
exit $code
