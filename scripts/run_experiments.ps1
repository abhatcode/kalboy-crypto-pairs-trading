# Runs every dev experiment in the report. The holdout is run separately, once.
param([string[]]$Timeframes = @('h1', 'm15', 'm5'), [int]$Workers = 12)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force logs | Out-Null

function Invoke-Run([string]$Name, [string]$Config, [string[]]$Extra = @()) {
    Write-Host "Running $Name"
    python -m kalboy walkforward --config $Config --workers $Workers @Extra *> "logs\wf_$Name.log"
    if ($LASTEXITCODE -ne 0) { throw "The walk-forward run $Name failed. See logs\wf_$Name.log" }
    Select-String -Path "logs\wf_$Name.log" -Pattern '^Return |^\d+ trades' | ForEach-Object { $_.Line }
}

$runs = @()
foreach ($tf in $Timeframes) {
    foreach ($v in 'base', 'tail', 'anchor', 'allpairs', 'base_kf', 'tail_kf') {
        Invoke-Run "${tf}_$v" "configs\${tf}_$v.json"
        $runs += "results\${tf}_${v}_dev"
    }
    python -m kalboy metalabel "results\${tf}_allpairs_dev" *> "logs\metalabel_$tf.log"
    Get-Content "logs\metalabel_$tf.log"
}
python -m kalboy report @runs --compare results\dev_comparison.csv

# cost sensitivity at 5, 2 and 0 bps per fill
$levels = @{ fee5 = @('0.0002', '3'); fee2 = @('0.0002', '0'); fee0 = @('0', '0') }
$sens = @()
foreach ($tf in $Timeframes) {
    foreach ($v in 'base', 'tail') {
        foreach ($lvl in 'fee5', 'fee2', 'fee0') {
            $label = "${tf}_${v}_${lvl}_dev"
            Invoke-Run $label "configs\${tf}_$v.json" @('--purpose', 'sensitivity', '--label', $label,
                '--set', "costs.taker_fee=$($levels[$lvl][0])", '--set', "costs.slippage_bps=$($levels[$lvl][1])")
            $sens += "results\$label"
        }
    }
}
$baseCost = foreach ($tf in $Timeframes) { foreach ($v in 'base', 'tail') { "results\${tf}_${v}_dev" } }
python -m kalboy report @baseCost @sens --sensitivity results\cost_sensitivity.csv

# final config: reproduction, then larger position sizes
Invoke-Run 'h1_tail_repro' 'configs\h1_tail.json' @('--purpose', 'reproduction', '--label', 'h1_tail_repro_dev')
$sizes = @{ lev5 = @('0.5', '1.5'); lev10 = @('1.0', '3.0'); lev20 = @('2.0', '6.0'); lev50 = @('5.0', '15.0') }
foreach ($name in 'lev5', 'lev10', 'lev20', 'lev50') {
    Invoke-Run "h1_tail_$name" 'configs\h1_tail.json' @('--purpose', 'sensitivity', '--label', "h1_tail_${name}_dev",
        '--set', "strategy.position_pct=$($sizes[$name][0])", '--set', "risk.max_concentration=$($sizes[$name][1])")
}
