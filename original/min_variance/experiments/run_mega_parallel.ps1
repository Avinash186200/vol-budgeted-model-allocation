# run_mega_parallel.ps1 -- run the 5 mega windows in parallel (one single-
# threaded process each = 5 cores, no oversubscription), then aggregate.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$PY = "<local-path>"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"

$offsets = @(0, 501, 1002, 1503, 2004)
$procs = @()
foreach ($o in $offsets) {
    $p = Start-Process -FilePath $PY -ArgumentList "mega_experiments.py","--window=$o" `
            -RedirectStandardOutput "results\mega_w$o.log" -RedirectStandardError "results\mega_w${o}_err.log" `
            -NoNewWindow -PassThru
    Write-Host "launched window $o pid $($p.Id)"
    $procs += $p
}
Write-Host "waiting for $($procs.Count) window processes ..."
$procs | Wait-Process
Write-Host "all windows done; aggregating ..."
& $PY mega_experiments.py --aggregate *> "results\mega_aggregate.log"
Write-Host "AGGREGATE DONE"