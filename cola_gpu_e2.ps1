# cola_gpu_e2.ps1: corre, una detrás de otra, las tareas del E2 que faltan con GPU, apenas termina la ablación.
# Uso (PowerShell, desde Trabajo\datos\wdsd, en una ventana aparte de la ablación):
#     powershell -ExecutionPolicy Bypass -File .\cola_gpu_e2.ps1
# Orden (por prioridad): análisis de la ablación (CPU), sanidad con etiquetas permutadas (~20 min), robustez y latencia
# de los 9 modelos que faltaban (~2-3 h), interpretabilidad del top 3 (~1-1.5 h), paper guía (~3.5 h) y auditoría de
# fuga (CPU). El paper guía solo corre si criterio_paper_guia_E2.md ya está en un commit de Repo_E2 (criterio a priori).
# Cada paso que falla queda anotado y la cola sigue con el siguiente. Todos los scripts saltan lo que ya terminó, así
# que se puede cortar con Ctrl+C y volver a lanzar.
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$conda = "C:\Users\ivanj\miniconda3\Scripts\conda.exe"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force "results\e2\logs" | Out-Null
Start-Transcript -Path ("results\e2\logs\cola_gpu_{0:yyyyMMdd_HHmm}.txt" -f (Get-Date)) -Append | Out-Null
$resumen = @()

function Correr([string[]]$a) {
    $t0 = Get-Date
    Write-Host "`n#### $(Get-Date -Format 'yyyy-MM-dd HH:mm') python $($a -join ' ')"
    & $conda run -n wdsd_dl --no-capture-output python @a
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $script:resumen += "{0,-70} código {1}, {2} min" -f ($a -join ' '), $LASTEXITCODE, $min
}

# 0. esperar a que la ablación tenga sus 15 corridas cerradas (resumen.json o fallo.json)
while ($true) {
    $n = @(Get-ChildItem "results\e2\ablacion" -Directory -ErrorAction SilentlyContinue | Where-Object {
        (Test-Path (Join-Path $_.FullName "resumen.json")) -or (Test-Path (Join-Path $_.FullName "fallo.json")) }).Count
    if ($n -ge 15) { break }
    Write-Host "$(Get-Date -Format HH:mm) ablación: $n de 15 corridas cerradas; reviso de nuevo en 5 min"
    Start-Sleep -Seconds 300
}
Start-Sleep -Seconds 90   # que el proceso de la ablación termine de escribir y suelte la GPU

# 1. ablación: análisis (CPU)
Correr @("analisis_ablacion_e2.py", "results/e2/corridas", "results/e2/ablacion", "results/e2/analisis_ablacion")
# 2. sanidad con etiquetas permutadas
Correr @("sanidad_e2.py", "datos")
Correr @("sanidad_e2.py", "correr")
Correr @("sanidad_e2.py", "analizar")
# 3. robustez y latencia de los 9 modelos que faltaban (se relanza solo en wdsd_mmdet y wdsd_yolo9)
Correr @("robustez_latencia_e2.py", "--entornos", "--faltantes", "--semillas", "42")
# 4. interpretabilidad del top 3
Correr @("interpretabilidad_e2.py", "--entornos")
Correr @("interpretabilidad_e2.py", "--analizar")
# 5. paper guía y auditoría, solo si el criterio a priori ya está en un commit
$commit = git -C "..\..\Repo_E2" log --format="%h %ad" --date=iso -1 -- criterio_paper_guia_E2.md 2>$null
if ($commit) {
    Write-Host "criterio_paper_guia_E2.md está en el commit $commit"
    Correr @("paper_guia_e2.py", "datos")
    Correr @("paper_guia_e2.py", "correr")
    Correr @("auditoria_fuga_e2.py")
    Correr @("analisis_paper_guia_e2.py", "results/e2/paper_guia", "datos_oficial/coco/test.json",
             "results/e2/auditoria_fuga/casi_gemelos_particion_oficial.csv", "results/e2/corridas.csv",
             "results/e2/analisis_paper_guia")
} else {
    $script:resumen += "paper guía y auditoría SALTADOS: criterio_paper_guia_E2.md no está en ningún commit de Repo_E2"
}

Write-Host "`n#### Cola terminada $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Resumen (código 0 = bien):"
$resumen | ForEach-Object { Write-Host "  $_" }
Stop-Transcript | Out-Null
