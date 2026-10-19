<#
.SYNOPSIS
  Build the API image and deploy it to the local Docker Desktop Kubernetes cluster.

.EXAMPLE
  .\scripts\k8s_deploy.ps1                 # build the tag in k8s/deployment.yaml, deploy, smoke test
  .\scripts\k8s_deploy.ps1 -SkipBuild      # redeploy the existing image
  .\scripts\k8s_deploy.ps1 -Monitoring     # also deploy Prometheus + Grafana
#>
param(
    [switch]$SkipBuild,
    [switch]$Monitoring,
    [string]$Context = "docker-desktop",
    [string]$Url = "http://localhost"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
# Single source of truth for the tag: the Deployment manifest.
$match = Select-String -Path k8s/deployment.yaml -Pattern 'image:\s*(heart-api:\S+)'
if (-not $match) { throw "no heart-api image found in k8s/deployment.yaml" }
$image = $match.Matches[0].Groups[1].Value

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Check($ok, $msg) { if (-not $ok) { throw $msg } }

Step "Using kube context '$Context'"
kubectl config use-context $Context | Out-Null
Check ($LASTEXITCODE -eq 0) "kube context '$Context' not found; enable Kubernetes in Docker Desktop"

if (-not $SkipBuild) {
    Step "Building $image"
    docker build -t $image .
    Check ($LASTEXITCODE -eq 0) "docker build failed"
}

# Docker Desktop's kind-based cluster runs its own containerd: images built locally are not
# visible to it until they are imported into the node (the same thing `kind load` does).
docker exec desktop-control-plane true 2>$null
if ($LASTEXITCODE -eq 0) {
    Step "Loading $image into the kind node"
    # cmd pipes binary data intact; Windows PowerShell 5.1 pipes would corrupt the tarball
    cmd /c "docker save $image | docker exec -i desktop-control-plane ctr -n k8s.io images import -"
    Check ($LASTEXITCODE -eq 0) "image import into the cluster failed"
}

Step "Applying manifests (kubectl apply -k k8s/)"
kubectl apply -k k8s/
Check ($LASTEXITCODE -eq 0) "kubectl apply failed"
if ($Monitoring) {
    Step "Applying monitoring stack (kubectl apply -k k8s/monitoring/)"
    kubectl apply -k k8s/monitoring/
    Check ($LASTEXITCODE -eq 0) "kubectl apply (monitoring) failed"
}

Step "Waiting for rollout"
kubectl -n heart-mlops rollout status deployment/heart-api --timeout=180s
Check ($LASTEXITCODE -eq 0) "rollout did not complete"
kubectl -n heart-mlops get pods,svc -o wide

Step "Smoke testing $Url"
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $python scripts/smoke_test.py --url $Url --wait 60
Check ($LASTEXITCODE -eq 0) "smoke test failed"

if ($Monitoring) {
    kubectl -n monitoring rollout status deployment/prometheus --timeout=180s
    kubectl -n monitoring rollout status deployment/grafana --timeout=180s
    Write-Host "Prometheus: http://localhost:9090   Grafana: http://localhost:3000" -ForegroundColor Green
}
Write-Host "`nDeployed $image -> $Url  (docs: $Url/docs)" -ForegroundColor Green
