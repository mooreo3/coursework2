$ErrorActionPreference = "Stop"

$MLFLOW_CONTAINER_NAME = "mlflow-tracking"
$MLFLOW_PORT = 5000
$MLFLOW_BACKEND_DIR = "$(Get-Location)\mlruns"
$MLFLOW_IMAGE = "ghcr.io/mlflow/mlflow"

$RUN_ID_PREFIX = if ($args.Length -gt 0) { $args[0] } else { "demo" }

Write-Host "Using RUN_ID prefix: $RUN_ID_PREFIX"
Write-Host "MLflow backend dir: $MLFLOW_BACKEND_DIR"
Write-Host "MLflow Docker image: $MLFLOW_IMAGE"

if (!(Test-Path $MLFLOW_BACKEND_DIR)) {
    New-Item -ItemType Directory -Path $MLFLOW_BACKEND_DIR | Out-Null
}

try {
    Write-Host "Ensuring MLflow image '$MLFLOW_IMAGE' is available (docker pull)..."
    docker pull $MLFLOW_IMAGE | Out-Null
    Write-Host "Image '$MLFLOW_IMAGE' is available."
}
catch {
    Write-Error "Failed to pull MLflow image '$MLFLOW_IMAGE'."
    Write-Error "Make sure Docker Desktop is running and you have network access."
    exit 1
}

$running = docker ps --format "{{.Names}}" | Select-String "^$MLFLOW_CONTAINER_NAME$" -ErrorAction SilentlyContinue

if ($running) {
    Write-Host "MLflow container '$MLFLOW_CONTAINER_NAME' already running."
} else {
    Write-Host "Starting MLflow tracking server in Docker..."

    docker run -d --rm `
        --name $MLFLOW_CONTAINER_NAME `
        -p ${MLFLOW_PORT}:5000 `
        -v "${MLFLOW_BACKEND_DIR}:/mlruns" `
        $MLFLOW_IMAGE `
        mlflow server `
            --backend-store-uri sqlite:///mlflow.db `
            --default-artifact-root /mlruns `
            --host 0.0.0.0 `
            --port 5000 | Out-Null

    Write-Host "MLflow server started on http://127.0.0.1:$MLFLOW_PORT"
}

$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:$MLFLOW_PORT"
Write-Host "MLFLOW_TRACKING_URI set to $env:MLFLOW_TRACKING_URI"

Write-Host ""
Write-Host "======================"
Write-Host "Running experiments..."
Write-Host "======================"

python train.py `
  --run-id "${RUN_ID_PREFIX}_msg_only" `
  --features msg_len `
  --n-estimators 100 `
  --max-depth 5

python train.py `
  --run-id "${RUN_ID_PREFIX}_msg_plus_time" `
  --features msg_len,hour,minute,second `
  --n-estimators 100 `
  --max-depth 10

python train.py `
  --run-id "${RUN_ID_PREFIX}_msg_plus_time_deep" `
  --features msg_len,hour,minute,second `
  --n-estimators 300 `
  --max-depth 20

Write-Host ""
Write-Host "======================"
Write-Host "Done."
Write-Host "Open MLflow at: http://127.0.0.1:$MLFLOW_PORT"
Write-Host "Experiment name: self_loop_detection_v2"
Write-Host "======================"
