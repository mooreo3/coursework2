$ErrorActionPreference = "Stop"

if ($args.Length -eq 0) {
    Write-Host "ERROR: You must provide a run ID."
    Write-Host "Example:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\serve-model.ps1 demo_msg_plus_time_deep"
    exit 1
}

$RUN_ID = $args[0]
$MODEL_DIR = "$(Get-Location)\models\mlflow_model_$RUN_ID"
$SERVE_PORT = 1234

if (!(Test-Path $MODEL_DIR)) {
    Write-Error "Model directory not found:"
    Write-Error "  $MODEL_DIR"
    Write-Error "Have you trained this run ID yet?"
    exit 1
}

Write-Host "Serving model from:"
Write-Host "  $MODEL_DIR"
Write-Host "On port:"
Write-Host "  $SERVE_PORT"

$existing = docker ps --format "{{.Names}}" | Select-String "mlflow-serve-$RUN_ID"
if ($existing) {
    Write-Host "Stopping existing container mlflow-serve-$RUN_ID..."
    docker stop "mlflow-serve-$RUN_ID" | Out-Null
}

$MLFLOW_IMAGE = "ghcr.io/mlflow/mlflow"
Write-Host "Ensuring MLflow image exists..."
docker pull $MLFLOW_IMAGE | Out-Null

Write-Host "Starting model serving container..."

docker run -d --rm `
  --name "mlflow-serve-$RUN_ID" `
  -p ${SERVE_PORT}:1234 `
  -v "${MODEL_DIR}:/model" `
  $MLFLOW_IMAGE `
  mlflow models serve `
    -m /model `
    -h 0.0.0.0 `
    -p 1234 `
    --no-conda | Out-Null

Write-Host ""
Write-Host "=============================="
Write-Host "✅ MODEL NOW SERVING"
Write-Host "=============================="
Write-Host "POST URL:"
Write-Host "  http://127.0.0.1:$SERVE_PORT/invocations"
Write-Host ""
Write-Host "Header:"
Write-Host "  Content-Type: application/json"
Write-Host ""
Write-Host "Press Ctrl+C to stop container manually later with:"
Write-Host "  docker stop mlflow-serve-$RUN_ID"
Write-Host "=============================="
