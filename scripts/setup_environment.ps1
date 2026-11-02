# Install main's shared CUDA environment and register its notebook kernel.
[CmdletBinding()]
param([string]$PythonVersion = '3.14')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
$lockPath = Join-Path $projectRoot 'requirements\training-cu130.lock.txt'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvPath = $uvCommand.Source
} else {
    $uvPath = Join-Path $projectRoot 'artifacts\tooling\bin\uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) {
        throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script.'
    }
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    # Preserve any existing non-environment files inside .venv.
    & $uvPath venv $environmentPath --python $PythonVersion --allow-existing --seed
    if ($LASTEXITCODE -ne 0) { throw 'Shared environment creation failed.' }
}
& $pythonPath -c "import sys; assert (3, 12) <= sys.version_info[:2] <= (3, 14), 'Use Python 3.12-3.14 for the shared main environment.'"
if ($LASTEXITCODE -ne 0) { throw 'The existing environment has an unsupported Python version.' }
& $uvPath pip sync $lockPath --python $pythonPath --torch-backend cu130
if ($LASTEXITCODE -ne 0) { throw 'Dependency synchronization failed.' }
& $uvPath pip check --python $pythonPath
if ($LASTEXITCODE -ne 0) { throw 'The environment has incompatible dependencies.' }
& $pythonPath -m ipykernel install --user --name mtrlearn --display-name 'Python (FaceGuard CUDA 13.0)'
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
& $pythonPath -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.__version__ == '2.10.0+cu130'; assert torchvision.__version__ == '0.25.0+cu130'; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
