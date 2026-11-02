# Create an isolated Conda environment and register the CUDA notebook kernel.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environments = (conda env list --json | ConvertFrom-Json).envs
$existing = @($environments | Where-Object { (Split-Path -Leaf $_) -eq 'mtrlearn' })
if ($existing.Count -eq 0) {
    conda create --name mtrlearn python=3.11 pip --override-channels --channel conda-forge --yes
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}
conda run --name mtrlearn python -m pip install --requirement (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
conda run --name mtrlearn python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'The environment has incompatible dependencies.' }
conda run --name mtrlearn python -m ipykernel install --user --name mtrlearn --display-name 'Python (mtrlearn CUDA)'
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
conda run --name mtrlearn python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
