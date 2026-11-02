# Run the recognition notebooks

Use [triplet_loss.ipynb](triplet_loss.ipynb) for B0 and [arcface.ipynb](arcface.ipynb) for B1. Both notebooks follow image processing, model training, loss graphs, and artifact export. Code, comments, and notebook explanations are in English.

From the repository root in PowerShell:

```powershell
./scripts/setup_mtrlearn.ps1
conda activate mtrlearn
python -m jupyterlab
```

Select **Python (mtrlearn CUDA)** as the notebook kernel. Alternatively, use `conda env create -f environment.yml` on a fresh machine, then register the kernel using the command in the notebook. The environment uses Python 3.11 and pinned CUDA 12.8 builds of PyTorch and torchvision. Do not install this requirements file into Conda base or a CPU-only environment.

Run one notebook at a time so the two training jobs do not compete for GPU memory. Each screens three margins at seed 42, then repeats its selected margin at seeds 123 and 2026. Each run starts from a fresh random encoder. The two notebooks reuse the same identity-disjoint manifests and evaluation pairs. Full training can take many hours; the notebook does not automatically estimate scientific results from its smoke run.

The default physical batch size is 32, P=8 and K=4. A disposable GPU pilot tests both objectives and falls back to the common batch-16 profile if necessary. Windows notebooks use zero data workers to avoid notebook spawn issues; importable scripts can use the default two workers. Source images and the original DuckDB catalog are read-only. Manifest preparation only removes duplicate entries from new CSVs.

Outputs are organized as follows:

```text
artifacts/
  shared/manifests/                 shared splits, exact-hash cache, pairs, probe
  shared/runtime_pilot.json         measured GPU memory profile
  triplet/                         B0 outputs
  arcface/                         B1 outputs
    sweep_val.csv                  dev-only margin and seed results
    selection.json                 selected margin and criterion
    graphs/                        processing preview and margin plot
    runs/margin_0.3_seed_42/
      config.json                  settings, environment, initialization hash
      history.csv                  training and dev metrics by epoch
      result.json                  best epoch, threshold, curriculum status
      graphs/                      loss, ROC, and score-distribution plots
      weights/best.pt              best full training checkpoint
      weights/last.pt              epoch-boundary resume checkpoint
      weights/embedding.pt         inference encoder and transform metadata
  smoke/                           isolated bounded verification outputs
```

Rerunning a notebook resumes interrupted runs from the last complete epoch and reuses completed matching runs. A changed configuration or dataset is rejected instead of silently overwriting results; use a new `artifacts_root` for a different experiment. An interrupted partial epoch is repeated. Load only checkpoints created by this pipeline, since resumable checkpoints include Python/NumPy RNG state.

`RUN_FINAL_TEST=False` keeps the held-out test unused during model selection. Enable it only after fixing the protocol and selecting margins on dev. ArcFace's training-class head is never evaluated against unseen dev/test identities and is excluded from inference exports. The Triplet result records whether the exported checkpoint actually received semi-hard updates, even if a later point in the same run completed the curriculum.

To verify the entire notebook flow with a tiny real-image subset:

```powershell
conda run -n mtrlearn python scripts/smoke_recognition_notebooks.py
```

The smoke run changes the training budget and curriculum gate, saves executed copies under `artifacts/smoke/notebooks/`, and leaves the source notebooks output-free. Its weights and metrics are software-test artifacts, not final recognition models.
