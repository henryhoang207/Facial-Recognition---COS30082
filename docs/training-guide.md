# Train models with the shared trainer

Use this guide to install the training environment, define a YAML configuration, run a training task, and resume it. All commands run from the repository root.

The shared API is in `src/training/`. Recognition A and C use it through their own command-line entry point. Other task owners supply a `TrainingTask` and data loaders. The existing liveness command keeps its separate training loop.

## Set up the environment

Use Python 3.12 through 3.14 and `uv`. Windows with Python 3.14 and an RTX 3060 has been tested. The lockfile includes Linux dependency markers; the Linux runtime has not been tested here.

For a new Windows environment, run:

```powershell
uv venv .venv --python 3.14
uv pip sync requirements/training-cu130.lock.txt --python .venv/Scripts/python.exe --torch-backend cu130
. ./.venv/Scripts/Activate.ps1
```

For a new Linux environment, run:

```bash
uv venv .venv --python 3.12
uv pip sync requirements/training-cu130.lock.txt --python .venv/bin/python --torch-backend cu130
source .venv/bin/activate
```

If `.venv` already exists, skip its creation and sync the dependencies. Activation makes `python` refer to the environment. If activation is unavailable, use `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux instead of `python` below.

Check the environment:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The locked PyTorch build is `2.10.0+cu130`, torchvision is `0.25.0+cu130`, and the bundled CUDA runtime is `13.0`. A CPU run works with `--device cpu`. A CUDA run requires `torch.cuda.is_available()` to be true. An explicit unavailable CUDA device raises an error.

The training environment uses `requirements/training-cu130.lock.txt`. The original `requirements.txt` remains available for existing workflows. To regenerate the training lock after an intentional dependency change, run:

```bash
uv pip compile requirements/training.in --universal --python-version 3.12 --torch-backend cu130 --emit-index-url --emit-find-links --output-file requirements/training-cu130.lock.txt
```

## Check the shared trainer with the example

Run the small synthetic regression example:

```bash
python -m examples.shared_training --config configs/training/example.yaml
```

The command prints progress and a JSON object containing `run_dir`, `best_checkpoint`, and `last_checkpoint`. Its output lives in a unique directory under `runs/examples/`. This example checks the trainer setup; its loss is not a face-recognition result.

Override settings for one run:

```bash
python -m examples.shared_training --config configs/training/example.yaml --epochs 3 --lr 0.02 --device cpu --set training.accumulation_steps=2 --set training.warmup_epochs=1
```

The complete runnable source is [examples/shared_training.py](../examples/shared_training.py). Its configuration is [configs/training/example.yaml](../configs/training/example.yaml).

## Set up YAML for a real task

Start from the configuration for that task:

| Task | Configuration | Entry point |
| --- | --- | --- |
| Shared synthetic example | `configs/training/example.yaml` | `python -m examples.shared_training` |
| Recognition A, cross-entropy | `configs/recognition/model_a.yaml` | `python -m src.recognition.classification.train` |
| Recognition C, ArcFace | `configs/recognition/model_c.yaml` | `python -m src.recognition.classification.train` |

Recognition configurations have `model`, `preprocess`, `data`, `training`, `verification`, `output_dir`, and `evaluator` sections. The shared trainer consumes `TrainingConfig`; the task's entry point owns the other sections. A configuration for one entry point is not automatically valid for another.

For a new task, keep all shared fields from the example's `training` section. Replace its synthetic dataset settings and `RegressionTask` with that task's data loader and objective. Use `parameter_groups: []` until you define module groups for the new task.

For recognition, check these YAML sections:

```yaml
data:
  root: ../../data/VGGFace2
  train_dir: train_data
  val_dir: val_data
  test_dir: test_data
  manifest: ../../data/manifests/vggface2_classification.csv
verification:
  root: ../../data/VGGFace2
  pairs: ../../data/manifests/verification_pairs_val.csv
  pairs_format: csv
  target_fpr: 0.01
  selection_metric: cosine
```

Keep the complete model and training sections from the supplied A or C YAML. With a manifest configured, `train_dir`, `val_dir`, and `test_dir` are unused folder-adapter settings.

Relative data and output paths are resolved from the YAML file's directory, including path values supplied through recognition CLI overrides. `--config` and `--resume` are resolved from the command's working directory. Use absolute paths when you move data to another machine or Drive mount. Moving a YAML file changes the base of its relative paths.

## Choose training settings

The shared settings are validated before training. Unknown YAML keys, duplicate keys, nonfinite values, and incompatible combinations raise errors.

| Purpose | YAML fields |
| --- | --- |
| Run length and loading | `epochs`, `batch_size`, `num_workers` |
| Optimizer | `optimizer`, `lr`, `weight_decay` |
| AdamW | `adam_betas`, `adam_eps`, `adam_amsgrad` |
| SGD | `sgd_momentum`, `sgd_dampening`, `sgd_nesterov` |
| Learning-rate schedule | `scheduler`, `eta_min`, `cosine_t_max` |
| Warmup | `warmup_epochs`, `warmup_start_factor` |
| Gradient updates | `accumulation_steps`, `clip_grad_norm` |
| Mixed precision | `amp`, `amp_init_scale` |
| Checkpoint selection and stopping | `monitor`, `mode`, `patience`, `min_delta`, `early_stopping_start_epoch` |
| Reproducibility and device | `seed`, `deterministic`, `device` |
| Output display | `progress` |
| Separate optimizer groups | `parameter_groups` |

Use `optimizer: adamw` or `optimizer: sgd`. Nesterov requires positive momentum and zero dampening. Optimizer-specific fields apply to the selected optimizer.

Use `scheduler: cosine` or `scheduler: none`. Linear warmup can precede either choice. By default, cosine spans the epochs remaining after warmup. An explicit `cosine_t_max` must cover those epochs. `eta_min` is an absolute floor for each optimizer group.

Use `mode: max` for `val_auc` or `val_accuracy`, and `mode: min` for `val_loss`. `patience: null` disables early stopping. `early_stopping_start_epoch: null` waits until warmup completes before counting epochs without improvement.

Best-checkpoint selection remains active during warmup.

Accumulation weights gradients by the actual number of samples represented by each microbatch. It also flushes an incomplete final window. Batch size 64 with accumulation 4 represents up to 256 images per optimizer update. BatchNorm statistics still come from individual microbatches.

## Set separate learning rates

Recognition defines two groups:

```yaml
parameter_groups:
  - name: backbone
    modules: [model.encoder.backbone]
    lr: null
    lr_scale: 1.0
    weight_decay: null
  - name: head
    modules: [model.encoder.projection, model.head]
    lr: null
    lr_scale: 1.0
    weight_decay: null
```

These paths are relative to the `TrainingTask`, including its `model` attribute. Other tasks use their own module paths. Groups must have unique names and disjoint trainable parameters. Unassigned trainable parameters enter the `default` group.

When `lr` is null, a group uses `training.lr * lr_scale`. When `weight_decay` is null, it inherits the shared value. An explicit LR cannot be combined with a scale other than 1.

Override recognition group rates with:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --group-lr backbone=0.0001 --group-lr head=0.001
```

## Prepare and validate dataset manifests

Create manifests on first use or after source data changes:

```bash
python -m src.datasets.prepare --config configs/datasets.yaml
```

Validate existing manifests before training:

```bash
python -m src.datasets.validate --config configs/datasets.yaml
```

Dataset preparation writes under `data/manifests/`. It reads existing source databases and images. Team photos in `data/faces/` are gallery data and are excluded from training. See [dataset preparation](dataset-preparation.md) for split rules and gallery review.

## Run recognition training

Run the supplied baseline:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml
```

Run Model C:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml
```

Override hyperparameters through arguments:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --epochs 30 --batch-size 32 --lr 0.0003 --device cuda --warmup-epochs 3 --accumulation-steps 2 --eta-min 0.000001 --set training.clip_grad_norm=5.0
```

Select SGD:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_c.yaml --optimizer sgd --set training.sgd_momentum=0.9 --set training.sgd_nesterov=true
```

Pass list values as quoted YAML:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --set 'training.adam_betas=[0.9,0.99]'
```

Recognition override precedence is YAML, then `--set`, then named flags, then `--group-lr`. Only explicitly supplied arguments override the YAML. They do not modify the file.

The supplied recognition configurations select checkpoints using verification `val_auc`. To run classification without verification, choose that policy explicitly:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --set verification=null --monitor val_loss --mode min
```

Random initialization is the baseline. Use ImageNet initialization only under the group's agreed course requirements. See [recognition training](recognition-training.md) for model and evaluation details.

## Inspect results and resume

Each recognition invocation creates a unique directory under `runs/recognition/`:

| File | Contents |
| --- | --- |
| `best.pt` | Selected inference weights, metadata, and validation metrics |
| `last.pt` | Full state for epoch-boundary resume, including the best-weight snapshot |
| `recognition_<a or c>_<date>.pt` | Export of the selected inference checkpoint |
| `history.csv` | Loss, metrics, actual optimizer updates, and learning rates per group |
| `training.png` | Recognition learning curves |
| `config.yaml`, `resolved_config.json` | Recorded configuration and task metadata |
| `environment.json`, `summary.json` | Runtime details and run summary |

The synthetic example writes the shared trainer files but does not export recognition-specific checkpoints or plots.

Resume recognition from an absolute checkpoint path:

```bash
python -m src.recognition.classification.train --config configs/recognition/model_a.yaml --resume /absolute/path/to/last.pt
```

Resume the example with its matching configuration and arguments:

```bash
python -m examples.shared_training --config configs/training/example.yaml --resume /absolute/path/to/last.pt
```

These resume commands apply to interrupted runs. A normally completed example has already reached its configured epoch limit.

Use `last.pt` for resume. Each resumed run gets a new output directory. Model structure, metadata, optimizer groups, schedule, accumulation, and other training hyperparameters must match. Device, progress display, and worker count may change. Epochs can increase only when no scheduler exists, including no warmup scheduler. A completed or early-stopped checkpoint cannot continue under the same epoch limit and stopping policy.

Resume occurs after completed epochs. The trainer flushes accumulated gradients before it saves the checkpoint. Custom stateful samplers, distributed training, and recovery partway through a batch are outside the current contract.

## Integrate another task

Implement one `TrainingTask` that owns the model and every trainable loss component. Its `step(batch)` returns:

- A scalar mean loss that retains its graph during training.
- `weight`, the number of samples or other units represented by that mean.
- Optional `MetricTotal(total, count)` values for metrics that can be aggregated.

The trainer owns backward, AMP, clipping, accumulation, optimizer updates, scheduling, stopping, and checkpoint state. A task must not perform those operations itself.

Call `seed_everything` before creating the model and loaders. Move the task to its device before creating the optimizer. Build the optimizer and scheduler with the shared factories, then call `Trainer.fit`. Use explicit loader generators, `seed_worker`, and nonpersistent workers for reproducible epoch-boundary resume. The [runnable example](../examples/shared_training.py) shows the complete sequence.

Supply an evaluator callback for metrics such as AUC that require the whole validation set. It receives `(task, device)` and returns named finite metrics. Do not average AUC across batches. Task packages own their dataset formats, sampling, inference interfaces, and evaluation protocol.

## Run checks

```bash
python scripts/check_training.py
python scripts/check_training.py --include-liveness
python scripts/check_docs.py
```

The training checks cover formatting, Ruff, CPU behavior, and CUDA behavior when a GPU is available. Documentation checks validate local Markdown links and scan project prose and comments for untranslated Vietnamese while allowing contributor names.

## Resolve common errors

| Error | Action |
| --- | --- |
| Missing manifest or pair file | Run dataset preparation and confirm paths relative to the YAML directory. |
| `monitor metric unavailable` | Configure the evaluator that produces the metric, or explicitly select an available validation metric. |
| Invalid Nesterov combination | Set positive `sgd_momentum` and zero `sgd_dampening`. |
| Unknown or overlapping optimizer group | Check task-relative module paths and keep groups disjoint. |
| `eta_min` exceeds a group LR | Lower the floor or correct the group LR. |
| Warmup is as long as the run | Keep `warmup_epochs < epochs`. |
| CUDA out of memory | Reduce `batch_size`; use accumulation if a larger effective batch is needed. |
| AMP skips updates | Inspect `train_optimizer_steps`; check loss/gradients and the configured initial scale. |
| Resume configuration mismatch | Restore the original configuration and overrides recorded with the run. |
| Test evaluation lacks calibration | Supply the validation report as the calibration source. |
