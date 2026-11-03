# Train Model C with margin warmup and collapse controls

Use [model_c_recovery.yaml](../configs/recognition/model_c_recovery.yaml) for Model C experiments. It keeps the selected bbox and verification protocols, CUDA BF16, batch size 64, and a maximum of 100 epochs. The original [model_c.yaml](../configs/recognition/model_c.yaml) retains fixed margin and the original analytical objective as a control. Model A's configuration is unchanged.

Both Model C configurations use `training.num_workers: 4`. Streaming loaders use those worker processes. Fully cached CUDA loaders use zero workers because no image reading or decoding remains. Use `--num-workers 6` to override the configured count for a run.

The controls below are implemented and tested as software behavior. Their effect on recognition quality requires the recovery experiments. See [the investigation](model-c-collapse-investigation.md) for the evidence behind them.

## Configure margin warmup

The recovery YAML contains:

```yaml
model:
  arcface_margin: 0.3
  arcface_margin_warmup:
    enabled: true
    start_margin: 0.0
    hold_epochs: 3
    ramp_epochs: 5
```

The schedule uses the absolute one-based epoch `e`:

```text
fraction = clamp((e - hold_epochs) / ramp_epochs, 0, 1)
margin = start_margin + fraction * (arcface_margin - start_margin)
```

| Epoch | Margin |
| --- | ---: |
| 1, 2, 3 | 0.00 |
| 4 | 0.06 |
| 5 | 0.12 |
| 6 | 0.18 |
| 7 | 0.24 |
| 8 and later | 0.30 |

The first ramp epoch advances by one ramp increment. With `hold_epochs: 0`, epoch 1 therefore uses the first increment rather than the start value. `ramp_epochs` must be positive, and `start_margin` must not exceed the target. A fixed zero margin is also supported by `arcface_margin: 0.0` with warmup disabled.

`training.warmup_epochs` schedules learning rates independently. Margin warmup changes neither the optimizer LR nor augmentation. Train and classification validation use the same margin within an epoch. Verification inference uses normalized embeddings and has no target margin.

`history.csv`, `best.pt` epoch metrics, and `last.pt` history record `arcface_margin`. Resume recomputes it from the saved epoch plus one and the unchanged model configuration. Standalone checkpoint loading restores the margin used at the checkpoint's epoch. Changing the schedule, gradient policy, or encoder design requires a new run.

When `early_stopping_start_epoch` is null, the patience counter waits until both task margin warmup and the shared LR warmup are complete. For the schedule above, task eligibility starts at epoch 9. An explicit `early_stopping_start_epoch` overrides this default readiness check. TAR/AUC best checkpoints can still be saved during warmup.

The recovery YAML breaks ties using `val_no_margin_loss` instead of margin-dependent `val_loss`. Margin loss has a changing objective during the ramp and is not directly comparable across its stages. Checkpoint priority remains TAR, then AUC, then the tie-breaker loss.

Run the recovery configuration:

```powershell
.venv/Scripts/python.exe -m src.recognition.classification.train --config configs/recognition/model_c_recovery.yaml
```

Override the schedule without changing the file:

```powershell
.venv/Scripts/python.exe -m src.recognition.classification.train --config configs/recognition/model_c_recovery.yaml --set model.arcface_margin_warmup.hold_epochs=2 --set model.arcface_margin_warmup.ramp_epochs=6
```

## Select collapse controls

All controls are explicit YAML fields and can be overridden with `--set`:

| Field | Default | Effect |
| --- | --- | --- |
| `model.arcface_gradient` | `analytic` | `straight_through` preserves the margin forward logits and backpropagates through the cosine logits instead of the angular derivative. This is an intentional surrogate gradient. |
| `model.prototype_regularization_weight` | `0.0` | Adds the configured weight times the L2 norm of the mean normalized class weight. |
| `model.rectify_wrong_class_logits` | `false` | Replaces negative wrong-class cosine logits with zero for the training loss. Target-margin logits and no-margin classification predictions retain their usual definitions. |
| `model.embedding_normalization` | `none` | `batchnorm` adds BatchNorm1d after the projection. |
| `model.embedding_bn_affine` | `true` | Controls learned scale and shift when embedding BatchNorm is enabled. |
| `model.projection_bias` | `true` | Controls the projection's bias independently of normalization. |

The prototype term is computed in FP32. `train_loss` and `val_loss` include it. `*_classification_loss` records margin cross-entropy without this penalty. `*_no_margin_loss` records cross-entropy over the unmodified cosine predictions.

The straight-through policy uses the original local forward fallback for extreme angles. With wrong-class rectification enabled, its backward retains the ReLU effect on wrong-class logits. At inference, no-margin predictions remain ordinary scaled cosine logits in all modes.

The recovery YAML includes `model.encoder.embedding_norm` in the head optimizer group. The original fixed-margin configuration leaves it unassigned, so newly introduced affine parameters enter the shared optimizer's `default` group when enabling BN there through CLI. Add the module to the intended group if using different group learning rates.

Embedding BN uses training batches to update running statistics. Checkpoint inference and verification use `eval()` and saved running statistics. Avoid singleton training batches when BN is enabled. Projection bias is a separate variable and remains enabled for the initial BN experiment.

For example, combine the recovery schedule with straight-through and BN:

```powershell
.venv/Scripts/python.exe -m src.recognition.classification.train --config configs/recognition/model_c_recovery.yaml --set model.arcface_gradient=straight_through --set model.embedding_normalization=batchnorm
```

This command is a candidate experiment, not evidence that the combination resolves collapse.

## Compare experiments before a full run

Run Model C only. Keep the original images, manifests, gallery photos, and old checkpoint directories intact. Use fresh initialization and optimizer state for every arm.

Use one fixed 10-epoch pilot budget, seed, deterministic setting, LR schedule, augmentation, precision, class mapping, and validation protocol across these arms:

| Arm | Gradient | Prototype penalty | Wrong-class rectification | Embedding BN | Margin |
| --- | --- | ---: | --- | --- | --- |
| Control | analytical | 0 | off | off | fixed 0.3 |
| Step 1 | straight-through | 0 | off | off | fixed 0.3 |
| Step 2a | analytical | 1.0 | off | off | fixed 0.3 |
| Step 2b | analytical | 0 | on | off | fixed 0.3 |
| Step 3a | analytical | 0 | off | on | fixed 0.3 |
| Step 3b | analytical | 0 | off | off | warmup 0 to 0.3 |
| Combined candidate | best supported policy | selected separately | selected separately | on | warmup 0 to 0.3 |

The penalty value 1.0 is a starting hypothesis, not a tuned optimum. Do not stack both step-2 penalties without a separate reason. The combined candidate builds on pilot evidence. Additional seeds are needed before claiming a robust improvement.

Use `--epochs 10 --set training.patience=null` for equal pilot budgets. Disable margin warmup explicitly for the fixed-margin arms. The original 100-epoch histories are historical context, not exact controls for a shorter cosine schedule. Log the full effective configuration for every arm.

Monitor these metrics at each epoch:

- `val_tar`, `val_auc`, and positive/negative cosine means and standard deviations.
- `val_embedding_mean_norm`, the norm of the mean of the unique verification-image unit embeddings. Values near 1 indicate strong directional concentration in that image pool. This pool is not balanced by identity.
- `val_prototype_mean_norm`, the norm of the mean of normalized class weights. Values near 1 indicate class-center concentration.
- `val_target_cosine`, `val_fallback_fraction`, and the actual `arcface_margin`.
- Classification accuracy, classification loss, no-margin loss, and total loss.

Training prototype metrics are averages over minibatch-time model states. Validation prototype metrics use the fixed post-training weights. A smaller concentration metric alone is insufficient: the model also needs genuine/impostor separation and useful TAR/AUC.

After the pilots, run the strongest candidate from fresh initialization for at most 100 epochs with patience 5 if its geometry and validation results justify it. Extending a 10-epoch scheduled checkpoint to 100 epochs is intentionally rejected by the shared resume rules. Do not bypass that check. Keep the held-out test set out of experiment selection.

Each invocation creates a unique directory under the configured output root. Back up reports, configs, and checkpoints through the group's existing workflow. The training command does not upload files to Google Drive.
