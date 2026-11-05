# Shared training design

The original liveness implementation owns its loop, ACER metrics, and checkpoints in `src/liveness/train.py`. At the start of this work, recognition was an empty package. The shared trainer was added as a separate package so other owners can adopt it without a forced migration. The existing BGR crop and `Embedder` interface remain the integration contracts.

## Usage defines the interface

```python
task = RecognitionTask(model, preprocess_config).to(device)
optimizer = make_optimizer(task, config)
trainer = Trainer(task, optimizer, config, new_run_dir(root, "recognition_a"))
result = trainer.fit(train_loader, val_loader)
```

A different task can accept a dictionary containing anchor, positive, and negative inputs and return triplet loss. The trainer does not distinguish classification from triplet batches. Inference uses `RecognitionEmbedder(checkpoint).embed(face)`, and verification depends only on `embed(face)`.

See the [shared training guide](training-guide.md) for a complete runnable example and configuration instructions.

## Recognition package boundaries

The ResNet-18 A/C implementation in `src/recognition/classification/` and the iResNet-34 B0/B1 notebook implementation in `src/recognition/identification/` use the same `Trainer.fit` engine and `TrainingTask` interface. B0/B1 retain `ExperimentConfig`, adapted through `trainer_config(seed)`, along with their P-by-K sampling, curriculum, margin sweeps, and identity-disjoint evaluation. `MetricLearningTask` owns the unchanged encoder, optional ArcFace head, objective, and reproducible mining state.

Both packages contain configuration, data, and model modules with distinct responsibilities. The shared `embed(face)` pair API remains in `verification/evaluator.py`, with its CLI in `verification/run.py`. The notebook-specific batched metrics are in `identification/embedding_evaluation.py`; `MetricEmbedder` adapts B0/B1 inference checkpoints for the common pair CLI. Data splits and evaluation thresholds remain specific to each protocol. See the [recognition package map](../src/recognition/README.md).

Both use the common version-1 checkpoint envelope. Training checkpoints contain task, optimizer, scaler, training configuration, metadata, RNG, and loader state; inference checkpoints contain the selected task snapshot and validation metrics. Optional policy state is saved alongside that envelope. Model-specific metadata and state-dict keys still differ, so the appropriate embedder loads each architecture. Legacy private-loop B0/B1 artifacts are retained but rejected for automatic resume; the notebook default is now `artifacts/main-cu130-shared-trainer/`.

## Designs considered

One design subclassed `Trainer` and required a `compute_step` hook. The selected design composes `Trainer` with `TrainingTask(nn.Module)` and `step(batch) -> StepResult`.

Both approaches respected task ownership. An independent review selected composition because it puts all trainable state, including ArcFace weights and trainable loss components, inside one module. Contributors implement the objective without inheriting methods that could override AMP or checkpoint handling. The public interface consists of the task step, `fit`, an optional evaluator callback, and optional `TrainingPolicy` hooks for experiment rules.

The selected design also provides explicit encoder access for recognition evaluation and inference. `RecognitionTask` has `model.encoder`; generic tasks do not need an encoder or classification logits. A universal `Trainer(model, criterion)` interface was rejected because triplet mining and multi-output liveness models do not share one forward/loss signature.

## State ownership

- The task owns the model and all trainable loss components.
- The trainer owns optimization, AMP, scheduling, aggregation, stopping, and checkpoints.
- An optional policy defines experiment-specific learning-rate adjustments, selection, stopping, and epoch annotations; its state is checkpointed by the trainer.
- Each invocation owns a unique directory. Checkpoint replacement is atomic within that directory.
- Dataset adapters own their input and batch formats. Recognition uses `src.common.face_crop.crop_face`.
- The pair evaluator owns the verification and threshold protocol, using embeddings rather than classifier identities.

Configuration is validated at each entry point: Pydantic YAML/CLI settings for A/C and a retained dataclass plus shared-config adapter for B0/B1. Loss and metrics are checked before a checkpoint advances. Resume restores completed-epoch state and rejects incompatible configuration or metadata. Test evaluation is separate from validation-based model selection.

`StepResult.optimize=False` records a valid no-signal batch without backward or an optimizer update. B0 uses this when no selected triplet contributes a positive hinge. This preserves the mining protocol and avoids weight-decay-only updates. AMP overflow skips and actual successful updates remain trainer responsibilities.

## Optimizer groups and accumulation

Optimizer groups specify task-relative module names, an absolute LR or scale, and an optional weight decay. Groups must have disjoint parameters. Remaining trainable parameters enter `default`. The trainer contains no recognition architecture names. A and C define backbone and head groups in their YAML, while other tasks define their own paths.

Gradient accumulation combines microbatch gradients according to their weights. It normalizes the result at the update boundary, then unscales, clips, and updates. The final incomplete window is flushed before validation and checkpointing. Linear warmup and cosine use PyTorch schedulers, with both scheduler and optimizer-group states saved for resume.

Default early stopping waits for actual warmup progress. If AMP skips every optimizer update in an epoch, the scheduler does not advance. Patience does not start merely because that physical epoch has passed.

## Verification history

`scripts/check_training.py` checks formatting, lint, training behavior, resume, weighted metrics, AMP, A/C inference, and threshold calibration. Existing liveness tests have also been run without migrating its implementation.

The following regressions were reproduced before their fixes and passed afterward:

- `test_scheduler_does_not_advance_when_amp_skips_all_updates`: the scheduler had advanced from epoch 0 to 1 even though AMP skipped the optimizer update. It now remains at 0.
- `test_resume_recovers_best_weights_from_last_checkpoint_alone`: resume previously required an adjacent `best.pt`. The full checkpoint now contains the best snapshot and can recreate the inference artifact.
- `test_early_stopping_waits_when_overflow_prevents_warmup_progress`: a five-epoch run had stopped after epoch 3 while AMP prevented warmup from progressing. Patience now waits for warmup, and that test completes all five epochs with scheduler epoch 0.

On October 31, 2026, 19 new tests passed on CPU/CUDA, together with six existing liveness tests. Black and Ruff passed. The environment then had 53 compatible packages, and a lockfile dry run required no changes.

On November 1, 2026, dataset support added DuckDB and pinned OpenCV to the compatible 4.13 release. Dataset validation and sample batches were checked against the local source data. See [dataset preparation](dataset-preparation.md) for results and the limits of those checks.

After the hyperparameter update on November 1, 2026, 47 training, recognition, and dataset tests passed on CPU/CUDA. The added checks assert specific AdamW/SGD updates, uneven accumulation, clipping after averaging, per-group LR, the cosine floor, CLI overrides, and exact resumed weights. Black and Ruff passed. The remaining pin-memory deprecation warnings originate inside PyTorch 2.10.

These checks establish implementation behavior. Full real-data training, recognition quality, and demo-laptop latency still require their own measurements.
