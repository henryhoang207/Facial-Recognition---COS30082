# Recognition package map

The package groups code by training and evaluation protocol. The similarly named files in its two subpackages serve different model families and are not duplicate implementations of one interface.

| Package | Models and workflow |
| --- | --- |
| `classification/` | ResNet-18 A/C, classification training through `src/training/`, YAML CLI configuration, and the app embedder |
| `verification/` | iResNet-34 B0/B1 notebook sweeps through the shared trainer, normalized 256-dimensional embeddings, identity-disjoint evaluation, and the shared pair-evaluation API |

## Classification modules

| Module | Responsibility |
| --- | --- |
| `classification/config.py` | A/C YAML configuration and validation |
| `classification/data.py` | Classification manifests and folder adapters |
| `classification/models.py` | ResNet-18 encoder, projection, and A/C heads |
| `classification/task.py` | Shared-trainer task adapter |
| `classification/train.py` | A/C training CLI |
| `classification/evaluate.py` | Held-out classification evaluation CLI |
| `classification/predictor.py` | Exported-checkpoint `RecognitionEmbedder` |
| `classification/verification.py` | Verification callback used during A/C training |

## Verification and metric-learning modules

| Module | Responsibility |
| --- | --- |
| `verification/config.py` | B0/B1 `ExperimentConfig`, its shared `TrainingConfig` adapter, and pair-evaluation `VerificationConfig` |
| `verification/data.py` | Identity-disjoint manifests, duplicate checks, image processing, and P-by-K sampling |
| `verification/models.py` | Randomly initialized iResNet-34 encoder |
| `verification/losses.py` | Squared-distance Triplet loss, mining, curriculum gate, and B1 ArcFace head |
| `verification/task.py` | `MetricLearningTask` with encoder, optional ArcFace head, loss, mining RNG, and curriculum state |
| `verification/training.py` | Shared-trainer orchestration, experiment policy, sweeps, resume, selection, and artifact export |
| `verification/predictor.py` | Common-checkpoint `MetricEmbedder` and encoder loading |
| `verification/notebook_utils.py` | Notebook previews, disposable GPU pilot, and smoke configuration |
| `verification/embedding_evaluation.py` | Batched embeddings, verification thresholds, open-set identification, bootstrap intervals, and plots |
| `verification/evaluator.py` | Shared `PairEvaluator` for objects exposing `embed(face)` |
| `verification/run.py` | Shared pair-comparison CLI for exported models |

For example, notebook imports use:

```python
from src.recognition.verification.config import ExperimentConfig
from src.recognition.verification.data import prepare_manifests
from src.recognition.verification.training import run_sweep
```

`classification/evaluate.py`, `verification/evaluator.py`, and `verification/embedding_evaluation.py` have separate contracts: classification scores over known classes, pair scoring through an embedder, and batched notebook evaluation on unseen identities. They remain separate modules within their owning packages.

Both packages use `src.training.Trainer` and `TrainingTask`. B0/B1 retain their dataclass configuration and adapt engine settings through `ExperimentConfig.trainer_config(seed)`. An optional shared `TrainingPolicy` carries experiment scheduling, selection, and stopping rules while the trainer performs backward, AMP, gradient clipping, optimizer updates, and checkpoint persistence.

A/C and B0/B1 use the same checkpoint envelope: `format_version=1`, `checkpoint_kind`, `task`, `training_config`, and `metadata`. `last.pt` is resumable; `best.pt` is selected inference state. B0/B1 also export `embedding.pt` in this inference schema with only encoder keys in `task`. Different architectures still require their own loader: `RecognitionEmbedder` for A/C and `MetricEmbedder` for B0/B1. The shared schema does not make ResNet-18 and iResNet-34 weights interchangeable.

Use `src.recognition.verification.predictor:MetricEmbedder` as the model factory for B0/B1 in `verification/run.py`. It accepts a shared BGR uint8 112-by-112 crop and returns a normalized 256-dimensional vector. Existing private-loop metric checkpoints are not resumed by the new engine; use the fresh notebook artifact root documented in the [notebook guide](../../notebooks/metrics-learn/RECOGNITION.md).

See the [recognition training guide](../../docs/recognition-training.md) for A/C and the shared pair CLI, or the [notebook guide](../../notebooks/metrics-learn/RECOGNITION.md) for B0/B1.
