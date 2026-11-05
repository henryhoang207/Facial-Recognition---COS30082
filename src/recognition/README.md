# Recognition package map

The package keeps the existing classification, verification, and preprocessing workflows separate. Metric-learning experiments live in the additive identification package. Similarly named modules belong to different model families and protocols.

| Package | Models and workflow |
| --- | --- |
| `classification/` | ResNet-18 A/C, classification training through `src/training/`, YAML CLI configuration, and the app embedder |
| `identification/` | iResNet-34 B0/B1 notebook sweeps through the shared trainer, normalized 256-dimensional embeddings, and identity-disjoint evaluation |
| `verification/` | Shared pair-evaluation API and exported-model comparison CLI |
| `preprocessing/` | Offline, resumable face detection and recognition manifest preparation |

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

## Identification and metric-learning modules

| Module | Responsibility |
| --- | --- |
| `identification/config.py` | B0/B1 `ExperimentConfig` and its shared `TrainingConfig` adapter |
| `identification/data.py` | Identity-disjoint manifests, duplicate checks, image processing, and P-by-K sampling |
| `identification/models.py` | Randomly initialized iResNet-34 encoder |
| `identification/losses.py` | Squared-distance Triplet loss, mining, curriculum gate, and B1 ArcFace head |
| `identification/task.py` | `MetricLearningTask` with encoder, optional ArcFace head, loss, mining RNG, and curriculum state |
| `identification/training.py` | Shared-trainer orchestration, experiment policy, sweeps, resume, selection, and artifact export |
| `identification/predictor.py` | Common-checkpoint `MetricEmbedder` and encoder loading |
| `identification/notebook_utils.py` | Notebook previews, disposable GPU pilot, and smoke configuration |
| `identification/embedding_evaluation.py` | Batched embeddings, verification thresholds, open-set identification, bootstrap intervals, and plots |

For example, notebook imports use:

```python
from src.recognition.identification.config import ExperimentConfig
from src.recognition.identification.data import prepare_manifests
from src.recognition.identification.training import run_sweep
```

`classification/evaluate.py`, `verification/evaluator.py`, and `identification/embedding_evaluation.py` have separate contracts: classification scores over known classes, pair scoring through an embedder, and batched notebook evaluation on unseen identities. They remain separate modules within their owning packages.

Classification and identification use `src.training.Trainer` and `TrainingTask`. B0/B1 retain their dataclass configuration and adapt engine settings through `ExperimentConfig.trainer_config(seed)`. An optional shared `TrainingPolicy` carries experiment scheduling, selection, and stopping rules while the trainer performs backward, AMP, gradient clipping, optimizer updates, and checkpoint persistence.

A/C and B0/B1 use the same checkpoint envelope: `format_version=1`, `checkpoint_kind`, `task`, `training_config`, and `metadata`. `last.pt` is resumable; `best.pt` is selected inference state. B0/B1 also export `embedding.pt` in this inference schema with only encoder keys in `task`. Different architectures still require their own loader: `RecognitionEmbedder` for A/C and `MetricEmbedder` for B0/B1. The shared schema does not make ResNet-18 and iResNet-34 weights interchangeable.

Use `src.recognition.identification.predictor:MetricEmbedder` as the model factory for B0/B1 in `verification/run.py`. It accepts a shared BGR uint8 112-by-112 crop and returns a normalized 256-dimensional vector. Existing private-loop metric checkpoints are not resumed by the new engine; use the fresh notebook artifact root documented in the [notebook guide](../../notebooks/metrics-learn/RECOGNITION.md).

## Shared verification and preprocessing modules

| Module | Responsibility |
| --- | --- |
| `verification/evaluator.py` | `VerificationConfig`, shared `PairEvaluator`, and pair score metrics for objects exposing `embed(face)` |
| `verification/run.py` | Shared pair-comparison CLI for exported models |
| `preprocessing/config.py` | Offline face-preparation settings |
| `preprocessing/detector.py` | Detector adapter for face-bbox proposals |
| `preprocessing/prepare.py` | Resumable preparation and export of recognition manifests |

Classification consumes the existing verification API. Identification supplies `MetricEmbedder` to that API through its model factory; it does not duplicate the evaluator or CLI. The offline bbox protocol remains separate from the notebooks' identity-disjoint experiment manifests.

See the [recognition training guide](../../docs/recognition-training.md) for A/C and the shared pair CLI, or the [notebook guide](../../notebooks/metrics-learn/RECOGNITION.md) for B0/B1.
