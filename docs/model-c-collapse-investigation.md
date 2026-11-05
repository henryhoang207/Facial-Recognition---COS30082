# Model C polar collapse investigation

Model C needs a change to its optimization or embedding design before another full training run. Its saved representations have two concentrated, nearly opposite directions: one for features and one for class weights. This is consistent with polar collapse. It is already present at epoch 2 and persists at epoch 100.

The initial investigation added read-only measurements and proposed controlled interventions. Its diagnosis did not change training or checkpoint weights. The controls have since been implemented as opt-in settings in the [recovery training guide](model-c-recovery-training.md). Their training effectiveness still requires the controlled experiments described below.

## Evidence and scope

The starting evidence is [diagnosis.md](../runs/diagnose_model_c_20261103/diagnosis.md), its [geometry measurements](../runs/diagnose_model_c_20261103/geometry.json), and the [three-round comparison](../runs/comparison_three_rounds_20261103/comparison.md). The follow-up independently decodes the same 960 classification-validation images, two per each of 480 identities, with sampling seed 123. It uses the crop and normalization recorded in each checkpoint.

These images belong to training identities. They diagnose geometry, not generalization to unseen identities. They are not the full validation set or the held-out test set. Pair counts below contain both directions of each pair and are correlated observations, not independent statistical trials.

| Follow-up measurement | C best, epoch 2 | C last, epoch 100 |
| --- | ---: | ---: |
| Different-identity cosine, FP32 inference | 0.978648 | 0.976460 |
| Different-identity cosine, BF16 inference | 0.978641 | 0.976455 |
| Norm of mean unit feature | 0.989286 | 0.988183 |
| Norm of mean unit class weight | 0.988375 | 0.989061 |
| Cosine between those two mean directions | -0.999822 | -0.999986 |
| Fraction of targets using the fallback branch | 73.02% | 39.69% |
| Fraction of targets using the angular branch | 26.98% | 60.31% |
| Mean target derivative within the angular branch | 0.039005 | 0.001879 |
| Different-identity cosine after subtracting projection bias | 0.978618 | 0.976055 |

The normalized class-weight mean would have norm 1 if every class weight pointed in exactly the same direction. Values near 0.989 confirm that this affects the classifier centers as well as the features. A healthy encoder does not require every different-person cosine to be zero, but it must retain useful separation between genuine and impostor distributions.

Subtracting the learned projection bias at inference barely changes the result. The bias norm is 0.328 at epoch 2 and 0.452 at epoch 100, versus mean raw feature norms of 92.98 and 185.04. Removing that bias from these trained checkpoints is not a repair. This does not establish what training without bias would do.

The same-image FP32-versus-BF16 embedding cosine averages above 0.999998 at both checkpoints. Collapse therefore exists in the saved FP32 representation and is not an inference precision artifact. A separate FP32 training experiment would still be needed to exclude an effect of BF16 on the original optimization trajectory.

## Why the current loss can sustain this geometry

In [ArcFaceHead](../src/recognition/classification/models.py), let `c` be a feature's cosine with its target class weight. Before scaling, the angular branch computes:

```text
phi(c) = c cos(m) - sqrt(1 - c²) sin(m)
dphi/dc = cos(m) + c sin(m) / sqrt(1 - c²)
```

At margin `m = 0.3`, the fallback boundary is:

```text
c_boundary = cos(pi - 0.3) = -0.955336489
```

Approaching this boundary from the angular side makes `dphi/dc` approach zero. The fallback uses `phi(c) = c - m sin(m)` with derivative 1. At the switch, this implementation also has a downward jump of about 0.043993 in the target cosine logit, or 1.319777 after scale 30.

The measured target cosines sit around this boundary. At epoch 100, the 60.31% of samples still in the angular branch have an average target derivative of only 0.001879. The other 39.69% use the fallback and have derivative 1. It would be incorrect to say that every sample has a vanishing gradient. The normalization Jacobian and network gradients also affect the complete update.

An autograd experiment using the actual head makes the common-direction problem concrete. All 480 class weights are deliberately equal, and a single scalar `c` moves the feature cosine against every class together. This is a degenerate diagnostic path, not a ResNet training simulation.

| Shared cosine `c` | Current margin loss | Current `dL/dc` |
| --- | ---: | ---: |
| 0.00 | 15.0373 | +1.3398 |
| -0.50 | 13.1796 | +6.4584 |
| -0.94 | 7.9373 | +25.7571 |

A positive derivative means gradient descent decreases the shared cosine. Along this path, the loss can improve while features move away from all class centers together. The fallback prevents continuing the original angular formula beyond its valid monotonic range, but does not remove the preceding weak-target-attraction region or its discontinuity.

This is a verified local mechanism compatible with the observed checkpoints. It does not prove exactly which early minibatch, initialization, LR, or architecture choice caused the original run to enter that region. The original analytic head test covers a healthy positive-cosine example, not training stability near this boundary.

The geometry matches the polar-collapse configuration studied by [Zhang et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Zhang_Unifying_Margin-Based_Softmax_Losses_in_Face_Recognition_WACV_2023_paper.pdf). Their remedies include penalizing the mean normalized prototype and rectifying negative wrong-class logits. Their experiments are not a guarantee for this implementation, which also has a fallback branch.

## Interventions in priority order

### Isolate the target-gradient policy

Test an explicitly configured straight-through margin adjustment against the current analytical gradient, leaving the forward logits, margin, scale, encoder, initialization, optimizer, and data unchanged:

```python
logits = cosine_logits + (margin_logits - cosine_logits).detach()
```

Here both logit tensors include the same scale. The forward output remains the current margin-adjusted logits, while backward follows the normalized cosine logits. This removes the extra angular derivative factor. The shared-cosine probe keeps the same forward loss but its common-direction derivative becomes approximately zero, within FP32 numerical error.

The [official InsightFace PyTorch loss](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py) applies its margin transformation under `torch.no_grad()`, giving a related straight-through gradient. The current local analytical gradient is therefore not equivalent to that reference's backward behavior. The reference also differs in its forward handling of extreme angles. The proposed expression preserves the local forward policy to isolate one variable.

This is a deliberate surrogate gradient, not the exact derivative of the forward ArcFace loss. Give it a named configuration value, record it in checkpoint metadata, and test forward values and backward behavior separately. It is the first causal experiment I recommend, not an already validated replacement.

### Constrain prototype concentration if analytical gradients are retained

Two targeted alternatives are:

- Add `lambda * norm(mean(normalize(class_weights)))` to classification loss. It penalizes the exact class-center concentration measured here. Treat `lambda` as an experimental hyperparameter and record classification loss separately from this penalty. Ordinary AdamW weight decay does not impose angular spread on normalized centers.
- Apply ReLU to wrong-class cosine logits before scaling, preserving the target margin logit. Negative wrong-class logits then stop rewarding movement toward the opposite pole. This changes the forward objective and needs its own named experiment.

These are the two approaches described in [the polar-collapse study](https://openaccess.thecvf.com/content/WACV2023/papers/Zhang_Unifying_Margin-Based_Softmax_Losses_in_Face_Recognition_WACV_2023_paper.pdf). Start each from a fresh seed. Neither method guarantees recovery when all centers have already become identical. Test them independently rather than adding every penalty at once.

### Test an embedding normalization layer

The local encoder ends with global pooling and a biased linear projection. The [ArcFace paper](https://arxiv.org/html/1801.07698v4#S4.SS1) uses a BN, dropout, FC, BN embedding design. [InsightFace IResNet](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py) includes a final `BatchNorm1d` with fixed unit scale and a learned shift.

Test a configurable `BatchNorm1d(128)` after the projection, with affine behavior explicitly defined. Keep A's encoder configuration unchanged for this C-specific experiment. Test projection bias removal separately if its effect needs to be attributed. L2 normalization already exists in the head and at inference, so adding it again does not remove a shared direction.

Batch normalization is a plausible structural improvement, not proof of a cure. Train its running statistics on training images only. Verification must use `eval()` and fixed running statistics so embeddings do not depend on which gallery images share the inference batch. The optimizer must cover every new trainable parameter.

### Introduce margin gradually and separate it from LR warmup

Test normalized softmax with margin 0 first, then increase the margin toward 0.3 over several epochs. An initial candidate is 3 epochs at zero margin followed by a 5-epoch linear ramp. Log the actual margin every epoch and restore its progress exactly on resume. This candidate schedule is a hypothesis, not a tuned setting.

At diagnosis time the schema required `arcface_margin > 0` and had no margin scheduler. The [recovery controls](model-c-recovery-training.md) now support zero margin and an epoch-based hold and ramp. `training.warmup_epochs` still only warms up learning rates; it does not change ArcFace margin. Validation margin loss changes meaning during a margin ramp, so do not use it to compare different ramp stages as though the objective were fixed.

An experiment supported immediately by the existing CLI is a fixed margin of 0.1. Compare it against a new margin-0.3 control with the same epoch budget. A lower AdamW LR such as 0.0003 and LR warmup of 3 epochs are separate experiments, not assumptions that AdamW itself is wrong. The [reference configuration](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/configs/base.py) supports both SGD and AdamW.

### Use the trained A encoder as an optional recovery experiment

The locally trained A encoder already has useful geometry. It can initialize C's encoder, with class weights initialized from normalized training-identity embedding means. This starts the target centers near the correct features instead of random directions. Build those means using training images only. Create a fresh optimizer and training run, rather than resuming a collapsed C optimizer state.

This is transfer initialization from the project's own A model and must be reported as such. Compare A continuation and C fine-tuning from the same A checkpoint if claiming a benefit from the ArcFace objective. Do not present this result as a comparison of A and C both trained from scratch.

## A controlled decision before another 100 epochs

Use the same split, crop protocol, 480-class mapping, batch order, augmentation seeds, and epoch budget across experiments. Keeping all classes avoids changing the full classifier geometry merely to make the diagnostic cheaper.

First compare the current head and the same-forward straight-through head for a fixed short budget, for example 10 epochs. Keep precision at BF16 in both arms. Log geometry within epoch 1 and after each epoch, since epoch 2 is already too late to see the original onset. A short run only screens for recurrence; it does not establish final recognition quality. A 10-epoch cosine schedule also differs from the old 100-epoch schedule, so use a new short control rather than treating the historical run as an exact control.

Track these quantities together:

- Classification train and validation accuracy, margin loss, and no-margin loss.
- Mean normalized feature norm, mean normalized class-weight norm, and their directional cosine.
- Positive-pair and negative-pair cosine distributions, including spread and tails.
- Target cosine, fraction in the fallback branch, angular derivative distribution, and raw feature norms.
- AUC and TAR at FPR 1% on the existing unseen-identity validation pairs.

A candidate must avoid the observed rising concentration and improve useful discrimination, not merely lower margin loss or move the threshold away from 0.998. Compare against the fresh control and then repeat promising results with additional seeds. Run BN, margin-ramp, regularization, optimizer, or FP32 controls only as needed to isolate the remaining uncertainty. Use the separate test protocol after choosing the method and validation threshold.

Start a new run for every changed objective, gradient policy, normalization design, or initialization. Keep the old checkpoints as evidence. More identical epochs, a different checkpoint tie-breaker, post-hoc sign flips, or another L2 normalization do not address this mechanism. CosFace is an alternative objective with different gradient behavior, but switching objectives must be explicit and must satisfy the coursework model definition.

## Reproduce the follow-up

The [follow-up script](../runs/diagnose_model_c_20261103/followup.py) reads checkpoints and writes a separate JSON report. Run it from the repository root:

```powershell
.venv/Scripts/python.exe -m runs.diagnose_model_c_20261103.followup --checkpoint runs/recognition/recognition_c_20261103T074533506303Z_a37c4003/best.pt --checkpoint runs/recognition/recognition_c_20261103T074533506303Z_a37c4003/last.pt --output runs/diagnose_model_c_20261103/followup.json
```

The [JSON report](../runs/diagnose_model_c_20261103/followup.json) contains the target-branch statistics, bias subtraction check, BF16 inference comparison, source checksum, and autograd experiment. It also evaluates wrong-class rectification on that diagnostic path. Paths, sampling seed, device, batch size, and images per identity are command-line inputs.

The `runs` directory is ignored by Git. Back up the script and JSON with the original diagnosis artifacts if sharing this report outside the current checkout.
