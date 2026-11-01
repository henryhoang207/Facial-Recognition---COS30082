"""Presentation attack detection metrics (ISO/IEC 30107-3 terms).

Scores are P(live); a face is accepted as live when score >= threshold.
- APCER: share of attacks accepted as live. Computed per attack type; the reported
  APCER is the worst type, as in the OULU-NPU protocol.
- BPCER: share of live faces rejected as spoof.
- ACER:  (APCER + BPCER) / 2. The main number for comparing models.
The threshold is always chosen on val (equal error rate point) and then fixed for test.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _rates(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    live = labels == 1
    accepted = scores >= threshold
    apcer = float(accepted[~live].mean()) if (~live).any() else float("nan")
    bpcer = float((~accepted[live]).mean()) if live.any() else float("nan")
    return apcer, bpcer


def eer_threshold(scores, labels) -> tuple[float, float]:
    """Threshold where APCER (pooled) equals BPCER, and that error rate."""
    scores, labels = _validate(scores, labels)
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    thr[0] = np.nextafter(scores.max(), np.inf)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def bpcer_at_apcer(scores, labels, target: float) -> float:
    """Lowest BPCER achievable while letting through at most `target` of attacks."""
    scores, labels = _validate(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    ok = fpr <= target
    return float(1 - tpr[ok].max()) if ok.any() else 1.0


def summarize(scores, labels, attacks, threshold: float) -> dict:
    scores, labels = _validate(scores, labels)
    attacks = np.asarray(attacks)
    if attacks.shape != labels.shape or not np.isfinite(threshold):
        raise ValueError("invalid attacks or threshold")
    apcer_pooled, bpcer = _rates(scores, labels, threshold)
    per_attack = {
        a: _rates(scores[attacks == a], labels[attacks == a], threshold)[0]
        for a in sorted(set(attacks[labels == 0]))
    }
    apcer = max(per_attack.values()) if per_attack else float("nan")
    both = (labels == 1).any() and (labels == 0).any()
    return {
        "n_live": int((labels == 1).sum()),
        "n_spoof": int((labels == 0).sum()),
        "threshold": float(threshold),
        "accuracy": float(((scores >= threshold) == (labels == 1)).mean()),
        "apcer": apcer,
        "apcer_pooled": apcer_pooled,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "auc": float(roc_auc_score(labels, scores)) if both else float("nan"),
        "eer": eer_threshold(scores, labels)[1] if both else float("nan"),
        "bpcer_at_apcer_5": (
            bpcer_at_apcer(scores, labels, 0.05) if both else float("nan")
        ),
        "bpcer_at_apcer_1": (
            bpcer_at_apcer(scores, labels, 0.01) if both else float("nan")
        ),
        "apcer_per_attack": per_attack,
    }


def _validate(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if (
        scores.ndim != 1
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
    ):
        raise ValueError("scores and labels must be finite aligned vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("metrics require both live=1 and spoof=0")
    return scores, labels.astype(int)
