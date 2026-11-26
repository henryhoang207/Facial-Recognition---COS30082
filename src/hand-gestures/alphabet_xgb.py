"""Train and load the XGBoost baseline for single-image ASL landmarks."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.datasets.catalog import write_json
from src.training.config import resolve_device

from .alphabet_features import augment_points, class_weights, pose_features


@torch.inference_mode()
def _features(points, config, device, *, augment=False):
    batches = torch.from_numpy(points).split(config.training.batch_size)
    features = []
    for batch in batches:
        batch = batch.to(device)
        if augment:
            batch = augment_points(
                batch,
                config.flip_probability,
                config.rotation_degrees,
                config.jitter_std,
            )
        features.append(pose_features(batch).cpu().numpy())
    return np.concatenate(features)


def train(config, points, labels, splits, metadata: dict, run: Path) -> None:
    import xgboost as xgb

    device = resolve_device(config.training.device)
    if device.type == "cuda" and not xgb.build_info().get("USE_CUDA"):
        raise RuntimeError("CUDA training needs an XGBoost build with GPU support")

    train_mask, val_mask = splits == "train", splits == "val"
    train_labels, val_labels = labels[train_mask], labels[val_mask]
    train_points = points[train_mask]
    class_count = len(metadata["classes"])
    if not len(val_labels):
        raise ValueError("XGBoost requires a validation set")
    train_weights = class_weights(train_labels, class_count)[train_labels]

    # Trees fit a fixed feature matrix, so augment one extra copy on the GPU.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(config.seed)
        train_features = np.concatenate(
            (
                _features(train_points, config, device),
                _features(train_points, config, device, augment=True),
            )
        )
    train_labels = np.tile(train_labels, 2)
    val_features = _features(points[val_mask], config, device)
    dtrain = xgb.QuantileDMatrix(
        train_features,
        label=train_labels,
        weight=np.tile(train_weights, 2),
        max_bin=128,
        nthread=4,
    )
    dval = xgb.QuantileDMatrix(
        val_features,
        label=val_labels,
        ref=dtrain,
        max_bin=128,
        nthread=4,
    )

    def macro_f1(probabilities, _data):
        predicted = probabilities.reshape(-1, class_count).argmax(axis=1)
        return "macro_f1", float(
            f1_score(
                val_labels,
                predicted,
                labels=range(class_count),
                average="macro",
                zero_division=0,
            )
        )

    params = {
        "objective": "multi:softprob",
        "num_class": class_count,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": str(device),
        "max_depth": 6,
        "max_bin": 128,
        "eta": 0.15,
        "seed": config.seed,
        "nthread": 4,
    }
    history = {}
    booster = xgb.train(
        params,
        dtrain,
        config.xgb_rounds,
        evals=[(dval, "val")],
        evals_result=history,
        custom_metric=macro_f1,
        verbose_eval=10,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=config.xgb_patience,
                metric_name="macro_f1",
                data_name="val",
                maximize=True,
                save_best=True,
            )
        ],
    )
    booster.set_attr(alphabet_schema=metadata["schema"])
    booster.save_model(run / "model.json")
    write_json(
        run / "history.json",
        {
            "xgboost": xgb.__version__,
            "params": params,
            "best_round": booster.best_iteration + 1,
            "history": history,
        },
    )


def load_alphabet_xgb(run: Path, metadata: dict, device: str = "cpu"):
    import xgboost as xgb

    target = resolve_device(device)
    if target.type == "cuda":
        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError(
                "CUDA prediction needs an XGBoost build with GPU support"
            )
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
    booster = xgb.Booster(params={"device": str(target), "nthread": 1})
    booster.load_model(run / "model.json")
    if (
        booster.num_features() != len(metadata["columns"])
        or booster.attr("alphabet_schema") != metadata["schema"]
    ):
        raise ValueError(f"alphabet feature schema mismatch: {run}")

    actual_device = json.loads(booster.save_config())["learner"]["generic_param"][
        "device"
    ]
    if actual_device != str(target):
        raise RuntimeError(
            f"XGBoost prediction requested {target} but uses {actual_device}"
        )

    if target.type == "cuda":

        @torch.inference_mode()
        def predict_cuda(points: np.ndarray) -> np.ndarray:
            features = pose_features(torch.from_numpy(points).to(target))
            # DMatrix stages NumPy features for the CUDA predictor without CuPy.
            return booster.predict(xgb.DMatrix(features.cpu().numpy(), nthread=1))

        return predict_cuda

    @torch.inference_mode()
    def predict(points: np.ndarray) -> np.ndarray:
        return booster.inplace_predict(pose_features(torch.from_numpy(points)).numpy())

    return predict
