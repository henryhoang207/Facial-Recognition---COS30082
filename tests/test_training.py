"""Behavioral checks for the reusable engine, independent of recognition."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    MetricTotal,
    StepResult,
    Trainer,
    TrainingConfig,
    TrainingPolicy,
    TrainingTask,
    make_optimizer,
    make_scheduler,
    seed_everything,
)
from src.training.config import read_yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def config(**updates) -> TrainingConfig:
    raw = read_yaml(REPO / "configs/recognition/model_a.yaml")["training"]
    raw.update(
        epochs=4,
        scheduler="none",
        device="cpu",
        amp=False,
        amp_dtype="fp16",
        deterministic=True,
        monitor="val_loss",
        mode="min",
        patience=None,
        progress=False,
        parameter_groups=[],
        checkpoint_tiebreakers=[],
        early_stopping_monitor=None,
    )
    raw.update(updates)
    return TrainingConfig.model_validate(raw)


class RegressionTask(TrainingTask):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 4), nn.Dropout(0.2), nn.Linear(4, 1))

    def step(self, batch):
        images, labels = batch
        predictions = self.network(images)
        return StepResult(F.mse_loss(predictions, labels), len(labels))


class FixedLossTask(TrainingTask):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))

    def step(self, batch):
        (values,) = batch
        return StepResult(
            values.square().mean() + self.parameter * 0,
            len(values),
            {"mae": MetricTotal(values.abs().sum().item(), len(values))},
        )


def loaders():
    x = torch.linspace(-1, 1, 12).unsqueeze(1)
    dataset = TensorDataset(x, 2 * x + 1)
    return tuple(
        DataLoader(
            dataset,
            batch_size=5,
            shuffle=train,
            generator=torch.Generator().manual_seed(73),
        )
        for train in (True, False)
    )


def run_regression(path, cfg, *, resume=None):
    seed_everything(18)
    task = RegressionTask().to(cfg.device)
    optimizer = make_optimizer(task, cfg)
    result = Trainer(
        task, optimizer, cfg, path, scheduler=make_scheduler(optimizer, cfg)
    ).fit(*loaders(), resume=resume)
    return result, torch.load(
        result.last_checkpoint, weights_only=True, map_location="cpu"
    )


def test_loss_and_metrics_weight_partial_batch(tmp_path):
    cfg = config(epochs=1)
    task = FixedLossTask()
    loader = DataLoader(
        TensorDataset(torch.arange(1, 6, dtype=torch.float32)), batch_size=2
    )
    result = Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "run").fit(
        loader, loader
    )
    assert result.history[0]["train_loss"] == pytest.approx(11)
    assert result.history[0]["val_loss"] == pytest.approx(11)
    assert result.history[0]["val_mae"] == pytest.approx(3)


def test_triplet_task_accepts_dictionary_batches_and_updates_trainable_loss(tmp_path):
    class TripletTask(TrainingTask):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def step(self, batch):
            anchor = self.scale * batch["anchor"]
            positive = self.scale * batch["positive"]
            negative = self.scale * batch["negative"]
            return StepResult(
                F.triplet_margin_loss(anchor, positive, negative, margin=1),
                len(anchor),
            )

    dataset = [
        {
            "anchor": torch.tensor([0.0, 0.0]),
            "positive": torch.tensor([0.1, 0.0]),
            "negative": torch.tensor([0.2, 0.0]),
        }
        for _ in range(4)
    ]
    loader = DataLoader(dataset, batch_size=2)
    cfg = config(epochs=2, lr=0.1, weight_decay=0)
    task = TripletTask()
    result = Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "triplet").fit(
        loader, loader
    )
    assert task.scale.item() > 1
    assert result.history[-1]["val_loss"] < result.history[0]["train_loss"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resume_restores_dropout_optimizer_and_shuffled_loader(tmp_path, device):
    if device == "cuda" and (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        pytest.skip("Native BF16 GPU unavailable")
    cfg = config(device=device, amp=device == "cuda", amp_dtype="bf16")
    full, full_state = run_regression(tmp_path / "full", cfg)
    initial, _ = run_regression(
        tmp_path / "initial", cfg.model_copy(update={"epochs": 2})
    )
    resumed, resumed_state = run_regression(
        tmp_path / "resumed", cfg, resume=initial.last_checkpoint
    )
    assert [row["epoch"] for row in resumed.history] == [1, 2, 3, 4]
    assert resumed.history == full.history
    for name, value in full_state["task"].items():
        torch.testing.assert_close(resumed_state["task"][name], value, rtol=0, atol=0)
    assert full_state["optimizer"]["state"]


def test_resume_recovers_best_weights_from_last_checkpoint_alone(tmp_path):
    initial, _ = run_regression(tmp_path / "initial", config(epochs=2))
    initial.best_checkpoint.unlink()
    legacy = torch.load(initial.last_checkpoint, weights_only=True)
    legacy.pop("best_row")
    torch.save(legacy, initial.last_checkpoint)
    result, _ = run_regression(
        tmp_path / "continued", config(), resume=initial.last_checkpoint
    )
    best = torch.load(result.best_checkpoint, weights_only=True)
    assert best["epoch"] == result.best_epoch
    assert best["checkpoint_kind"] == "inference"


def test_best_checkpoint_and_early_stopping_use_epoch_metric(tmp_path):
    cfg = config(epochs=8, monitor="val_auc", mode="max", patience=2)
    values = iter([0.6, 0.8, 0.7, 0.65])
    task = FixedLossTask()
    loader = DataLoader(TensorDataset(torch.ones(3)), batch_size=2)
    trainer = Trainer(
        task,
        make_optimizer(task, cfg),
        cfg,
        tmp_path / "run",
        evaluator=lambda task, device: {"val_auc": next(values)},
    )
    result = trainer.fit(loader, loader)
    assert result.best_epoch == 2
    assert len(result.history) == 4
    best = torch.load(result.best_checkpoint, weights_only=True)
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert best["best"] == 0.8 and best["epoch"] == 2
    assert last["bad_epochs"] == 2 and last["epoch"] == 4


def test_missing_monitor_and_nonfinite_loss_fail_without_checkpoint(tmp_path):
    for value, monitor, message in [
        (1.0, "val_auc", "unavailable"),
        (float("nan"), "val_loss", "nonfinite"),
    ]:
        cfg = config(epochs=1, monitor=monitor)
        task = FixedLossTask()
        loader = DataLoader(TensorDataset(torch.tensor([value])), batch_size=1)
        directory = tmp_path / monitor
        with pytest.raises(ValueError, match=message):
            Trainer(task, make_optimizer(task, cfg), cfg, directory).fit(loader, loader)
        assert not (directory / "last.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("amp_dtype", ["fp16", "bf16"])
def test_cuda_amp_training_updates_weights(tmp_path, amp_dtype):
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("Native BF16 GPU unavailable")
    cfg = config(
        epochs=1,
        device="cuda",
        amp=True,
        amp_dtype=amp_dtype,
        deterministic=False,
        accumulation_steps=2,
        amp_init_scale=1.0,
    )
    seed_everything(18)
    task = RegressionTask().cuda()
    observed_dtypes = []
    task.network[-1].register_forward_hook(
        lambda module, inputs, output: observed_dtypes.append(output.dtype)
    )
    before = task.network[0].weight.detach().clone()
    result = Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "cuda").fit(
        *loaders()
    )
    assert not torch.equal(task.network[0].weight, before)
    checkpoint = torch.load(
        result.last_checkpoint, weights_only=True, map_location="cpu"
    )
    if amp_dtype == "fp16":
        assert checkpoint["scaler"]["scale"] > 0
    else:
        assert checkpoint["scaler"] == {}
    assert set(observed_dtypes) == {
        torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    }
    assert all(parameter.dtype == torch.float32 for parameter in task.parameters())
    environment = json.loads((result.run_dir / "environment.json").read_text())
    assert environment["precision"] == f"{amp_dtype}-mixed"
    assert environment["gradient_scaling"] == (amp_dtype == "fp16")
    assert result.history[0]["train_loss"] > 0
    assert result.history[0]["train_optimizer_steps"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_bf16_rejects_unsupported_gpu_instead_of_changing_precision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda including_emulation=False: False
    )
    cfg = config(device="cuda", amp=True, amp_dtype="bf16")
    task = RegressionTask().cuda()
    with pytest.raises(ValueError, match="does not support native BF16"):
        Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "unsupported")


def test_bf16_selection_with_cpu_or_disabled_amp_uses_fp32(tmp_path):
    cfg = config(epochs=1, amp=True, amp_dtype="bf16")
    result, payload = run_regression(tmp_path / "cpu", cfg)
    assert payload["scaler"] == {}
    environment = json.loads((result.run_dir / "environment.json").read_text())
    assert environment["precision"] == "fp32"
    assert environment["gradient_scaling"] is False


def test_resume_rejects_a_precision_change(tmp_path):
    initial, _ = run_regression(tmp_path / "initial-fp16", config(epochs=1))
    with pytest.raises(ValueError, match="amp_dtype"):
        run_regression(
            tmp_path / "changed-bf16",
            config(amp_dtype="bf16"),
            resume=initial.last_checkpoint,
        )


def test_older_checkpoint_without_amp_dtype_keeps_fp16_default(tmp_path):
    initial, payload = run_regression(tmp_path / "older-config", config(epochs=1))
    del payload["training_config"]["amp_dtype"]
    torch.save(payload, initial.last_checkpoint)
    result, saved = run_regression(
        tmp_path / "continued-config", config(epochs=2), resume=initial.last_checkpoint
    )
    assert len(result.history) == 2
    assert saved["training_config"]["amp_dtype"] == "fp16"


@pytest.mark.parametrize("amsgrad,expected", [(False, -0.0133333333), (True, -0.04)])
def test_adamw_custom_betas_epsilon_and_amsgrad_affect_updates(amsgrad, expected):
    cfg = config(
        lr=0.1, weight_decay=0, adam_betas=(0, 0), adam_eps=0.5, adam_amsgrad=amsgrad
    )
    task = FixedLossTask()
    optimizer = make_optimizer(task, cfg)
    for gradient in (2.0, -1.0):
        task.parameter.grad = torch.tensor(gradient)
        optimizer.step()
    assert task.parameter.item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "nesterov,dampening,expected", [(True, 0.0, -0.65), (False, 0.25, -0.45)]
)
def test_sgd_momentum_nesterov_and_dampening_affect_updates(
    nesterov, dampening, expected
):
    cfg = config(
        optimizer="sgd",
        lr=0.1,
        weight_decay=0,
        sgd_momentum=0.5,
        sgd_nesterov=nesterov,
        sgd_dampening=dampening,
    )
    task = FixedLossTask()
    optimizer = make_optimizer(task, cfg)
    for _ in range(2):
        task.parameter.grad = torch.tensor(2.0)
        optimizer.step()
    assert task.parameter.item() == pytest.approx(expected)


def test_parameter_groups_control_learning_rate_decay_and_cover_remaining_parameters():
    task = nn.Module()
    task.backbone = nn.Linear(1, 1, bias=False)
    task.head = nn.Linear(1, 1, bias=False)
    task.temperature = nn.Parameter(torch.ones(()))
    nn.init.ones_(task.backbone.weight)
    nn.init.ones_(task.head.weight)
    cfg = config(
        optimizer="sgd",
        lr=0.1,
        weight_decay=0,
        parameter_groups=[
            {"name": "backbone", "modules": ["backbone"], "lr_scale": 0.1},
            {"name": "head", "modules": ["head"], "lr": 0.2, "weight_decay": 0.5},
        ],
    )
    optimizer = make_optimizer(task, cfg)
    for parameter in task.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert task.backbone.weight.item() == pytest.approx(0.99)
    assert task.head.weight.item() == pytest.approx(0.7)
    assert task.temperature.item() == pytest.approx(0.9)
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone",
        "head",
        "default",
    ]


@pytest.mark.parametrize(
    "scheduler,expected,last",
    [
        ("cosine", [0.1, 0.55, 1.0, 0.775, 0.325], 0.1),
        ("none", [0.1, 0.55, 1.0, 1.0, 1.0], 1.0),
    ],
)
def test_linear_warmup_and_cosine_floor(scheduler, expected, last):
    cfg = config(
        epochs=5,
        lr=1.0,
        scheduler=scheduler,
        warmup_epochs=2,
        warmup_start_factor=0.1,
        eta_min=0.1,
    )
    task = FixedLossTask()
    optimizer = make_optimizer(task, cfg)
    schedule = make_scheduler(optimizer, cfg)
    rates = []
    for _ in range(5):
        rates.append(optimizer.param_groups[0]["lr"])
        task.parameter.grad = torch.tensor(1.0)
        optimizer.step()
        schedule.step()
    assert rates == pytest.approx(expected)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(last)


class ScalarRegressionTask(TrainingTask):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def step(self, batch):
        images, targets = batch
        return StepResult(F.mse_loss(self.weight * images, targets), len(targets))


@pytest.mark.parametrize(
    "accumulation,batch_size,expected,steps", [(3, 5, 0.6, 1), (2, 4, -1.0, 2)]
)
def test_accumulation_weights_uneven_and_final_microbatches_like_large_batches(
    tmp_path, accumulation, batch_size, expected, steps
):
    dataset = TensorDataset(torch.arange(1, 6, dtype=torch.float32), torch.ones(5))
    cfg = config(
        epochs=1,
        optimizer="sgd",
        lr=0.1,
        weight_decay=0,
        accumulation_steps=accumulation,
    )
    task = ScalarRegressionTask()
    micro = DataLoader(dataset, batch_size=2)
    result = Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "micro").fit(
        micro, micro
    )
    large_task = ScalarRegressionTask()
    large_cfg = cfg.model_copy(update={"accumulation_steps": 1})
    large = DataLoader(dataset, batch_size=batch_size)
    Trainer(
        large_task, make_optimizer(large_task, large_cfg), large_cfg, tmp_path / "large"
    ).fit(large, large)
    assert task.weight.item() == pytest.approx(expected)
    assert large_task.weight.item() == pytest.approx(expected)
    assert result.history[0]["train_optimizer_steps"] == steps


def test_accumulation_clips_the_averaged_gradient_once(tmp_path):
    cfg = config(
        epochs=1,
        optimizer="sgd",
        lr=0.1,
        weight_decay=0,
        accumulation_steps=3,
        clip_grad_norm=1,
    )
    task = ScalarRegressionTask()
    loader = DataLoader(
        TensorDataset(torch.arange(1, 6, dtype=torch.float32), torch.ones(5)),
        batch_size=2,
    )
    Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "clipped").fit(
        loader, loader
    )
    assert task.weight.item() == pytest.approx(0.1, abs=1e-6)


def test_early_stopping_waits_until_warmup_is_over(tmp_path):
    cfg = config(epochs=7, warmup_epochs=3, patience=2, monitor="val_auc", mode="max")
    task = FixedLossTask()
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    optimizer = make_optimizer(task, cfg)
    trainer = Trainer(
        task,
        optimizer,
        cfg,
        tmp_path / "warmup-stop",
        scheduler=make_scheduler(optimizer, cfg),
        evaluator=lambda task, device: {"val_auc": 0.5},
    )
    result = trainer.fit(loader, loader)
    assert len(result.history) == 5
    assert result.best_epoch == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_early_stopping_waits_when_overflow_prevents_warmup_progress(tmp_path):
    class OverflowTask(TrainingTask):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.zeros(()))

        def step(self, batch):
            return StepResult(self.parameter.sqrt(), len(batch[0]))

    cfg = config(
        epochs=5,
        warmup_epochs=2,
        patience=1,
        device="cuda",
        amp=True,
        deterministic=False,
    )
    task = OverflowTask().cuda()
    optimizer = make_optimizer(task, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    result = Trainer(
        task, optimizer, cfg, tmp_path / "overflow-warmup", scheduler=scheduler
    ).fit(loader, loader)
    assert len(result.history) == 5
    assert scheduler.last_epoch == 0


def test_warmup_accumulation_and_group_learning_rates_resume_exactly(tmp_path):
    cfg = config(
        epochs=5,
        warmup_epochs=2,
        scheduler="cosine",
        eta_min=1e-5,
        accumulation_steps=2,
        parameter_groups=[
            {"name": "backbone", "modules": ["network.0"], "lr_scale": 0.5},
            {"name": "head", "modules": ["network.2"]},
        ],
    )
    full, full_payload = run_regression(tmp_path / "full-warmup", cfg)
    seed_everything(18)
    task = RegressionTask()
    optimizer = make_optimizer(task, cfg)
    calls = 0

    def interrupt(task, device):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return {}

    partial_dir = tmp_path / "interrupted"
    trainer = Trainer(
        task,
        optimizer,
        cfg,
        partial_dir,
        scheduler=make_scheduler(optimizer, cfg),
        evaluator=interrupt,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.fit(*loaders())
    resumed, resumed_payload = run_regression(
        tmp_path / "resumed-warmup", cfg, resume=partial_dir / "last.pt"
    )
    assert resumed.history == full.history
    assert full.history[0]["lr_backbone"] == pytest.approx(0.00005)
    assert full.history[0]["lr_head"] == pytest.approx(0.0001)
    for name, value in full_payload["task"].items():
        torch.testing.assert_close(resumed_payload["task"][name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"adam_betas": [1.0, 0.99]}, "betas"),
        ({"optimizer": "sgd", "sgd_nesterov": True}, "Nesterov"),
        ({"warmup_epochs": 4}, "warmup_epochs"),
        ({"accumulation_steps": 0}, "accumulation_steps"),
        ({"scheduler": "cosine", "eta_min": 1}, "eta_min"),
        ({"scheduler": "cosine", "cosine_t_max": 2}, "cosine_t_max"),
    ],
)
def test_invalid_hyperparameters_fail_at_config_boundary(updates, message):
    with pytest.raises(ValueError, match=message):
        config(**updates)


def test_optimizer_rejects_unknown_or_overlapping_module_groups():
    task = RegressionTask()
    missing = config(parameter_groups=[{"name": "missing", "modules": ["unknown"]}])
    with pytest.raises(ValueError, match="unknown parameter group module"):
        make_optimizer(task, missing)
    overlapping = config(
        parameter_groups=[
            {"name": "all", "modules": ["network"]},
            {"name": "part", "modules": ["network.0"]},
        ]
    )
    with pytest.raises(ValueError, match="overlap"):
        make_optimizer(task, overlapping)


def test_cosine_floor_uses_actual_group_rates_and_rejects_uncovered_low_rate():
    task = RegressionTask()
    cfg = config(
        lr=0.001,
        scheduler="cosine",
        eta_min=0.01,
        parameter_groups=[
            {"name": "backbone", "modules": ["network.0"], "lr": 0.1},
            {"name": "head", "modules": ["network.2"], "lr": 0.2},
        ],
    )
    optimizer = make_optimizer(task, cfg)
    schedule = make_scheduler(optimizer, cfg)
    for _ in range(4):
        for parameter in task.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        schedule.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [0.01, 0.01]
    )
    incomplete = cfg.model_copy(update={"parameter_groups": cfg.parameter_groups[:1]})
    with pytest.raises(ValueError, match="actual optimizer group"):
        make_scheduler(make_optimizer(task, incomplete), incomplete)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_scheduler_does_not_advance_when_amp_skips_all_updates(tmp_path):
    class OverflowTask(TrainingTask):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.zeros(()))

        def step(self, batch):
            # Finite forward loss with an infinite derivative at zero.
            return StepResult(self.parameter.sqrt(), len(batch[0]))

    cfg = config(
        epochs=1, scheduler="cosine", device="cuda", amp=True, deterministic=False
    )
    task = OverflowTask().cuda()
    optimizer = make_optimizer(task, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    Trainer(task, optimizer, cfg, tmp_path / "overflow", scheduler=scheduler).fit(
        loader, loader
    )
    assert scheduler.last_epoch == 0


def test_zero_signal_skips_adamw_decay_and_state_but_keeps_metrics(tmp_path):
    class ZeroSignalTask(TrainingTask):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.ones(()))

        def step(self, batch):
            return StepResult(
                self.parameter * 0,
                len(batch[0]),
                {"inactive": MetricTotal(len(batch[0]), len(batch[0]))},
                optimize=False,
            )

    cfg = config(epochs=1, lr=0.1, weight_decay=0.5, accumulation_steps=2)
    task = ZeroSignalTask()
    optimizer = make_optimizer(task, cfg)
    loader = DataLoader(TensorDataset(torch.ones(5)), batch_size=2)
    result = Trainer(task, optimizer, cfg, tmp_path / "zero-signal").fit(loader, loader)
    assert task.parameter.item() == 1
    assert optimizer.state_dict()["state"] == {}
    assert result.history[0]["train_optimizer_steps"] == 0
    assert result.history[0]["train_inactive"] == 1
    assert result.history[0]["train_loss"] == 0


def test_zero_signal_batches_do_not_dilute_accumulated_gradients(tmp_path):
    class SelectiveTask(ScalarRegressionTask):
        def step(self, batch):
            images, targets, active = batch
            result = super().step((images, targets))
            return StepResult(result.loss, result.weight, optimize=bool(active.item()))

    cfg = config(
        epochs=1, optimizer="sgd", lr=0.1, weight_decay=0, accumulation_steps=2
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor([1.0, 100.0, 3.0]),
            torch.ones(3),
            torch.tensor([True, False, True]),
        ),
        batch_size=1,
    )
    task = SelectiveTask()
    result = Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "selective").fit(
        loader, loader
    )
    assert task.weight.item() == pytest.approx(0.4)
    assert result.history[0]["train_optimizer_steps"] == 1


def test_whole_validation_evaluator_needs_no_batch_validation_step(tmp_path):
    class TrainOnlyTask(FixedLossTask):
        def step(self, batch):
            assert self.training
            return super().step(batch)

    cfg = config(epochs=2, monitor="val_auc", mode="max")
    task = TrainOnlyTask()
    loader = DataLoader(TensorDataset(torch.ones(4)), batch_size=2)

    def evaluate(task, device):
        assert not task.training and not torch.is_grad_enabled()
        return {"val_auc": 0.75}

    result = Trainer(
        task,
        make_optimizer(task, cfg),
        cfg,
        tmp_path / "whole-validation",
        evaluator=evaluate,
    ).fit(loader)
    assert all(row["val_auc"] == 0.75 for row in result.history)
    assert all("val_loss" not in row for row in result.history)
    no_eval = TrainOnlyTask()
    with pytest.raises(ValueError, match="evaluator is required"):
        Trainer(no_eval, make_optimizer(no_eval, cfg), cfg, tmp_path / "no-eval").fit(
            loader
        )


class StatefulTask(TrainingTask):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(()))
        self.generator = torch.Generator().manual_seed(1729)
        self.seen = 0

    def get_extra_state(self):
        return {"seen": self.seen, "generator": self.generator.get_state()}

    def set_extra_state(self, state):
        self.seen = state["seen"]
        self.generator.set_state(state["generator"])

    def step(self, batch):
        assert self.training
        self.seen += 1
        target = torch.rand((), generator=self.generator)
        return StepResult((self.parameter - target).square(), len(batch[0]))


class StatefulPolicy(TrainingPolicy):
    def __init__(self):
        self.completed = 0
        self.batch_calls = 0

    def on_epoch_start(self, trainer, epoch, loader):
        assert trainer.current_epoch == epoch
        assert self.completed == epoch - 1

    def on_batch_start(self, trainer, epoch, batch_index, batches):
        assert batches == 2
        self.batch_calls += 1
        for group in trainer.optimizer.param_groups:
            group["lr"] = 0.02 / (epoch + batch_index)

    def on_epoch_end(self, trainer, epoch, row):
        assert "val_auc" in row
        self.completed = epoch
        row["strategy"] = "random" if epoch == 1 else "semi_hard"

    def should_stop(self, trainer, epoch):
        return self.completed >= 3

    def state_dict(self):
        return {"completed": self.completed, "batch_calls": self.batch_calls}

    def load_state_dict(self, state):
        self.completed = state["completed"]
        self.batch_calls = state["batch_calls"]


def test_policy_and_task_extra_state_resume_and_finalize_without_extra_epoch(tmp_path):
    cfg = config(epochs=4, monitor="val_auc", mode="max", patience=1)

    def run(path, *, resume=None, interrupt=False):
        task = StatefulTask()
        policy = StatefulPolicy()

        def evaluate(task, device):
            if interrupt and policy.completed == 2:
                raise RuntimeError("policy interruption")
            return {"val_auc": 0.75}

        trainer = Trainer(
            task,
            make_optimizer(task, cfg),
            cfg,
            path,
            evaluator=evaluate,
            policy=policy,
        )
        loader = DataLoader(
            TensorDataset(torch.ones(4)),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(98),
        )
        return trainer.fit(loader, resume=resume), task, policy

    full, full_task, full_policy = run(tmp_path / "policy-full")
    with pytest.raises(RuntimeError, match="policy interruption"):
        run(tmp_path / "policy-interrupted", interrupt=True)
    resumed, resumed_task, resumed_policy = run(
        tmp_path / "policy-resumed", resume=tmp_path / "policy-interrupted/last.pt"
    )
    assert resumed.history == full.history
    assert len(full.history) == 3
    assert full_policy.batch_calls == resumed_policy.batch_calls == 6
    assert full_task.seen == resumed_task.seen == 6
    torch.testing.assert_close(
        full_task.parameter, resumed_task.parameter, rtol=0, atol=0
    )
    best = torch.load(full.best_checkpoint, weights_only=True)
    last = torch.load(full.last_checkpoint, weights_only=True)
    assert best["task"]["_extra_state"]["seen"] == 2
    assert last["task"]["_extra_state"]["seen"] == 6
    assert last["policy_state"] == full_policy.state_dict()
    assert best["val_metrics"]["strategy"] == "random"
    finalized, finalized_task, finalized_policy = run(
        tmp_path / "policy-finalized", resume=full.last_checkpoint
    )
    assert finalized.history == full.history
    assert finalized_task.seen == 6
    assert finalized_policy.batch_calls == 6
    assert finalized.best_checkpoint.exists() and finalized.last_checkpoint.exists()
    assert (finalized.run_dir / "history.csv").exists()
    task = StatefulTask()
    with pytest.raises(ValueError, match="training policy type differs"):
        Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "wrong-policy").fit(
            DataLoader(TensorDataset(torch.ones(2))),
            DataLoader(TensorDataset(torch.ones(2))),
            resume=full.last_checkpoint,
        )


def test_trainer_advances_epoch_aware_batch_sampler(tmp_path):
    class EpochBatchSampler(torch.utils.data.BatchSampler):
        def __init__(self, dataset):
            super().__init__(torch.utils.data.SequentialSampler(dataset), 2, False)
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    dataset = TensorDataset(torch.ones(4))
    sampler = EpochBatchSampler(dataset)
    loader = DataLoader(dataset, batch_sampler=sampler)
    cfg = config(epochs=3)
    task = FixedLossTask()
    Trainer(task, make_optimizer(task, cfg), cfg, tmp_path / "epoch-sampler").fit(
        loader, loader
    )
    assert sampler.epochs == [0, 1, 2]


def test_checkpoint_writes_through_file_stream_and_round_trips(tmp_path, monkeypatch):
    directory = tmp_path / "training_session"
    original_save = torch.save
    writes = []

    def save(payload, stream, *args, **kwargs):
        assert hasattr(stream, "write") and not stream.closed
        writes.append(stream.name)
        return original_save(payload, stream, *args, **kwargs)

    monkeypatch.setattr(torch, "save", save)
    cfg = config(epochs=1)
    task = FixedLossTask()
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    result = Trainer(task, make_optimizer(task, cfg), cfg, directory).fit(
        loader, loader
    )
    with result.last_checkpoint.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["checkpoint_kind"] == "training"
    assert result.best_checkpoint.is_file()
    assert len(writes) == 2


def selection_config(**updates):
    return config(
        epochs=12,
        patience=2,
        monitor="val_tar",
        mode="max",
        checkpoint_tiebreakers=[
            {"metric": "val_auc", "mode": "max"},
            {"metric": "val_loss", "mode": "min", "min_delta": 0.0001},
        ],
        **updates,
    )


def selection_run(path, cfg, rows, *, resume=None):
    task = FixedLossTask()
    values = iter(rows)
    loader = DataLoader(
        TensorDataset(torch.ones(2)),
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    def epoch(loader, *, train):
        if train:
            with torch.no_grad():
                task.parameter.add_(1)
            return {"loss": 1.0, "optimizer_steps": 1}
        return next(values)

    trainer = Trainer(task, make_optimizer(task, cfg), cfg, path)
    trainer._epoch = epoch
    return trainer.fit(loader, loader, resume=resume)


def test_secondary_metrics_save_best_after_auc_plateaus_and_reset_patience(tmp_path):
    rows = [
        {"tar": 0.8, "auc": 0.99, "loss": 1.0},
        {"tar": 0.81, "auc": 0.99, "loss": 1.2},
        {"tar": 0.81, "auc": 0.99, "loss": 0.8},
        {"tar": 0.80, "auc": 1.0, "loss": 0.1},
        {"tar": 0.81, "auc": 0.99, "loss": 0.8},
    ]
    result = selection_run(tmp_path / "multi", selection_config(), rows)
    assert result.best_epoch == 3 and len(result.history) == 5
    best = torch.load(result.best_checkpoint, weights_only=True)
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert best["selection_metrics"] == {
        "val_tar": 0.81,
        "val_auc": 0.99,
        "val_loss": 0.8,
    }
    assert last["bad_epochs"] == 2
    assert best["task"]["parameter"].item() == 3
    assert last["task"]["parameter"].item() == 5


def test_auc_tie_breaker_precedes_loss_and_unchanged_metrics_keep_earlier_epoch(
    tmp_path,
):
    rows = [
        {"tar": 0.9, "auc": 0.98, "loss": 0.5},
        {"tar": 0.9, "auc": 0.99, "loss": 1.0},
        {"tar": 0.9, "auc": 0.99, "loss": 0.99995},
        {"tar": 0.9, "auc": 0.99, "loss": 1.0},
    ]
    result = selection_run(tmp_path / "priority", selection_config(), rows)
    assert result.best_epoch == 2 and len(result.history) == 4


def test_missing_secondary_metric_fails_before_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="val_auc"):
        selection_run(
            tmp_path / "missing", selection_config(), [{"tar": 0.9, "loss": 1.0}]
        )
    assert not (tmp_path / "missing/best.pt").exists()


def test_separate_stopping_metric_does_not_reset_for_checkpoint_tiebreakers(tmp_path):
    rows = [
        {"tar": 0.8, "auc": 0.99, "loss": 1.0},
        {"tar": 0.81, "auc": 0.99, "loss": 1.0},
        {"tar": 0.82, "auc": 0.99, "loss": 1.0},
    ]
    cfg = selection_config(early_stopping_monitor={"metric": "val_auc", "mode": "max"})
    result = selection_run(tmp_path / "separate", cfg, rows)
    assert result.best_epoch == 3 and len(result.history) == 3
    saved = torch.load(result.last_checkpoint, weights_only=True)
    assert saved["stopping_best"] == {"val_auc": 0.99} and saved["bad_epochs"] == 2


def test_multi_metric_resume_keeps_selection_and_stopping_state(tmp_path):
    rows = [
        {"tar": 0.8, "auc": 0.99, "loss": 1.0},
        {"tar": 0.81, "auc": 0.99, "loss": 1.0},
        {"tar": 0.81, "auc": 0.99, "loss": 0.8},
        {"tar": 0.80, "auc": 1.0, "loss": 0.1},
        {"tar": 0.81, "auc": 0.99, "loss": 0.8},
    ]
    cfg = selection_config()
    full = selection_run(tmp_path / "full-metrics", cfg, rows)
    first = selection_run(
        tmp_path / "first-metrics", cfg.model_copy(update={"epochs": 2}), rows[:2]
    )
    resumed = selection_run(
        tmp_path / "resumed-metrics", cfg, rows[2:], resume=first.last_checkpoint
    )
    assert resumed.history == full.history
    assert resumed.best_epoch == full.best_epoch == 3
    saved = torch.load(resumed.last_checkpoint, weights_only=True)
    assert saved["bad_epochs"] == 2
    cfg = selection_config(early_stopping_monitor={"metric": "val_auc", "mode": "max"})
    first = selection_run(
        tmp_path / "first-stop", cfg.model_copy(update={"epochs": 2}), rows[:2]
    )
    resumed = selection_run(
        tmp_path / "resumed-stop", cfg, rows[2:], resume=first.last_checkpoint
    )
    assert len(resumed.history) == 3
    assert torch.load(resumed.last_checkpoint, weights_only=True)["bad_epochs"] == 2


def test_duplicate_checkpoint_metrics_are_rejected():
    with pytest.raises(ValueError, match="metric names must be unique"):
        config(checkpoint_tiebreakers=[{"metric": "val_loss", "mode": "min"}])


def test_policy_preserves_task_schedule_tiebreakers_and_stopping_warmup(tmp_path):
    class ScheduledTask(FixedLossTask):
        def __init__(self):
            super().__init__()
            self.scheduled_epochs = []

        def on_epoch_start(self, epoch):
            self.scheduled_epochs.append(epoch)
            return {"scheduled": float(epoch)}

        def early_stopping_ready(self, epoch):
            return epoch >= 3

    class AnnotationPolicy(TrainingPolicy):
        def on_epoch_end(self, trainer, epoch, row):
            row["annotation"] = "scheduled"

    cfg = config(
        epochs=8,
        patience=2,
        checkpoint_tiebreakers=[{"metric": "scheduled", "mode": "max"}],
        early_stopping_monitor={"metric": "val_loss", "mode": "min"},
    )
    task = ScheduledTask()
    loader = DataLoader(TensorDataset(torch.ones(2)), batch_size=2)
    result = Trainer(
        task,
        make_optimizer(task, cfg),
        cfg,
        tmp_path / "scheduled-policy",
        policy=AnnotationPolicy(),
    ).fit(loader, loader)
    assert task.scheduled_epochs == [1, 2, 3, 4]
    assert result.best_epoch == 4
    assert all(row["annotation"] == "scheduled" for row in result.history)
    best = torch.load(result.best_checkpoint, weights_only=True)
    last = torch.load(result.last_checkpoint, weights_only=True)
    assert best["selection_metrics"] == {"val_loss": 1.0, "scheduled": 4.0}
    assert last["stopping_best"] == {"val_loss": 1.0}
    assert last["bad_epochs"] == 2
