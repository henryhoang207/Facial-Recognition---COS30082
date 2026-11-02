"""Generate report sections and a comparison ROC from measured final outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .provenance import file_sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def markdown_table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ]


def write_report(fig_dir, frozen_path, dataset_summary, output):
    fig_dir, frozen_path = Path(fig_dir), Path(frozen_path)
    decision = json.loads(frozen_path.read_text())
    dataset = json.loads(Path(dataset_summary).read_text())
    table = pd.read_csv(fig_dir / "liveness_results.csv")
    if set(table.model) != {"mobilenetv3", "deeppixbis"}:
        raise ValueError("report requires final measurements for both models")
    training_rows, configurations, runtimes = [], [], []
    expected_runs = set()
    figure, axis = plt.subplots(figsize=(5, 5))
    for candidate in decision["candidates"]:
        path, model = Path(candidate["checkpoint"]), candidate["model"]
        if file_sha256(path) != candidate["sha256"]:
            raise ValueError("frozen checkpoint content changed")
        measured = json.loads((fig_dir / f"liveness_{model}_test.json").read_text())
        if measured["checkpoint_sha256"] != candidate["sha256"]:
            raise ValueError("results do not match frozen checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        training = config["training"]
        expected_runs.add(config["run_id"])
        summary = json.loads((path.parent / "summary.json").read_text())
        training_rows.append(
            [
                model,
                training["lr"],
                training["batch_size"],
                training["accumulation_steps"],
                training["seed"],
                checkpoint["epoch"],
                summary["epochs_in_checkpoint"],
                f"{candidate['val_acer']:.2%}",
            ]
        )
        configurations.append(
            [
                model,
                "ImageNet" if config["pretrained"] else "random",
                config["img_size"],
                config["crop_scale"],
                config["freeze_blocks"],
            ]
        )
        runtimes.append(
            [
                model,
                f"{summary['seconds_this_invocation'] / 3600:.3f}",
                "current training invocation; see invocations.jsonl for interruptions",
            ]
        )
        scores = pd.read_csv(fig_dir / f"liveness_{model}_test_scores.csv")
        fpr, tpr, _ = roc_curve(scores.label, scores.score)
        axis.plot(fpr, tpr, label=model)
    if set(table.run_id) != expected_runs:
        plt.close(figure)
        raise ValueError("comparison table contains runs outside the frozen set")
    axis.set(xlabel="Pooled APCER", ylabel="Live acceptance rate", title="Test ROC")
    axis.legend()
    figure.tight_layout()
    figure.savefig(fig_dir / "liveness_comparison_roc.png", dpi=150)
    plt.close(figure)

    lines = [
        "# FaceGuard public data anti spoofing results",
        "",
        "## Methodology",
        "",
        "This phase compared MobileNetV3-Large binary classification with "
        "DeepPixBiS pixel-wise binary supervision in PyTorch. Both used the same "
        "processed CelebA-Spoof protocol. Images were converted from BGR to RGB "
        "and normalized with ImageNet statistics. Live was label 1 and spoof "
        "was label 0. Print and replay attacks were retained; masks were excluded.",
        "",
        f"Exported split counts: {json.dumps(dataset['splits'], sort_keys=True)}. "
        "Test subjects were kept out of training and validation. Refer to the "
        "dataset summary, preparation configuration and rejection log for "
        "holdout settings and the exact exclusions.",
        "",
        *markdown_table(
            ["Model", "Initialization", "Input size", "Crop scale", "Frozen blocks"],
            configurations,
        ),
        "",
        "The default protocol uses AdamW with weight decay 0.0001, cosine decay "
        "and balanced live/spoof sampling. MobileNet's frozen blocks retain "
        "fixed normalization statistics. DeepPixBiS minimizes equally weighted "
        "pixel and image BCE losses; inference averages pixel probabilities. "
        "The selected runs' full resolved configurations accompany the weights.",
        "",
        *markdown_table(
            [
                "Model",
                "LR",
                "Microbatch",
                "Accumulation",
                "Seed",
                "Best epoch",
                "Epochs completed",
                "Val ACER",
            ],
            training_rows,
        ),
        "",
        *markdown_table(["Model", "Hours", "Timing scope"], runtimes),
        "",
        "Checkpoints minimized validation ACER. Each deployed threshold was "
        "calibrated at that checkpoint's validation equal-error operating point. "
        "Final evaluation used the saved threshold unchanged.",
        "",
        "## Results",
        "",
        "APCER is reported separately for print and replay. The worst-attack "
        "APCER is used in ACER, which averages it with BPCER. EER and BPCER at "
        "pooled APCER 5% are descriptive ROC diagnostics; they do not recalibrate "
        "the deployment threshold.",
        "",
    ]
    result_rows = []
    for row in table.itertuples():
        values = [
            row.accuracy,
            row.apcer_print,
            row.apcer_replay,
            row.bpcer,
            row.acer,
            row.eer,
            row.bpcer_at_apcer_5,
        ]
        result_rows.append(
            [
                row.model,
                row.test_set,
                *[f"{value:.2%}" for value in values],
                f"{row.auc:.4f}",
                row.n_live,
                row.n_spoof,
            ]
        )
    lines += markdown_table(
        [
            "Model",
            "Test set",
            "Accuracy",
            "Print APCER",
            "Replay APCER",
            "BPCER",
            "ACER",
            "EER",
            "BPCER at pooled APCER 5%",
            "AUC",
            "Live N",
            "Spoof N",
        ],
        result_rows,
    )
    lines += [
        "",
        "## Discussion",
        "",
        f"Deployment selection status: `{decision['status']}`. Selection "
        "uses validation ACER, with measured demo-laptop CPU latency as "
        "the tie-break within two percentage points. Test outcomes do "
        "not choose the model or threshold.",
        "",
    ]
    for row in table[table.test_set == "all"].itertuples():
        target_status = "meeting" if row.acer <= 0.1 else "exceeding"
        lines += [
            f"For {row.model}, the hardest attack group was {row.worst_attack}. "
            f"Test ACER was {row.acer:.2%}, {target_status} the proposed 10% "
            "target. This target is an experimental goal, not a guaranteed "
            "assessment requirement.",
            "",
        ]
    timing_file = frozen_path.parent / "cpu_benchmark.json"
    if timing_file.exists():
        timings = json.loads(timing_file.read_text())
        benchmark_rows = [
            [
                candidate["model"],
                timing["median_ms"],
                timing["p95_ms"],
                timing["hostname"],
                timing["threads"],
            ]
            for candidate in decision["candidates"]
            for timing in timings
            if timing["checkpoint_sha256"] == candidate["sha256"]
            and timing["machine_role"] == "demo_laptop"
        ]
        lines += markdown_table(
            ["Model", "CPU median ms", "CPU p95 ms", "Machine", "Threads"],
            benchmark_rows,
        ) + [""]
    lines += [
        "One training seed per model was the minimum requirement. These "
        "public-data results do not establish performance on team members, new "
        "cameras or the live app. Face collection, recorded-data adaptation, "
        "webcam rejection trials and attendance integration are deferred. "
        "Representative error crops support inspection but do not by themselves "
        "establish the visual cause of failures. The laptop benchmark remains "
        "outstanding whenever selection is pending.",
        "",
        "Figures include training curves, model and combined ROC curves, score "
        "distributions, confusion matrices and representative errors. Dataset "
        "faces and derived crops must remain outside the public repository.",
        "",
        "## References",
        "",
        "- Howard et al. Searching for MobileNetV3. ICCV 2019. "
        "https://arxiv.org/abs/1905.02244",
        "- George and Marcel. Deep Pixel-wise Binary Supervision for Face "
        "Presentation Attack Detection. ICB 2019. https://arxiv.org/abs/1907.04047",
        "- Zhang et al. CelebA-Spoof. ECCV 2020. "
        "https://github.com/ZhangYuanhan-AI/CelebA-Spoof",
        "",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(args.fig_dir, args.frozen, args.dataset_summary, args.out))


if __name__ == "__main__":
    main()
