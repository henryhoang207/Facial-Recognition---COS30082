"""Rebuild the output-free Colab notebook and its two model configurations."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def markdown(source):
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
    )


def code(source):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "outputs": [],
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
    )


markdown("""
    # FaceGuard anti spoofing on Google Colab

    **Hoan · PyTorch · MobileNetV3-Large versus DeepPixBiS · public data only**

    This notebook stages the supplied processed CelebA-Spoof dataset, audits the data,
    runs both pilots, trains resumable models, freezes validation-based choices, evaluates
    the held-out test split and exports measured report sections. Face collection and webcam
    integration are deferred. No trained results are included in this source notebook.

    Before starting:
    1. Outputs use the group's [FaceGuard folder](https://drive.google.com/drive/folders/1vOxqufQJaj1zPhzzUZMdHj6VF3vLCnGX)
       at `My Drive/COS30082 - Project/AI/FaceGuard`. Other members can add that folder
       as a shortcut named `FaceGuard` directly under their own My Drive.
    2. The [extracted dataset folder](https://drive.google.com/drive/folders/1D3b5u61grPype3UDh5Jn5phbrX4Bl5Xa)
       is at `My Drive/COS30082 - Project/Data/anti_spoof/CelebA_Spoof_processed`.
       It already contains `celeba_spoof.duckdb` and `Data/train` and `Data/test`.
       Other members can add this folder as a shortcut named `CelebA_Spoof_processed`
       under their own My Drive. Existing ZIP shortcuts are accepted as a fallback.
       If using the backup source ZIP, place it inside the group's FaceGuard folder.
    3. Use a GPU runtime for training. Archive preparation can run on a CPU runtime to conserve GPU time.
    4. Review the crop grid before enabling the full runs. Allow substantial staging/training time;
       many small files must be copied from Drive and free Colab resources vary.
""")
code("""
    from pathlib import Path
    import json, os, subprocess, sys, time, zipfile, shutil, platform, importlib.metadata

    from google.colab import drive
    drive.mount('/content/drive')
    MY_DRIVE = Path('/content/drive/MyDrive')
    GROUP_FOLDER_URL = 'https://drive.google.com/drive/folders/1vOxqufQJaj1zPhzzUZMdHj6VF3vLCnGX'
    DATASET_URL = 'https://drive.google.com/drive/folders/1D3b5u61grPype3UDh5Jn5phbrX4Bl5Xa'

    def resolve_drive_locations(my_drive):
        group_candidates = [my_drive / 'COS30082 - Project/AI/FaceGuard', my_drive / 'FaceGuard']
        group = next((path for path in group_candidates if path.is_dir()), None)
        if group is None:
            raise FileNotFoundError('Open ' + GROUP_FOLDER_URL + ' and add a shortcut named FaceGuard under My Drive, then rerun this cell.')
        filename = 'CelebA_Spoof_processed.zip'
        folder_candidates = [my_drive / 'COS30082 - Project/Data/anti_spoof/CelebA_Spoof_processed',
                             my_drive / 'CelebA_Spoof_processed', group / 'data/CelebA_Spoof_processed']
        dataset_candidates = [*folder_candidates, my_drive / filename, group / 'data' / filename,
                              group / filename, my_drive / 'FaceGuard/data' / filename]
        dataset = next((path for path in dataset_candidates if (
            path.is_file() or (path.is_dir() and (path / 'celeba_spoof.duckdb').is_file()
                              and (path / 'Data').is_dir()))), None)
        if dataset is None:
            raise FileNotFoundError('Open ' + DATASET_URL + ' and add a shortcut named CelebA_Spoof_processed under My Drive. The folder must contain celeba_spoof.duckdb and Data/. Then rerun this cell.')
        return group, dataset

    DRIVE, DATA_SOURCE = resolve_drive_locations(MY_DRIVE)
    SOURCE_ZIP = DRIVE / 'faceguard_colab_source.zip'
    REPO = Path('/content/faceguard')
    REPO_URL = 'https://github.com/henryhoang207/Facial-Recognition---COS30082.git'
    REPO_REF = 'hoan/antispoofing-colab'
    EXPERIMENT = 'antispoofing_public_v1'  # change only when starting a different experiment
    OUT = DRIVE / 'experiments' / EXPERIMENT
    LOCAL_DATA = Path('/content/data/CelebA_Spoof_processed')
    MANIFEST_DIR = OUT / 'manifests'
    FIG_DIR = OUT / 'results'
    for directory in (OUT, MANIFEST_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    def command(*args):
        subprocess.run([str(arg) for arg in args], cwd=REPO, check=True)

    if not REPO.exists():
        if SOURCE_ZIP.exists():
            with zipfile.ZipFile(SOURCE_ZIP) as archive:
                for member in archive.infolist():
                    path = Path(member.filename.replace('\\\\', '/'))
                    if path.is_absolute() or '..' in path.parts or ':' in str(path):
                        raise ValueError('Unsafe source bundle entry')
                archive.extractall(REPO)
        else:
            subprocess.run(['git', 'clone', '--branch', REPO_REF, REPO_URL, str(REPO)], check=True)
    staging_module = REPO / 'src/liveness/archive.py'
    if staging_module.exists() and 'def stage_directory(' not in staging_module.read_text() and (REPO / '.git').exists():
        subprocess.run(['git', 'pull', '--ff-only'], cwd=REPO, check=True)
    if not staging_module.exists():
        raise RuntimeError('This source checkout lacks the Colab workflow; use the supplied source ZIP or feature branch.')
    if 'def stage_directory(' not in staging_module.read_text():
        raise RuntimeError('Replace the backup source ZIP with the updated folder-support version, or use the published feature branch.')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    print('Source:', REPO, '| persistent outputs:', OUT)
    print('Group folder:', GROUP_FOLDER_URL, '| dataset:', DATA_SOURCE)
""")
markdown("""
    ## Environment

    Keep Colab's installed Torch/Torchvision pair. Install only the additional packages.
    Do not apply the CUDA 13 Windows training lockfile here. If the binary compatibility
    check fails, reconnect to a clean supported Colab runtime before staging data.
""")
code("""
    command(sys.executable, '-m', 'pip', 'install', '-q', 'numpy>=1.26', 'pandas>=2.1',
            'scikit-learn>=1.4', 'matplotlib>=3.8', 'Pillow>=10', 'tqdm>=4.66',
            'PyYAML==6.0.3', 'pydantic==2.12.5', 'duckdb==1.4.4', 'opencv-python==4.13.0.92')
    import torch, torchvision, cv2, numpy as np, pandas as pd, yaml
    from torchvision.ops import nms
    nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), 0.5)
    assert hasattr(torch.amp, 'GradScaler') and hasattr(cv2, 'CascadeClassifier')
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                   'cuda': torch.version.cuda, 'gpu_available': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (OUT / 'environment.json').write_text(json.dumps(environment, indent=2))
    (OUT / 'requirements-observed.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    if (REPO / 'SOURCE_REVISION.json').exists():
        shutil.copy2(REPO / 'SOURCE_REVISION.json', OUT / 'source_revision.json')
    else:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=REPO).strip()
        (OUT / 'source_revision.json').write_text(json.dumps({'commit': revision}))
    print(environment)
""")
markdown("""
    ## Stage and audit the processed dataset

    The extracted-folder path skips unzipping. Staging reads the DuckDB catalog,
    selects live/print/replay images and bounding boxes, checks free disk space,
    and copies only those files locally. Completed file copies are reused after
    an interruption in the same runtime. Source files on Drive remain unchanged.
    ZIP inputs still work as a fallback. No evaluation split is silently reduced.
""")
code("""
    if not DATA_SOURCE.exists():
        raise FileNotFoundError(f'Dataset source is unavailable: {DATA_SOURCE}')
    command(sys.executable, '-m', 'src.liveness.archive', '--source', DATA_SOURCE, '--root', LOCAL_DATA)
    shutil.copy2(LOCAL_DATA / 'staging.json', OUT / 'staging.json')
    DATA_CONFIG = OUT / 'dataset_config.yaml'
    data_settings = yaml.safe_load((REPO / 'configs/datasets.yaml').read_text())
    data_settings.update(output_dir=str(MANIFEST_DIR), seed=42, workers=4)
    data_settings['liveness'].update(root=str(LOCAL_DATA), val_fraction=0.15, attacks=['print', 'replay'])
    DATA_CONFIG.write_text(yaml.safe_dump(data_settings, sort_keys=False))
    command(sys.executable, '-m', 'src.datasets.prepare', '--config', DATA_CONFIG, '--only', 'liveness')
    command(sys.executable, '-m', 'src.datasets.validate', '--config', DATA_CONFIG, '--only', 'liveness')
    MANIFEST = MANIFEST_DIR / 'celeba_liveness.csv'
    records = pd.read_csv(MANIFEST, dtype={'subject': str})
    display(records.groupby(['split', 'attack']).size().unstack(fill_value=0))
    expected = {'train': 75780, 'val': 13367, 'test': 49646}
    actual = records.groupby('split').size().to_dict()
    print('Reference counts:', expected, '| actual counts:', actual)
    print('Exclusion log:', MANIFEST_DIR / 'celeba_rejected.csv')
""")
code("""
    import matplotlib.pyplot as plt
    from src.common.face_crop import crop_face
    examples = records.groupby(['split', 'attack']).head(2)
    columns = 6
    figure, axes = plt.subplots((len(examples)+columns-1)//columns, columns,
                                figsize=(15, 3*((len(examples)+columns-1)//columns)), squeeze=False)
    for axis in axes.flat:
        axis.axis('off')
    for axis, row in zip(axes.flat, examples.itertuples()):
        image = cv2.imread(str(LOCAL_DATA / row.path))
        face = crop_face(image, (row.x1, row.y1, row.x2, row.y2), size=224, scale=2.)
        axis.imshow(face[:, :, ::-1])
        axis.set_title(f'{row.split} / {row.attack}')
    figure.tight_layout()
    figure.savefig(FIG_DIR / 'crop_audit.png', dpi=120)
    plt.show()
""")
markdown("""
    Check that each face is centred, correctly scaled, and includes surrounding context.
    Stop and fix incorrect boxes before training. Explain count differences using the rejection
    log (the new audit can exclude corrupt or duplicated images). Then update the two fields below.
""")
code("""
    CROPS_REVIEWED = False  # set True after inspecting the grid above
    COUNTS_EXPLANATION = ''  # required only when actual counts differ from the reference
    if not CROPS_REVIEWED:
        raise RuntimeError('Inspect the crop grid, then set CROPS_REVIEWED=True.')
    if actual != expected and not COUNTS_EXPLANATION.strip():
        raise RuntimeError('Explain the split-count differences from the audit log before training.')
    (OUT / 'data_review.json').write_text(json.dumps({'crop_reviewed': True, 'actual': actual,
          'reference': expected, 'count_explanation': COUNTS_EXPLANATION}, indent=2))
""")
markdown("""
    ## Reusable training and recovery

    On a new session, rerun setup and staging with the same EXPERIMENT. Completed jobs are reused
    and interrupted jobs resume from their most recent `last.pt`. Every epoch writes recovery
    state to Drive. A reconnect can lose the current epoch. Keep the same planned epoch count
    when resuming a cosine schedule. No padding, synthetic images or pretrained anti-spoofing
    checkpoints are used for final results.
""")
code("""
    if not torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > GPU before training, then rerun setup.')
    from src.liveness.config import defaults, LivenessConfig

    def settings(model, key, pilot=False, lr_scale=1.):
        config = defaults(model)
        config.update(manifest=[str(MANIFEST)], data_root=str(LOCAL_DATA),
                      output_dir=str(OUT / 'runs' / key), fig_dir=str(FIG_DIR / key))
        config['training'].update(device='cuda', lr=config['training']['lr']*lr_scale)
        if pilot:
            config.update(max_train=2000, max_val=600)
            config['training'].update(epochs=1)
        return config

    def run_job(key, config):
        config = LivenessConfig.model_validate(config).model_dump(mode='json')
        root = Path(config['output_dir'])
        root.mkdir(parents=True, exist_ok=True)
        for export in sorted(root.glob('*/export.json'), reverse=True):
            previous = json.loads((export.parent / 'resolved_config.json').read_text())
            if previous != config:
                raise ValueError('Completed run has different settings. Choose a new EXPERIMENT.')
            return Path(json.loads(export.read_text())['checkpoint'])
        if (OUT / 'frozen_candidates.json').exists():
            raise RuntimeError('Candidates are frozen. New training is disabled for this experiment.')
        configs = OUT / 'configs'
        configs.mkdir(exist_ok=True)
        path = configs / (key + '.yaml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        recoveries = sorted(root.glob('*/last.pt'))
        arguments = [sys.executable, '-m', 'src.liveness.train', '--config', path]
        if recoveries:
            arguments += ['--resume', recoveries[-1]]
        started = time.perf_counter()
        try:
            command(*arguments)
        finally:
            with (OUT / 'invocations.jsonl').open('a') as stream:
                stream.write(json.dumps({'job': key, 'elapsed_seconds': time.perf_counter()-started})+'\\n')
        export = sorted(root.glob('*/export.json'))[-1]
        return Path(json.loads(export.read_text())['checkpoint'])
""")
markdown("""
    ## Pilot both models

    Each pilot uses fixed stratified subsets of 2,000 training and 600 validation images.
    Inspect finite losses, crop/label correctness, memory use and time before the full runs.
    Pilots cannot be selected as final checkpoints.
""")
code("""
    for model in ['mobilenetv3', 'deeppixbis']:
        key = 'pilot_' + model
        pilot = run_job(key, settings(model, key, pilot=True))
        display(pd.read_csv(pilot.parent / 'history.csv'))
        profile = json.loads((pilot.parent / 'runtime.json').read_text())
        elapsed = json.loads((pilot.parent / 'summary.json').read_text())['seconds_this_invocation']
        multiplier = max(actual['train']/profile['training_examples'], actual['val']/profile['validation_examples'])
        print('Peak allocated GPU GiB:', (profile['gpu_peak_allocated_bytes'] or 0)/1024**3)
        print('Rough 15-epoch fit estimate in hours:', round(elapsed*multiplier*15/3600, 2))
        print('Estimate excludes staging and is approximate; early stopping can shorten it.')
        print('Pilot checkpoint:', pilot)
""")
markdown("""
    ## Full training and bounded validation-only tuning

    Both candidates use all eligible training data, the same validation subjects and seed 42.
    MobileNet uses microbatch 64; DeepPixBiS uses 32 with two-step accumulation. If the initial
    validation ACER exceeds 10%, inspect the errors/crops before enabling its single lower-LR
    run. Test scores are not available at this stage and never determine tuning.
""")
code("""
    base_checkpoints = {}
    for model in ['mobilenetv3', 'deeppixbis']:
        key = 'full_' + model
        base_checkpoints[model] = run_job(key, settings(model, key))
        checkpoint = torch.load(base_checkpoints[model], map_location='cpu', weights_only=True)
        print(model, 'validation ACER:', checkpoint['val_metrics']['val_acer'])
""")
code("""
    VALIDATION_AUDIT_COMPLETE = False  # enable only after checking any high-validation-error run
    candidate_checkpoints = list(base_checkpoints.values())
    for model, checkpoint_path in base_checkpoints.items():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['val_metrics']['val_acer'] > .10 and VALIDATION_AUDIT_COMPLETE:
            key = 'tune_' + model
            candidate_checkpoints.append(run_job(key, settings(model, key, lr_scale=1/3)))
    print('Candidates ready:', candidate_checkpoints)
""")
markdown("""
    ## Freeze before looking at the test split

    Keep the best validation run for each architecture. Within two percentage points, the
    final deployment choice waits for the real demo-laptop CPU benchmark; Colab CPU results
    cannot replace it. Both frozen candidates can still receive their final test evaluation.
""")
code("""
    from src.liveness.select import select_candidates
    FROZEN = OUT / 'frozen_candidates.json'
    decision = select_candidates(candidate_checkpoints)
    if FROZEN.exists():
        previous = json.loads(FROZEN.read_text())
        if {item['sha256'] for item in previous['candidates']} != {item['sha256'] for item in decision['candidates']}:
            raise RuntimeError('Candidates were already frozen. Do not retune this experiment after testing.')
        decision = previous
    else:
        FROZEN.write_text(json.dumps(decision, indent=2))
    print(decision['status'])
""")
markdown("""
    ## Final held-out test evaluation

    The evaluator verifies the frozen checkpoint hash, actual image hashes and portable
    subject provenance. It uses each checkpoint's saved validation threshold unchanged.
    Outputs include per-attack APCER, BPCER, ACER, accuracy, AUC, ROC diagnostics and figures.
""")
code("""
    for candidate in decision['candidates']:
        command(sys.executable, '-m', 'src.liveness.evaluate', '--checkpoint', candidate['checkpoint'],
                '--manifest', MANIFEST, '--data-root', LOCAL_DATA, '--fig-dir', FIG_DIR,
                '--frozen', FROZEN, '--num-workers', '2')
    display(pd.read_csv(FIG_DIR / 'liveness_results.csv'))
""")
markdown("""
    ## Export report sections and a small laptop benchmark pack

    The report is populated from real measurements and includes outstanding limitations.
    The benchmark pack contains only a few validation images; keep it private with your
    checkpoints. Its images must not enter the public source repository.
""")
code("""
    command(sys.executable, '-m', 'src.liveness.report', '--fig-dir', FIG_DIR, '--frozen', FROZEN,
            '--dataset-summary', MANIFEST_DIR / 'liveness_summary.json', '--out', OUT / 'liveness_report.md')
    benchmark_records = records[records.split == 'val'].groupby('attack').head(3)
    with zipfile.ZipFile(OUT / 'benchmark_inputs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as pack:
        pack.writestr('manifest.csv', benchmark_records.to_csv(index=False))
        for row in benchmark_records.itertuples():
            pack.write(LOCAL_DATA / row.path, row.path)
    from IPython.display import Markdown, display
    display(Markdown((OUT / 'liveness_report.md').read_text()))
    print('All artifacts saved in:', OUT)
""")
markdown("""
    ## Laptop handoff

    Download the two exported `liveness_*.pt` files and `benchmark_inputs.zip`, extract the
    pack, and run from the implemented repository on your laptop (CPU):

    ```text
    python -m src.liveness.benchmark --checkpoint MODEL1.pt MODEL2.pt --manifest benchmark_inputs/manifest.csv --data-root benchmark_inputs --out cpu_benchmark.json
    ```

    Upload `cpu_benchmark.json` to this experiment folder. The optional cell below updates
    only the validation/latency decision, preserving the original frozen checkpoint set.
    Forward the selected checkpoint and `LivenessPredictor.predict_frame(frame, box)` usage
    to the app owner. Webcam and attendance tests remain deferred.

    References: [MobileNetV3](https://arxiv.org/abs/1905.02244),
    [DeepPixBiS](https://arxiv.org/abs/1907.04047),
    [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof),
    [Colab resource and storage guidance](https://research.google.com/colaboratory/faq.html).
""")
code("""
    timing_file = OUT / 'cpu_benchmark.json'
    if timing_file.exists():
        updated = select_candidates(candidate_checkpoints, json.loads(timing_file.read_text()))
        if {item['sha256'] for item in updated['candidates']} != {item['sha256'] for item in decision['candidates']}:
            raise ValueError('Benchmark update must keep the original frozen candidate set.')
        updated['created_utc'] = decision['created_utc']
        FROZEN.write_text(json.dumps(updated, indent=2))
        decision = updated
        command(sys.executable, '-m', 'src.liveness.report', '--fig-dir', FIG_DIR, '--frozen', FROZEN,
                '--dataset-summary', MANIFEST_DIR / 'liveness_summary.json', '--out', OUT / 'liveness_report.md')
        print('Deployment decision:', decision['status'], decision['selected_sha256'])
    else:
        print('Laptop benchmark not uploaded yet. Current decision:', decision['status'])
""")


def main():
    for index, cell in enumerate(CELLS):
        cell["id"] = f"liveness-{index:03d}"
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "liveness_training.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "notebooks/liveness_training.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
