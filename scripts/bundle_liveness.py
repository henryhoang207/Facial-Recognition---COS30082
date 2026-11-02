"""Package the notebook and its source without datasets, weights or credentials."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".ipynb"}


def bundle(output: Path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    files = sorted(
        {
            name
            for name in names
            if name and (Path(name).suffix in TEXT_SUFFIXES or name == ".gitignore")
        }
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {name: (ROOT / name).read_bytes() for name in files}
    descriptor = {
        "commit": revision,
        "files_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payload.items()
        },
    }
    destination = output / "faceguard_colab_source.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
        archive.writestr("SOURCE_REVISION.json", json.dumps(descriptor, indent=2))
    shutil.copy2(
        ROOT / "notebooks/liveness_training.ipynb", output / "liveness_training.ipynb"
    )
    (output / "START_HERE.md").write_text(
        "# Run FaceGuard training on Google Colab\n\n"
        "1. Use the group folder My Drive/COS30082 - Project/AI/FaceGuard. "
        "Other members can add it as a shortcut named FaceGuard under My Drive. "
        "Place the backup source ZIP there only if not using the published branch.\n"
        "2. The extracted dataset is at My Drive/COS30082 - Project/Data/"
        "anti_spoof/CelebA_Spoof_processed. Other members can add that folder "
        "as a shortcut named CelebA_Spoof_processed under My Drive. "
        "No ZIP download or extraction is needed.\n"
        "3. Upload liveness_training.ipynb to Google Colab.\n"
        "4. Run the setup and data audit; inspect the crop grid.\n"
        "5. Select a GPU runtime, then run the pilots and full training.\n\n"
        "Checkpoints, logs, evaluation plots and measured report sections will "
        "be saved in the group FaceGuard/experiments/antispoofing_public_v1/ folder. "
        "Rerunning the same experiment recovers interrupted jobs and reuses "
        "completed ones. The notebook explains the laptop benchmark handoff.\n\n"
        "This package contains implemented code, not trained final models. "
        "Real-data results are produced by running the notebook. Face collection "
        "and live webcam integration are deferred.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_zip": str(destination),
                "files": len(files),
                "commit": revision,
                "bytes": destination.stat().st_size,
            }
        )
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    bundle(parser.parse_args().out)
