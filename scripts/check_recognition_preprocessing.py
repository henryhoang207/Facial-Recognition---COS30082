"""Reproduce current preprocessing integration issues without changing raw code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".venv" / "research-deps"), str(ROOT / "data" / "VGGFace2")]
# Existing notebook uses this workaround. This diagnostic does not change the
# user's Python installation and does not establish a training-ready environment.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import duckdb
from PIL import Image
import processing_pipeline as pipeline


def main():
    database = ROOT / "data" / "VGGFace2" / "vggface2.duckdb"
    results = {}
    image = Image.new("RGB", (160, 180), (140, 110, 90))
    for erase in [False, True]:
        pipeline.random.seed(42)
        pipeline.torch.manual_seed(42)
        transform = pipeline.VGGFace2Transform(is_train=True, random_erasing=erase)
        try:
            with patch.object(pipeline.random, "random", return_value=0.1):
                value = transform(image)
            result = {"passed": True, "shape": list(value.shape)}
        except Exception as exc:
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        results[f"transform_random_erasing_{erase}"] = result
    loader = pipeline.build_dataloader(database, architecture="resnet", sample_limit=200)
    results["resnet_loader_default"] = {"image_size": loader.dataset.transform.img_size,
                                         "normalization": loader.dataset.transform.norm_type,
                                         "mean": loader.dataset.transform.norm_mean,
                                         "aspect_mode": loader.dataset.transform.aspect_mode,
                                         "unique_identities_in_first_200": loader.dataset.num_classes}
    with duckdb.connect(str(database), read_only=True) as connection:
        results["smoke_test_identity_coverage"] = {str(limit): connection.execute(f"SELECT count(DISTINCT identity_id) FROM (SELECT identity_id FROM vggface2_images WHERE split='train' ORDER BY image_id LIMIT {limit})").fetchone()[0] for limit in [100, 200, 1000]}
    results["original_artifact_sha256"] = {filename: hashlib.sha256((database.parent / filename).read_bytes()).hexdigest() for filename in ["processing_pipeline.py", "vggface2.ipynb", "vggface2.duckdb"]}
    output = ROOT / "reports" / "recognition" / "research" / "preprocessing_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
