"""Report frozen MLP and baseline decisions on labeled webcam attempts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .prepare import rows


def summarize(paths: list[Path]) -> dict:
    seen, sessions, signatures = set(), set(), set()
    counts = Counter()
    decisions = {method: Counter() for method in ("mlp", "cosine")}
    for path in paths:
        for row in rows(path):
            kind = row["kind"]
            if kind not in {"correct", "wrong_known", "unknown"}:
                raise ValueError(f"invalid attempt kind: {kind}")
            if (
                not row["session"]
                or not row["subject"]
                or not row["attempt_id"]
                or row["attempt_id"] in seen
                or row["label"] != str(int(kind == "correct"))
                or row["is_enrolled"] != str(int(kind != "unknown"))
                or (
                    kind != "unknown"
                    and (row["subject"] == row["candidate"]) != (kind == "correct")
                )
            ):
                raise ValueError(
                    f"duplicate attempt or invalid label: {row['attempt_id']}"
                )
            seen.add(row["attempt_id"])
            sessions.add(row["session"])
            signatures.add(
                tuple(
                    row[key]
                    for key in (
                        "encoder_sha256",
                        "mlp_sha256",
                        "gallery_sha256",
                        "frame_count",
                        "gallery_size",
                        "mlp_threshold",
                        "cosine_threshold",
                        "minimum_consensus",
                    )
                )
            )
            counts[kind] += 1
            for method, accepted in decisions.items():
                value = row[f"{method}_accepted"]
                if value not in {"true", "false"}:
                    raise ValueError(f"invalid decision: {value}")
                accepted[kind] += int(value == "true")
    if len(signatures) != 1:
        raise ValueError(
            "supply nonempty CSVs recorded with the same gallery and model settings"
        )
    enrolled = counts["correct"] + counts["wrong_known"]
    return {
        "attempts": len(seen),
        "sessions": len(sessions),
        "attempt_types": dict(counts),
        "methods": {
            method: {
                "correct_accepted": accepted["correct"],
                "wrong_known_accepted": accepted["wrong_known"],
                "unknown_accepted": accepted["unknown"],
                "enrolled_attempts": enrolled,
                "unknown_attempts": counts["unknown"],
                "correct_accept_rate_enrolled": (
                    accepted["correct"] / enrolled if enrolled else None
                ),
                "wrong_known_accept_rate_enrolled": (
                    accepted["wrong_known"] / enrolled if enrolled else None
                ),
                "unknown_accept_rate": (
                    accepted["unknown"] / counts["unknown"]
                    if counts["unknown"]
                    else None
                ),
            }
            for method, accepted in decisions.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    print(json.dumps(summarize(parser.parse_args().csv), indent=2))
