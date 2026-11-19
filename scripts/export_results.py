"""Write docs/results/runs.jsonl: one line per scored run, from results/*/summary.json.

    uv run python scripts/export_results.py results docs/results/runs.jsonl

Tracking-database ids are dropped, and model paths are cut down to their last three parts.
"""

import json
import sys
from pathlib import Path

PRIVATE = ("train_run_id", "train_experiment_id")


def public(value):
    """A path with no machine-specific prefix; anything else unchanged."""
    if isinstance(value, str) and value.startswith("/"):
        return "/".join(Path(value).parts[-3:])
    return value


def main(results: str, out: str) -> None:
    lines = []
    for path in sorted(Path(results).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        row = {"run": path.parent.name} | {k: public(v) for k, v in summary.items() if k not in PRIVATE}
        lines.append(json.dumps(row))
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} runs -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
