"""Record scored Needle runs in MLflow, from the project venv (which has mlflow).

    .venv/bin/python experiments/needle/log.py needle3-20l-v2 needle3-20l-v2-eval2
"""

import argparse
import json
from pathlib import Path

from tempus import tracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="folders under results/ written by run.py")
    parser.add_argument("--group", default="ladder")
    args = parser.parse_args()
    for name in args.names:
        out = Path("results") / name
        summary = json.loads((out / "summary.json").read_text())
        tracking.log_eval(name, summary, out, summary["eval_set"], args.group)
        print(f"logged {name}")


if __name__ == "__main__":
    main()
