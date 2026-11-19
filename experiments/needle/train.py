"""Finetune Needle 3 with the vendor's CLI and build one archive per depth.

Runs `cactus finetune` (LoRA, vendor defaults unless given) and `cactus build --layers N`,
logs both to runs/<name>/train.log, samples GPU memory, and writes runs/<name>/run.json with
the settings, wall times, peak GPU memory, the loss curve and archive sizes.

    CACTUS_TELEMETRY=0 DO_NOT_TRACK=1 python train.py --data needle-data/train-v2-reasoning.jsonl \\
        --checkpoint base/checkpoints/needle3.safetensors --name s0 --seed 0 --layers 20 8
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

CACTUS = str(Path(sys.executable).with_name("cactus"))
UNSCORED = Path(__file__).with_name("finetune_unscored.py")
LINE = re.compile(r"^\s*(step|epoch)\s+(\d+)/\d+\s+loss ([\d.]+)(?:\s+val ([\d.]+))?")


class GpuPeak(threading.Thread):
    """Samples used GPU memory (MiB) every few seconds until stopped."""

    def __init__(self, every: float = 2.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stopped = every, 0, threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(self.every):
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout
            self.peak = max(self.peak, int(used.split()[0]))


def logged(command: list[str], log: Path) -> tuple[float, list[str]]:
    """Run a command, appending its output to `log`; return wall seconds and output lines."""
    started, lines = time.monotonic(), []
    with log.open("a") as sink:
        sink.write(f"$ {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sink.write(line)
            sink.flush()
            lines.append(line.rstrip())
        if process.wait():
            raise SystemExit(f"failed ({process.returncode}): {' '.join(command)}, see {log}")
    return round(time.monotonic() - started, 1), lines


def losses(lines: list[str]) -> dict[str, list]:
    """Training loss per logged step, and training and validation loss per epoch."""
    curve = {"step": [], "epoch": []}
    for line in lines:
        if match := LINE.match(line):
            kind, number, loss, val = match.groups()
            row = [int(number), float(loss)] + ([float(val)] if val else [])
            curve[kind].append(row)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="base needle3.safetensors")
    parser.add_argument("--name", required=True, help="written to runs/<name>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3, help="the CLI's default")
    parser.add_argument("--lr", type=float, default=1e-4, help="the CLI's default")
    parser.add_argument("--lora-rank", type=int, default=16, help="the CLI's default")
    parser.add_argument("--layers", type=int, nargs="+", default=[20], help="one archive per depth")
    parser.add_argument(
        "--no-vendor-score", action="store_true",
        help="skip finetune's closing generation pass over the validation split (finetune_unscored.py)",
    )
    args = parser.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    log, adapter = out / "train.log", out / "adapter.safetensors"
    command = [sys.executable, str(UNSCORED)] if args.no_vendor_score else [CACTUS, "finetune"]
    finetune = [
        *command, str(args.data), "--checkpoint", str(args.checkpoint),
        "--epochs", str(args.epochs), "--lr", str(args.lr), "--lora-rank", str(args.lora_rank),
        "--seed", str(args.seed),
        "--checkpoint-dir", str(out), "--out", str(adapter),
    ]
    gpu = GpuPeak()
    gpu.start()
    train_seconds, lines = logged(finetune, log)
    gpu.stopped.set()
    archives = {}
    for layers in args.layers:
        archive = out / f"{layers}l.cact"
        build = [
            CACTUS, "build", str(args.checkpoint), "--lora", str(adapter),
            "--layers", str(layers), "--out", str(archive),
        ]
        seconds, _ = logged(build, log)
        archives[layers] = {"path": str(archive), "bytes": archive.stat().st_size, "build_seconds": seconds}
    record = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "vendor_score": not args.no_vendor_score,
        "train_seconds": train_seconds,
        "peak_gpu_mib": gpu.peak,
        "held_out": next((line.strip() for line in lines if line.strip().startswith("accuracy")), None),
        "loss": losses(lines),
        "archives": archives,
    }
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "loss"}, indent=1))


if __name__ == "__main__":
    main()
