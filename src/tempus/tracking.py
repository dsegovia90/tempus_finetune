"""Log runs to MLflow so results can be compared over time.

Eval runs go to the experiment `eval`, training runs to `train`. The store is a SQLite file
in the project folder unless MLFLOW_TRACKING_URI is set.
View it with: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_URI = "sqlite:///mlflow.db"
SCALARS = (
    "seconds", "latency_ms_p50", "latency_ms_p95", "parameters_millions", "weights_gb", "peak_vram_gb",
    "server_prompt_ms_p50", "server_generation_ms_p50", "server_prompt_tokens_p50",
    "server_generated_tokens_p50", "truncated",
)

GLOSSARY = """\
Each run scores one model on the eval set. Metric names are `metric/slice`.

**Metrics** (all are shares of messages, 0 to 1, except precision and recall)

| Metric | Meaning |
|---|---|
| `exact` | The model's answer is completely right: every time found, right span, right time, right order. The headline number. |
| `valid` | The reply is in the required JSON format (whether or not the content is right). |
| `fenced` | The reply was wrapped in Markdown backticks. Tolerated, but a finetune should score 0. |
| `grounded` | Every span the model returned really appears in the message. |
| `precision` | Of the times the model returned, the share that are correct. |
| `recall` | Of the times it should have found, the share it found. |
| `seconds` | Wall time to answer the whole eval set: in batches of 16 with transformers, one message at a time with llama.cpp. |
| `latency_ms_p50` | Median time to answer one message alone, in milliseconds, measured by the client. The response-time number. |
| `latency_ms_p95` | 95th percentile of the same: the slow end. |
| `parameters_millions` | Model size: number of weights, in millions. |
| `weights_gb` | Size of the weights in GB: GPU memory they occupy with transformers, the GGUF file with llama.cpp. |
| `peak_vram_gb` | Most GPU memory used while scoring (batches of 16), in GB. Transformers only. |
| `server_prompt_ms_p50` | llama.cpp only: median time the server spent reading the prompt, in ms (same messages as `latency_ms_p50`). |
| `server_generation_ms_p50` | llama.cpp only: median time the server spent generating the reply, in ms. The rest of the latency is client and HTTP. |
| `server_prompt_tokens_p50` | llama.cpp only: median prompt length, in tokens. |
| `server_generated_tokens_p50` | llama.cpp only: median reply length, in tokens, end-of-turn included. |
| `truncated` | llama.cpp only: replies cut off at the token cap instead of ending their turn (a count, ideally 0). |

**Slices** (kinds of message)

| Slice | Example | What it tests |
|---|---|---|
| `all` | | Every message. `exact/all` must reach 0.98 to stop the ladder. |
| `implicit` | "dinner at 7" | World knowledge decides am or pm. `exact/implicit` must reach 0.95. |
| `cue` | "3 in the afternoon" | A part-of-day word decides am or pm. |
| `explicit` | "3pm", "15:00" | The time states am or pm itself. |
| `range` | "2 to 4pm" | Two times, often sharing one am/pm marker. |
| `fallback` | "at 3?" | No context at all: am, and a bare 12 is noon. |
| `none` | "back in 5", "room 3" | No clock time; the right answer is an empty list. |

**Parameters**: `eval_set` names the messages a run was scored on: `core+eval` is the main
530-message set; `eval2` is the second set, built from scenarios no model or decision has seen.
`base_model` is the published model a run started from, so a finetune and its
untrained baseline share the same value (group or filter by it to compare them). `kind` is `base` (untrained model) or `finetune` (trained model); `with_spec` says whether the
labelling spec was given as a system prompt (normally only to base models). `train_run_id` is the run in the `train`
experiment that produced a finetune; its description links there. `format` is the output format the model
writes: `json`, or `compact` (one `text=HH:MM` per line, about half the tokens). `engine` is what
ran the model: `transformers` (plain `generate`) or `llama.cpp` (`llama-server`, given the exact
same prompt tokens). `quant` is the weight format: `bf16`, or the GGUF quantizations `Q8_0` (8-bit)
and `Q4_K_M` (about 4.5-bit). `device` is `gpu` (every layer on the GPU) or `cpu` (no GPU at all,
4 threads). `grammar` is true when a GBNF grammar forced the reply into the output format.
"""


def log_eval(name: str, summary: dict, out: Path, eval_set: str, group: str | None = None) -> None:
    """Record one scored model: its settings, per-slice metrics and result files, tagged `group`."""
    mlflow = _connect()
    experiment = mlflow.set_experiment("eval")
    # The experiment description shown at the top of the MLflow page.
    mlflow.MlflowClient().set_experiment_tag(experiment.experiment_id, "mlflow.note.content", GLOSSARY)
    kind = summary.get("kind") or ("base" if summary["with_spec"] else "finetune")
    prompt = "prompted with the labelling spec" if summary["with_spec"] else "message only"
    description = f"{'Base model' if kind == 'base' else 'Finetune'}, {prompt}. Model: {summary['model']}."
    if summary.get("gguf"):
        description += f" Served by llama.cpp from {summary['gguf']}."
    if summary.get("train_run_id"):
        link = f"#/experiments/{summary['train_experiment_id']}/runs/{summary['train_run_id']}"
        description += f" [Training run]({link})."
    with mlflow.start_run(run_name=name, description=description, tags={"group": group} if group else None):
        mlflow.log_params(
            {
                "model": summary["model"],
                "base_model": summary.get("base_model", summary["model"]),
                "kind": kind,
                "with_spec": summary["with_spec"],
                "format": summary.get("format", "json"),
                "examples": summary["examples"],
                "eval_set": eval_set,
                "train_run_id": summary.get("train_run_id") or "none",
                "engine": summary.get("engine", "transformers"),
                "quant": summary.get("quant", "bf16"),
                "device": summary.get("device", "gpu"),
                "grammar": summary.get("grammar", False),
            }
        )
        for key in SCALARS:
            if key in summary:
                mlflow.log_metric(key, summary[key])
        for slice_name, metrics in summary["metrics"].items():
            for metric, value in metrics.items():
                if metric != "n":
                    mlflow.log_metric(f"{metric}/{slice_name}", value)
        mlflow.log_artifacts(str(out))


@contextmanager
def train_run(name: str, params: dict) -> Iterator[dict]:
    """Open a run in the `train` experiment, record its settings, and yield its MLflow ids."""
    mlflow = _connect()
    mlflow.set_experiment("train")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        yield {"mlflow_run_id": run.info.run_id, "mlflow_experiment_id": run.info.experiment_id}


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log numbers to the active run."""
    import mlflow

    mlflow.log_metrics(metrics, step=step)


def metrics_callback():
    """A Trainer callback that logs every training and validation report to the active run."""
    from transformers import TrainerCallback

    class LogToMlflow(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            numbers = {k: v for k, v in logs.items() if isinstance(v, int | float)}
            log_metrics(numbers, step=state.global_step)

    return LogToMlflow()


def _connect():
    import mlflow  # imported here so the rest of the CLI works without it

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI))
    return mlflow
