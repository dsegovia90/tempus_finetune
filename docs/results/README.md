# Per-run results

`runs.jsonl` has one line per scored model run: its settings and its metrics, exported from each
run's summary file. It covers the runs scored on the training machine. It does not hold the
rule-based baseline, the Claude Opus reference, the 8-bit classifier scores quoted on its model
card, or the timings taken on a Raspberry Pi and on a Mac. The `tag-bert-mini-v2-s1` to `-s3`
rows were scored on an Apple M3 Pro, so their latency is that machine's; the `-int8-`, `-fp16-`
and `-gpu-` tagger rows were scored on the training machine. Fields:

- `run`: the run's name. Suffixes: none = main eval set (530 messages), `-eval2` = the fresh
  set (378), `-neg` = the hard-negatives set (1,021 messages with no clock time).
- `model`, `base_model`, `format`, `engine`, `quant`, `device`, `eval_set`: how it was scored.
- `examples`, `seconds`, `latency_ms_p50`, `latency_ms_p95`: size of the set, time for the
  whole set, and per-message latency.
- `metrics`: per slice (`all`, `implicit`, `cue`, `explicit`, `range`, `fallback`, `none`):
  `exact`, `valid`, `fenced`, `grounded`, `precision`, `recall`. On the hard-negatives set,
  `1 - exact` for `all` is the false-alarm rate.

Per-message replies are not included.

`kind` is present only where it differs from the usual rule (`with_spec` true = untrained base
model given the labelling spec, false = finetune given only the message). Latency fields of
the `needle3-*` runs were mostly measured while training ran on the same machine; use the
clean figures in the main README instead. Regenerate the file with
`uv run python scripts/export_results.py results docs/results/runs.jsonl`, which reads the `results/` folder
that `tempus eval` writes when you score your own runs (it is not in the repository).
