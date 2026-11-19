// Eval mode: every eval message through the engine, compared with the engine's reference replies
// (native llama.cpp for the GGUF models, the Python pipeline for the tagger).

const WARMUP = 3; // untimed calls first, as the native latency runs do

// Linear interpolation between ranks, as numpy's default and Python's statistics.median.
const percentile = (values, p) => {
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (p / 100) * (sorted.length - 1);
  const low = Math.floor(rank);
  return sorted[low] + (sorted[Math.min(low + 1, sorted.length - 1)] - sorted[low]) * (rank - low);
};

/** Each message is sent exactly as the demo sends it; `engine.check` adds engine-specific problems. */
export async function runEval(engine, { limit, onProgress } = {}) {
  const { reference, items: all } = await engine.evalSet();
  const items = all.slice(0, limit || all.length);
  for (const item of items.slice(0, WARMUP)) await engine.complete(item.text);

  const rows = [];
  const started = performance.now();
  for (const item of items) {
    rows.push({ item, out: await engine.complete(item.text) });
    onProgress?.(rows.length, items.length);
  }

  const ms = rows.map((r) => r.out.ms);
  const differ = rows.filter((r) => r.out.reply !== r.item.expected);
  return {
    reference,
    n: rows.length,
    matches: rows.length - differ.length,
    mismatches: differ.map((r) => ({ id: r.item.id, text: r.item.text, expected: r.item.expected, browser: r.out.reply })),
    problems: rows.flatMap((r) => engine.check(r.item, r.out).map((problem) => ({ id: r.item.id, problem }))),
    restarts: engine.restarts,
    seconds: (performance.now() - started) / 1000,
    msMedian: percentile(ms, 50),
    msP95: percentile(ms, 95),
    msMean: ms.reduce((a, b) => a + b, 0) / ms.length,
  };
}
