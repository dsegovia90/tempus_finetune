// The model-only tagger (src/tempus/tagger.py) in ONNX Runtime Web. A small encoder tags every
// token as outside a time, beginning one or continuing one, and at each beginning token gives
// the hour and the minute (in 5-minute steps). No code finds numbers or reads digits; code only
// turns the tags back into spans of the message and prints the reply.
import * as ort from '../vendor/onnxruntime-web/ort.wasm.min.mjs';
import { WordPiece } from '../tagger/wordpiece.js';
import { locate } from './source.js';

const here = (path) => new URL(path, import.meta.url).href;
const CACHE = 'tempus-tagger'; // Cache Storage, where the browser offers it (secure contexts)
const [OUTSIDE, BEGIN, INSIDE] = [0, 1, 2];
const TAGS = ['outside', 'begins', 'continues'];
const WIDTHS = { tag: 3, hour: 24, minute: 12 }; // classes per output
const MINUTE_STEP = 5;
const RAW_TITLE = 'Model output: its top choice for each token (the times above are assembled from these by code)';

ort.env.wasm.wasmPaths = here('../vendor/onnxruntime-web/');
ort.env.wasm.numThreads = 1; // one short message through a tiny model: worker threads cost more than they save

export class Tagger {
  /** Where the weights will come from, and whether this browser already holds them. */
  static async source({ repo, variant }) {
    const found = await locate({ repo, file: `${variant}.onnx`, local: `tagger/${variant}.onnx` });
    const cached = !!(self.caches && (await (await caches.open(CACHE)).match(found.url)));
    return { ...found, cached };
  }

  /** `url` is the export (from `Tagger.source`), `variant` its name (int8, fp16); `fresh` empties the cache first. */
  static async load({ url, repo, variant, fresh = false, onProgress }) {
    if (fresh && self.caches) await caches.delete(CACHE);
    const started = performance.now();
    const vocabulary = await locate({ repo, file: 'vocab.txt', local: 'tagger/vocab.txt' });
    const [model, vocab] = await Promise.all([download(url, onProgress), download(vocabulary.url)]);
    const engine = new Tagger(variant);
    engine.tokenizer = new WordPiece(new TextDecoder().decode(vocab.bytes).split('\n'));
    engine.session = await ort.InferenceSession.create(model.bytes, { executionProviders: ['wasm'] });
    engine.info = {
      variant,
      modelBytes: model.bytes.byteLength,
      vocabBytes: vocab.bytes.byteLength,
      fromCache: model.fromCache,
      loadMs: performance.now() - started,
      onnxruntime: ort.env.versions.web,
      threads: ort.env.wasm.numThreads,
      isolated: self.crossOriginIsolated,
    };
    return engine;
  }

  constructor(variant) {
    this.variant = variant;
  }

  /** Status line parts. */
  describe() {
    const { info } = this;
    return [
      `Backend: ONNX Runtime Web ${info.onnxruntime} (WASM, ${info.threads} thread)`,
      `model: ${(info.modelBytes / 2 ** 20).toFixed(1)} MB ${info.variant}`,
      `loaded in ${(info.loadMs / 1000).toFixed(2)} s${info.fromCache ? ' (cached)' : ''}`,
    ];
  }

  /**
   * The reply, which code assembles, and for the raw box what the model itself gave: every token
   * with its highest-scoring tag and, where a time begins, its highest-scoring hour and minute.
   */
  async complete(message) {
    const started = performance.now();
    const { ids, offsets } = this.tokenizer.encode(message);
    const shape = [1, ids.length];
    const out = await this.session.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), shape),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(ids.length).fill(1n), shape),
    });
    const [tags, hours, minutes] = ['tag', 'hour', 'minute'].map((name) => argmax(out[name].data, WIDTHS[name]));
    const reply = decode(message, offsets, tags, hours, minutes);
    const ms = performance.now() - started;
    const tokens = offsets.map(([start, end], i) => ({ text: message.slice(start, end), tag: tags[i], hour: hours[i], minute: minutes[i] * MINUTE_STEP }));
    return { reply, raw: rawText(tokens), rawTitle: RAW_TITLE, ms, ids };
  }

  /** The eval messages with the Python pipeline's replies and token ids for this export (from export_tagger.py). */
  evalSet() {
    return fetch(here(`../eval/tagger-${this.variant}.json`)).then((r) => r.json());
  }

  /** The tokens must be the ones Python made. */
  check(item, out) {
    return JSON.stringify(out.ids) === JSON.stringify(item.ids) ? [] : [`token ids differ from Python's`];
  }
}

/** The class with the highest score for each token, from scores laid out token by token. */
function argmax(scores, width) {
  const best = [];
  for (let at = 0; at < scores.length; at += width) {
    let top = 0;
    for (let k = 1; k < width; k++) if (scores[at + k] > scores[at + top]) top = k;
    best.push(top);
  }
  return best;
}

/** As `decode` in tagger.py: a span runs from a beginning token over the continuing tokens after it. */
function decode(message, offsets, tags, hours, minutes) {
  const clock = (h, m) => `${String(h).padStart(2, '0')}:${String(m * MINUTE_STEP).padStart(2, '0')}`;
  const spans = [];
  let open = null;
  offsets.forEach(([start, end], i) => {
    if (start === end) return; // [CLS], [SEP]
    if (tags[i] === BEGIN) spans.push((open = { start, end, clock: clock(hours[i], minutes[i]) }));
    else if (tags[i] === INSIDE && open) open.end = end;
    else open = null;
  });
  return spans.map((s) => `${message.slice(s.start, s.end)}=${s.clock}`).join('\n') || 'none';
}

/** The bytes at `url`, from Cache Storage when there, with download progress otherwise. */
async function download(url, onProgress) {
  const cache = self.caches && (await caches.open(CACHE));
  const cached = await cache?.match(url);
  if (cached) return { bytes: new Uint8Array(await cached.arrayBuffer()), fromCache: true };
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get('Content-Length'));
  const chunks = [];
  let loaded = 0;
  for (const reader = response.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  chunks.reduce((offset, chunk) => (bytes.set(chunk, offset), offset + chunk.length), 0);
  await cache?.put(url, new Response(bytes));
  return { bytes, fromCache: false };
}

function rawText(tokens) {
  const shown = tokens.filter((t) => t.text);
  const width = Math.max(5, ...shown.map((t) => t.text.length));
  const lines = shown.map(
    (t) =>
      `${t.text.padEnd(width)}  ${TAGS[t.tag]}` +
      (t.tag === BEGIN ? `  hour ${String(t.hour).padStart(2, '0')}, minute ${String(t.minute).padStart(2, '0')}` : ''),
  );
  return `${'token'.padEnd(width)}  tag\n${lines.join('\n')}`;
}
