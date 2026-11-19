// A finetuned GGUF model in wllama, with the exact settings of the native llama.cpp client
// (src/tempus/llama.py): one slot, 1024-token context, greedy, 160 new tokens, no prompt cache.
// Each model brings its chat template (prompts/NAME.json) and native replies (eval/NAME.json),
// both from export_eval.py.
import { Wllama } from '../vendor/wllama/index.js';
import { locate } from './source.js';

const here = (path) => new URL(path, import.meta.url).href;
const json = (path) => fetch(here(path)).then((r) => r.json());

const WASM = { default: here('../vendor/wllama/wasm/wllama.wasm') };
// Browsers without JSPI or Memory64 (Safari) need the slower Asyncify build.
const COMPAT = { wasm: here('../vendor/wllama-compat/wllama.wasm'), worker: here('../vendor/wllama-compat/wllama.js') };

const SAMPLING = { max_tokens: 160, temperature: 0, top_k: 1, cache_prompt: false };

export class Gguf {
  /**
   * `url` is the GGUF file (from `Gguf.source`), `prompt` its template in prompts/, `reference`
   * its native replies in eval/. `restartEvery` restarts the wllama worker after that many replies (see
   * README: llama.cpp's prompt state cache fills the wasm heap for Qwen's hybrid layers).
   * `cpu` skips WebGPU; `threads` overrides wllama's default (half the cores); `fresh` drops the
   * cached weights first, to time a cold load.
   */
  static async load({ url, prompt, reference, restartEvery = Infinity, cpu = false, threads, fresh = false, onProgress }) {
    const engine = new Gguf({ url, reference, restartEvery, cpu, threads });
    engine.template = await json(`../prompts/${prompt}.json`);
    engine.info = await engine.#start({ fresh, onProgress });
    return engine;
  }

  /** Where the weights will come from, and whether this browser already holds them. */
  static async source({ repo, file }) {
    const found = await locate({ repo, file: `${file}.gguf` });
    const cached = (await new Wllama(WASM).cacheManager.list()).some((e) => e.metadata.originalURL === found.url);
    return { ...found, cached };
  }

  constructor(options) {
    this.options = options;
    this.calls = 0;
    this.restarts = 0;
  }

  /** Status line parts. */
  describe() {
    const { info } = this;
    return [
      `Backend: ${info.backend}${info.compat ? ' (compat build)' : ''}`,
      `threads: ${info.threads}`,
      `cross-origin isolated: ${info.isolated ? 'yes' : 'no'}`,
      `model: ${(info.modelBytes / 2 ** 20).toFixed(0)} MB`,
      `loaded in ${(info.loadMs / 1000).toFixed(1)} s${info.fromCache ? ' (cached)' : ''}`,
    ];
  }

  /**
   * One greedy completion, capped and stopped at end of turn like the native client. wllama 3.13
   * takes text prompts only, so the message goes between the template's text halves and llama.cpp
   * tokenizes it, parsing special tokens and adding BOS where the GGUF asks (checked in #start).
   */
  async complete(message) {
    if (this.calls === this.options.restartEvery) {
      await this.wllama.exit();
      await this.#start({});
      this.calls = 0;
      this.restarts++;
    }
    this.calls++;
    const prompt = this.template.prefix + message + this.template.suffix;
    const started = performance.now();
    const result = await this.wllama.createCompletion({ prompt, ...SAMPLING });
    const ms = performance.now() - started;
    const choice = result.choices[0];
    const reply = choice.text.trim();
    return { reply, raw: reply, ms, promptTokens: result.usage.prompt_tokens, finish: choice.finish_reason };
  }

  /** The eval messages with the native llama.cpp replies and prompt ids. */
  evalSet() {
    return json(`../eval/${this.options.reference}.json`);
  }

  /**
   * Problems beyond the reply text: the prompt length llama.cpp reports must equal the ids the
   * native client sent (a missing or doubled BOS, or a mis-parsed special token, changes it), and
   * the reply must end at an end-of-generation token (the model's end of turn), not the cap.
   */
  check(item, out) {
    const problems = [];
    if (out.promptTokens !== item.ids.length) problems.push(`prompt ${out.promptTokens} tokens, native ${item.ids.length}`);
    if (out.finish !== 'stop') problems.push(`finished by ${out.finish}`);
    return problems;
  }

  async #start({ fresh = false, onProgress }) {
    this.log = [];
    const keep = (...args) => this.log.push(args.join(' '));
    const logger = { debug: keep, log: keep, warn: (...a) => (keep(...a), console.warn(...a)), error: console.error };
    const wllama = (this.wllama = new Wllama(WASM, { logger }));
    wllama.setCompat(COMPAT);

    if (fresh) await wllama.cacheManager.clear();
    const cachedEntry = async () => (await wllama.cacheManager.list()).find((e) => e.metadata.originalURL === this.options.url);
    const fromCache = !!(await cachedEntry());
    const started = performance.now();
    await wllama.loadModelFromUrl(this.options.url, {
      n_ctx: 1024,
      n_parallel: 1,
      n_threads: this.options.threads,
      ...(this.options.cpu && { n_gpu_layers: 0 }),
      progressCallback: onProgress,
    });
    const loadMs = performance.now() - started;

    // The template text leaves BOS to llama.cpp exactly when the GGUF adds it.
    const context = wllama.getLoadedContextInfo();
    if (context.add_bos_token !== this.template.add_bos) throw new Error(`GGUF add_bos_token ${context.add_bos_token}, template expects ${this.template.add_bos}`);

    const offload = this.log.join('\n').match(/offloaded (\d+)\/(\d+) layers to GPU/);
    return {
      backend: offload && +offload[1] > 0 ? `WebGPU (${offload[1]}/${offload[2]} layers)` : 'CPU (WASM)',
      threads: wllama.getNumThreads(),
      multiThread: wllama.isMultithread(),
      isolated: self.crossOriginIsolated,
      compat: wllama.getWorkerResources().compat,
      modelBytes: (await cachedEntry())?.size,
      fromCache,
      loadMs,
      addBos: context.add_bos_token,
      eogTokens: context.list_tokens_eog,
      llamaCpp: Wllama.getLibllamaVersion(),
    };
  }
}
