// The models in the drop-down: value -> label, licence, `source()` and `load(options)`. `source()` says
// where the weights come from ({ url, from, cached }; see source.js), so the page can ask before
// a download. Every engine has the same interface: `load({ ...options, url })` resolves to an engine with `describe()` (status line parts), `complete(message)`
// -> { reply, raw, ms }, `evalSet()` -> { reference, items: [{ id, text, expected }] } and
// `check(item, out)` -> extra problems. Engines are imported on demand, so a page loads one runtime.

const gguf = (model) => ({
  source: async () => (await import('./gguf.js')).Gguf.source(model),
  load: async (options) => (await import('./gguf.js')).Gguf.load({ ...options, ...model }),
});
const APACHE = { name: 'Apache 2.0', url: 'https://www.apache.org/licenses/LICENSE-2.0' };
const LFM_LICENCE = {
  name: 'LFM Open License v1.0',
  url: 'https://huggingface.co/dsego/tempus-lfm2.5-230m/blob/main/LICENSE',
  note: 'commercial use only for organisations under $10 million annual revenue',
};
const BERT = 'dsego/tempus-bert-mini-tagger';
const tagger = (variant) => ({
  source: async () => (await import('./tagger.js')).Tagger.source({ repo: BERT, variant }),
  load: async (options) => (await import('./tagger.js')).Tagger.load({ ...options, repo: BERT, variant }),
});

// Per GGUF model: `repo` on Hugging Face, `file` there and in models/, `prompt` template in
// prompts/, `reference` replies in eval/, and `restartEvery` (worker restart to empty llama.cpp's
// prompt state cache, see README).
const lfm = (quant) => gguf({ repo: 'dsego/tempus-lfm2.5-230m', prompt: 'lfm2.5-230m-v2', file: `lfm2.5-230m-v2-${quant}`, reference: `lfm2.5-230m-v2-${quant}`, restartEvery: 256 });
// Qwen's cached states are tens of MB each (LFM2.5's about 0.6 MB), so it restarts far more often.
// The trimmed file gives the untrimmed file's native replies, so both share one reference.
const qwen = (file) => gguf({ repo: 'dsego/tempus-qwen3.5-0.8b', prompt: 'qwen3.5-0.8b-v2', file, reference: 'qwen3.5-0.8b-v2-Q4_K_M', restartEvery: 16 });

export const DEFAULT_MODEL = 'lfm2.5-230m-v2-Q8_0';

export const MODELS = {
  'lfm2.5-230m-v2-Q8_0': { label: 'LFM2.5-230M finetune, 247 MB', licence: LFM_LICENCE, ...lfm('Q8_0') },
  'lfm2.5-230m-v2-Q4_K_M': { label: 'LFM2.5-230M finetune, 153 MB (Q4_K_M, about a point less accurate)', licence: LFM_LICENCE, ...lfm('Q4_K_M') },
  'qwen3.5-0.8b-v2-Q4_K_M': { label: 'Qwen3.5 0.8B finetune, 542 MB (more accurate on unfamiliar situations)', licence: APACHE, ...qwen('qwen3.5-0.8b-v2-Q4_K_M') },
  'qwen3.5-0.8b-v2-trim-Q4_K_M': { label: 'Qwen3.5 0.8B finetune, 448 MB (trimmed vocabulary)', licence: APACHE, ...qwen('qwen3.5-0.8b-v2-trim-Q4_K_M') },
  'tagger-int8': { label: 'BERT-mini tagger (not generative, 11M parameters), 11 MB', licence: APACHE, ...tagger('int8') },
};
