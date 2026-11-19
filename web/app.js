// Page wiring. Query switches: ?model=NAME (a key of MODELS), ?cpu=1 / ?cpu=0 (skip or use WebGPU),
// ?threads=N, ?eval=1, ?limit=N (eval subset), ?fresh=1 (drop the cached model first, to time a cold load),
// ?download=1 (download without asking, for scripted runs).
import { DEFAULT_MODEL, MODELS } from './engines/index.js';
import { runEval } from './eval.js';

const EXAMPLES = [
  'dinner at 7?',
  "I'm free tomorrow from noon to 4pm",
  'got paged at 2, been up ever since',
  'standup moved to 9:30, room 3',
  'can we do 3 in the afternoon instead of the 30 minute slot at 11',
  'see you in 2 hours',
];

const $ = (selector) => document.querySelector(selector);
const query = new URLSearchParams(location.search);
// Where WebGPU is the wrong default for the Qwen model, both seen on real devices: on Android it
// stalls while loading (Pixel 10), and in Firefox it works but takes about 38 s per reply against
// under 1 s on CPU. CPU is the default there; ?cpu=0 opts back in.
const CPU_FIRST = /Android|Firefox/.test(navigator.userAgent);
const options = {
  model: MODELS[query.get('model')] ? query.get('model') : DEFAULT_MODEL,
  cpu: query.has('cpu') ? query.get('cpu') === '1' : CPU_FIRST,
  threads: query.has('threads') ? Number(query.get('threads')) : undefined,
  fresh: query.get('fresh') === '1',
};

function showStatus(engine, replyMs) {
  const parts = engine.describe();
  if (replyMs !== undefined) parts.push(`last reply: ${replyMs.toFixed(replyMs < 10 ? 1 : 0)} ms`);
  $('#status').textContent = parts.join(' · ');
}

function showReply({ reply, raw, rawTitle }) {
  const lines = reply === 'none' ? [] : reply.split('\n').map((line) => line.split(/=(?=[^=]*$)/));
  const wellFormed = reply === 'none' || lines.every((pair) => pair.length === 2);
  $('#none').hidden = reply !== 'none';
  $('#times').hidden = !wellFormed || !lines.length;
  $('#raw-output').hidden = false; // always shown: what the model gave, before parsing
  $('#raw-heading').textContent = rawTitle ?? 'Raw model output';
  $('#raw').textContent = raw;
  $('#times tbody').replaceChildren(
    ...(wellFormed ? lines : []).map(([text, time]) => {
      const row = document.createElement('tr');
      for (const value of [text, time]) row.insertCell().textContent = value;
      return row;
    }),
  );
}

function showEval(result) {
  $('#eval-heading').textContent = `Eval against ${result.reference}`;
  $('#eval-summary').textContent =
    `${result.matches} / ${result.n} replies identical to the reference · median ${result.msMedian.toFixed(1)} ms, ` +
    `p95 ${result.msP95.toFixed(1)} ms per message · other problems: ${result.problems.length}`;
  $('#eval-mismatches').hidden = !result.mismatches.length;
  $('#eval-mismatches tbody').replaceChildren(
    ...result.mismatches.map((m) => {
      const row = document.createElement('tr');
      for (const value of [m.text, m.expected, m.browser]) row.insertCell().textContent = value;
      return row;
    }),
  );
  $('#eval-json').textContent = JSON.stringify(result, null, 1);
}

function setupOptions() {
  const form = $('#options');
  form.model.replaceChildren(
    ...Object.entries(MODELS).map(([value, { label }]) => new Option(label, value, false, value === options.model)),
  );
  form.cpu.checked = options.cpu;
  form.threads.checked = options.threads === 1;
  form.eval.checked = query.get('eval') === '1';
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const next = new URLSearchParams();
    next.set('cpu', form.cpu.checked ? '1' : '0'); // explicit, so unticking overrides the CPU-first default
    for (const name of ['threads', 'eval']) if (form[name].checked) next.set(name, '1');
    next.set('model', form.model.value);
    location.search = next;
  });
  // Choosing a model applies at once; the checkboxes wait for the button.
  form.model.addEventListener('change', () => form.requestSubmit());
}

/** Ask before a download. Weights this browser already holds load without a question. */
async function agreed(entry, source) {
  if (source.cached || query.get('download') === '1') return true;
  $('#consent-text').textContent =
    `“${entry.label}” will be downloaded from ${source.from} and stored in this browser, ` +
    'so later visits load it without downloading again.';
  const link = Object.assign(document.createElement('a'), { href: entry.licence.url, textContent: entry.licence.name, target: '_blank', rel: 'noopener' });
  $('#consent-licence').replaceChildren('Model licence: ', link, entry.licence.note ? ` (${entry.licence.note}).` : '.');
  const dialog = $('#consent');
  dialog.returnValue = '';
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener('close', resolve, { once: true }));
  return dialog.returnValue === 'yes';
}

async function main() {
  setupOptions();
  const entry = MODELS[options.model];
  const source = await entry.source();
  while (!(await agreed(entry, source))) {
    $('#progress').hidden = true;
    $('#progress-label').textContent = 'No model loaded.';
    $('#download').hidden = false;
    await new Promise((resolve) => $('#download').addEventListener('click', resolve, { once: true }));
  }
  $('#progress').hidden = false;
  $('#download').hidden = true;
  const onProgress = ({ loaded, total }) => {
    $('#progress').value = loaded / total;
    $('#progress-label').textContent =
      loaded < total ? `Downloading the model: ${(loaded / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MB` : 'Starting the model…';
  };
  const engine = await entry.load({ ...options, url: source.url, onProgress });
  window.__model = engine;
  $('#loading').hidden = true;
  showStatus(engine);

  let busy = false; // one reply at a time: an engine may restart its worker between replies
  const ask = async (message) => {
    if (busy) return;
    busy = $('#send').disabled = true;
    const out = await engine.complete(message);
    showReply(out);
    showStatus(engine, out.ms);
    busy = $('#send').disabled = false;
  };
  $('#ask').addEventListener('submit', (event) => {
    event.preventDefault();
    if ($('#message').value.trim()) ask($('#message').value);
  });
  $('#examples').replaceChildren(
    ...EXAMPLES.map((example) => {
      const button = document.createElement('button');
      button.textContent = example;
      button.addEventListener('click', () => {
        $('#message').value = example;
        ask(example);
      });
      return button;
    }),
  );

  if (query.get('eval') === '1') {
    $('#eval').hidden = false;
    busy = true;
    const result = await runEval(engine, {
      limit: Number(query.get('limit')) || undefined,
      onProgress: (done, total) => ($('#eval-progress').textContent = `${done} / ${total} messages`),
    });
    window.__evalResult = { ...result, model: options.model, info: engine.info };
    showEval(window.__evalResult);
  }
  busy = $('#message').disabled = $('#send').disabled = false;
}

main().catch((error) => {
  console.error(error);
  $('#loading').hidden = false;
  $('#progress').hidden = true;
  $('#progress-label').textContent = `Failed: ${error.message}`;
  window.__evalResult = { error: String(error) };
});
