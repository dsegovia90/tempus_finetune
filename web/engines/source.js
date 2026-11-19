// Where a model file comes from: a copy in models/ beside the page when there is one (local
// development, or a host that serves its own copy), otherwise the published file on Hugging Face.
const HUB = 'https://huggingface.co';

/** `repo` and `file` name the published file; `local` is its path under models/. */
export async function locate({ repo, file, local = file }) {
  const beside = new URL(`../models/${local}`, import.meta.url).href;
  // A host that answers every unknown path with its index page says 200 here, with HTML.
  const found = await fetch(beside, { method: 'HEAD' }).then(
    (r) => r.ok && !(r.headers.get('Content-Type') ?? '').startsWith('text/html'),
    () => false,
  );
  if (found) return { url: beside, from: 'this site' };
  if (!repo) throw new Error(`models/${local} is not on this site, and this model is not published`);
  return { url: `${HUB}/${repo}/resolve/main/${file}`, from: 'Hugging Face' };
}
