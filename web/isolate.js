// Cross-origin isolation on a plain static host. wllama's multi-threading needs the page to be
// cross-origin isolated, which takes two response headers a static host may not let you set.
// Loaded as a page script, this file registers itself as a service worker and reloads once;
// as the service worker, it adds the headers to this site's own responses. Where the host
// already sends them (web/serve.py does), it does nothing.
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', (event) => {
    // Other origins (the model files) pass through untouched: they are fetched with CORS,
    // which isolation allows, and rewrapping them would hide their Content-Length.
    if (new URL(event.request.url).origin !== self.location.origin) return;
    event.respondWith(
      fetch(event.request).then((response) => {
        const headers = new Headers(response.headers);
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Resource-Policy', 'same-origin');
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
      }),
    );
  });
} else if (!window.crossOriginIsolated && window.isSecureContext && 'serviceWorker' in navigator) {
  const RELOADED = 'tempus-isolation-reload'; // at most one reload per tab, whatever the browser does
  navigator.serviceWorker.register(document.currentScript.src).then(async () => {
    await navigator.serviceWorker.ready;
    if (sessionStorage.getItem(RELOADED)) return;
    sessionStorage.setItem(RELOADED, '1');
    location.reload();
  });
}
