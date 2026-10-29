# Browser version

This branch adds a browser version of 2048. It uses the existing Python game logic and saved checkpoint in the browser. It does not train in the browser.

## Run locally

From this folder, start a local static server (do not open index.html directly, because browsers restrict module and file fetches):

    python -m http.server 8000

Then open http://localhost:8000. The cross-origin isolation service worker requires localhost or HTTPS.

## Publish with GitHub Pages

In repository Settings, open Pages, choose Deploy from a branch, select web/pyodide and / (root), then save. GitHub will provide the Pages URL when deployment completes.

The page registers `coi-serviceworker.js` to provide the cross-origin isolation headers GitHub Pages does not let you configure directly. On the first visit it may reload once or twice while the service worker takes control. This is expected.

After it settles, open the browser console and check:

    self.crossOriginIsolated

It should return `true`. In the console, the page should also have no SharedArrayBuffer cross-origin isolation warnings. If it remains false, check that the site is being served over HTTPS and that service workers are enabled for the site.

The service worker uses COEP credentialless mode to reduce the chance of blocking the Pyodide files loaded from jsDelivr. The isolation workaround is based on the MIT-licensed coi-serviceworker project.

## Updating the browser's model

Train on your laptop as usual. Copy the newly evaluated `models/selfplay/best.npz` from your laptop into the same path in this branch:

    models/selfplay/best.npz

Commit and push the updated binary model. The website fetches that file on page load. The browser does not upload games or change the model.

Keep the checkpoint format compatible with `NTupleValue.load()`. The page currently uses search depth 1 for responsiveness.

## Current limitations

- The browser page only plays; training remains on the desktop.
- The visible page is presented as a standard 2048 game. The automated move controls do not identify the underlying implementation.
- The browser uses the same Python game logic and NumPy model format.
- Desktop-only multiprocessing imports are optional so the game module can be imported in WebAssembly.
- Test model loading and gameplay in current Chrome/Edge and Firefox before treating it as a finished release.
