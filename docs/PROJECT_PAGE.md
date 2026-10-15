# Project page

The repository contains two connected interfaces:

- `project/`: a buildless static paper page with a browser-local evidence viewer.
- `app.py`: the latest V5 Gradio workbench for bank building, evidence browsing,
  and model-backed question answering.

The static evidence viewer opens `slidebank-ui-tree-v1` JSON and optionally a
local image folder. It can inspect real exports without running Python or
calling a model. Imported files are read in the browser and are not sent to a
server. No patient images or synthetic inference results are preloaded.

## Preview

```bash
python -m http.server --directory project 8080
```

Open <http://127.0.0.1:8080>. Select **Open evidence JSON**, then optionally
**Add image folder**. Choose a region and an anchor to inspect patch views and
signals. The viewer supports captions and coordinates without image files.
Images with ambiguous file matches stay unresolved. **Clear** discards the
current tree and releases image object URLs.

## GitHub Pages

The project page is hosted at `https://gexinh.github.io/SlideBank/` from the
`gexinh/SlideBank` repository.

For a fork or new repository, select **Settings → Pages → Source → GitHub
Actions**. The included `.github/workflows/pages.yml` deploys only `project/`
on relevant pushes to `main`, and can also be run manually. It follows GitHub's
[official static-site workflow](https://github.com/actions/starter-workflows/blob/main/pages/static.yml).
The application uses relative asset paths so it works under `/SlideBank/`.

## Attach a hosted workbench

Deploy `app.py` on a Python host with your intended data and runtime resources.
Use [WORKBENCH.md](WORKBENCH.md) for paths, credentials, and local model setup.
For an external host, set `SLIDEBANK_HOST=0.0.0.0` and the service's assigned
port. Follow the host's own dependency and storage requirements.

Set the deployed HTTPS URL in `project/config.js`:

```javascript
window.SLIDEBANK_CONFIG = Object.freeze({
  demoUrl: "https://your-hosted-workbench.example",
  mainFigure: "main-figure.png"
});
```

The paper page then displays a workbench iframe and a direct-open link. Until
that URL is configured, the page shows working local setup instructions.
The host must permit iframe embedding; the direct link remains available when
embedding is restricted. GitHub Pages itself does not execute Gradio or models.
Preserve the existing `mainFigure` value when changing `demoUrl`; the figure
preparation script keeps it set to `main-figure.png`.

## Main figure

See [assets/README.md](../assets/README.md). The figure preparation script
synchronizes the original PDF and web PNG into both asset directories. Both the
README and static HTML use the paper figure directly.
