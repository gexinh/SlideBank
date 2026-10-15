# Project page

The static project page and Python workbench serve complementary roles:

- `project/`: a buildless static paper page with a browser-local evidence viewer.
- `app.py`: a Gradio workbench for bank building, evidence browsing,
  and model-backed question answering.

The evidence viewer automatically opens an included slide example. It loads a
published evidence tree and derived images, so visitors can inspect the global
overview, choose a region, and browse anchor patches immediately. Browsing
requires no model inference.

The **Your own evidence** section opens `slidebank-ui-tree-v1` JSON and an
optional local image folder. Imported files stay in the browser. The viewer
does not request server paths from an imported JSON file.

## Preview

```bash
python -m http.server --directory project 8080
```

Open <http://127.0.0.1:8080>. The default example opens automatically; use the
sample selector and **Reset example** to return to it. Click a region and an
anchor to inspect patch views, signals, and coordinates.

For a local export, expand **Your own evidence**, select **Open evidence JSON**,
then **Add image folder**. Images with ambiguous file matches stay unresolved.
**Clear** discards imported data and releases image object URLs. See the
[demo data guide](../demos/README.md) for the included slide's provenance.

## GitHub Pages

The project page is hosted at [gexinh.github.io/SlideBank](https://gexinh.github.io/SlideBank/) from the
`gexinh/SlideBank` repository.

For a fork or new repository, select **Settings → Pages → Source → GitHub
Actions**. The included `.github/workflows/pages.yml` deploys only `project/`
on relevant pushes to `main`, and can also be run manually. It follows GitHub's
[official static-site workflow](https://github.com/actions/starter-workflows/blob/main/pages/static.yml).
The application uses relative asset paths so it works under `/SlideBank/`.

## Attach a hosted workbench

The [SlideBank Space](https://huggingface.co/spaces/gexinh/SlideBank) runs the
Python workbench separately from GitHub Pages. See [HOSTING.md](HOSTING.md) for
the hosted model, ZeroGPU runtime, and deployment files. The page embeds this
workbench and provides a direct-open link.

Set the deployed HTTPS URL in `project/config.js`:

```javascript
window.SLIDEBANK_CONFIG = Object.freeze({
  demoUrl: "https://gexinh-slidebank.hf.space",
  qaEnabled: true,
  mainFigure: "main-figure.png"
});
```

With a hosted URL configured, the page displays a workbench iframe and a
direct-open link. Otherwise, it shows local setup instructions.
The host must permit iframe embedding; the direct link remains available when
embedding is restricted. GitHub Pages itself does not execute Gradio or models.
Preserve the existing `mainFigure` value when changing `demoUrl`; the figure
preparation script keeps it set to `main-figure.png`.

## Main figure

See [assets/README.md](../assets/README.md). The figure preparation script
synchronizes the original PDF and web PNG into both asset directories. Both the
README and static HTML use the paper figure directly.

## Included example assets

`demos/manifest.json` lists the example evidence tree and its exact image paths.
The project page uses the matching files in `project/demos/`; the retrieval bank
is required by the Python workbench and is not served by the static viewer.
Keep the manifest, evidence tree, and derived images together when updating an
example. The viewer accepts only the manifest's same-origin image paths for
published examples; imported trees use local files instead.
