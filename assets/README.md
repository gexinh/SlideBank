# Paper figure assets

`overview.svg` is a conceptual repository illustration, not an extracted paper
figure. Both the README and project page use it until the original main figure
is provided.

Place the original PDF at `assets/main-figure.pdf`, then run from the repo root:

```bash
python scripts/prepare_main_figure.py
```

If you place the PDF elsewhere, including in the repository root, pass its path:

```bash
python scripts/prepare_main_figure.py "path/to/main figure.pdf"
```

The script uses Poppler's `pdftoppm` to render page 1 at 160 DPI. It preserves
the PDF at `assets/main-figure.pdf`, writes `assets/main-figure.png`, copies both
into `project/assets/`, switches the README image and caption, and updates
`project/config.js` to display the PNG. Override the render resolution
with `--dpi 200` if needed.

Install Poppler with your system package manager or
`conda install -c conda-forge poppler`. Review the image before publishing.
The figure retains the paper's applicable asset terms; the source-code MIT grant
does not replace them.
