# Paper figure assets

`main-figure.pdf` contains the paper's main figure. `main-figure.png` is its
200-DPI rendering for the README and project page.

To update the figure, replace `assets/main-figure.pdf`, then run from the repo root:

```bash
python scripts/prepare_main_figure.py --dpi 200
```

If you place the PDF elsewhere, including in the repository root, pass its path:

```bash
python scripts/prepare_main_figure.py "path/to/main figure.pdf"
```

The script uses Poppler's `pdftoppm` to render page 1 at 160 DPI. It preserves
the PDF at `assets/main-figure.pdf`, writes `assets/main-figure.png`, copies both
into `project/assets/`, updates the README caption, and sets
`project/config.js` to display the PNG. Override the render resolution
with `--dpi 200` if needed.

Install Poppler with your system package manager or
`conda install -c conda-forge poppler`.
The figure retains the paper's applicable asset terms; the source-code MIT grant
does not replace them.
