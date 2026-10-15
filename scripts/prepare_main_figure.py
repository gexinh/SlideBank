#!/usr/bin/env python
"""Render the supplied paper figure and synchronize README/Pages assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path, dpi: int = 160) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Main figure PDF not found: {source}")
    with source.open("rb") as handle:
        signature = handle.read(5)
    if signature != b"%PDF-":
        raise ValueError("The input must be a PDF file.")
    if not 50 <= dpi <= 600:
        raise ValueError("DPI must be between 50 and 600.")
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError("Install Poppler first (for example: conda install -c conda-forge poppler).")
    assets = ROOT / "assets"
    pages_assets = ROOT / "project" / "assets"
    assets.mkdir(exist_ok=True)
    pages_assets.mkdir(parents=True, exist_ok=True)
    # Convert successfully before changing any visible repository assets.
    with tempfile.TemporaryDirectory(prefix="slidebank-figure-") as temp:
        output = Path(temp) / "figure"
        subprocess.run(
            [renderer, "-f", "1", "-l", "1", "-singlefile", "-png", "-r", str(dpi), str(source), str(output)],
            check=True, capture_output=True, text=True,
        )
        pdf = assets / "main-figure.pdf"
        png = assets / "main-figure.png"
        if source != pdf.resolve():
            shutil.copyfile(source, pdf)
        shutil.copyfile(output.with_suffix(".png"), png)
        for asset in (pdf, png):
            shutil.copyfile(asset, pages_assets / asset.name)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(
        r"(?<=<!-- main-figure-caption:start -->).*?(?=<!-- main-figure-caption:end -->)",
        "\n*Main figure from the SlideBank paper. [Original PDF](assets/main-figure.pdf).*\n",
        text, flags=re.S,
    )
    readme.write_text(text, encoding="utf-8")
    page_config = ROOT / "project" / "config.js"
    config_text = page_config.read_text(encoding="utf-8")
    config_text = re.sub(r'mainFigure:\s*[\'"][^\'"]+[\'"]', 'mainFigure: "main-figure.png"', config_text)
    page_config.write_text(config_text, encoding="utf-8")
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=ROOT / "assets" / "main-figure.pdf")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    try:
        pdf, png = prepare(args.pdf, args.dpi)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Figure preparation failed: {exc}\n")
    print(f"Prepared {pdf.relative_to(ROOT)} and {png.relative_to(ROOT)}")
    print("Updated README and project/assets; review the rendered figure before committing.")


if __name__ == "__main__":
    main()
