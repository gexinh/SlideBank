#!/usr/bin/env python3
"""Prepare the public Hugging Face Space files without uploading anything."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKER = ".slidebank-space-manifest.json"


def stage_space(out_dir: Path) -> int:
    out_dir = out_dir.resolve()
    if out_dir == ROOT or ROOT.is_relative_to(out_dir):
        raise ValueError("Choose an output directory outside the source tree or inside build/.")
    if out_dir.is_relative_to(ROOT) and not out_dir.is_relative_to(ROOT / "build"):
        raise ValueError("Output directories within this repository must be inside build/.")

    files: dict[str, Path] = {}
    for name in ("app.py", "space_app.py", "pyproject.toml", "LICENSE"):
        files[name] = ROOT / name
    for folder, suffixes in (
        ("slidebank", {".py", ".json"}),
        ("configs", {".yaml"}),
        ("demos", {".json", ".jsonl", ".png", ".md"}),
        ("hf_space", {".py"}),
    ):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in ("THIRD_PARTY.md", "HOSTING.md"):
        files[f"docs/{name}"] = ROOT / "docs" / name
    files["scripts/export_ui_tree.py"] = ROOT / "scripts/export_ui_tree.py"
    files["README.md"] = ROOT / "hf_space/README.md"
    files["requirements.txt"] = ROOT / "hf_space/requirements.txt"
    for path in files.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Expected a regular source file: {path.relative_to(ROOT)}")

    previous: list[str] = []
    marker = out_dir / MARKER
    if out_dir.exists() and any(out_dir.iterdir()):
        if not marker.is_file():
            raise ValueError("Refusing to overwrite a nonempty directory not created by this script.")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("schema") != "slidebank-space-stage-v1":
            raise ValueError("Unrecognized Space staging manifest.")
        previous = record["files"]
        for name in previous:
            path = out_dir / name
            if not path.resolve().is_relative_to(out_dir) or path.is_symlink():
                raise ValueError("Invalid path in the previous staging manifest.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sorted(files.items()):
        target = out_dir / name
        if not target.resolve().is_relative_to(out_dir) or target.is_symlink():
            raise ValueError(f"Invalid output path: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in set(previous) - files.keys():
        (out_dir / name).unlink(missing_ok=True)
    marker.write_text(json.dumps({"schema": "slidebank-space-stage-v1", "files": sorted(files)}, indent=2) + "\n", encoding="utf-8")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Prepared {stage_space(args.out_dir)} public Space files in {args.out_dir}.")


if __name__ == "__main__":
    main()
