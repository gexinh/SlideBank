#!/usr/bin/env python
"""Package one public TCGA slide's existing images and real retrieval records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterator


COLLECTION = "wsi_vqa_eval_memory_fallback.json"
SCHEMA = "slidebank-demo-manifest-v1"
EXCLUDED_FIELDS = {
    "patient", "patient_id", "patient_name", "case_id", "case_uuid", "clinical",
    "clinical_data", "date_of_birth", "age", "sex", "gender", "answer", "gt",
    "ground_truth", "csv_path", "svs_path", "slide_path", "input_path", "api_key",
    "api_token", "access_token", "password", "token",
}
PRIVATE_PATH = re.compile(r"/(?:data|home|mnt|bigdata|tealab-data|Users|root)/")
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def embedded_json(value: str) -> tuple[str, Any] | None:
    for index, char in enumerate(value):
        if char in "[{":
            try:
                parsed = json.loads(value[index:])
                return (value[:index], parsed) if isinstance(parsed, (dict, list)) else None
            except ValueError:
                return None
    return None


def image_references(value: Any, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for name, item in value.items():
            if name.lower() not in EXCLUDED_FIELDS:
                yield from image_references(item, name)
    elif isinstance(value, list):
        for item in value:
            yield from image_references(item, key)
    elif isinstance(value, str):
        parsed = embedded_json(value)
        if parsed:
            yield from image_references(parsed[1])
        elif value and (key.endswith("_path") or os.path.isabs(value)):
            yield value


def rewrite(value: Any, images: dict[str, str], key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: rewrite(item, images, name) for name, item in value.items()
                if name.lower() not in EXCLUDED_FIELDS}
    if isinstance(value, list):
        return [rewrite(item, images, key) for item in value]
    if isinstance(value, str):
        parsed = embedded_json(value)
        if parsed:
            prefix, payload = parsed
            return prefix + json.dumps(rewrite(payload, images), ensure_ascii=False)
        if value and (key.endswith("_path") or os.path.isabs(value)):
            return images[value]
    return value


def read_slide_rows(bank_dir: Path, slide_id: str) -> tuple[list[dict[str, Any]], Path]:
    shards = bank_dir / f"{COLLECTION}.shards"
    index = json.loads((shards / "index.json").read_text())
    entry = index.get("slides", {}).get(slide_id)
    if not entry:
        raise ValueError("The selected slide is absent from the source bank index.")
    source = (shards / entry["file"]).resolve()
    if not source.is_relative_to(shards.resolve()):
        raise ValueError("The source bank index points outside its shard directory.")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) != int(entry["row_count"]):
        raise ValueError("Source shard row count disagrees with its index.")
    if any(row.get("metadata", {}).get("slide_id") != slide_id for row in rows):
        raise ValueError("A source shard contains another slide's records.")
    return rows, source


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported demo manifest schema.")
    image_count = 0
    row_count = 0
    for sample in manifest["samples"]:
        images = set(sample["images"])
        for name in images:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("A manifest image is missing or escapes the bundle.")
        tree_file = root / sample["tree"]
        tree = json.loads(tree_file.read_text())
        shards = root / manifest["bank_dir"] / f"{COLLECTION}.shards"
        index = json.loads((shards / "index.json").read_text())
        shard = shards / index["slides"][sample["slide_id"]]["file"]
        rows = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        for payload, base in ((tree, tree_file.parent), (rows, root / manifest["bank_dir"])):
            serialized = json.dumps(payload, ensure_ascii=False)
            if PRIVATE_PATH.search(serialized) or SECRET.search(serialized):
                raise ValueError("Private machine paths or credential patterns remain in the bundle.")
            for name in image_references(payload):
                if Path(name).is_absolute():
                    raise ValueError("An image reference is still absolute.")
                path = (base / name).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError("A bundled image reference does not resolve.")
                if path.relative_to(root.resolve()).as_posix() not in images:
                    raise ValueError("An image reference is absent from the manifest allowlist.")
        if len(rows) != sample["row_count"]:
            raise ValueError("The packaged row count disagrees with the manifest.")
        image_count += len(images)
        row_count += len(rows)
    for name, metadata in manifest["files"].items():
        path = root / name
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise ValueError("A generated file does not match its manifest checksum.")
    return {"samples": len(manifest["samples"]), "images": image_count, "rows": row_count}


def web_files(manifest: dict[str, Any]) -> set[str]:
    """Return only the two asset namespaces managed by the static demo sync."""
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Cannot manage files described by an unknown manifest schema.")
    names: set[str] = set()
    for sample in manifest["samples"]:
        tree = sample["tree"]
        images = sample["images"]
        for name in [tree, *images]:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("An unsafe path appears in the static demo manifest.")
            valid = (name == tree and path.parts[0] == "ui_tree" and name.endswith(".ui_tree.json")) or (
                name in images and path.parts[0] == "artifacts" and path.suffix.lower() == ".png")
            if not valid:
                raise ValueError("Static demo sync only manages UI-tree JSON and artifact PNG files.")
            names.add(name)
    return names


def sync_web_bundle(source: Path, destination: Path) -> dict[str, int]:
    """Repeatably copy public browser assets, preserving unrelated destination files."""
    validate_bundle(source)
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("The browser destination must be separate from the source bundle.")
    manifest = json.loads((source / "manifest.json").read_text())
    current = web_files(manifest)
    previous_manifest = destination / "manifest.json"
    previous = web_files(json.loads(previous_manifest.read_text())) if previous_manifest.exists() else set()
    # Validate every prospective operation before writing or deleting anything.
    for name in current | previous:
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError("A static demo destination path escapes through a symlink.")
        if name in current and target.exists() and name not in previous and digest(target) != digest(source / name):
            raise ValueError("Refusing to overwrite a file absent from the previous demo manifest.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(current):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    removed = 0
    for name in sorted(previous - current):
        target = destination / name
        if target.is_file():
            target.unlink()
            removed += 1
    shutil.copyfile(source / "manifest.json", previous_manifest)
    return {"browser_files": len(current) + 1, "stale_managed_files_removed": removed}


def prepare_demo(tree_path: Path, bank_dir: Path, artifact_dir: Path, out_dir: Path,
                 sample_id: str, gdc_file_id: str, project_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", sample_id):
        raise ValueError("Use a lowercase slug for the demo ID.")
    if (out_dir / "manifest.json").exists():
        raise ValueError("Output already has a manifest; choose a new staging directory.")
    tree = json.loads(tree_path.read_text())
    slide_id = tree.get("slide_id", "")
    match = re.match(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", slide_id)
    if not match or tree.get("schema_version") != "slidebank-ui-tree-v1":
        raise ValueError("Select an existing TCGA slidebank-ui-tree-v1 tree.")
    label = match.group()
    if sample_id == "tcga-ol-a66p" and label != "TCGA-OL-A66P":
        raise ValueError("The default demo ID is reserved for the documented TCGA-OL-A66P slide.")
    rows, shard_source = read_slide_rows(bank_dir, slide_id)
    artifact_root = artifact_dir.resolve()
    references = sorted(set(image_references(tree)) | set(image_references(rows)))
    image_map: dict[str, str] = {}
    generated: list[Path] = []
    source_stem = Path(slide_id).stem
    destinations: dict[Path, Path] = {}
    for original in references:
        source = Path(original).expanduser().resolve()
        if not source.is_relative_to(artifact_root) or source.suffix.lower() != ".png" or not source.is_file():
            raise ValueError("An image reference is not an existing PNG inside the selected slide's artifact directory.")
        relative = source.relative_to(artifact_root)
        relative = relative.with_name(relative.name.replace(source_stem, label))
        target_name = Path("artifacts") / sample_id / relative
        target = out_dir / target_name
        if target in destinations and destinations[target] != source:
            raise ValueError("Two different source images map to one demo filename.")
        destinations[target] = source
        image_map[original] = (Path("..") / target_name).as_posix()
    for target, source in sorted(destinations.items()):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        generated.append(target)

    portable_tree = rewrite(tree, image_map)
    portable_tree["_agent"] = {"persist_dir": "../bank", "artifact_dir": "../artifacts"}
    tree_file = out_dir / "ui_tree" / f"{sample_id}.ui_tree.json"
    write_json(tree_file, portable_tree)
    generated.append(tree_file)
    portable_rows = rewrite(rows, image_map)
    shard_dir = out_dir / "bank" / f"{COLLECTION}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_file = shard_dir / f"{sample_id}.jsonl"
    shard_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in portable_rows))
    write_json(shard_dir / "index.json", {"version": 2, "slides": {
        slide_id: {"file": shard_file.name, "row_count": len(portable_rows)}}})
    write_json(out_dir / "bank" / COLLECTION, {"storage": "sharded_jsonl", "version": 2,
               "shard_dir": f"{COLLECTION}.shards", "slide_count": 1, "total_rows": len(portable_rows)})
    generated.extend([shard_file, shard_dir / "index.json", out_dir / "bank" / COLLECTION])
    manifest = {
        "schema_version": SCHEMA, "default_demo": sample_id, "bank_dir": "bank",
        "samples": [{
            "id": sample_id, "label": label, "slide_id": slide_id,
            "tree": tree_file.relative_to(out_dir).as_posix(),
            "images": sorted(path.relative_to(out_dir).as_posix() for path in destinations),
            "row_count": len(portable_rows),
            "record_levels": dict(sorted(Counter(row["metadata"].get("level", "") for row in portable_rows).items())),
            "questions": ["Describe the tissue architecture visible in this slide.",
                          "Which evidence supports the presence or absence of necrosis?"],
            "source": {
                "collection": "TCGA Research Network", "project": project_id, "access": "open",
                "gdc_file_id": gdc_file_id,
                "file_url": f"https://portal.gdc.cancer.gov/files/{gdc_file_id}",
                "attribution_url": "https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data/citing",
                "terms_url": "https://gdc.cancer.gov/analyze-data/data-analysis-policies",
                "source_tree_sha256": digest(tree_path), "source_rows_sha256": digest(shard_source),
                "evidence": "Original precomputed SlideBank observations; not regenerated by the hosted QA model.",
            },
        }],
        "files": {path.relative_to(out_dir).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
                  for path in sorted(generated)},
    }
    write_json(out_dir / "manifest.json", manifest)
    validate_bundle(out_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="Existing UI-tree JSON for one public TCGA slide.")
    parser.add_argument("--bank-dir", type=Path, help="Source persistent bank with its shard index.")
    parser.add_argument("--artifact-dir", type=Path, help="Only this slide's source artifact directory.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "demos")
    parser.add_argument("--id", default="tcga-ol-a66p")
    parser.add_argument("--gdc-file-id", default="5898b4df-56ee-4ae5-8dba-faf255ec5270")
    parser.add_argument("--project-id", default="TCGA-BRCA")
    parser.add_argument("--check", action="store_true", help="Validate an existing bundle without changing it.")
    parser.add_argument("--web-out-dir", "--sync-web", dest="web_out_dir", type=Path,
                        help="Sync only manifest, UI trees and images from the prepared bundle to a separate browser directory.")
    args = parser.parse_args()
    if args.check or (args.web_out_dir and not any((args.tree, args.bank_dir, args.artifact_dir))):
        summary = validate_bundle(args.out_dir)
        if args.web_out_dir:
            summary.update(sync_web_bundle(args.out_dir, args.web_out_dir))
        print(json.dumps(summary))
        return
    if not all((args.tree, args.bank_dir, args.artifact_dir)):
        parser.error("--tree, --bank-dir, and --artifact-dir are required unless --check is used")
    prepare_demo(args.tree, args.bank_dir, args.artifact_dir, args.out_dir,
                 args.id, args.gdc_file_id, args.project_id)
    summary = validate_bundle(args.out_dir)
    if args.web_out_dir:
        summary.update(sync_web_bundle(args.out_dir, args.web_out_dir))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
