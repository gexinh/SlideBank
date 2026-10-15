#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slidebank.data import safe_name


SCHEMA_VERSION = "slidebank-ui-tree-v1"
METRIC_KEYS = [
    "dominant_category",
    "tissue_ratio",
    "selected_region_count",
    "malignant_region_count",
    "suspicion_level",
    "suspicion_score",
    "tile_total_count",
    "tile_non_background_count",
    "tile_tumor_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SlideBank readable JSONL records into a UI-friendly global/region/patch tree."
    )
    parser.add_argument(
        "--readable-dir",
        default="",
        help="Directory containing per-slide readable *.jsonl files.",
    )
    parser.add_argument(
        "--bank-dir",
        default="",
        help="Memory-bank directory. Used to infer --readable-dir=<bank-dir>/readable and default output dir.",
    )
    parser.add_argument("--readable-file", default="", help="Convert one readable JSONL file.")
    parser.add_argument("--slide-id", default="", help="Optional exact slide id or prefix filter.")
    parser.add_argument("--out-dir", default="", help="Output directory for *.ui_tree.json files.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of slides to export.")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON instead of pretty JSON.")
    parser.add_argument(
        "--no-anchor-signals",
        action="store_true",
        help="Do not include per-anchor signal_screening blocks.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_json_payload(text: Any, prefix: str = "") -> Any:
    raw = str(text or "").strip()
    if prefix and raw.startswith(prefix):
        raw = raw[len(prefix):].strip()
    if not prefix and raw and raw[0] not in "[{":
        match = re.search(r"([\[{].*)$", raw, flags=re.S)
        if match:
            raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def record_subtype(record: Dict[str, Any]) -> str:
    record_id = str(record.get("id", ""))
    if "::global::" in record_id:
        return record_id.rsplit("::global::", 1)[1]
    if record_id.endswith("::global"):
        return "summary"
    return ""


def catalog_from_records(records: List[Dict[str, Any]], subtype: str, default: Any) -> Any:
    for record in records:
        if record_subtype(record) != subtype:
            continue
        parsed = parse_json_payload(record.get("text", ""))
        return parsed if parsed is not None else default
    return default


def first_record_text(records: List[Dict[str, Any]], subtype: str) -> str:
    for record in records:
        if record_subtype(record) == subtype:
            return str(record.get("text", "")).strip()
    return ""


def infer_slide_id(records: List[Dict[str, Any]], path: Path) -> str:
    for record in records:
        slide_id = str(record.get("slide_id", "")).strip()
        if slide_id:
            return slide_id
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def normalize_path(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    # Legacy banks may contain paths relative to the builder's working
    # directory. Resolve only files that are present there; leave other
    # relative paths intact so an existing portable bundle keeps its layout.
    if text and not Path(text).is_absolute() and Path(text).expanduser().is_file():
        return str(Path(text).expanduser().resolve())
    return text or None


def metric_subset(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def build_signal_lookup(ontology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    signals = ontology.get("signals", []) if isinstance(ontology, dict) else []
    lookup: Dict[str, Dict[str, Any]] = {}
    if not isinstance(signals, list):
        return lookup
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("signal_id", ""))
        if not signal_id:
            continue
        lookup[signal_id] = {
            "signal_id": signal_id,
            "signal_name": signal.get("signal_name"),
            "description": signal.get("description"),
            "weight": signal.get("weight"),
        }
    return lookup


def summarize_signal_overview(
    signal_screenings: List[Dict[str, Any]],
    signal_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    present: Dict[str, Dict[str, Any]] = {}
    for item in signal_screenings:
        for signal in item.get("signal_screening", []) or []:
            status = str(signal.get("status", "unknown")).lower()
            status_counts[status] += 1
            if status == "present":
                signal_id = str(signal.get("signal_id", ""))
                base = dict(signal_lookup.get(signal_id, {"signal_id": signal_id}))
                base["confidence"] = signal.get("confidence")
                base["rationale"] = signal.get("rationale")
                present[signal_id] = base
    return {
        "present_count": int(status_counts.get("present", 0)),
        "absent_count": int(status_counts.get("absent", 0)),
        "present_signals": list(present.values()),
        "candidate_signal_types": list(signal_lookup.values()),
    }


def grid_shape_from_tiles(tiles_survey: List[Dict[str, Any]]) -> Optional[List[int]]:
    rows: List[int] = []
    cols: List[int] = []
    for item in tiles_survey:
        grid = item.get("grid_rc")
        if isinstance(grid, list) and len(grid) == 2:
            try:
                rows.append(int(grid[0]))
                cols.append(int(grid[1]))
            except Exception:
                pass
    if not rows or not cols:
        return None
    return [max(rows) + 1, max(cols) + 1]


def best_signal(signal_screening: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rank = {"present": 3, "unknown": 2, "absent": 1}

    def score(signal: Dict[str, Any]) -> Tuple[int, float]:
        status_score = rank.get(str(signal.get("status", "")).lower(), 0)
        try:
            confidence = float(signal.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return status_score, confidence

    signals = [signal for signal in signal_screening if isinstance(signal, dict)]
    return max(signals, key=score) if signals else None


def candidate_signals_from_package(package: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for signal in package.get("candidate_signals", []) or []:
        if not isinstance(signal, dict):
            continue
        result.append(
            {
                "signal_id": signal.get("signal_id"),
                "signal_name": signal.get("signal_name"),
                "description": signal.get("description"),
                "confidence": signal.get("confidence"),
            }
        )
    return result


def anchor_summary_text(evidence: Dict[str, Any], anchor: Dict[str, Any]) -> str:
    summary = str(evidence.get("summary") or "").strip()
    if summary:
        return summary
    desc = str(anchor.get("anchor_description") or "").strip()
    if not desc:
        return ""
    return desc.split(" Cues:", 1)[0].strip()


def anchor_confidence(
    evidence: Dict[str, Any],
    anchor: Dict[str, Any],
    screening: Dict[str, Any],
) -> Any:
    if evidence.get("confidence") is not None:
        return evidence.get("confidence")
    values: List[float] = []
    for view in evidence.get("views", []) or []:
        try:
            values.append(float(view.get("selection_confidence")))
        except Exception:
            pass
    if values:
        return round(max(values), 4)
    best = best_signal(screening.get("signal_screening", []) or [])
    if best and best.get("confidence") is not None:
        return best.get("confidence")
    return anchor.get("focus_confidence") or anchor.get("label_score")


def view_rows(evidence: Dict[str, Any], patch_by_key: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view in evidence.get("views", []) or []:
        evidence_id = str(evidence.get("evidence_id", ""))
        view_id = str(view.get("view_id", ""))
        patch_meta = patch_by_key.get((evidence_id, view_id), {})
        rows.append(
            {
                "view_id": view_id,
                "mag": view.get("mag") or patch_meta.get("mag"),
                "bbox_xywh_level0": view.get("patch_xywh_level0") or patch_meta.get("patch_xywh_level0"),
                "patch_path": normalize_path(view.get("patch_path") or patch_meta.get("patch_path")),
                "caption": view.get("caption", ""),
                "selection_confidence": view.get("selection_confidence") or patch_meta.get("selection_confidence"),
            }
        )
    return rows


def signal_screening_rows(
    screening: Dict[str, Any],
    signal_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for signal in screening.get("signal_screening", []) or []:
        signal_id = str(signal.get("signal_id", ""))
        base = dict(signal_lookup.get(signal_id, {"signal_id": signal_id}))
        base.update(
            {
                "status": signal.get("status"),
                "confidence": signal.get("confidence"),
                "supporting_views": signal.get("supporting_views", []),
                "rationale": signal.get("rationale"),
            }
        )
        rows.append(base)
    return rows


def signal_screening_summary(screening: Dict[str, Any]) -> Dict[str, Any]:
    rows = [row for row in screening.get("signal_screening", []) or [] if isinstance(row, dict)]
    counts: Counter[str] = Counter(str(row.get("status", "unknown")).lower() for row in rows)
    present = []
    for row in rows:
        if str(row.get("status", "")).lower() != "present":
            continue
        present.append(row.get("signal_name") or row.get("signal_id"))
    return {
        "num_present": int(counts.get("present", 0)),
        "num_absent": int(counts.get("absent", 0)),
        "present_signals": present,
    }


def candidate_signal_names(screening: Dict[str, Any], max_count: int = 4) -> List[str]:
    rows = [row for row in screening.get("signal_screening", []) or [] if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            str(row.get("status", "")).lower() == "present",
            float(row.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    names: List[str] = []
    for row in rows:
        name = str(row.get("signal_name") or row.get("signal_id") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= max_count:
            break
    return names


def build_ui_tree(records: List[Dict[str, Any]], readable_path: Path, include_anchor_signals: bool = True) -> Dict[str, Any]:
    slide_id = infer_slide_id(records, readable_path)
    summary = first_record_text(records, "summary")
    metrics_raw = parse_json_payload(first_record_text(records, "metrics"), "Structured global metrics:") or {}
    qc_raw = parse_json_payload(first_record_text(records, "qc"), "Global segmentation QC:") or {}
    thumbnail_meta = catalog_from_records(records, "thumbnail_meta", {}) or {}
    tiles_survey = catalog_from_records(records, "tiles_survey", []) or []
    tiles_catalog = catalog_from_records(records, "tiles_catalog", []) or []
    packages = catalog_from_records(records, "tile_anchor_packages", []) or []
    anchors_catalog = catalog_from_records(records, "anchors_catalog", []) or []
    patch_views = catalog_from_records(records, "patch_views_catalog", []) or []
    evidence_items = catalog_from_records(records, "evidence_items", []) or []
    signal_screenings = catalog_from_records(records, "signal_screenings", []) or []
    signal_ontology = catalog_from_records(records, "signal_ontology", {}) or {}

    signal_lookup = build_signal_lookup(signal_ontology)
    evidence_by_id = {str(item.get("evidence_id", "")): item for item in evidence_items if isinstance(item, dict)}
    anchor_by_id = {str(item.get("anchor_id", "")): item for item in anchors_catalog if isinstance(item, dict)}
    screening_by_evidence = {
        str(item.get("evidence_id", "")): item for item in signal_screenings if isinstance(item, dict)
    }
    patch_by_key = {
        (str(item.get("evidence_id", "")), str(item.get("view_id", ""))): item
        for item in patch_views
        if isinstance(item, dict)
    }
    tiles_by_id = {str(item.get("tile_id", "")): item for item in tiles_catalog if isinstance(item, dict)}
    packages_by_tile = {str(item.get("tile_id", "")): item for item in packages if isinstance(item, dict)}

    tile_ids = sorted(set(tiles_by_id) | set(packages_by_tile))
    tiles_out: List[Dict[str, Any]] = []
    for tile_id in tile_ids:
        tile = tiles_by_id.get(tile_id, {})
        package = packages_by_tile.get(tile_id, {})
        anchor_refs = package.get("anchors", []) or []
        if not anchor_refs:
            anchor_refs = [a for a in anchors_catalog if a.get("tile_id") == tile_id]

        anchors_summary: List[Dict[str, Any]] = []
        anchors_out: List[Dict[str, Any]] = []
        for anchor_ref in anchor_refs:
            anchor_id = str(anchor_ref.get("anchor_id", ""))
            evidence_id = str(anchor_ref.get("evidence_id", ""))
            anchor = {**anchor_ref, **anchor_by_id.get(anchor_id, {})}
            evidence = evidence_by_id.get(evidence_id, {})
            screening = screening_by_evidence.get(evidence_id, {})
            views = view_rows(evidence, patch_by_key)
            summary_text = anchor_summary_text(evidence, anchor)
            confidence = anchor_confidence(evidence, anchor, screening)

            anchors_summary.append(
                {
                    "anchor_id": anchor_id,
                    "evidence_id": evidence_id,
                    "anchor_bbox_xywh": anchor.get("anchor_bbox_xywh"),
                    "summary": summary_text,
                    "confidence": confidence,
                    "num_views": len(views),
                    "candidate_signals": candidate_signal_names(screening),
                    "signal_screening_summary": signal_screening_summary(screening),
                }
            )
            anchor_obj: Dict[str, Any] = {
                "anchor_id": anchor_id,
                "evidence_id": evidence_id,
                "anchor_bbox_xywh": anchor.get("anchor_bbox_xywh"),
                "views": views,
            }
            if include_anchor_signals:
                anchor_obj["signal_screening"] = signal_screening_rows(screening, signal_lookup)
            anchors_out.append(anchor_obj)

        tiles_out.append(
            {
                "tile_id": tile_id,
                "grid_rc": tile.get("grid_rc"),
                "bbox_xywh_level0": tile.get("bbox_level0_xywh") or package.get("tile_bbox_xywh"),
                "thumb_bbox_xywh": tile.get("thumb_bbox_xywh"),
                "tile_image_path": normalize_path(tile.get("tile_png_path")),
                "coarse_label": tile.get("coarse_label") or package.get("coarse_label"),
                "label_score": tile.get("label_score") or package.get("label_score"),
                "tissue_ratio": tile.get("tissue_ratio"),
                "tile_summary": tile.get("tile_desc") or package.get("tile_desc"),
                "candidate_signals": candidate_signals_from_package(package),
                "anchors_summary": anchors_summary,
                "anchors": anchors_out,
            }
        )

    thumbnail = {
        "thumbnail_path": normalize_path(thumbnail_meta.get("thumbnail_path")),
        "debug_grid_overlay_path": normalize_path(
            thumbnail_meta.get("debug_grid_overlay_path") or qc_raw.get("debug_grid_overlay_path")
        ),
        "slide_dimensions_level0": thumbnail_meta.get("level0_size"),
        "grid_shape": grid_shape_from_tiles(tiles_survey),
    }

    return {
        "slide_id": slide_id,
        "schema_version": SCHEMA_VERSION,
        "global": {
            "summary": summary,
            "metrics": metric_subset(metrics_raw),
            "signal_overview": summarize_signal_overview(signal_screenings, signal_lookup),
            "thumbnail": thumbnail,
        },
        "tiles": tiles_out,
    }


def candidate_files(args: argparse.Namespace) -> List[Path]:
    if args.readable_file:
        return [Path(args.readable_file).expanduser()]
    readable_dir = Path(args.readable_dir).expanduser() if args.readable_dir else None
    if readable_dir is None and args.bank_dir:
        readable_dir = Path(args.bank_dir).expanduser() / "readable"
    if readable_dir is None:
        raise SystemExit("Provide --readable-dir, --readable-file, or --bank-dir.")
    files = sorted(readable_dir.glob("*.jsonl"))
    if args.slide_id:
        query = args.slide_id
        filtered = []
        for path in files:
            name = path.name[:-6] if path.name.endswith(".jsonl") else path.name
            if name == query or name.startswith(query) or query in name:
                filtered.append(path)
        files = filtered
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    return files


def default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser()
    if args.bank_dir:
        return Path(args.bank_dir).expanduser() / "ui_tree"
    if args.readable_dir:
        return Path(args.readable_dir).expanduser().parent / "ui_tree"
    return Path(args.readable_file).expanduser().parent.parent / "ui_tree"


def main() -> None:
    args = parse_args()
    files = candidate_files(args)
    if not files:
        raise SystemExit("No readable JSONL files matched the request.")
    out_dir = default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    indent = None if args.compact else 2

    exported = []
    for readable_path in files:
        records = load_jsonl(readable_path)
        tree = build_ui_tree(records, readable_path, include_anchor_signals=not args.no_anchor_signals)
        out_name = f"{safe_name(Path(tree['slide_id']).stem)}.ui_tree.json"
        out_path = out_dir / out_name
        out_path.write_text(
            json.dumps(tree, ensure_ascii=False, indent=indent, separators=(",", ":") if args.compact else None)
            + "\n",
            encoding="utf-8",
        )
        exported.append(
            {
                "slide_id": tree["slide_id"],
                "out_path": str(out_path),
                "tiles": len(tree["tiles"]),
                "anchors": sum(len(tile.get("anchors", [])) for tile in tree["tiles"]),
                "views": sum(len(anchor.get("views", [])) for tile in tree["tiles"] for anchor in tile.get("anchors", [])),
            }
        )
        print(
            f"exported {tree['slide_id']} -> {out_path} "
            f"(tiles={exported[-1]['tiles']}, anchors={exported[-1]['anchors']}, views={exported[-1]['views']})"
        )

    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "exports": exported}, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
