from __future__ import annotations

import base64
import html
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


LABEL_COLORS = {
    "tumor": "#DC2626",
    "carcinoma": "#DC2626",
    "stroma": "#2563EB",
    "background": "#9CA3AF",
    "normal": "#10B981",
    "uncertain": "#F59E0B",
}
ANCHOR_COLOR = "#7C3AED"
PATCH_COLOR = "#10B981"
SELECTED_COLOR = "#F59E0B"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#6B7280"


def compact_text(value: Any, max_chars: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def full_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_artifact_path(value: Any) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser()) if raw else ""


def load_bank_json(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("global"), dict) or not isinstance(data.get("tiles"), list):
        raise ValueError("The selected JSON is not a SlideBank UI-tree file.")
    if data.get("schema_version") != "slidebank-ui-tree-v1":
        raise ValueError("Expected schema_version='slidebank-ui-tree-v1'; export the bank with scripts/export_ui_tree.py.")

    def resolve_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item and (key.endswith("_path") or key in {"persist_dir", "artifact_dir"}):
                    artifact = Path(item).expanduser()
                    value[key] = str(artifact if artifact.is_absolute() else source.parent / artifact)
                elif isinstance(item, (dict, list)):
                    resolve_paths(item)
        elif isinstance(value, list):
            for item in value:
                resolve_paths(item)

    resolve_paths(data)
    return data


def slide_prefix(slide_id: str) -> str:
    match = re.match(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)", slide_id)
    if match:
        return match.group(1)
    return Path(slide_id).stem.split(".", 1)[0]


def build_demo_registry(ui_tree_dir: str | Path, limit: int = 0) -> Dict[str, Dict[str, str]]:
    root = Path(ui_tree_dir).expanduser()
    files = sorted(root.glob("*.ui_tree.json"))
    ordered = files
    if limit and limit > 0:
        ordered = ordered[:limit]
    registry: Dict[str, Dict[str, str]] = {}
    for path in ordered:
        try:
            data = load_bank_json(path)
        except Exception:
            continue
        label = slide_prefix(str(data.get('slide_id', path.stem)))
        suffix = 2
        unique = label
        while unique in registry:
            suffix += 1
            unique = f"{label} {suffix}"
        registry[unique] = {"bank_json_path": str(path), "slide_id": str(data.get("slide_id", ""))}
    return registry


def tile_choices(bank: Optional[Dict[str, Any]]) -> List[str]:
    choices = []
    for tile in (bank or {}).get("tiles", []) or []:
        tile_id = str(tile.get("tile_id", "")).strip()
        if not tile_id:
            continue
        label = str(tile.get("coarse_label") or "unknown")
        score = tile.get("label_score")
        suffix = f" · {label}"
        if isinstance(score, (int, float)):
            suffix += f" · {score:.2f}"
        choices.append(tile_id + suffix)
    return choices


def parse_tile_choice(choice: str) -> str:
    return str(choice or "").split(" · ", 1)[0].strip()


def anchor_choices(tile: Optional[Dict[str, Any]]) -> List[str]:
    choices = []
    for item in (tile or {}).get("anchors_summary", []) or []:
        anchor_id = str(item.get("anchor_id", "")).strip()
        if not anchor_id:
            continue
        conf = item.get("confidence")
        views = item.get("num_views")
        parts = [anchor_id]
        if isinstance(conf, (int, float)):
            parts.append(f"confidence {conf:.2f}")
        if views is not None:
            parts.append(f"{views} views")
        choices.append(" · ".join(parts))
    return choices


def parse_anchor_choice(choice: str) -> str:
    return str(choice or "").split(" · ", 1)[0].strip()


def find_tile(bank: Optional[Dict[str, Any]], tile_id: str) -> Optional[Dict[str, Any]]:
    query = parse_tile_choice(tile_id)
    for tile in (bank or {}).get("tiles", []) or []:
        if str(tile.get("tile_id", "")) == query:
            return tile
    tiles = (bank or {}).get("tiles", []) or []
    return tiles[0] if tiles else None


def find_anchor(tile: Optional[Dict[str, Any]], anchor_id: str) -> Optional[Dict[str, Any]]:
    query = parse_anchor_choice(anchor_id)
    for anchor in (tile or {}).get("anchors", []) or []:
        if str(anchor.get("anchor_id", "")) == query:
            return anchor
    anchors = (tile or {}).get("anchors", []) or []
    return anchors[0] if anchors else None


def find_anchor_summary(tile: Optional[Dict[str, Any]], anchor_id: str) -> Dict[str, Any]:
    query = parse_anchor_choice(anchor_id)
    for item in (tile or {}).get("anchors_summary", []) or []:
        if str(item.get("anchor_id", "")) == query:
            return item
    return {}


def first_tile_id(bank: Optional[Dict[str, Any]]) -> str:
    tiles = (bank or {}).get("tiles", []) or []
    return str(tiles[0].get("tile_id", "")) if tiles else ""


def first_anchor_id(tile: Optional[Dict[str, Any]]) -> str:
    anchors = (tile or {}).get("anchors", []) or []
    return str(anchors[0].get("anchor_id", "")) if anchors else ""


def _open_image(path: str, size: Tuple[int, int] = (960, 560)) -> Image.Image:
    path = normalize_artifact_path(path)
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    image = Image.new("RGB", size, "#F3F4F6")
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "Image not available", fill=MUTED_COLOR)
    return image


def _draw_text_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: str = "#FFFFFF") -> None:
    x, y = xy
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=fill,
        outline="#E5E7EB",
    )
    draw.text((x, y), text, fill=TEXT_COLOR, font=font)


def _hex_to_rgba(color: str, alpha: int) -> Tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def render_global_overlay(bank: Optional[Dict[str, Any]], selected_tile_id: str = "") -> Image.Image:
    if not bank:
        return _open_image("", size=(1100, 620))
    thumb = (bank.get("global", {}) or {}).get("thumbnail", {}) or {}
    image = _open_image(thumb.get("thumbnail_path") or thumb.get("debug_grid_overlay_path"))
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    selected = parse_tile_choice(selected_tile_id)
    for tile in bank.get("tiles", []) or []:
        bbox = tile.get("thumb_bbox_xywh")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x, y, w, h = [int(round(float(v))) for v in bbox]
        label = str(tile.get("coarse_label") or "uncertain").lower()
        color = LABEL_COLORS.get(label, LABEL_COLORS["uncertain"])
        tile_id = str(tile.get("tile_id", ""))
        draw.rectangle((x, y, x + w, y + h), fill=_hex_to_rgba(color, 34), outline=color, width=2)
        if tile_id == selected:
            draw.rectangle((x + 2, y + 2, x + w - 2, y + h - 2), outline=SELECTED_COLOR, width=5)
        if w > 58 and h > 24:
            _draw_text_box(draw, (x + 6, y + 6), tile_id)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _level0_to_tile_px(tile: Dict[str, Any], bbox: Any, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    tile_bbox = tile.get("bbox_xywh_level0")
    if not (isinstance(tile_bbox, list) and len(tile_bbox) == 4 and isinstance(bbox, list) and len(bbox) == 4):
        return None
    tx, ty, tw, th = [float(v) for v in tile_bbox]
    bx, by, bw, bh = [float(v) for v in bbox]
    if tw <= 0 or th <= 0:
        return None
    iw, ih = image_size
    x1 = int(round((bx - tx) / tw * iw))
    y1 = int(round((by - ty) / th * ih))
    x2 = int(round((bx + bw - tx) / tw * iw))
    y2 = int(round((by + bh - ty) / th * ih))
    return x1, y1, x2, y2


def render_tile_overlay(
    bank: Optional[Dict[str, Any]],
    tile_id: str,
    selected_anchor_id: str = "",
    selected_patch_id: str = "",
) -> Image.Image:
    tile = find_tile(bank, tile_id)
    if not tile:
        return _open_image("", size=(900, 620))
    image = _open_image(tile.get("tile_image_path"))
    draw = ImageDraw.Draw(image)
    selected_anchor = parse_anchor_choice(selected_anchor_id)
    selected_patch = str(selected_patch_id or "")
    for summary in tile.get("anchors_summary", []) or []:
        anchor_id = str(summary.get("anchor_id", ""))
        rect = _level0_to_tile_px(tile, summary.get("anchor_bbox_xywh"), image.size)
        if not rect:
            continue
        color = SELECTED_COLOR if anchor_id == selected_anchor else ANCHOR_COLOR
        draw.rectangle(rect, outline=color, width=5 if anchor_id == selected_anchor else 3)
        _draw_text_box(draw, (max(0, rect[0] + 4), max(0, rect[1] + 4)), anchor_id)
    for anchor in tile.get("anchors", []) or []:
        for view in anchor.get("views", []) or []:
            patch_key = f"{anchor.get('anchor_id')}::{view.get('view_id')}"
            rect = _level0_to_tile_px(tile, view.get("bbox_xywh_level0") or view.get("patch_xywh_level0"), image.size)
            if not rect:
                continue
            color = SELECTED_COLOR if patch_key == selected_patch else PATCH_COLOR
            draw.rectangle(rect, outline=color, width=3 if patch_key == selected_patch else 2)
    return image


def signal_overview_markdown(bank: Optional[Dict[str, Any]]) -> str:
    if not bank:
        return "### Global Slide Overview\nNo bank loaded."
    global_data = bank.get("global", {}) or {}
    metrics = global_data.get("metrics", {}) or {}
    signals = global_data.get("signal_overview", {}) or {}
    candidates = signals.get("candidate_signal_types", []) or []
    present = signals.get("present_signals", []) or []
    return (
        "### Global Slide Overview\n"
        f"- Slide ID: `{bank.get('slide_id', 'N/A')}`\n"
        f"- Dominant category: {metrics.get('dominant_category', 'N/A')}\n"
        f"- Suspicion level: {metrics.get('suspicion_level', 'N/A')}\n"
        f"- Tissue ratio: {metrics.get('tissue_ratio', 'N/A')}\n"
        f"- Candidate signal types: {len(candidates)}\n"
        f"- Present signals: {signals.get('present_count', signals.get('num_present', len(present)))}\n\n"
        "### Summary\n"
        f"{compact_text(global_data.get('summary', ''), 520) or 'No global summary available.'}"
    )


def render_tile_summary(bank: Optional[Dict[str, Any]], tile_id: str) -> str:
    tile = find_tile(bank, tile_id)
    if not tile:
        return "### Selected Grid Detail\nNo grid selected."
    lines = [
        "### Selected Grid Detail",
        f"- Grid ID: `{tile.get('tile_id', 'N/A')}`",
        f"- Grid coordinate: {tile.get('grid_rc', 'N/A')}",
        f"- Coarse label: {tile.get('coarse_label', 'N/A')}",
        f"- Label score: {tile.get('label_score', 'N/A')}",
        f"- Tissue ratio: {tile.get('tissue_ratio', 'N/A')}",
        "",
        "### Grid Summary",
        compact_text(tile.get("tile_summary", ""), 520) or "No grid summary available.",
        "",
        "### Anchor Summaries",
    ]
    for item in tile.get("anchors_summary", []) or []:
        summary = compact_text(item.get("summary", ""), 150)
        conf = item.get("confidence")
        conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
        signal_summary = item.get("signal_screening_summary", {}) or {}
        present = signal_summary.get("num_present", 0)
        absent = signal_summary.get("num_absent", 0)
        lines.append(
            f"- `{item.get('anchor_id')}` · confidence {conf_text} · "
            f"{item.get('num_views', 0)} views · signals {present} present / {absent} absent: {summary}"
        )
    return "\n".join(lines)


def render_signal_details(bank: Optional[Dict[str, Any]], tile_id: str, anchor_id: str) -> str:
    tile = find_tile(bank, tile_id)
    anchor = find_anchor(tile, anchor_id)
    if not anchor:
        return "No signal details available."
    rows = []
    for signal in anchor.get("signal_screening", []) or []:
        rows.append(
            "| "
            + " | ".join(
                [
                    str(signal.get("signal_name") or signal.get("signal_id") or ""),
                    str(signal.get("status", "")),
                    str(signal.get("confidence", "")),
                    compact_text(signal.get("rationale", ""), 160),
                ]
            )
            + " |"
        )
    if not rows:
        return "No signal details available."
    return "\n".join(
        [
            "| Signal | Status | Confidence | Rationale |",
            "|---|---:|---:|---|",
            *rows,
        ]
    )


def _image_data_uri(path: str) -> str:
    path = normalize_artifact_path(path)
    if not path or not os.path.exists(path):
        return ""
    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def _patch_card(view: Dict[str, Any], anchor_id: str) -> str:
    title = f"{view.get('view_id', 'View')} · {view.get('mag', '')}"
    confidence = view.get("selection_confidence")
    conf_text = f"confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
    caption = compact_text(view.get("caption", ""), 700)
    src = _image_data_uri(str(view.get("patch_path", "")))
    image_html = f"<img src='{src}' alt='{html.escape(title)}'>" if src else "<div class='missing-patch'>Patch unavailable</div>"
    return (
        "<div class='patch-card' tabindex='0'>"
        f"{image_html}"
        "<div class='patch-meta'>"
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(conf_text)}</span>"
        "</div>"
        f"<div class='patch-tip'><strong>{html.escape(anchor_id)}</strong><br>{html.escape(caption)}</div>"
        "</div>"
    )


def render_patch_cards(bank: Optional[Dict[str, Any]], tile_id: str, anchor_id: str) -> str:
    tile = find_tile(bank, tile_id)
    anchor = find_anchor(tile, anchor_id)
    if not anchor:
        return "<div class='empty-panel'>No anchor selected.</div>"
    views = anchor.get("views", []) or []
    cards = [_patch_card(view, str(anchor.get("anchor_id", ""))) for view in views]
    visible = "".join(cards[:4]) or "<div class='empty-panel'>No patch views available.</div>"
    extra = "".join(cards[4:])
    extra_html = ""
    if extra:
        extra_html = f"<details class='extra-patches'><summary>Show {len(cards) - 4} more patch views</summary><div class='patch-grid'>{extra}</div></details>"
    summary = html.escape(compact_text(find_anchor_summary(tile, anchor.get("anchor_id", "")).get("summary", ""), 280))
    return (
        "<div class='patch-panel'>"
        f"<div class='anchor-line'><strong>{html.escape(str(anchor.get('anchor_id', 'Anchor')))}</strong><span>{summary}</span></div>"
        f"<div class='patch-grid'>{visible}</div>"
        f"{extra_html}"
        "</div>"
    )


def tile_from_thumbnail_click(bank: Optional[Dict[str, Any]], x: int, y: int) -> str:
    for tile in (bank or {}).get("tiles", []) or []:
        bbox = tile.get("thumb_bbox_xywh")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        tx, ty, tw, th = [float(v) for v in bbox]
        if tx <= x <= tx + tw and ty <= y <= ty + th:
            return str(tile.get("tile_id", ""))
    return first_tile_id(bank)
