from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LABELS = ["A", "B", "C", "D"]


@dataclass
class MCQRow:
    idx: int
    row_id: str
    slide: str
    question: str
    options: List[str]
    answer: str = ""
    gt: str = ""
    image: str = ""


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")


def load_mcq_rows(csv_path: str, max_rows: int = 0) -> List[MCQRow]:
    rows: List[MCQRow] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get("Question") or row.get("question") or "").strip()
            slide = (row.get("Slide") or row.get("slide") or "").strip()
            if not question or not slide:
                continue
            item = MCQRow(
                idx=idx,
                row_id=(
                    row.get("RowId")
                    or row.get("row_id")
                    or row.get("Unnamed: 0")
                    or row.get("ID")
                    or str(idx)
                ).strip(),
                slide=slide,
                question=question,
                options=[
                    (row.get("A") or "").strip(),
                    (row.get("B") or "").strip(),
                    (row.get("C") or "").strip(),
                    (row.get("D") or "").strip(),
                ],
                answer=(row.get("Answer") or row.get("answer") or "").strip(),
                gt=(row.get("GT") or row.get("gt") or "").strip(),
                image=(row.get("Image") or row.get("image") or "").strip(),
            )
            rows.append(item)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def build_svs_index(svs_root: str, pattern: str = "*/*.svs") -> Dict[str, str]:
    paths = sorted(glob.glob(os.path.join(svs_root, pattern), recursive=True))
    index: Dict[str, str] = {}
    for path in paths:
        base = os.path.basename(path)
        prefix = base.split("-01Z-")[0]
        index.setdefault(prefix, path)
    return index


def find_svs_for_slide(slide_prefix: str, svs_index: Dict[str, str]) -> Optional[str]:
    if slide_prefix in svs_index:
        return svs_index[slide_prefix]
    for key, path in svs_index.items():
        if key.startswith(slide_prefix) or slide_prefix.startswith(key):
            return path
    return None


def infer_gt_label(answer: str, options: List[str], provided_gt: str = "") -> str:
    gt = (provided_gt or "").strip().upper()
    if gt in LABELS:
        return gt
    answer_norm = norm_text(answer)
    for idx, option in enumerate(options):
        if answer_norm and answer_norm == norm_text(option):
            return LABELS[idx]
    match = re.search(r"\b([ABCD])\b", (answer or "").upper())
    return match.group(1) if match else ""


def pick_option_from_response(response: str, options: List[str]) -> Tuple[str, str, float]:
    response = response or ""
    stripped = response.strip()
    match = re.fullmatch(r"\(?\s*([ABCD])\s*\)?\.?", stripped, flags=re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        idx = LABELS.index(label)
        return options[idx], label, 1.0

    for line in reversed([line.strip() for line in response.splitlines() if line.strip()]):
        match = re.fullmatch(
            r"(?:final\s+answer|answer|option|choice)?\s*[:\-]?\s*\(?\s*([ABCD])\s*\)?\.?",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            label = match.group(1).upper()
            idx = LABELS.index(label)
            return options[idx], label, 1.0

    match = re.search(
        r"\b(?:final\s+answer|answer|option|choice)\s*[:\-]?\s*\(?\s*([ABCD])\s*\)?\b",
        response,
        flags=re.IGNORECASE,
    )
    if match:
        label = match.group(1).upper()
        idx = LABELS.index(label)
        return options[idx], label, 1.0

    match = re.search(r"\b([ABCD])[\).:]", response)
    if match:
        label = match.group(1).upper()
        idx = LABELS.index(label)
        return options[idx], label, 1.0

    response_norm = norm_text(response)
    for idx, option in enumerate(options):
        option_norm = norm_text(option)
        if option_norm and option_norm in response_norm:
            return option, LABELS[idx], 0.99

    best_idx = 0
    best_score = -1.0
    for idx, option in enumerate(options):
        score = SequenceMatcher(None, norm_text(option), response_norm).ratio()
        if score > best_score:
            best_idx = idx
            best_score = score
    return options[best_idx], LABELS[best_idx], float(best_score)


def resolve_thumbnail_path(
    row_image: str,
    svs_path: str,
    thumbnail_root: str,
    artifact_dir: str = "",
) -> str:
    if row_image and os.path.exists(row_image):
        return row_image
    if svs_path and thumbnail_root:
        thumb = os.path.join(thumbnail_root, os.path.basename(svs_path).replace(".svs", ".png"))
        if os.path.exists(thumb):
            return thumb
    if svs_path and artifact_dir:
        stem = os.path.splitext(os.path.basename(svs_path))[0]
        slide_artifacts = os.path.join(artifact_dir, safe_name(stem))
        candidates = sorted(glob.glob(os.path.join(slide_artifacts, "thumb", "*thumbnail*.png")))
        if candidates:
            return candidates[0]
    return row_image or ""


def rows_by_slide(rows: Iterable[MCQRow]) -> Dict[str, List[MCQRow]]:
    grouped: Dict[str, List[MCQRow]] = {}
    for row in rows:
        grouped.setdefault(row.slide, []).append(row)
    return grouped


def ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
