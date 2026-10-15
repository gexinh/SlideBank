from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .bank import open_memory_bank
from .data import (
    build_svs_index,
    ensure_parent,
    find_svs_for_slide,
    infer_gt_label,
    load_mcq_rows,
    pick_option_from_response,
    resolve_thumbnail_path,
    rows_by_slide,
)
from .retrieval import RetrievalConfig, build_answer_prompt, retrieve_tool_context
from .vlm import VLMClient, VLMConfig


@dataclass
class InferenceConfig:
    csv_path: str
    svs_root: str
    thumbnail_root: str
    persist_dir: str
    artifact_dir: str
    out_jsonl: str
    out_summary: str
    svs_pattern: str = "*/*.svs"
    max_samples: int = 0
    max_slides: int = 0
    mode: str = "single"
    quiet_memory: bool = True
    vlm: VLMConfig = field(default_factory=VLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_inference(config: InferenceConfig) -> Dict[str, Any]:
    start = time.perf_counter()
    rows = load_mcq_rows(config.csv_path)
    grouped = rows_by_slide(rows)
    slide_prefixes = sorted(grouped.keys())
    if config.max_slides > 0:
        slide_prefixes = slide_prefixes[: config.max_slides]
    svs_index = build_svs_index(config.svs_root, pattern=config.svs_pattern)
    memory_bank = open_memory_bank(config.persist_dir, rebuild=False, quiet=config.quiet_memory)
    vlm = VLMClient(config.vlm)

    predictions: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    errors = 0
    history_by_slide: Dict[str, List[Dict[str, str]]] = {}

    for slide_prefix in slide_prefixes:
        svs_path = find_svs_for_slide(slide_prefix, svs_index)
        slide_rows = grouped.get(slide_prefix, [])
        if not svs_path:
            for row in slide_rows:
                predictions.append({"row_id": row.row_id, "slide": row.slide, "error": "MISSING_SVS"})
                errors += 1
            continue
        slide_id = os.path.basename(svs_path)
        if not memory_bank.has_slide(slide_id):
            for row in slide_rows:
                predictions.append({"row_id": row.row_id, "slide": row.slide, "svs": svs_path, "error": "MISSING_MEMORY"})
                errors += 1
            continue
        memory_bank.active_slide_id = slide_id
        for row in slide_rows:
            if config.max_samples > 0 and total >= config.max_samples:
                break
            total += 1
            image_path = resolve_thumbnail_path(row.image, svs_path, config.thumbnail_root, config.artifact_dir)
            gt_label = infer_gt_label(row.answer, row.options, row.gt)
            if not image_path or not os.path.exists(image_path):
                predictions.append(
                    {
                        "row_id": row.row_id,
                        "slide": row.slide,
                        "svs": svs_path,
                        "question": row.question,
                        "error": f"IMAGE_NOT_FOUND: {image_path}",
                    }
                )
                errors += 1
                continue

            retrieval = retrieve_tool_context(memory_bank, slide_id, row.question, config.retrieval)
            previous = history_by_slide.get(slide_id, []) if config.mode == "multi" else []
            prompt = build_answer_prompt(row.question, row.options, retrieval["tool_logs"], previous_turns=previous)
            try:
                raw = vlm.generate(image_path=image_path, prompt=prompt)
                pred_text, pred_label, score = pick_option_from_response(raw, row.options)
                is_correct = bool(gt_label and pred_label == gt_label)
                correct += int(is_correct)
                record = {
                    "row_id": row.row_id,
                    "slide": row.slide,
                    "svs": svs_path,
                    "image": image_path,
                    "question": row.question,
                    "options": dict(zip(["A", "B", "C", "D"], row.options)),
                    "answer": row.answer,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "pred_text": pred_text,
                    "raw_output": raw,
                    "score": score,
                    "correct": is_correct,
                    "route": retrieval["route"],
                    "retrieval_calls": retrieval["retrieval_calls"],
                    "retrieval_errors": retrieval["retrieval_errors"],
                    "tool_logs": retrieval["tool_logs"],
                }
                predictions.append(record)
                if config.mode == "multi":
                    history_by_slide.setdefault(slide_id, []).append(
                        {
                            "turn_id": str(len(history_by_slide.get(slide_id, [])) + 1),
                            "question": row.question,
                            "pred_label": pred_label,
                            "pred_text": pred_text,
                        }
                    )
            except Exception as exc:
                errors += 1
                predictions.append(
                    {
                        "row_id": row.row_id,
                        "slide": row.slide,
                        "svs": svs_path,
                        "image": image_path,
                        "question": row.question,
                        "gt_label": gt_label,
                        "error": f"{type(exc).__name__}: {exc}",
                        "route": retrieval.get("route", {}),
                        "retrieval_errors": retrieval.get("retrieval_errors", []),
                    }
                )
        if config.max_samples > 0 and total >= config.max_samples:
            break

    _write_jsonl(config.out_jsonl, predictions)
    summary: Dict[str, Any] = {
        "csv_path": config.csv_path,
        "svs_root": config.svs_root,
        "persist_dir": config.persist_dir,
        "artifact_dir": config.artifact_dir,
        "mode": config.mode,
        "vlm_backend": config.vlm.backend,
        "api_provider": config.vlm.api_provider
        or {
            "api": "gemini",
            "gemini_api": "gemini",
            "openai_api": "openai",
            "openai_compatible_api": "openai_compatible",
        }.get(config.vlm.backend, ""),
        "vlm_model": config.vlm.model_path,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "errors": errors,
        "out_jsonl": config.out_jsonl,
        "elapsed_sec": round(time.perf_counter() - start, 3),
    }
    _write_json(config.out_summary, summary)
    return summary
