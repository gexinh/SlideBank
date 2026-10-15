from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List

from .data import build_svs_index, find_svs_for_slide, load_mcq_rows, rows_by_slide, safe_name
from .wsi_agent.loader import RealWSILoader
from .wsi_agent.memory_bank import HierarchicalMemoryBank


@dataclass
class PatchVLMConfig:
    backend: str = "local_hf"
    model_name: str = "wisdomik/Quilt-Llava-v1.5-7b"
    model_path: str = "wisdomik/Quilt-Llava-v1.5-7b"
    quilt_llava_repo: str = ""
    device_map: str = "auto"
    dtype: str = "fp16"
    batch_size: int = 32
    max_new_tokens: int = 224
    api_provider: str = ""
    api_key: str = ""
    api_key_env: str = ""
    api_key_file: str = ""
    api_base_url: str = ""
    api_version: str = "v1beta"
    api_timeout_sec: int = 300
    gemini_api_key: str = ""
    gemini_api_env: str = "GEMINI_API_KEY"
    gemini_timeout: int = 120


@dataclass
class BankBuildConfig:
    csv_path: str
    svs_root: str
    persist_dir: str
    artifact_dir: str
    svs_pattern: str = "*/*.svs"
    max_slides: int = 0
    rebuild: bool = False
    force_ingest: bool = False
    use_llm_patch_describer: bool = True
    patch_description_strict: bool = True
    thumbnail_tile_size: int = 384
    anchor_crop_size_level0: int = 4096
    patch_crop_size_level0: int = 512
    quiet: bool = False
    patch_vlm: PatchVLMConfig = field(default_factory=PatchVLMConfig)


def make_loader(config: BankBuildConfig) -> RealWSILoader:
    patch = config.patch_vlm
    return RealWSILoader(
        use_llm_patch_describer=bool(config.use_llm_patch_describer),
        patch_description_strict=bool(config.patch_description_strict),
        tile_llm_backend=str(patch.backend),
        tile_llm_model_name=str(patch.model_name),
        tile_llm_model_path=str(patch.model_path),
        tile_llm_device_map=str(patch.device_map),
        tile_llm_dtype=str(patch.dtype),
        quilt_llava_repo=str(patch.quilt_llava_repo),
        gemini_api_key=str(patch.gemini_api_key),
        gemini_api_env=str(patch.gemini_api_env),
        gemini_request_timeout=int(patch.gemini_timeout),
        api_provider=str(patch.api_provider),
        api_key=str(patch.api_key),
        api_key_env=str(patch.api_key_env),
        api_key_file=str(patch.api_key_file),
        api_base_url=str(patch.api_base_url),
        api_version=str(patch.api_version),
        api_timeout_sec=int(patch.api_timeout_sec),
        patch_llm_max_new_tokens=int(patch.max_new_tokens),
        patch_llm_batch_size=int(patch.batch_size),
        thumbnail_tile_size=int(config.thumbnail_tile_size),
        anchor_crop_size_level0=int(config.anchor_crop_size_level0),
        patch_crop_size_level0=int(config.patch_crop_size_level0),
        verbose=not bool(config.quiet),
    )


def open_memory_bank(persist_dir: str, rebuild: bool = False, quiet: bool = False) -> HierarchicalMemoryBank:
    return HierarchicalMemoryBank(
        collection_name="wsi_vqa_eval_memory",
        persist_directory=persist_dir,
        reset_collection=bool(rebuild),
        verbose=not bool(quiet),
    )


def build_bank(config: BankBuildConfig) -> Dict[str, Any]:
    # Artifact paths become part of the persistent records and UI export. Bind
    # them to this run's working directory before handing them to the loader,
    # so a tree exported elsewhere still points to the same files. Leave the
    # caller's config intact for reuse by a CLI, notebook, or UI session.
    config = replace(config, **{
        key: os.path.abspath(os.path.expanduser(os.path.expandvars(getattr(config, key))))
        for key in ("csv_path", "svs_root", "persist_dir", "artifact_dir")
    })
    start = time.perf_counter()
    os.makedirs(config.persist_dir, exist_ok=True)
    os.makedirs(config.artifact_dir, exist_ok=True)
    if config.rebuild:
        fallback = os.path.join(config.persist_dir, "wsi_vqa_eval_memory_fallback.json")
        for path in [fallback, f"{fallback}.jsonl", f"{fallback}.shards"]:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)

    rows = load_mcq_rows(config.csv_path)
    grouped = rows_by_slide(rows)
    slide_prefixes = sorted(grouped.keys())
    if config.max_slides > 0:
        slide_prefixes = slide_prefixes[: config.max_slides]

    svs_index = build_svs_index(config.svs_root, pattern=config.svs_pattern)
    loader = make_loader(config)
    memory_bank = open_memory_bank(config.persist_dir, rebuild=False, quiet=config.quiet)

    summary: Dict[str, Any] = {
        "csv_path": config.csv_path,
        "svs_root": config.svs_root,
        "persist_dir": config.persist_dir,
        "artifact_dir": config.artifact_dir,
        "patch_backend": config.patch_vlm.backend,
        "patch_api_provider": config.patch_vlm.api_provider,
        "patch_model": config.patch_vlm.model_path or config.patch_vlm.model_name,
        "slides_requested": len(slide_prefixes),
        "slides_ingested": 0,
        "slides_reused": 0,
        "missing_svs": [],
        "errors": [],
    }

    for idx, slide_prefix in enumerate(slide_prefixes, start=1):
        svs_path = find_svs_for_slide(slide_prefix, svs_index)
        if not svs_path:
            summary["missing_svs"].append(slide_prefix)
            print(f"[{idx}/{len(slide_prefixes)}] missing SVS: {slide_prefix}")
            continue
        slide_id = os.path.basename(svs_path)
        if not config.force_ingest and memory_bank.has_slide(slide_id):
            summary["slides_reused"] += 1
            print(f"[{idx}/{len(slide_prefixes)}] reuse memory: {slide_id}")
            continue
        artifact_subdir = os.path.join(config.artifact_dir, safe_name(os.path.splitext(slide_id)[0]))
        os.makedirs(artifact_subdir, exist_ok=True)
        try:
            print(f"[{idx}/{len(slide_prefixes)}] ingest: {slide_id}")
            slide_data = loader.load(svs_path, out_dir=artifact_subdir)
            memory_bank.ingest_slide(slide_data)
            summary["slides_ingested"] += 1
        except Exception as exc:
            message = f"{slide_id}: {type(exc).__name__}: {exc}"
            summary["errors"].append(message)
            print(f"[{idx}/{len(slide_prefixes)}] ERROR {message}")

    summary["elapsed_sec"] = round(time.perf_counter() - start, 3)
    summary_path = os.path.join(config.persist_dir, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["summary_path"] = summary_path
    return summary
