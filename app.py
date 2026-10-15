from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from scripts.export_ui_tree import build_ui_tree, load_jsonl
from slidebank.bank import BankBuildConfig, PatchVLMConfig, build_bank
from slidebank.config import load_config
from slidebank.data import safe_name
from slidebank.agent_inference_v5 import answer_question_v5
from slidebank.ui_v3_backend import (
    anchor_choices,
    build_demo_registry,
    find_anchor,
    find_tile,
    first_anchor_id,
    first_tile_id,
    load_bank_json,
    parse_anchor_choice,
    parse_tile_choice,
    render_global_overlay,
    render_patch_cards,
    render_signal_details,
    render_tile_overlay,
    render_tile_summary,
    signal_overview_markdown,
    tile_choices,
    tile_from_thumbnail_click,
)


APP_ROOT = Path(__file__).resolve().parent
HOSTED_MODE = os.environ.get("SLIDEBANK_HOSTED", "").strip().lower() in {"1", "true", "yes"}
if HOSTED_MODE:
    from hf_space import runtime as hosted_runtime


def _configured_path(variable: str, default: str) -> Path:
    path = Path(os.environ.get(variable, default)).expanduser()
    return path if path.is_absolute() else APP_ROOT / path


UI_TREE_DIR = _configured_path("SLIDEBANK_UI_TREE_DIR", "outputs/ui_tree")
BUILD_CONFIG_PATH = _configured_path("SLIDEBANK_BUILD_CONFIG", "configs/build.yaml")
INFER_CONFIG_PATH = _configured_path("SLIDEBANK_INFER_CONFIG", "configs/infer.yaml")
DEFAULT_HOST = os.environ.get("SLIDEBANK_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SLIDEBANK_PORT", os.environ.get("GRADIO_SERVER_PORT", "7864")))
RUN_BANK_DIR = _configured_path("SLIDEBANK_RUN_DIR", "outputs/workbench_runs")
GRADIO_TEMP_DIR = _configured_path("GRADIO_TEMP_DIR", "outputs/gradio_tmp")
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TEMP_DIR))


def _workbench_defaults(config_path: Path) -> Dict[str, Any]:
    defaults = load_config(str(config_path)) if config_path.exists() else {}
    for section, keys in {
        "dataset": ("csv_path", "svs_root", "thumbnail_root"),
        "bank": ("persist_dir", "artifact_dir"),
        "output": ("jsonl", "summary"),
    }.items():
        values = defaults.get(section, {})
        if not isinstance(values, dict):
            continue
        for key in keys:
            if values.get(key):
                path = Path(values[key]).expanduser()
                values[key] = str(path if path.is_absolute() else APP_ROOT / path)
    return defaults


DEMO_REGISTRY = build_demo_registry(UI_TREE_DIR)
BUILD_DEFAULTS = _workbench_defaults(BUILD_CONFIG_PATH)
INFER_DEFAULTS = _workbench_defaults(INFER_CONFIG_PATH)
API_PROVIDERS = ["Google", "OpenAI", "Custom Endpoint"]
LOCAL_MODEL_CHOICES = ["Quilt-LLaVA", "Qwen2.5-VL-7B-Instruct"]


def _default_demo() -> Optional[str]:
    return next(iter(DEMO_REGISTRY.keys()), None)


def _slide_prefix_from_demo(demo_slide_name: str) -> str:
    entry = DEMO_REGISTRY.get(demo_slide_name, {})
    slide_id = str(entry.get("slide_id", ""))
    if "-01Z-" in slide_id:
        return slide_id.split("-01Z-", 1)[0]
    return demo_slide_name.strip()


def _runtime_config(
    model_backend: str,
    local_model: str,
    api_provider: str,
    api_model: str,
    custom_api_model: str,
    api_key: str,
    api_base_url: str,
) -> Dict[str, Any]:
    if HOSTED_MODE:
        return hosted_runtime.hosted_runtime(INFER_DEFAULTS)
    return {
        "model_backend": model_backend,
        "local_model": local_model,
        "api_provider": api_provider,
        "api_model": custom_api_model.strip() if api_model == "Custom" and custom_api_model.strip() else api_model,
        "custom_api_model": custom_api_model,
        "api_key": str(api_key or "").strip(),
        "api_base_url": str(api_base_url or "").strip(),
    }


def _runtime_label(runtime: Dict[str, Any]) -> str:
    if runtime.get("hosted"):
        return f"Hosted demo / {runtime.get('local_model', 'Qwen2.5-VL-3B-Instruct')}"
    if str(runtime.get("model_backend", "")) == "API Model":
        provider = str(runtime.get("api_provider", "") or "API").strip()
        model = str(runtime.get("api_model", "") or "Custom").strip()
        return f"API Model / {provider} / {model}"
    model = str(runtime.get("local_model", "") or "Local Model").strip()
    return f"Local Model / {model}"


def _validate_send_runtime(runtime: Dict[str, Any]) -> str:
    if HOSTED_MODE:
        return hosted_runtime.availability_error()
    backend = str(runtime.get("model_backend", "Local Model"))
    if backend == "API Model":
        provider = str(runtime.get("api_provider", "")).strip()
        model = str(runtime.get("api_model", "")).strip()
        if not str(runtime.get("api_key", "")).strip():
            return (
                "API key is required for inference with API Model. "
                "Please fill the API Key field on the left, then press Send again."
            )
        if model == "Custom" or not model:
            return "Enter an API Model ID before sending with API Model."
        if provider == "Custom Endpoint" and not str(runtime.get("api_base_url", "")).strip():
            return "Custom Endpoint inference requires an API Base URL."
        if provider not in API_PROVIDERS:
            return "Choose a supported API provider: Google, OpenAI, or Custom Endpoint."
    elif str(runtime.get("local_model", "")) not in LOCAL_MODEL_CHOICES:
        return "Choose a supported local model."
    return ""


def _config_section(name: str) -> Dict[str, Any]:
    section = BUILD_DEFAULTS.get(name, {}) if isinstance(BUILD_DEFAULTS, dict) else {}
    return section if isinstance(section, dict) else {}


def _infer_config_section(name: str) -> Dict[str, Any]:
    section = INFER_DEFAULTS.get(name, {}) if isinstance(INFER_DEFAULTS, dict) else {}
    return section if isinstance(section, dict) else {}


def _attach_agent_paths(bank: Dict[str, Any], persist_dir: str = "", artifact_dir: str = "") -> Dict[str, Any]:
    if not bank:
        return bank
    result = dict(bank)
    existing = result.get("_agent", {}) if isinstance(result.get("_agent"), dict) else {}
    bank_cfg = _infer_config_section("bank")
    thumb_path = str(((bank.get("global", {}) or {}).get("thumbnail", {}) or {}).get("thumbnail_path", ""))
    inferred_root = thumb_path.split(f"{os.sep}artifacts{os.sep}", 1)[0] if f"{os.sep}artifacts{os.sep}" in thumb_path else ""
    persist = str(os.environ.get("SLIDEBANK_BANK_DIR") or persist_dir or existing.get("persist_dir") or inferred_root or bank_cfg.get("persist_dir", "")).strip()
    artifact = str(os.environ.get("SLIDEBANK_ARTIFACT_DIR") or artifact_dir or existing.get("artifact_dir") or (str(Path(inferred_root) / "artifacts") if inferred_root else "") or bank_cfg.get("artifact_dir", "") or (str(Path(persist) / "artifacts") if persist else "")).strip()
    if persist:
        path = Path(persist).expanduser()
        persist = str(path if path.is_absolute() else APP_ROOT / path)
    if artifact:
        path = Path(artifact).expanduser()
        artifact = str(path if path.is_absolute() else APP_ROOT / path)
    result["_agent"] = {
        "persist_dir": persist,
        "artifact_dir": artifact,
    }
    return result


def _selected_api_model(api_model: str, custom_api_model: str) -> str:
    model_name = custom_api_model.strip() if api_model == "Custom" and custom_api_model.strip() else api_model
    return model_name


def _patch_vlm_config(
    model_backend: str,
    local_model: str,
    api_provider: str,
    api_model: str,
    custom_api_model: str,
    api_key: str,
    api_base_url: str,
) -> PatchVLMConfig:
    patch_cfg = _config_section("patch_vlm")
    if model_backend == "API Model":
        provider_map = {"Google": "gemini", "OpenAI": "openai", "Custom Endpoint": "openai_compatible"}
        if api_provider not in provider_map:
            raise RuntimeError("Choose Google, OpenAI, or Custom Endpoint.")
        provider = provider_map.get(api_provider, "gemini")
        key = str(api_key or "").strip()
        if not key:
            raise RuntimeError("Run Bank with API Model requires an API key.")
        if provider == "openai_compatible" and not str(api_base_url or "").strip():
            raise RuntimeError("Custom Endpoint requires an API Base URL.")
        model_name = _selected_api_model(api_model, custom_api_model)
        if not model_name or model_name == "Custom":
            raise RuntimeError("Enter an API Model ID before building the bank.")
        return PatchVLMConfig(
            backend="api",
            model_name=model_name,
            model_path=model_name,
            device_map=str(patch_cfg.get("device_map", "auto")),
            dtype=str(patch_cfg.get("dtype", "fp16")),
            batch_size=int(patch_cfg.get("batch_size", 32)),
            max_new_tokens=int(patch_cfg.get("max_new_tokens", 224)),
            api_provider=provider,
            api_key=key,
            api_base_url=str(api_base_url or "").strip(),
            api_version=str(patch_cfg.get("api_version", "v1beta")),
            api_timeout_sec=int(patch_cfg.get("api_timeout_sec", 300)),
            gemini_api_env=str(patch_cfg.get("gemini_api_env", "GEMINI_API_KEY")),
            gemini_timeout=int(patch_cfg.get("gemini_timeout", 120)),
        )

    if local_model not in LOCAL_MODEL_CHOICES:
        raise RuntimeError("Choose a supported local model.")

    # API model IDs configure the API branch only. The local preset can use an
    # independent checkpoint, or the existing fields in an explicit local config.
    explicit_local = str(patch_cfg.get("backend", "")) == "local_hf"
    quilt_name = str(
        patch_cfg.get("local_model_name")
        or (patch_cfg.get("model_name") if explicit_local else "")
        or "wisdomik/Quilt-Llava-v1.5-7b"
    )
    quilt_path = str(
        patch_cfg.get("local_model_path")
        or (patch_cfg.get("model_path") if explicit_local else "")
        or quilt_name
    )
    model_aliases = {
        "Quilt-LLaVA": (quilt_name, quilt_path),
        "Qwen2.5-VL-7B-Instruct": ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
    }
    model_name, model_path = model_aliases.get(local_model, (local_model, local_model))
    return PatchVLMConfig(
        backend="local_hf",
        model_name=model_name,
        model_path=model_path,
        quilt_llava_repo=str(patch_cfg.get("quilt_llava_repo", "")),
        device_map=str(patch_cfg.get("device_map", "auto")),
        dtype=str(patch_cfg.get("dtype", "fp16")),
        batch_size=int(patch_cfg.get("batch_size", 32)),
        max_new_tokens=int(patch_cfg.get("max_new_tokens", 224)),
        gemini_api_env=str(patch_cfg.get("gemini_api_env", "GEMINI_API_KEY")),
        gemini_timeout=int(patch_cfg.get("gemini_timeout", 120)),
    )


def _bank_loaded(bank: Optional[Dict[str, Any]]) -> bool:
    return bool(bank and bank.get("tiles"))


def _tile_outputs(
    bank: Optional[Dict[str, Any]],
    tile_id: str,
    anchor_id: str = "",
    patch_id: str = "",
) -> Tuple[Any, str, Any, str, str, str]:
    tile = find_tile(bank, tile_id)
    tile_id = str(tile.get("tile_id", "")) if tile else ""
    if not anchor_id:
        anchor_id = first_anchor_id(tile)
    anchor = find_anchor(tile, anchor_id)
    anchor_id = str(anchor.get("anchor_id", "")) if anchor else ""
    choices = anchor_choices(tile)
    anchor_value = next((choice for choice in choices if parse_anchor_choice(choice) == anchor_id), choices[0] if choices else None)
    return (
        render_tile_overlay(bank, tile_id, anchor_id, patch_id),
        render_tile_summary(bank, tile_id),
        gr.update(choices=choices, value=anchor_value),
        render_patch_cards(bank, tile_id, anchor_id),
        render_signal_details(bank, tile_id, anchor_id),
        anchor_id,
    )


def _initial_bank() -> Tuple[Dict[str, Any], str]:
    demo_name = _default_demo()
    if not demo_name:
        return {}, ""
    try:
        return _attach_agent_paths(load_bank_json(DEMO_REGISTRY[demo_name]["bank_json_path"])), demo_name
    except Exception:
        return {}, demo_name


def first_evidence_tile_id(bank: Optional[Dict[str, Any]]) -> str:
    for tile in (bank or {}).get("tiles", []) or []:
        if tile.get("anchors"):
            return str(tile.get("tile_id", ""))
    return first_tile_id(bank)


INITIAL_BANK, INITIAL_DEMO = _initial_bank()
INITIAL_TILE_ID = first_evidence_tile_id(INITIAL_BANK)
INITIAL_TILE = find_tile(INITIAL_BANK, INITIAL_TILE_ID)
INITIAL_ANCHOR_ID = first_anchor_id(INITIAL_TILE)
INITIAL_TILE_CHOICES = tile_choices(INITIAL_BANK)
INITIAL_TILE_VALUE = next(
    (choice for choice in INITIAL_TILE_CHOICES if parse_tile_choice(choice) == INITIAL_TILE_ID),
    INITIAL_TILE_CHOICES[0] if INITIAL_TILE_CHOICES else None,
)
INITIAL_ANCHOR_CHOICES = anchor_choices(INITIAL_TILE)
INITIAL_ANCHOR_VALUE = next(
    (choice for choice in INITIAL_ANCHOR_CHOICES if parse_anchor_choice(choice) == INITIAL_ANCHOR_ID),
    INITIAL_ANCHOR_CHOICES[0] if INITIAL_ANCHOR_CHOICES else None,
)
INITIAL_STATUS = (
    "Evidence tree loaded. Browse the evidence, then configure the model and memory-bank paths for QA."
    if INITIAL_BANK
    else "Export a bank with scripts/export_ui_tree.py, then set SLIDEBANK_UI_TREE_DIR or upload its JSON. See docs/WORKBENCH.md."
)
INITIAL_CHAT = [
    {
        "role": "assistant",
        "content": "The evidence tree is loaded. Browse its grid cells and anchors below, or configure your model and memory-bank paths to ask a question.",
    }
] if INITIAL_BANK else []
if HOSTED_MODE:
    availability = hosted_runtime.availability_error()
    INITIAL_STATUS = availability or "ZeroGPU QA is available. Requests share a GPU queue and usage quota."
    INITIAL_CHAT = [{
        "role": "assistant",
        "content": (
            "The prebuilt evidence bank is ready to explore. "
            + (availability or "Ask a question to retrieve evidence and run the hosted Qwen2.5-VL model.")
        ),
    }]


def on_image_source_change(image_source: str) -> Tuple[Any, Any]:
    return (
        gr.update(visible=image_source == "Upload UI-tree JSON"),
        gr.update(visible=image_source == "Configured evidence trees"),
    )


def on_model_backend_change(model_backend: str) -> Tuple[Any, Any]:
    return (
        gr.update(visible=model_backend == "Local Model"),
        gr.update(visible=model_backend == "API Model"),
    )


def on_api_provider_change(api_provider: str) -> Any:
    return gr.update(visible=api_provider == "Custom Endpoint")


def _bank_ui_response(
    bank: Dict[str, Any],
    status: str,
    runtime: Dict[str, Any],
    history: List[Dict[str, str]],
) -> Tuple[
    Dict[str, Any],
    str,
    str,
    str,
    Any,
    Any,
    str,
    Any,
    str,
    Any,
    str,
    str,
    str,
    Dict[str, Any],
    List[Dict[str, str]],
]:
    if not _bank_loaded(bank):
        return (
            {},
            "",
            "",
            "",
            render_global_overlay(None),
            gr.update(choices=[], value=None),
            signal_overview_markdown(None),
            render_tile_overlay(None, ""),
            "### Selected Grid Detail\nNo grid selected.",
            gr.update(choices=[], value=None),
            "<div class='empty-panel'>No anchor selected.</div>",
            "No signal details available.",
            status,
            runtime,
            history,
        )
    tile_id = first_evidence_tile_id(bank)
    tile = find_tile(bank, tile_id)
    anchor_id = first_anchor_id(tile)
    tile_choice_list = tile_choices(bank)
    tile_value = next(
        (choice for choice in tile_choice_list if parse_tile_choice(choice) == tile_id),
        tile_choice_list[0] if tile_choice_list else None,
    )
    tile_image, tile_md, anchor_update, patch_html, signal_details, anchor_id = _tile_outputs(bank, tile_id, anchor_id)
    return (
        bank,
        tile_id,
        anchor_id,
        "",
        render_global_overlay(bank, tile_id),
        gr.update(choices=tile_choice_list, value=tile_value),
        signal_overview_markdown(bank),
        tile_image,
        tile_md,
        anchor_update,
        patch_html,
        signal_details,
        status,
        runtime,
        history,
    )


def _demo_csv_for_slide(slide_prefix: str, out_csv: Path) -> int:
    dataset_cfg = _config_section("dataset")
    source_csv = Path(str(dataset_cfg.get("csv_path", ""))).expanduser()
    if not source_csv.is_file():
        raise RuntimeError(f"Configured CSV does not exist: {source_csv}")

    matched: List[Dict[str, str]] = []
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"CSV has no header: {source_csv}")
        slide_keys = [key for key in ["Slide", "slide", "slide_id", "Slide ID"] if key in fieldnames]
        if not slide_keys:
            raise RuntimeError("CSV must contain a Slide or slide column.")
        for row in reader:
            slide_value = str(row.get(slide_keys[0], "")).strip()
            if slide_value == slide_prefix or slide_value.startswith(slide_prefix) or slide_prefix.startswith(slide_value):
                matched.append(row)
    if not matched:
        raise RuntimeError(f"No CSV rows matched demo slide prefix: {slide_prefix}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched)
    return len(matched)


def _export_ui_tree_from_readable(readable_dir: Path, slide_prefix: str, out_dir: Path) -> Tuple[Dict[str, Any], Path]:
    if not readable_dir.is_dir():
        raise RuntimeError(f"Readable bank directory was not created: {readable_dir}")
    files = sorted(readable_dir.glob("*.jsonl"))
    matching = [path for path in files if path.stem == slide_prefix or path.stem.startswith(slide_prefix)]
    readable_path = matching[0] if matching else (files[0] if files else None)
    if readable_path is None:
        raise RuntimeError(f"No readable JSONL files found in {readable_dir}")

    records = load_jsonl(readable_path)
    tree = build_ui_tree(records, readable_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(Path(tree['slide_id']).stem)}.ui_tree.json"
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return tree, out_path


def load_prebuilt_bank(
    image_source: str,
    uploaded_file: Any,
    demo_slide_name: str,
    model_backend: str,
    local_model: str,
    api_provider: str,
    api_model: str,
    custom_api_model: str,
    api_key: str,
    api_base_url: str,
    chat_history: List[Dict[str, str]],
) -> Tuple[
    Dict[str, Any],
    str,
    str,
    str,
    Any,
    Any,
    str,
    Any,
    str,
    Any,
    str,
    str,
    str,
    Dict[str, Any],
    List[Dict[str, str]],
]:
    runtime = _runtime_config(model_backend, local_model, api_provider, api_model, custom_api_model, api_key, api_base_url)
    history = list(chat_history or [])
    try:
        if HOSTED_MODE and image_source != "Configured evidence trees":
            raise RuntimeError("The hosted workbench uses the prebuilt evidence bank. Uploads are available in the local workbench.")
        if image_source == "Configured evidence trees":
            if not demo_slide_name or demo_slide_name not in DEMO_REGISTRY:
                raise RuntimeError("Export an evidence tree and set SLIDEBANK_UI_TREE_DIR before loading a configured slide.")
            bank = _attach_agent_paths(load_bank_json(DEMO_REGISTRY[demo_slide_name]["bank_json_path"]))
            status = "Evidence tree loaded. Browse below, then configure the model and memory-bank paths for QA."
        else:
            file_path = getattr(uploaded_file, "name", uploaded_file)
            file_path = str(file_path or "").strip()
            if not file_path:
                raise RuntimeError("Upload a UI-tree JSON exported with scripts/export_ui_tree.py.")
            if Path(file_path).suffix.lower() == ".json":
                bank = _attach_agent_paths(load_bank_json(file_path))
                status = "Uploaded evidence tree loaded. Image paths must be accessible on this server; configure the memory bank for QA."
            else:
                raise RuntimeError("Upload a UI-tree JSON file. Build raw WSI inputs with scripts/build_bank.py first.")
        history.append({"role": "assistant", "content": status})
        return _bank_ui_response(bank, status, runtime, history)
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
        history.append({"role": "assistant", "content": status})
        return _bank_ui_response({}, status, runtime, history)


def run_bank_extraction(
    image_source: str,
    uploaded_file: Any,
    demo_slide_name: str,
    model_backend: str,
    local_model: str,
    api_provider: str,
    api_model: str,
    custom_api_model: str,
    api_key: str,
    api_base_url: str,
    chat_history: List[Dict[str, str]],
    current_bank: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    str,
    str,
    str,
    Any,
    Any,
    str,
    Any,
    str,
    Any,
    str,
    str,
    str,
    Dict[str, Any],
    List[Dict[str, str]],
]:
    runtime = _runtime_config(model_backend, local_model, api_provider, api_model, custom_api_model, api_key, api_base_url)
    history = list(chat_history or [])
    try:
        if image_source != "Configured evidence trees":
            raise RuntimeError("Rebuild uses a configured slide and the local WSI paths in configs/build.yaml. Use Load Evidence to inspect uploaded JSON.")

        if not demo_slide_name or demo_slide_name not in DEMO_REGISTRY:
            raise RuntimeError("Choose a configured evidence tree to rebuild. For a new slide, run scripts/build_bank.py first.")

        slide_prefix = _slide_prefix_from_demo(demo_slide_name)
        run_tag = f"{safe_name(slide_prefix)}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = RUN_BANK_DIR / run_tag
        persist_dir = run_dir / "memory_bank"
        artifact_dir = persist_dir / "artifacts"
        filtered_csv = run_dir / "inputs" / "selected_slide.csv"
        row_count = _demo_csv_for_slide(slide_prefix, filtered_csv)

        dataset_cfg = _config_section("dataset")
        bank_cfg = _config_section("bank")
        patch_vlm = _patch_vlm_config(
            model_backend=model_backend,
            local_model=local_model,
            api_provider=api_provider,
            api_model=api_model,
            custom_api_model=custom_api_model,
            api_key=api_key,
            api_base_url=api_base_url,
        )
        build_cfg = BankBuildConfig(
            csv_path=str(filtered_csv),
            svs_root=str(dataset_cfg.get("svs_root", "")),
            svs_pattern=str(dataset_cfg.get("svs_pattern", "*/*.svs")),
            persist_dir=str(persist_dir),
            artifact_dir=str(artifact_dir),
            max_slides=1,
            rebuild=True,
            force_ingest=True,
            use_llm_patch_describer=bool(bank_cfg.get("use_llm_patch_describer", True)),
            patch_description_strict=bool(bank_cfg.get("patch_description_strict", True)),
            thumbnail_tile_size=int(bank_cfg.get("thumbnail_tile_size", 384)),
            anchor_crop_size_level0=int(bank_cfg.get("anchor_crop_size_level0", 4096)),
            patch_crop_size_level0=int(bank_cfg.get("patch_crop_size_level0", 512)),
            quiet=False,
            patch_vlm=patch_vlm,
        )
        for required_field in ["csv_path", "svs_root", "persist_dir", "artifact_dir"]:
            if not getattr(build_cfg, required_field):
                raise RuntimeError(f"Missing required build config: {required_field}")

        summary = build_bank(build_cfg)
        if summary.get("errors"):
            raise RuntimeError("; ".join(str(item) for item in summary["errors"][:3]))
        if int(summary.get("slides_ingested", 0)) < 1 and int(summary.get("slides_reused", 0)) < 1:
            raise RuntimeError(f"No slides were ingested. Missing SVS: {summary.get('missing_svs', [])}")

        bank, ui_tree_path = _export_ui_tree_from_readable(persist_dir / "readable", slide_prefix, run_dir / "ui_tree")
        bank = _attach_agent_paths(bank, persist_dir=str(persist_dir), artifact_dir=str(artifact_dir))
        status = (
            f"Bank extraction complete for {slide_prefix}. "
            f"Rows={row_count}; summary={summary.get('summary_path')}; ui_tree={ui_tree_path}"
        )
        history.append({"role": "assistant", "content": status})
        return _bank_ui_response(bank, status, runtime, history)
    except Exception as exc:
        status = f"{type(exc).__name__}: {exc}"
        history.append({"role": "assistant", "content": status})
        if _bank_loaded(current_bank):
            return _bank_ui_response(current_bank, status, runtime, history)
        return (
            {},
            "",
            "",
            "",
            render_global_overlay(None),
            gr.update(choices=[], value=None),
            signal_overview_markdown(None),
            render_tile_overlay(None, ""),
            "### Selected Grid Detail\nNo grid selected.",
            gr.update(choices=[], value=None),
            "<div class='empty-panel'>No anchor selected.</div>",
            "No signal details available.",
            status,
            runtime,
            history,
        )


def on_tile_select(tile_choice: str, bank: Dict[str, Any]) -> Tuple[str, str, str, Any, Any, str, Any, str, str, str]:
    if not _bank_loaded(bank) or not tile_choice:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Please run or load the evidence bank first.",
        )
    tile_id = parse_tile_choice(tile_choice)
    tile = find_tile(bank, tile_id)
    anchor_id = first_anchor_id(tile)
    tile_image, tile_md, anchor_update, patch_html, signal_details, anchor_id = _tile_outputs(bank, tile_id, anchor_id)
    return (
        tile_id,
        anchor_id,
        "",
        render_global_overlay(bank, tile_id),
        tile_image,
        tile_md,
        anchor_update,
        patch_html,
        signal_details,
        f"Grid {tile_id} selected.",
    )


def on_global_image_select(bank: Dict[str, Any], evt: gr.SelectData) -> Tuple[str, str, str, Any, Any, str, Any, str, str, str, Any]:
    if not _bank_loaded(bank):
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Please run or load the evidence bank first.",
            gr.skip(),
        )
    try:
        x, y = evt.index
    except Exception:
        x, y = 0, 0
    tile_id = tile_from_thumbnail_click(bank, int(x), int(y))
    tile = find_tile(bank, tile_id)
    anchor_id = first_anchor_id(tile)
    tile_choice_list = tile_choices(bank)
    tile_value = next((choice for choice in tile_choice_list if parse_tile_choice(choice) == tile_id), tile_choice_list[0] if tile_choice_list else None)
    tile_image, tile_md, anchor_update, patch_html, signal_details, anchor_id = _tile_outputs(bank, tile_id, anchor_id)
    return (
        tile_id,
        anchor_id,
        "",
        render_global_overlay(bank, tile_id),
        tile_image,
        tile_md,
        anchor_update,
        patch_html,
        signal_details,
        f"Grid {tile_id} selected.",
        gr.update(value=tile_value),
    )


def on_anchor_select(anchor_choice: str, bank: Dict[str, Any], tile_id: str) -> Tuple[str, str, Any, str, str, str]:
    if not _bank_loaded(bank) or not anchor_choice:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Please run or load the evidence bank first.",
        )
    anchor_id = parse_anchor_choice(anchor_choice)
    return (
        anchor_id,
        "",
        render_tile_overlay(bank, tile_id, anchor_id),
        render_patch_cards(bank, tile_id, anchor_id),
        render_signal_details(bank, tile_id, anchor_id),
        f"Anchor {anchor_id} selected.",
    )


def _hosted_error_message(error: Exception) -> str:
    text = str(error).lower()
    if "zerogpu" in text and "quota" in text:
        return (
            "Your remaining ZeroGPU allowance is insufficient for this request's reserved GPU duration. "
            "Sign in to Hugging Face to use your account's allowance, or try again after the quota resets. "
            "Evidence browsing is still available."
        )
    return (
        "Live QA could not finish. ZeroGPU capacity, quota, or the model runtime may be unavailable. "
        "Please retry later; evidence browsing is still available."
    )


def submit_chat(
    message: str,
    chat_history: List[Dict[str, str]],
    bank: Dict[str, Any],
    selected_tile_id: str,
    selected_anchor_id: str,
    model_backend: str,
    local_model: str,
    api_provider: str,
    api_model: str,
    custom_api_model: str,
    api_key: str,
    api_base_url: str,
) -> Tuple[List[Dict[str, str]], str, Any, str, Dict[str, Any], str]:
    history = list(chat_history or [])
    question = str(message or "").strip()
    if not question:
        return history, "", gr.skip(), gr.skip(), gr.skip(), "Please enter a question."
    runtime = _runtime_config(model_backend, local_model, api_provider, api_model, custom_api_model, api_key, api_base_url)
    if HOSTED_MODE and len(question) > 2000:
        return history, question, gr.skip(), gr.skip(), runtime, "Please limit hosted questions to 2,000 characters."
    if not _bank_loaded(bank):
        status = "Load an exported evidence tree before asking a question. See docs/WORKBENCH.md."
        return history, "", gr.skip(), gr.skip(), runtime, status
    validation_error = _validate_send_runtime(runtime)
    if validation_error:
        assistant_text = f"### Runtime Configuration Needed\n{validation_error}\n\nSelected backend: `{_runtime_label(runtime)}`"
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        return (
            history,
            "",
            render_tile_overlay(bank, selected_tile_id, selected_anchor_id),
            render_patch_cards(bank, selected_tile_id, selected_anchor_id),
            runtime,
            validation_error,
        )
    history_before_turn = list(history)
    try:
        answer_function = hosted_runtime.answer_on_gpu if HOSTED_MODE else answer_question_v5
        answer = answer_function(
            message=question,
            bank=bank,
            runtime=runtime,
            defaults=INFER_DEFAULTS,
            previous_messages=history_before_turn,
        )
        assistant_text = f"{answer.answer_markdown}\n\n{answer.trace_markdown}"
        status = (
            f"Agent inference complete in {answer.elapsed_sec:.1f}s. "
            f"backend={_runtime_label(runtime)}; "
            f"channel={answer.question_channel}; stage={answer.final_stage}; route={answer.evidence_route}"
        )
        runtime = dict(runtime)
        runtime["agent_route"] = answer.route
        runtime["question_channel"] = answer.question_channel
        runtime["final_stage"] = answer.final_stage
        runtime["evidence_route"] = answer.evidence_route
    except Exception as exc:
        if HOSTED_MODE:
            status = _hosted_error_message(exc)
            assistant_text = status
            print(f"Hosted QA failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            assistant_text = f"### Agent Error\n{type(exc).__name__}: {exc}"
            status = f"Agent inference failed: {type(exc).__name__}: {exc}"
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    return (
        history,
        "",
        render_tile_overlay(bank, selected_tile_id, selected_anchor_id),
        render_patch_cards(bank, selected_tile_id, selected_anchor_id),
        runtime,
        status,
    )


def clear_chat() -> List[Dict[str, str]]:
    return []


CSS = """
.gradio-container { max-width: 1760px !important; background: #F8FAFC; color: #111827; }
#title h1 { margin: 0 0 4px 0; }
#top_controls, #chat_panel, #global_panel, #grid_panel, #patch_panel {
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  background: #FFFFFF;
  padding: 12px;
}
#top_controls { min-height: 492px; }
#chat_panel { min-height: 492px; }
#chatbot { border: 1px solid #E5E7EB; border-radius: 8px; min-height: 460px; }
#chatbot .message,
#chatbot .message-content,
#chatbot .prose,
#chatbot .md,
#chatbot p,
#chatbot li {
  max-height: none !important;
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: normal !important;
  -webkit-line-clamp: unset !important;
  line-clamp: unset !important;
}
#chat_input_row { align-items: end; }
#patch_cards .patch-panel { width: 100%; }
#patch_cards .anchor-line { display: flex; gap: 12px; align-items: baseline; margin-bottom: 10px; color: #111827; }
#patch_cards .anchor-line span { color: #6B7280; }
#patch_cards .patch-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
#patch_cards .patch-card { position: relative; border: 1px solid #E5E7EB; border-radius: 8px; background: #FFFFFF; overflow: visible; min-height: 180px; }
#patch_cards .patch-card img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 8px 8px 0 0; }
#patch_cards .patch-meta { display: flex; justify-content: space-between; gap: 8px; padding: 8px; font-size: 13px; color: #111827; }
#patch_cards .patch-meta span { color: #6B7280; white-space: nowrap; }
#patch_cards .patch-tip { display: none; position: absolute; z-index: 20; left: 8px; right: 8px; bottom: 48px; padding: 10px; border: 1px solid #CBD5E1; border-radius: 8px; background: #111827; color: #FFFFFF; font-size: 12px; line-height: 1.45; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18); }
#patch_cards .patch-card:hover .patch-tip, #patch_cards .patch-card:focus .patch-tip { display: block; }
#patch_cards .missing-patch, #patch_cards .empty-panel { display: flex; align-items: center; justify-content: center; min-height: 160px; color: #6B7280; background: #F3F4F6; border-radius: 8px; }
#patch_cards .extra-patches { margin-top: 12px; }
#patch_cards .extra-patches summary { cursor: pointer; color: #2563EB; font-weight: 600; margin-bottom: 10px; }
@media (max-width: 980px) {
  #top_controls, #chat_panel { min-height: 0; }
  #patch_cards .patch-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


with gr.Blocks(title="SlideBank Workbench", css=CSS) as demo:
    gr.Markdown("# SlideBank Workbench", elem_id="title")
    gr.Markdown(
        "Explore the prebuilt **slide → grid → anchor patches** evidence bank. "
        "Live questions use **Qwen2.5-VL-3B-Instruct on ZeroGPU**, a demonstration model distinct from the paper's pathology backbones. "
        "GPU requests may queue and are subject to Hugging Face usage quota."
        if HOSTED_MODE else
        "Load an exported evidence tree to explore **slide → grid → anchor patches**. Connect your own local model or API for question answering. [Setup guide](https://github.com/gexinh/SlideBank/blob/main/docs/WORKBENCH.md)."
    )
    bank_state = gr.State(INITIAL_BANK)
    selected_tile_id_state = gr.State(INITIAL_TILE_ID)
    selected_anchor_id_state = gr.State(INITIAL_ANCHOR_ID)
    selected_patch_id_state = gr.State("")
    runtime_config_state = gr.State({})

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=340, elem_id="top_controls"):
            gr.Markdown("### Evidence bank" if HOSTED_MODE else "### Input & Runtime Controls")
            image_source = gr.Radio(label="Evidence Source", choices=["Configured evidence trees", "Upload UI-tree JSON"], value="Configured evidence trees", visible=not HOSTED_MODE)
            upload_file = gr.File(
                label="Upload exported UI-tree JSON",
                file_types=[".json"],
                visible=False,
            )
            demo_slide = gr.Dropdown(
                label="Demo Slide" if HOSTED_MODE else "Configured Slide",
                choices=list(DEMO_REGISTRY.keys()),
                value=INITIAL_DEMO or None,
                interactive=True,
            )
            model_backend = gr.Radio(label="Model Backend", choices=["Local Model", "API Model"], value="Local Model", visible=not HOSTED_MODE)
            with gr.Group(visible=not HOSTED_MODE) as local_model_group:
                local_model = gr.Dropdown(
                    label="Local Model",
                    choices=LOCAL_MODEL_CHOICES,
                    value="Quilt-LLaVA",
                )
            with gr.Group(visible=False) as api_model_group:
                api_provider = gr.Dropdown(label="API Provider", choices=API_PROVIDERS, value="Google")
                api_model = gr.State("Custom")
                custom_api_model = gr.Textbox(label="API Model ID", placeholder="Enter the model ID offered by your provider", value="", lines=1)
                api_key = gr.Textbox(label="API Key", value="", type="password", lines=1)
                api_base_url = gr.Textbox(label="API Base URL", value="", lines=1, visible=False)
            load_bank_btn = gr.Button("Load Evidence", variant="primary")
            run_bank_btn = gr.Button("Rebuild Configured Slide", variant="secondary", visible=not HOSTED_MODE)
            gr.Markdown("Rebuild reads local WSI inputs from `configs/build.yaml` and calls the selected model. For a new WSI, use `scripts/build_bank.py` first.", visible=not HOSTED_MODE)
            status_box = gr.Textbox(label="Status", value=INITIAL_STATUS, interactive=False, lines=2)

        with gr.Column(scale=2, min_width=620, elem_id="chat_panel"):
            gr.Markdown("### Inference Chat")
            chatbot = gr.Chatbot(
                label="Conversation",
                value=INITIAL_CHAT,
                type="messages",
                height=460,
                max_height=720,
                autoscroll=True,
                resizable=True,
                show_copy_button=True,
                elem_id="chatbot",
            )
            with gr.Row(elem_id="chat_input_row"):
                chat_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask a question about the selected slide evidence.",
                    lines=2,
                    scale=8,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, interactive=not HOSTED_MODE or not hosted_runtime.availability_error())
            clear_chat_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

    with gr.Column(elem_id="global_panel"):
        gr.Markdown("### Global Slide Overview")
        with gr.Row():
            with gr.Column(scale=3):
                global_image = gr.Image(
                    label="Thumbnail Grid",
                    value=render_global_overlay(INITIAL_BANK, INITIAL_TILE_ID),
                    interactive=False,
                    height=390,
                )
            with gr.Column(scale=2):
                global_summary = gr.Markdown(signal_overview_markdown(INITIAL_BANK))
                tile_selector = gr.Dropdown(
                    label="Grid Cell",
                    choices=INITIAL_TILE_CHOICES,
                    value=INITIAL_TILE_VALUE,
                    interactive=True,
                )

    with gr.Column(elem_id="grid_panel"):
        gr.Markdown("### Selected Grid Detail")
        with gr.Row():
            with gr.Column(scale=3):
                tile_image = gr.Image(
                    label="Grid Overlay",
                    value=render_tile_overlay(INITIAL_BANK, INITIAL_TILE_ID, INITIAL_ANCHOR_ID),
                    interactive=False,
                    height=390,
                )
            with gr.Column(scale=2):
                tile_summary = gr.Markdown(render_tile_summary(INITIAL_BANK, INITIAL_TILE_ID))
                anchor_selector = gr.Radio(
                    label="Anchor Summaries",
                    choices=INITIAL_ANCHOR_CHOICES,
                    value=INITIAL_ANCHOR_VALUE,
                    interactive=True,
                )
                with gr.Accordion("Signal Details", open=False):
                    signal_details = gr.Markdown(render_signal_details(INITIAL_BANK, INITIAL_TILE_ID, INITIAL_ANCHOR_ID))

    with gr.Column(elem_id="patch_panel"):
        gr.Markdown("### Anchor Patch Views")
        patch_cards = gr.HTML(render_patch_cards(INITIAL_BANK, INITIAL_TILE_ID, INITIAL_ANCHOR_ID), elem_id="patch_cards")

    if not HOSTED_MODE:
        image_source.change(on_image_source_change, inputs=[image_source], outputs=[upload_file, demo_slide])
        model_backend.change(on_model_backend_change, inputs=[model_backend], outputs=[local_model_group, api_model_group])
        api_provider.change(on_api_provider_change, inputs=[api_provider], outputs=[api_base_url])

    bank_inputs = [
        image_source,
        upload_file,
        demo_slide,
        model_backend,
        local_model,
        api_provider,
        api_model,
        custom_api_model,
        api_key,
        api_base_url,
        chatbot,
    ]
    bank_outputs = [
        bank_state,
        selected_tile_id_state,
        selected_anchor_id_state,
        selected_patch_id_state,
        global_image,
        tile_selector,
        global_summary,
        tile_image,
        tile_summary,
        anchor_selector,
        patch_cards,
        signal_details,
        status_box,
        runtime_config_state,
        chatbot,
    ]
    if not HOSTED_MODE:
        run_bank_btn.click(run_bank_extraction, inputs=bank_inputs + [bank_state], outputs=bank_outputs)
    load_bank_btn.click(load_prebuilt_bank, inputs=bank_inputs, outputs=bank_outputs)
    demo_slide.change(load_prebuilt_bank, inputs=bank_inputs, outputs=bank_outputs)

    tile_selector.change(
        on_tile_select,
        inputs=[tile_selector, bank_state],
        outputs=[
            selected_tile_id_state,
            selected_anchor_id_state,
            selected_patch_id_state,
            global_image,
            tile_image,
            tile_summary,
            anchor_selector,
            patch_cards,
            signal_details,
            status_box,
        ],
    )
    global_image.select(
        on_global_image_select,
        inputs=[bank_state],
        outputs=[
            selected_tile_id_state,
            selected_anchor_id_state,
            selected_patch_id_state,
            global_image,
            tile_image,
            tile_summary,
            anchor_selector,
            patch_cards,
            signal_details,
            status_box,
            tile_selector,
        ],
    )
    anchor_selector.change(
        on_anchor_select,
        inputs=[anchor_selector, bank_state, selected_tile_id_state],
        outputs=[
            selected_anchor_id_state,
            selected_patch_id_state,
            tile_image,
            patch_cards,
            signal_details,
            status_box,
        ],
        preprocess=False,
    )

    chat_inputs = [
        chat_input,
        chatbot,
        bank_state,
        selected_tile_id_state,
        selected_anchor_id_state,
        model_backend,
        local_model,
        api_provider,
        api_model,
        custom_api_model,
        api_key,
        api_base_url,
    ]
    chat_outputs = [chatbot, chat_input, tile_image, patch_cards, runtime_config_state, status_box]
    send_btn.click(submit_chat, inputs=chat_inputs, outputs=chat_outputs)
    chat_input.submit(submit_chat, inputs=chat_inputs, outputs=chat_outputs)
    clear_chat_btn.click(clear_chat, inputs=None, outputs=[chatbot])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name=DEFAULT_HOST, server_port=DEFAULT_PORT, share=False)
