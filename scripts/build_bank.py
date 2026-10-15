#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slidebank.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SlideBank memory bank from WSI slides.")
    parser.add_argument("--config", default="", help="Optional JSON/YAML config.")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--svs-root", default=None)
    parser.add_argument("--svs-pattern", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--no-llm-patch-describer", action="store_true")
    parser.add_argument("--allow-patch-fallback", action="store_true")
    parser.add_argument(
        "--patch-backend",
        choices=["local_hf", "api", "gemini_api", "openai_api", "openai_compatible_api"],
        default=None,
    )
    parser.add_argument("--patch-model-name", default=None)
    parser.add_argument("--patch-model-path", default=None)
    parser.add_argument("--patch-quilt-llava-repo", default=None, help="External Quilt-LLaVA source checkout; alternatively set SLIDEBANK_QUILT_LLAVA_REPO.")
    parser.add_argument("--patch-device-map", default=None)
    parser.add_argument("--patch-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.add_argument("--patch-api-provider", choices=["gemini", "openai", "openai_compatible"], default=None)
    parser.add_argument("--patch-api-key", default=None)
    parser.add_argument("--patch-api-key-env", default=None)
    parser.add_argument("--patch-api-key-file", default=None)
    parser.add_argument("--patch-api-base-url", default=None)
    parser.add_argument("--patch-api-version", default=None)
    parser.add_argument("--patch-api-timeout-sec", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from slidebank.bank import BankBuildConfig, PatchVLMConfig, build_bank

    cfg = load_config(args.config)
    dataset = cfg.get("dataset", {})
    bank = cfg.get("bank", {})
    patch_cfg = cfg.get("patch_vlm", {})

    patch = PatchVLMConfig(
        backend=args.patch_backend or patch_cfg.get("backend", "local_hf"),
        model_name=args.patch_model_name or patch_cfg.get("model_name", "wisdomik/Quilt-Llava-v1.5-7b"),
        model_path=args.patch_model_path
        or patch_cfg.get("model_path", "wisdomik/Quilt-Llava-v1.5-7b"),
        quilt_llava_repo=args.patch_quilt_llava_repo or patch_cfg.get("quilt_llava_repo", ""),
        device_map=args.patch_device_map or patch_cfg.get("device_map", "auto"),
        dtype=args.patch_dtype or patch_cfg.get("dtype", "fp16"),
        batch_size=args.patch_batch_size or int(patch_cfg.get("batch_size", 32)),
        max_new_tokens=int(patch_cfg.get("max_new_tokens", 224)),
        api_provider=args.patch_api_provider or patch_cfg.get("api_provider", ""),
        api_key=args.patch_api_key or "",
        api_key_env=args.patch_api_key_env or patch_cfg.get("api_key_env", ""),
        api_key_file=args.patch_api_key_file or patch_cfg.get("api_key_file", ""),
        api_base_url=args.patch_api_base_url or patch_cfg.get("api_base_url", ""),
        api_version=args.patch_api_version or patch_cfg.get("api_version", "v1beta"),
        api_timeout_sec=args.patch_api_timeout_sec
        if args.patch_api_timeout_sec is not None
        else int(patch_cfg.get("api_timeout_sec", 300)),
        gemini_api_key="",
        gemini_api_env=patch_cfg.get("gemini_api_env", "GEMINI_API_KEY"),
        gemini_timeout=int(patch_cfg.get("gemini_timeout", 120)),
    )

    build_cfg = BankBuildConfig(
        csv_path=args.csv_path or dataset.get("csv_path", ""),
        svs_root=args.svs_root or dataset.get("svs_root", ""),
        svs_pattern=args.svs_pattern or dataset.get("svs_pattern", "*/*.svs"),
        persist_dir=args.persist_dir or bank.get("persist_dir", ""),
        artifact_dir=args.artifact_dir or bank.get("artifact_dir", ""),
        max_slides=args.max_slides if args.max_slides is not None else int(bank.get("max_slides", 0)),
        rebuild=bool(args.rebuild or bank.get("rebuild", False)),
        force_ingest=bool(args.force_ingest or bank.get("force_ingest", False)),
        use_llm_patch_describer=not args.no_llm_patch_describer and bool(bank.get("use_llm_patch_describer", True)),
        patch_description_strict=not args.allow_patch_fallback and bool(bank.get("patch_description_strict", True)),
        thumbnail_tile_size=int(bank.get("thumbnail_tile_size", 384)),
        anchor_crop_size_level0=int(bank.get("anchor_crop_size_level0", 4096)),
        patch_crop_size_level0=int(bank.get("patch_crop_size_level0", 512)),
        quiet=bool(args.quiet or bank.get("quiet", False)),
        patch_vlm=patch,
    )
    for field in ["csv_path", "svs_root", "persist_dir", "artifact_dir"]:
        if not getattr(build_cfg, field):
            raise SystemExit(f"Missing required build config: {field}")
    summary = build_bank(build_cfg)
    print(f"Build summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
