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
    parser = argparse.ArgumentParser(description="Run SlideBank inference from an existing memory bank.")
    parser.add_argument("--config", default="", help="Optional JSON/YAML config.")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--svs-root", default=None)
    parser.add_argument("--svs-pattern", default=None)
    parser.add_argument("--thumbnail-root", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--out-summary", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--mode", choices=["single", "multi"], default=None)
    parser.add_argument(
        "--vlm-backend",
        choices=["quilt_llava", "local_hf", "api", "gemini_api", "openai_api", "openai_compatible_api"],
        default=None,
    )
    parser.add_argument("--model", default=None, help="Model name/path. Alias for --model-path.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--quilt-llava-repo", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--api-provider", choices=["gemini", "openai", "openai_compatible"], default=None)
    parser.add_argument("--api-key", default=None, help="API key value. Prefer env/file injection for shared runs.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable that holds the API key.")
    parser.add_argument("--api-key-file", default=None, help="Optional shell-style file with export NAME='key'.")
    parser.add_argument("--api-base-url", default=None, help="Base URL for OpenAI-compatible APIs.")
    parser.add_argument("--api-version", default=None, help="Provider API version when supported, e.g. Gemini v1beta.")
    parser.add_argument("--api-timeout-sec", type=int, default=None)
    parser.add_argument("--signal-topk", type=int, default=None)
    parser.add_argument("--signal-views-per-signal", type=int, default=None)
    parser.add_argument("--region-topk", type=int, default=None)
    parser.add_argument("--detail-topk", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from slidebank.inference import InferenceConfig, run_inference
    from slidebank.retrieval import RetrievalConfig
    from slidebank.vlm import VLMConfig

    cfg = load_config(args.config)
    dataset = cfg.get("dataset", {})
    bank = cfg.get("bank", {})
    out = cfg.get("output", {})
    vlm_cfg = cfg.get("vlm", {})
    retrieval_cfg = cfg.get("retrieval", {})

    model_name = (
        args.model
        or args.model_path
        or vlm_cfg.get("model")
        or vlm_cfg.get("model_path")
        or "wisdomik/Quilt-Llava-v1.5-7b"
    )
    vlm = VLMConfig(
        backend=args.vlm_backend or vlm_cfg.get("backend", "quilt_llava"),
        model_path=model_name,
        quilt_llava_repo=args.quilt_llava_repo or vlm_cfg.get("quilt_llava_repo", ""),
        device_map=args.device_map or vlm_cfg.get("device_map", "auto"),
        dtype=args.dtype or vlm_cfg.get("dtype", "fp16"),
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else int(vlm_cfg.get("max_new_tokens", 64)),
        api_provider=args.api_provider or vlm_cfg.get("api_provider", ""),
        api_key=args.api_key or "",
        api_key_env=args.api_key_env or vlm_cfg.get("api_key_env", ""),
        api_key_file=args.api_key_file or vlm_cfg.get("api_key_file", ""),
        api_base_url=args.api_base_url or vlm_cfg.get("api_base_url", ""),
        api_version=args.api_version or vlm_cfg.get("api_version", "v1beta"),
        api_timeout_sec=args.api_timeout_sec
        if args.api_timeout_sec is not None
        else int(vlm_cfg.get("api_timeout_sec", 300)),
        gemini_api_key="",
        gemini_api_env=vlm_cfg.get("gemini_api_env", "GEMINI_API_KEY"),
    )
    retrieval = RetrievalConfig(
        signal_topk=args.signal_topk if args.signal_topk is not None else int(retrieval_cfg.get("signal_topk", 4)),
        signal_views_per_signal=args.signal_views_per_signal
        if args.signal_views_per_signal is not None
        else int(retrieval_cfg.get("signal_views_per_signal", 2)),
        signal_context_max_chars=int(retrieval_cfg.get("signal_context_max_chars", 3200)),
        region_topk=args.region_topk if args.region_topk is not None else int(retrieval_cfg.get("region_topk", 3)),
        detail_topk=args.detail_topk if args.detail_topk is not None else int(retrieval_cfg.get("detail_topk", 2)),
        anchor_topk=int(retrieval_cfg.get("anchor_topk", 2)),
    )
    infer_cfg = InferenceConfig(
        csv_path=args.csv_path or dataset.get("csv_path", ""),
        svs_root=args.svs_root or dataset.get("svs_root", ""),
        svs_pattern=args.svs_pattern or dataset.get("svs_pattern", "*/*.svs"),
        thumbnail_root=args.thumbnail_root or dataset.get("thumbnail_root", ""),
        persist_dir=args.persist_dir or bank.get("persist_dir", ""),
        artifact_dir=args.artifact_dir or bank.get("artifact_dir", ""),
        out_jsonl=args.out_jsonl or out.get("jsonl", "outputs/predictions.jsonl"),
        out_summary=args.out_summary or out.get("summary", "outputs/summary.json"),
        max_samples=args.max_samples if args.max_samples is not None else int(out.get("max_samples", 0)),
        max_slides=args.max_slides if args.max_slides is not None else int(out.get("max_slides", 0)),
        mode=args.mode or out.get("mode", "single"),
        vlm=vlm,
        retrieval=retrieval,
    )
    for field in ["csv_path", "svs_root", "persist_dir", "artifact_dir"]:
        if not getattr(infer_cfg, field):
            raise SystemExit(f"Missing required inference config: {field}")
    summary = run_inference(infer_cfg)
    print(f"Inference summary: {summary['out_jsonl']} | acc={summary['accuracy']:.3f} | errors={summary['errors']}")


if __name__ == "__main__":
    main()
