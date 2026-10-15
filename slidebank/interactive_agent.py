from __future__ import annotations

import html
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .bank import open_memory_bank
from .retrieval import RetrievalConfig, retrieve_tool_context
from .vlm import VLMClient, VLMConfig


@dataclass
class AgentAnswer:
    answer_markdown: str
    trace_markdown: str
    raw_output: str
    route: Dict[str, Any]
    retrieval_errors: List[str]
    elapsed_sec: float


_MEMORY_BANK_CACHE: Dict[str, Any] = {}
_VLM_CACHE: Dict[Tuple[Any, ...], VLMClient] = {}


def normalize_artifact_path(value: Any) -> str:
    raw = os.path.expanduser(os.path.expandvars(str(value or "").strip()))
    if not raw:
        return ""
    return raw


def infer_bank_paths(bank: Dict[str, Any], defaults: Dict[str, Any]) -> Tuple[str, str]:
    agent = bank.get("_agent", {}) if isinstance(bank, dict) else {}
    persist_dir = str(agent.get("persist_dir", "")).strip()
    artifact_dir = str(agent.get("artifact_dir", "")).strip()
    bank_defaults = defaults.get("bank", {}) if isinstance(defaults.get("bank", {}), dict) else {}
    if not persist_dir:
        persist_dir = str(bank_defaults.get("persist_dir", "")).strip()
    if not artifact_dir:
        artifact_dir = str(bank_defaults.get("artifact_dir", "")).strip()

    thumb = ((bank.get("global", {}) or {}).get("thumbnail", {}) or {}).get("thumbnail_path")
    thumb = normalize_artifact_path(thumb)
    marker = f"{os.sep}artifacts{os.sep}"
    if thumb and marker in thumb:
        root = thumb.split(marker, 1)[0]
        if not persist_dir:
            persist_dir = root
        if not artifact_dir:
            artifact_dir = os.path.join(root, "artifacts")
    return persist_dir, artifact_dir


def thumbnail_path_for_bank(bank: Dict[str, Any]) -> str:
    thumb = ((bank.get("global", {}) or {}).get("thumbnail", {}) or {})
    for key in ["thumbnail_path", "debug_grid_overlay_path"]:
        path = normalize_artifact_path(thumb.get(key))
        if path and os.path.exists(path):
            return path
    return ""


def _model_name_from_runtime(runtime: Dict[str, Any]) -> str:
    model = str(runtime.get("api_model", "")).strip()
    custom = str(runtime.get("custom_api_model", "")).strip()
    if model == "Custom" and custom:
        model = custom
    aliases = {
        "Gemini 1.5 Pro": "gemini-1.5-pro",
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
        "GPT-4o": "gpt-4o",
        "GPT-4.1": "gpt-4.1",
    }
    return aliases.get(model, model)


def make_vlm_config(runtime: Dict[str, Any], defaults: Dict[str, Any]) -> VLMConfig:
    vlm_defaults = defaults.get("vlm", {}) if isinstance(defaults.get("vlm", {}), dict) else {}
    backend = str(runtime.get("model_backend", "Local Model"))
    if backend == "API Model":
        provider_map = {"Google": "gemini", "OpenAI": "openai", "Custom Endpoint": "openai_compatible"}
        provider_label = str(runtime.get("api_provider", "Google"))
        if provider_label == "Anthropic":
            raise RuntimeError("Anthropic API inference is not wired for the agent yet.")
        api_key = str(runtime.get("api_key", "")).strip()
        if not api_key:
            raise RuntimeError("API agent inference requires an API key.")
        provider = provider_map.get(provider_label, "gemini")
        if provider == "openai_compatible" and not str(runtime.get("api_base_url", "")).strip():
            raise RuntimeError("Custom Endpoint agent inference requires an API Base URL.")
        model_name = _model_name_from_runtime(runtime) or str(vlm_defaults.get("model", "gemini-1.5-pro"))
        return VLMConfig(
            backend="api",
            model_path=model_name,
            api_provider=provider,
            api_key=api_key,
            api_base_url=str(runtime.get("api_base_url", "")).strip(),
            api_version=str(vlm_defaults.get("api_version", "v1beta")),
            api_timeout_sec=int(vlm_defaults.get("api_timeout_sec", 300)),
            max_new_tokens=max(512, int(vlm_defaults.get("max_new_tokens", 1024))),
        )

    local_model = str(runtime.get("local_model", "Quilt-LLaVA"))
    if local_model == "Mock Local Model":
        raise RuntimeError("Mock Local Model does not run real agent inference. Choose Quilt-LLaVA or API Model.")
    if local_model in {"Quilt-LLaVA", "Pathology Local VLM"}:
        return VLMConfig(
            backend="quilt_llava",
            model_path=str(vlm_defaults.get("local_model_path") or vlm_defaults.get("model_path") or "wisdomik/Quilt-Llava-v1.5-7b"),
            quilt_llava_repo=str(vlm_defaults.get("quilt_llava_repo") or os.getenv("SLIDEBANK_QUILT_LLAVA_REPO", "")),
            device_map=str(vlm_defaults.get("device_map", "auto")),
            dtype=str(vlm_defaults.get("dtype", "fp16")),
            max_new_tokens=max(256, int(vlm_defaults.get("max_new_tokens", 256))),
        )
    model_aliases = {
        "Qwen2.5-VL": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    model_path = model_aliases.get(local_model, local_model)
    return VLMConfig(
        backend="local_hf",
        model_path=model_path,
        device_map=str(vlm_defaults.get("device_map", "auto")),
        dtype=str(vlm_defaults.get("dtype", "fp16")),
        max_new_tokens=max(256, int(vlm_defaults.get("max_new_tokens", 256))),
    )


def make_retrieval_config(defaults: Dict[str, Any]) -> RetrievalConfig:
    cfg = defaults.get("retrieval", {}) if isinstance(defaults.get("retrieval", {}), dict) else {}
    return RetrievalConfig(
        signal_topk=int(cfg.get("signal_topk", 4)),
        signal_views_per_signal=int(cfg.get("signal_views_per_signal", 2)),
        signal_context_max_chars=int(cfg.get("signal_context_max_chars", 3200)),
        region_topk=int(cfg.get("region_topk", 2)),
        detail_topk=int(cfg.get("detail_topk", 2)),
        anchor_topk=int(cfg.get("anchor_topk", 2)),
    )


def get_memory_bank(persist_dir: str):
    key = os.path.abspath(os.path.expanduser(str(persist_dir)))
    if key not in _MEMORY_BANK_CACHE:
        _MEMORY_BANK_CACHE[key] = open_memory_bank(key, rebuild=False, quiet=True)
    return _MEMORY_BANK_CACHE[key]


def get_vlm_client(config: VLMConfig) -> VLMClient:
    key = (
        config.backend,
        config.model_path,
        config.quilt_llava_repo,
        config.device_map,
        config.dtype,
        config.max_new_tokens,
        config.api_provider,
        config.api_key,
        config.api_base_url,
        config.api_version,
    )
    if key not in _VLM_CACHE:
        client = VLMClient(config)
        _VLM_CACHE[key] = client
    return _VLM_CACHE[key]


def _recent_dialogue(messages: List[Dict[str, str]], max_turns: int = 4) -> str:
    turns: List[str] = []
    for msg in messages or []:
        role = str(msg.get("role", "")).strip()
        content = " ".join(str(msg.get("content", "")).split())
        if role not in {"user", "assistant"} or not content:
            continue
        if content.startswith("Prebuilt evidence bank loaded") or content.startswith("Bank extraction complete"):
            continue
        turns.append(f"{role}: {content[:900]}")
    if not turns:
        return ""
    return "\n".join(turns[-2 * max_turns :])


def build_freeform_agent_prompt(
    question: str,
    tool_logs: List[Dict[str, str]],
    previous_messages: List[Dict[str, str]],
) -> str:
    tool_lines: List[str] = []
    for call in tool_logs:
        step = str(call.get("tool_step", ""))
        tool_name = str(call.get("tool", ""))
        tool_lines.append(f"Tool {step} INPUT [{tool_name}]: {call.get('input', '')}")
        tool_lines.append(f"Tool {step} OUTPUT [{tool_name}]: {call.get('output', '')}")
    dialogue = _recent_dialogue(previous_messages)
    dialogue_block = f"=== Previous Dialogue ===\n{dialogue}\n=== End Previous Dialogue ===\n\n" if dialogue else ""
    return (
        "You are a pathology WSI QA agent.\n"
        "Use the slide thumbnail image and memory-bank tool outputs to answer the user's question.\n"
        "Do not simply restate the retrieved evidence. Synthesize a direct answer.\n"
        "Do not use external reports or outside patient information.\n"
        "If the memory bank does not contain enough evidence, say what is uncertain.\n"
        "Cite relevant grid, anchor, evidence, or view IDs when they support the answer.\n\n"
        f"{dialogue_block}"
        "=== Memory-Bank Tool Calls For Current Question ===\n"
        f"{chr(10).join(tool_lines)}\n"
        "=== End Tool Calls ===\n\n"
        f"User question: {question}\n\n"
        "Answer in concise natural language:"
    )


def _tool_trace_markdown(route: Dict[str, Any], tool_logs: List[Dict[str, str]], errors: List[str]) -> str:
    lines = [
        "<details>",
        "<summary>Agent Trace</summary>",
        "",
        f"Route: `{html.escape(str(route.get('route', 'unknown')))}`",
    ]
    if errors:
        lines.append("")
        lines.append("Retrieval warnings:")
        for err in errors:
            lines.append(f"- {html.escape(str(err))}")
    for call in tool_logs:
        lines.append("")
        lines.append(f"**Tool {html.escape(str(call.get('tool_step', '')))}: `{html.escape(str(call.get('tool', '')) )}`**")
        lines.append("")
        lines.append("Input:")
        lines.append("```text")
        lines.append(str(call.get("input", ""))[:1200])
        lines.append("```")
        lines.append("Output:")
        lines.append("```json")
        lines.append(str(call.get("output", ""))[:2400])
        lines.append("```")
    lines.append("</details>")
    return "\n".join(lines)


def answer_question_with_agent(
    question: str,
    bank: Dict[str, Any],
    runtime: Dict[str, Any],
    defaults: Dict[str, Any],
    previous_messages: Optional[List[Dict[str, str]]] = None,
) -> AgentAnswer:
    started = time.perf_counter()
    if not bank or not bank.get("slide_id"):
        raise RuntimeError("No SlideBank memory is loaded.")
    persist_dir, _artifact_dir = infer_bank_paths(bank, defaults)
    if not persist_dir:
        raise RuntimeError("Cannot resolve memory bank persist_dir for this slide.")
    image_path = thumbnail_path_for_bank(bank)
    if not image_path:
        raise RuntimeError("Cannot resolve a thumbnail image for VLM inference.")

    slide_id = str(bank.get("slide_id", "")).strip()
    memory_bank = get_memory_bank(persist_dir)
    if not memory_bank.has_slide(slide_id):
        raise RuntimeError(f"Slide is not present in memory bank: {slide_id}")
    retrieval = retrieve_tool_context(
        memory_bank=memory_bank,
        slide_id=slide_id,
        question=question,
        config=make_retrieval_config(defaults),
    )
    prompt = build_freeform_agent_prompt(
        question=question,
        tool_logs=retrieval["tool_logs"],
        previous_messages=previous_messages or [],
    )
    vlm = get_vlm_client(make_vlm_config(runtime, defaults))
    raw = vlm.generate(image_path=image_path, prompt=prompt).strip()
    if not raw:
        raw = "The agent did not return a non-empty answer."
    trace = _tool_trace_markdown(
        route=retrieval.get("route", {}),
        tool_logs=retrieval.get("tool_logs", []),
        errors=retrieval.get("retrieval_errors", []),
    )
    return AgentAnswer(
        answer_markdown=f"### Answer\n{raw}",
        trace_markdown=trace,
        raw_output=raw,
        route=retrieval.get("route", {}),
        retrieval_errors=retrieval.get("retrieval_errors", []),
        elapsed_sec=round(time.perf_counter() - started, 3),
    )
