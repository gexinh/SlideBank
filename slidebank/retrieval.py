from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .wsi_agent.memory_bank import HierarchicalMemoryBank


@dataclass
class RetrievalConfig:
    signal_topk: int = 4
    signal_views_per_signal: int = 2
    signal_context_max_chars: int = 3200
    region_topk: int = 3
    detail_topk: int = 2
    anchor_topk: int = 2
    compact_tool_chars: int = 2200


def compact_json(obj: Any, max_chars: int = 2200) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _extract_anchor_ids(signal_payload: Dict[str, Any], max_count: int) -> List[str]:
    anchors: List[str] = []
    for item in signal_payload.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        anchor_id = str(item.get("anchor_id", "")).strip()
        if anchor_id and anchor_id not in anchors:
            anchors.append(anchor_id)
        if len(anchors) >= max_count:
            break
    return anchors


def retrieve_tool_context(
    memory_bank: HierarchicalMemoryBank,
    slide_id: str,
    question: str,
    config: RetrievalConfig,
) -> Dict[str, Any]:
    route = memory_bank.route_question(question)
    logs: List[Dict[str, str]] = []
    errors: List[str] = []
    calls = 0

    calls += 1
    try:
        global_view = memory_bank.consult_global_view(slide_id)
        if isinstance(global_view, dict) and global_view.get("error"):
            errors.append(str(global_view["error"]))
    except Exception as exc:
        global_view = {"error": f"consult_global_view: {type(exc).__name__}: {exc}"}
        errors.append(str(global_view["error"]))
    logs.append(
        {
            "tool_step": "1",
            "tool": "consult_global_view",
            "input": "Get global pathology overview for this slide.",
            "output": compact_json(
                {
                    "route": route,
                    "summary": global_view.get("summary", ""),
                    "overall_impression": global_view.get("overall_impression", ""),
                },
                1600,
            ),
        }
    )

    calls += 1
    try:
        signal_payload = memory_bank.build_signal_patch_context(
            question=question,
            route=route,
            max_signals=max(1, int(config.signal_topk)),
            max_views_per_signal=max(1, int(config.signal_views_per_signal)),
            max_chars=max(500, int(config.signal_context_max_chars)),
        )
        if isinstance(signal_payload, dict) and signal_payload.get("error"):
            errors.append(str(signal_payload["error"]))
    except Exception as exc:
        signal_payload = {"error": f"build_signal_patch_context: {type(exc).__name__}: {exc}"}
        errors.append(str(signal_payload["error"]))
    logs.append(
        {
            "tool_step": "2",
            "tool": "build_signal_patch_context",
            "input": "Retrieve question-aligned signals, evidence, and patch views from memory bank.",
            "output": compact_json(
                {
                    "context": signal_payload.get("context", ""),
                    "route": signal_payload.get("route", {}),
                    "evidence_top": (signal_payload.get("evidence", []) or [])[: max(1, int(config.signal_topk))],
                },
                config.compact_tool_chars,
            ),
        }
    )

    calls += 1
    region_payload: Dict[str, Any] = {}
    details: List[Dict[str, Any]] = []
    try:
        region_payload = memory_bank.scan_regions(question)
        if isinstance(region_payload, dict) and region_payload.get("error"):
            errors.append(str(region_payload["error"]))
        for item in (region_payload.get("matches", []) or [])[: max(1, int(config.region_topk))]:
            region_id = str(item.get("region_id", "")).strip() if isinstance(item, dict) else ""
            if not region_id:
                continue
            calls += 1
            detail = memory_bank.inspect_details(region_id=region_id, feature_query=question)
            if isinstance(detail, dict):
                findings = detail.get("findings", [])
                if isinstance(findings, list):
                    detail["findings"] = findings[: max(1, int(config.detail_topk))]
                details.append(detail)
    except Exception as exc:
        errors.append(f"scan_regions/inspect_details: {type(exc).__name__}: {exc}")
    logs.append(
        {
            "tool_step": "3",
            "tool": "scan_regions+inspect_details",
            "input": "Find relevant regions and inspect detailed morphology.",
            "output": compact_json(
                {
                    "regions": (region_payload.get("matches", []) or [])[: max(1, int(config.region_topk))],
                    "details": details,
                },
                config.compact_tool_chars,
            ),
        }
    )

    target_signals = route.get("target_signals", [])
    signal_query = str(target_signals[0]) if isinstance(target_signals, list) and target_signals else ""
    anchors = _extract_anchor_ids(signal_payload, max_count=max(1, int(config.anchor_topk)))
    anchor_payloads: List[Dict[str, Any]] = []
    for anchor_id in anchors:
        calls += 1
        try:
            evidence = memory_bank.inspect_evidence(anchor_id=anchor_id, signal_query=signal_query)
            if isinstance(evidence, dict):
                anchor_payloads.append(evidence)
        except Exception as exc:
            errors.append(f"inspect_evidence({anchor_id}): {type(exc).__name__}: {exc}")
    logs.append(
        {
            "tool_step": "4",
            "tool": "inspect_evidence",
            "input": "Follow key anchors and verify evidence-to-signal support.",
            "output": compact_json({"anchor_ids": anchors, "anchor_evidence": anchor_payloads}, config.compact_tool_chars),
        }
    )

    return {
        "route": route,
        "tool_logs": logs,
        "retrieval_errors": errors,
        "retrieval_calls": calls,
    }


def build_answer_prompt(
    question: str,
    options: List[str],
    tool_logs: List[Dict[str, str]],
    previous_turns: List[Dict[str, str]] | None = None,
) -> str:
    labels = ["A", "B", "C", "D"]
    option_lines = "\n".join(f"{labels[idx]}. {options[idx]}" for idx in range(4))
    tool_lines: List[str] = []
    for call in tool_logs:
        tool_lines.append(f"Tool {call.get('tool_step')} INPUT [{call.get('tool')}]: {call.get('input', '')}")
        tool_lines.append(f"Tool {call.get('tool_step')} OUTPUT [{call.get('tool')}]: {call.get('output', '')}")
    history = ""
    if previous_turns:
        lines = ["=== Previous Question Turns On This Slide ==="]
        for turn in previous_turns:
            lines.append(
                f"Turn {turn.get('turn_id')} | Question: {turn.get('question')} | "
                f"Final answer: {turn.get('pred_label')} | Matched option text: {turn.get('pred_text')}"
            )
        lines.append("=== End Previous Question Turns ===\n")
        history = "\n".join(lines)
    return (
        "You are a pathology WSI QA agent.\n"
        "Use only slide image evidence and memory-bank tool outputs.\n"
        "Do not use external reports or outside knowledge.\n"
        "Output only one uppercase letter from A, B, C, D.\n\n"
        f"{history}"
        "=== Tool Calls For Current Question ===\n"
        f"{chr(10).join(tool_lines)}\n"
        "=== End Tool Calls ===\n\n"
        f"Question: {question}\n"
        f"{option_lines}\n"
        "Answer:"
    )
