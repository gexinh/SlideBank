from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .interactive_agent import (
    get_memory_bank,
    get_vlm_client,
    infer_bank_paths,
    make_vlm_config,
    thumbnail_path_for_bank,
)


LABELS = ["A", "B", "C", "D"]
MEMORY_LEVEL_ORDER = ["signal", "view", "evidence", "global"]
ALLOWED_TASK_LAYERS = {
    "WSI_GLOBAL_ONLY",
    "WSI_HIERARCHICAL",
    "WSI_SPATIAL_MULTI_SCALE",
    "WSI_PROXY_SIGNAL_EVIDENCE",
}


@dataclass
class MCQInput:
    question: str
    options: List[str]


@dataclass
class AgentV5Result:
    answer_markdown: str
    trace_markdown: str
    raw_output: str
    pred_label: str
    pred_text: str
    score: float
    question: str
    options: List[str]
    question_channel: str
    question_intent: str
    final_stage: str
    route: Dict[str, Any]
    evidence_route: str
    evidence: List[Dict[str, Any]]
    context_prefix: str
    route_decision_raw: str
    route_decision_source: str
    prompt: str
    elapsed_sec: float


def norm_text(value: str) -> str:
    text = (value or "").strip().lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(value: Any, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def compact_json(obj: Any, max_chars: int = 2400) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def pick_option_from_response(response: str, options: List[str]) -> Tuple[str, str, float]:
    if len(options) != 4:
        return "", "", 0.0
    resp = response or ""
    resp_n = norm_text(resp)
    match = re.search(r"\b([ABCD])\b", resp.upper())
    if match:
        label = match.group(1)
        return options[LABELS.index(label)], label, 1.0
    for idx, option in enumerate(options):
        option_n = norm_text(option)
        if option_n and option_n in resp_n:
            return option, LABELS[idx], 0.99
    best_i = 0
    best_s = -1.0
    for idx, option in enumerate(options):
        option_n = norm_text(option)
        score = SequenceMatcher(None, option_n, resp_n).ratio() if option_n else -1.0
        if score > best_s:
            best_i = idx
            best_s = score
    return options[best_i], LABELS[best_i], float(best_s)


def parse_mcq_message(message: str) -> MCQInput:
    text = str(message or "").strip()
    if not text:
        return MCQInput(question="", options=[])
    patterns = [
        r"(?im)^\s*A[\.\):]\s*(?P<a>.+?)\s*$",
        r"(?im)^\s*B[\.\):]\s*(?P<b>.+?)\s*$",
        r"(?im)^\s*C[\.\):]\s*(?P<c>.+?)\s*$",
        r"(?im)^\s*D[\.\):]\s*(?P<d>.+?)\s*$",
    ]
    values: Dict[str, str] = {}
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            values.update({key: value.strip() for key, value in match.groupdict().items() if value})
    if set(values) == {"a", "b", "c", "d"}:
        first_option = re.search(r"(?im)^\s*A[\.\):]\s*", text)
        question = text[: first_option.start()].strip() if first_option else text
        question = re.sub(r"(?im)^\s*(Question|Q)\s*[:：]\s*", "", question).strip()
        return MCQInput(question=question, options=[values["a"], values["b"], values["c"], values["d"]])
    return MCQInput(question=text, options=[])


def option_lines(options: List[str]) -> str:
    return "\n".join(f"{LABELS[idx]}. {options[idx]}" for idx in range(min(4, len(options))))


def route_answer_channel(question: str) -> Dict[str, str]:
    q = (question or "").lower()
    size_patterns = [
        r"tumou?r size",
        r"size of (the )?(tumou?r|carcinoma|mass|lesion)",
        r"(tumou?r|carcinoma|mass|lesion).{0,24}(size|diameter|dimension|largest|maximum)",
        r"(largest|maximum).{0,24}(tumou?r|carcinoma|mass|lesion)",
        r"maximum diameter",
        r"largest dimension",
        r"largest focus",
    ]
    if any(re.search(pattern, q) for pattern in size_patterns):
        return {"channel": "global_size", "intent": "tumor_size"}
    biomarker_patterns = [
        r"\bihc\b",
        r"immunohistochem",
        r"biomarker",
        r"\ber\b",
        r"\bpr\b",
        r"\bher2\b",
        r"ki[- ]?67",
        r"\bpd-?l1\b",
        r"hormone receptor",
        r"receptor status",
        r"\bfish\b",
    ]
    if any(re.search(pattern, q) for pattern in biomarker_patterns):
        return {"channel": "thumbnail_only", "intent": "ihc_biomarker"}
    return {"channel": "detailed", "intent": "detail_default"}


def load_router_concepts() -> Dict[str, Dict[str, Any]]:
    ontology_path = Path(__file__).resolve().parent / "wsi_agent" / "signal_ontology.json"
    try:
        data = json.loads(ontology_path.read_text(encoding="utf-8"))
        concepts = data.get("router_concepts", [])
    except Exception:
        concepts = []
    if not isinstance(concepts, list):
        concepts = []
    out: Dict[str, Dict[str, Any]] = {}
    for item in concepts:
        if not isinstance(item, dict):
            continue
        concept_id = str(item.get("id", "")).strip()
        if concept_id:
            out[concept_id] = item
    return out


ROUTER_CONCEPTS = load_router_concepts()


def dedupe_nonempty_str_list(values: Any) -> List[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,\n]+", values)
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    out: List[str] = []
    for value in raw_values:
        item = str(value).strip()
        if item and item not in out:
            out.append(item)
    return out


def normalize_target_levels(levels: List[str]) -> List[str]:
    allowed = {"global", "evidence", "view", "signal"}
    out: List[str] = []
    for level in levels:
        item = str(level).strip().lower()
        if item in allowed and item not in out:
            out.append(item)
    out.sort(key=lambda item: MEMORY_LEVEL_ORDER.index(item))
    return out


def levels_from_task_layer(task_layer: str) -> List[str]:
    task = str(task_layer or "").strip().upper()
    if task == "WSI_GLOBAL_ONLY":
        return ["global"]
    if task == "WSI_PROXY_SIGNAL_EVIDENCE":
        return ["signal", "evidence", "global"]
    if task == "WSI_SPATIAL_MULTI_SCALE":
        return ["view", "evidence", "global"]
    if task == "WSI_HIERARCHICAL":
        return ["signal", "evidence", "view", "global"]
    return ["global"]


def infer_task_layer_from_levels(levels: List[str]) -> str:
    level_set = set(levels)
    if level_set == {"global"}:
        return "WSI_GLOBAL_ONLY"
    if "signal" in level_set or "evidence" in level_set:
        return "WSI_HIERARCHICAL"
    if "view" in level_set:
        return "WSI_SPATIAL_MULTI_SCALE"
    return "WSI_GLOBAL_ONLY"


def render_router_concepts_for_prompt(max_concepts: int = 24, max_signals: int = 6) -> str:
    lines: List[str] = []
    for idx, (concept_id, cfg) in enumerate(ROUTER_CONCEPTS.items()):
        if idx >= max_concepts:
            break
        signals = dedupe_nonempty_str_list(cfg.get("target_signals", []))
        sig_text = ", ".join(signals[:max_signals]) if signals else "N/A"
        lines.append(f"- {concept_id}: {cfg.get('name', concept_id)} | signals={sig_text}")
    return "\n".join(lines)


def infer_concept_from_target_signals(target_signals: List[str]) -> str:
    target_set = set(dedupe_nonempty_str_list(target_signals))
    best_id = ""
    best_overlap = 0
    best_cover = -1.0
    for concept_id, cfg in ROUTER_CONCEPTS.items():
        signals = dedupe_nonempty_str_list(cfg.get("target_signals", []))
        if not signals:
            continue
        sig_set = set(signals)
        overlap = len(target_set & sig_set)
        if overlap <= 0:
            continue
        cover = float(overlap) / float(max(1, len(sig_set)))
        if overlap > best_overlap or (overlap == best_overlap and cover > best_cover):
            best_id = concept_id
            best_overlap = overlap
            best_cover = cover
    return best_id


def extract_json_object_loose(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def build_memory_routing_prompt(
    question: str,
    options: List[str],
    global_context: str,
    policy_prefix: str,
    signal_candidates: Optional[List[str]] = None,
) -> str:
    context_prefix = f"Global slide context (coarse): {global_context}\n\n" if global_context else ""
    candidates = dedupe_nonempty_str_list(signal_candidates or [])
    candidates_prefix = ""
    if candidates:
        candidates_prefix = "Available signal ids in current memory (prefer these): " + ", ".join(candidates[:32]) + "\n"
    question_block = f"Question: {question}\n"
    if len(options) == 4:
        question_block += option_lines(options) + "\n"
    return (
        policy_prefix
        + context_prefix
        + candidates_prefix
        + "You are a memory-router for pathology MCQ.\n"
        + "Step 1: choose ONE target_concept id from Concept guide.\n"
        + "Step 2: select target_signals only from that concept's mapped signals (or [] if N/A).\n"
        + "Select the MINIMAL memory level policy required to answer reliably.\n"
        + "Levels: global, evidence, view, signal.\n"
        + "Concept guide:\n"
        + render_router_concepts_for_prompt()
        + "\n\n"
        + "Return ONLY strict JSON with keys:\n"
        + "{\"task_layer\":\"WSI_GLOBAL_ONLY|WSI_HIERARCHICAL|WSI_SPATIAL_MULTI_SCALE|WSI_PROXY_SIGNAL_EVIDENCE\","
        + "\"target_levels\":[\"global|evidence|view|signal\"],"
        + "\"target_concept\":\"<concept_id>\","
        + "\"target_signals\":[],"
        + "\"source_policy\":\"WSI_ONLY\","
        + "\"route\":\"agent_memory_router\","
        + "\"reason\":\"short\"}\n\n"
        + question_block
    )


def parse_agent_memory_route(
    response: str,
    fallback_route: Dict[str, Any],
    signal_candidates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    obj = extract_json_object_loose(response)
    task_layer = str(obj.get("task_layer", "")).strip().upper()
    target_levels_raw = obj.get("target_levels", [])
    memory_level = str(obj.get("memory_level", "")).strip().lower()
    route_name = str(obj.get("route", "")).strip() or "agent_memory_router"
    target_concept = str(obj.get("target_concept", "") or obj.get("concept_id", "") or obj.get("concept", "")).strip()
    if target_concept and target_concept not in ROUTER_CONCEPTS:
        target_concept = ""

    if isinstance(target_levels_raw, list):
        levels = [str(value) for value in target_levels_raw]
    elif isinstance(target_levels_raw, str) and target_levels_raw.strip():
        levels = [item.strip() for item in re.split(r"[,\s]+", target_levels_raw) if item.strip()]
    else:
        levels = []
    if memory_level in {"global", "evidence", "view", "signal"}:
        if memory_level == "global":
            levels = ["global"]
        elif memory_level == "evidence":
            levels = ["evidence", "global"]
        elif memory_level == "view":
            levels = ["view", "evidence", "global"]
        else:
            levels = ["signal", "view", "evidence", "global"]

    target_signals = dedupe_nonempty_str_list(obj.get("target_signals", []))
    allowed_signal_set = set(dedupe_nonempty_str_list(signal_candidates or []))
    if allowed_signal_set:
        target_signals = [sid for sid in target_signals if sid in allowed_signal_set]

    concept_cfg = ROUTER_CONCEPTS.get(target_concept, {})
    concept_signals = dedupe_nonempty_str_list(concept_cfg.get("target_signals", []))
    if allowed_signal_set and concept_signals:
        concept_signals = [sid for sid in concept_signals if sid in allowed_signal_set]
    concept_levels = normalize_target_levels(dedupe_nonempty_str_list(concept_cfg.get("target_levels", [])))
    concept_task_layer = str(concept_cfg.get("task_layer", "")).strip().upper()
    if concept_task_layer not in ALLOWED_TASK_LAYERS:
        concept_task_layer = ""

    if target_concept and concept_signals:
        if target_signals:
            concept_signal_set = set(concept_signals)
            target_signals = [sid for sid in target_signals if sid in concept_signal_set]
        if not target_signals:
            target_signals = list(concept_signals)

    if not levels and concept_levels:
        levels = concept_levels
    if task_layer in ALLOWED_TASK_LAYERS and not levels:
        levels = levels_from_task_layer(task_layer)
    levels = normalize_target_levels(levels)
    if not levels:
        return dict(fallback_route)
    if task_layer not in ALLOWED_TASK_LAYERS:
        task_layer = concept_task_layer or infer_task_layer_from_levels(levels)
    if not target_concept and target_signals:
        target_concept = infer_concept_from_target_signals(target_signals)

    out = {
        "route": route_name,
        "target_signals": target_signals,
        "target_levels": levels,
        "task_layer": task_layer,
        "source_policy": "WSI_ONLY",
    }
    if target_concept:
        out["target_concept"] = target_concept
    return out


def build_global_first_prompt(question: str, options: List[str], global_context: str, policy_prefix: str) -> str:
    context_prefix = f"Global slide context (coarse only): {global_context}\n\n" if global_context else ""
    if len(options) == 4:
        return (
            policy_prefix
            + context_prefix
            + "You are doing a global-first triage for pathology MCQ.\n"
            + "Use only global context and image for this first pass.\n"
            + "Return ONLY compact JSON: {\"answer\":\"A|B|C|D\",\"confidence\":0.0-1.0}\n\n"
            + f"Question: {question}\n"
            + option_lines(options)
            + "\n"
        )
    return (
        policy_prefix
        + context_prefix
        + "You are doing a global-first triage for a pathology slide question.\n"
        + "Use only global context and image for this first pass. Answer concisely.\n\n"
        + f"Question: {question}\nAnswer:"
    )


def build_thumbnail_only_prompt(question: str, options: List[str], policy_prefix: str) -> str:
    if len(options) == 4:
        return (
            "You are given a multiple-choice question about a pathology slide. "
            "Answer with only one letter from A, B, C, D.\n\n"
            f"Question: {question}\n"
            f"{option_lines(options)}\n"
            "Answer:"
        )
    return (
        policy_prefix
        + "You are given a question about a pathology slide. Use the slide thumbnail image only. "
        + "Answer concisely and state uncertainty when the image is insufficient.\n\n"
        + f"Question: {question}\nAnswer:"
    )


def parse_global_answer_confidence(response: str, options: List[str], fallback_confidence: float = 0.55) -> Tuple[str, str, float]:
    raw = str(response or "").strip()
    pred_text, pred_label, _score = pick_option_from_response(raw, options)
    conf: Optional[float] = None
    obj = extract_json_object_loose(raw)
    if obj:
        answer = str(obj.get("answer") or obj.get("option") or obj.get("pred") or obj.get("choice") or "").strip().upper()
        if answer in LABELS and len(options) == 4:
            pred_label = answer
            pred_text = options[LABELS.index(answer)]
        if obj.get("confidence") is not None:
            try:
                conf = float(obj.get("confidence"))
            except Exception:
                conf = None
    if conf is None:
        match = re.search(r"confidence[^0-9]{0,12}([01](?:\.\d+)?)", raw.lower())
        if match:
            try:
                conf = float(match.group(1))
            except Exception:
                conf = None
    if conf is None:
        conf = float(fallback_confidence)
    return pred_text, pred_label, max(0.0, min(1.0, float(conf)))


def global_context_summary(global_view: Dict[str, Any]) -> str:
    summary = str(global_view.get("summary", "")).strip()
    impression = str(global_view.get("overall_impression", "")).strip()
    return " ".join([item for item in [summary, impression] if item]).strip()


def signal_candidates_from_global(global_view: Dict[str, Any]) -> List[str]:
    ontology = global_view.get("signal_ontology", {})
    signals = ontology.get("signals", []) if isinstance(ontology, dict) else []
    candidates: List[str] = []
    if isinstance(signals, list):
        for sig in signals:
            if isinstance(sig, dict):
                sid = str(sig.get("signal_id", "")).strip()
            else:
                sid = str(sig or "").strip()
            if sid and sid not in candidates:
                candidates.append(sid)
    return candidates


def mcq_final_prompt(policy_prefix: str, context_prefix: str, question: str, options: List[str]) -> str:
    return (
        policy_prefix
        + context_prefix
        + "You are given a multiple-choice question about a pathology slide. "
        "Answer with only one letter from A, B, C, D.\n\n"
        f"Question: {question}\n"
        f"{option_lines(options)}\n"
        "Answer:"
    )


def freeform_final_prompt(policy_prefix: str, context_prefix: str, question: str) -> str:
    return (
        policy_prefix
        + context_prefix
        + "You are answering a pathology whole-slide question using only the slide image and selected memory context. "
        "Give a direct concise answer, cite relevant grid/anchor/view ids when they support the answer, and state uncertainty when needed.\n\n"
        f"Question: {question}\nAnswer:"
    )


def _evidence_summary(evidence: List[Dict[str, Any]], max_items: int = 6) -> str:
    if not evidence:
        return "No structured evidence was returned."
    lines: List[str] = []
    for item in evidence[:max_items]:
        level = str(item.get("level", "")).strip() or "evidence"
        anchor = str(item.get("anchor_id", "")).strip()
        view = str(item.get("view_id", "")).strip()
        signal = str(item.get("signal_id", "")).strip()
        text = compact_text(item.get("text", ""), 360)
        parts = [level]
        if anchor:
            parts.append(f"anchor={anchor}")
        if view:
            parts.append(f"view={view}")
        if signal:
            parts.append(f"signal={signal}")
        if text:
            parts.append(text)
        lines.append("- " + " | ".join(parts))
    if len(evidence) > max_items:
        lines.append(f"- ... {len(evidence) - max_items} more evidence items")
    return "\n".join(lines)


def trace_markdown(result: AgentV5Result) -> str:
    route_for_display = dict(result.route or {})
    lines = [
        "<details>",
        "<summary>V5 Agent Trace</summary>",
        "",
        f"Question channel: `{html.escape(result.question_channel)}`",
        f"Question intent: `{html.escape(result.question_intent)}`",
        f"Final stage: `{html.escape(result.final_stage)}`",
        f"Evidence route: `{html.escape(result.evidence_route)}`",
        f"Route source: `{html.escape(result.route_decision_source or 'policy/rule')}`",
        "",
        "Route:",
        "```json",
        compact_json(route_for_display, 1800),
        "```",
    ]
    if result.route_decision_raw:
        lines.extend(["", "Router raw output:", "```text", compact_text(result.route_decision_raw, 1800), "```"])
    lines.extend(["", "Selected evidence:", _evidence_summary(result.evidence), "</details>"])
    return "\n".join(lines)


def _answer_markdown(raw: str, pred_label: str, pred_text: str, score: float, options: List[str]) -> str:
    if len(options) == 4 and pred_label:
        return (
            "### Answer\n"
            f"**{html.escape(pred_label)}. {html.escape(pred_text)}**\n\n"
            f"Model raw output: `{html.escape(compact_text(raw, 600))}`\n\n"
            f"Parse score: `{score:.2f}`"
        )
    return "### Answer\n" + (raw or "The model did not return a non-empty answer.")


def _global_evidence(context_summary: str, score: float = 0.0) -> List[Dict[str, Any]]:
    if not context_summary:
        return []
    return [
        {
            "level": "global",
            "anchor_id": "",
            "evidence_id": "",
            "signal_id": "",
            "signal_status": "",
            "view_id": "",
            "mag": "",
            "patch_xywh_level0": [],
            "text": context_summary,
            "score": float(score),
            "score_semantic": 0.0,
            "score_lexical": 0.0,
            "score_confidence": float(score),
            "score_signal": 0.0,
        }
    ]


def answer_question_v5(
    message: str,
    bank: Dict[str, Any],
    runtime: Dict[str, Any],
    defaults: Dict[str, Any],
    *,
    question: str = "",
    options: Optional[List[str]] = None,
    previous_messages: Optional[List[Dict[str, str]]] = None,
) -> AgentV5Result:
    started = time.perf_counter()
    parsed = parse_mcq_message(message)
    question_text = str(question or parsed.question).strip()
    option_values = [str(item).strip() for item in (options or parsed.options or []) if str(item).strip()]
    if len(option_values) != 4:
        option_values = []
    if not question_text:
        raise RuntimeError("No question was provided.")
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

    global_view = memory_bank.consult_global_view(slide_id)
    if not isinstance(global_view, dict) or global_view.get("error"):
        raise RuntimeError(f"consult_global_view failed for {slide_id}: {global_view}")
    context_summary = global_context_summary(global_view)
    route_signal_candidates = signal_candidates_from_global(global_view)

    retrieval_cfg = defaults.get("retrieval", {}) if isinstance(defaults.get("retrieval", {}), dict) else {}
    signal_topk = int(retrieval_cfg.get("signal_topk", 4))
    signal_views_per_signal = int(retrieval_cfg.get("signal_views_per_signal", 2))
    signal_context_max_chars = int(retrieval_cfg.get("signal_context_max_chars", 3200))
    memory_route_mode = str(retrieval_cfg.get("memory_route_mode", "agent") or "agent").strip().lower()
    global_conf_fallback = float(retrieval_cfg.get("global_conf_fallback", 0.55))

    policy_prefix = "Use only WSI evidence (image + memory context). Do not use external report or metadata assumptions.\n\n"
    vlm = get_vlm_client(make_vlm_config(runtime, defaults))
    question_route = route_answer_channel(question_text)
    question_channel = str(question_route.get("channel", "detailed"))
    question_intent = str(question_route.get("intent", "detail_default"))

    evidence: List[Dict[str, Any]] = []
    route: Dict[str, Any] = {}
    route_decision_raw = ""
    route_decision_source = ""
    evidence_route = ""
    context_prefix = ""
    prompt = ""
    raw = ""
    pred_label = ""
    pred_text = ""
    score = 0.0
    final_stage = "detailed" if question_channel == "detailed" else question_channel

    if question_channel == "thumbnail_only":
        route = {
            "route": f"question_policy:{question_intent}|thumbnail_only",
            "target_signals": [],
            "target_levels": ["global"],
            "task_layer": "WSI_GLOBAL_ONLY",
            "source_policy": "WSI_ONLY",
        }
        evidence_route = str(route["route"])
        prompt = build_thumbnail_only_prompt(question_text, option_values, policy_prefix)
        raw = vlm.generate(image_path=image_path, prompt=prompt).strip()
        if len(option_values) == 4:
            pred_text, pred_label, score = pick_option_from_response(raw, option_values)
        evidence = _global_evidence(context_summary)
    elif question_channel == "global_size":
        route = {
            "route": f"question_policy:{question_intent}|global_size",
            "target_signals": [],
            "target_levels": ["global"],
            "task_layer": "WSI_GLOBAL_ONLY",
            "source_policy": "WSI_ONLY",
        }
        evidence_route = str(route["route"])
        prompt = build_global_first_prompt(question_text, option_values, context_summary, policy_prefix)
        raw = vlm.generate(image_path=image_path, prompt=prompt).strip()
        if len(option_values) == 4:
            pred_text, pred_label, score = parse_global_answer_confidence(raw, option_values, global_conf_fallback)
        evidence = _global_evidence(context_summary, score)
        final_stage = "global"
    else:
        fallback_route = memory_bank.route_question(question_text)
        route = dict(fallback_route)
        route_decision_source = "rule"
        if memory_route_mode == "agent":
            route_prompt = build_memory_routing_prompt(
                question=question_text,
                options=option_values,
                global_context=context_summary,
                policy_prefix=policy_prefix,
                signal_candidates=route_signal_candidates,
            )
            try:
                route_decision_raw = vlm.generate(image_path=image_path, prompt=route_prompt).strip()
                parsed_route = parse_agent_memory_route(route_decision_raw, fallback_route, route_signal_candidates)
                if parsed_route == fallback_route:
                    route_decision_source = "agent_fallback_rule"
                else:
                    route = parsed_route
                    route_decision_source = "agent"
            except Exception as exc:
                route_decision_raw = f"MEMORY_ROUTE_ERROR: {type(exc).__name__}: {exc}"
                route_decision_source = "agent_error_fallback_rule"
        if route_decision_source:
            route["route"] = f"{route.get('route', 'memory_router')}|src={route_decision_source}"
        evidence_route = f"question_policy:{question_intent}|{route.get('route', '')}"

        global_only_evidence = _global_evidence(context_summary)
        try:
            signal_payload = memory_bank.build_signal_patch_context(
                question=question_text,
                route=route,
                max_signals=max(1, signal_topk),
                max_views_per_signal=max(1, signal_views_per_signal),
                max_chars=max(500, signal_context_max_chars),
            )
            context_prefix = str(signal_payload.get("context", "")).strip()
            if context_prefix:
                context_prefix += "\n\n"
            payload_evidence = signal_payload.get("evidence", [])
            evidence = payload_evidence if isinstance(payload_evidence, list) else []
            payload_route = signal_payload.get("route", {})
            if isinstance(payload_route, dict):
                route.setdefault("task_layer", payload_route.get("task_layer", ""))
                route.setdefault("source_policy", payload_route.get("source_policy", "WSI_ONLY"))
            if not evidence and global_only_evidence:
                evidence_route = f"{evidence_route or 'signal_router'}_global_fallback_empty_hits"
                evidence = global_only_evidence
                context_prefix = (
                    "Slide-level context from memory (signal aggregation empty; fallback to global): "
                    f"{context_summary}\n\n"
                )
        except Exception as exc:
            if global_only_evidence:
                evidence_route = "global_fallback_after_signal_router_error"
                route = {
                    "route": evidence_route,
                    "target_signals": [],
                    "target_levels": ["global"],
                    "task_layer": "WSI_GLOBAL_ONLY",
                    "source_policy": "WSI_ONLY",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                evidence = global_only_evidence
                context_prefix = (
                    "Slide-level context from memory (signal aggregation failed; fallback to global): "
                    f"{context_summary}\n\n"
                )
            else:
                raise

        if str(route.get("source_policy", "WSI_ONLY")) != "WSI_ONLY":
            policy_prefix = ""
        if len(option_values) == 4:
            prompt = mcq_final_prompt(policy_prefix, context_prefix, question_text, option_values)
        else:
            prompt = freeform_final_prompt(policy_prefix, context_prefix, question_text)
        raw = vlm.generate(image_path=image_path, prompt=prompt).strip()
        if len(option_values) == 4:
            pred_text, pred_label, score = pick_option_from_response(raw, option_values)

    if not raw:
        raw = "The agent did not return a non-empty answer."
    answer_md = _answer_markdown(raw, pred_label, pred_text, score, option_values)
    result = AgentV5Result(
        answer_markdown=answer_md,
        trace_markdown="",
        raw_output=raw,
        pred_label=pred_label,
        pred_text=pred_text,
        score=score,
        question=question_text,
        options=option_values,
        question_channel=question_channel,
        question_intent=question_intent,
        final_stage=final_stage,
        route=route,
        evidence_route=evidence_route,
        evidence=evidence,
        context_prefix=context_prefix,
        route_decision_raw=route_decision_raw,
        route_decision_source=route_decision_source,
        prompt=prompt,
        elapsed_sec=round(time.perf_counter() - started, 3),
    )
    result.trace_markdown = trace_markdown(result)
    return result
