from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tifffile
except Exception:
    tifffile = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

try:
    import openslide
except Exception:
    openslide = None

try:
    from .tumor_size_tool import quantify_tumor_size_from_thumbnail
except Exception:
    from tumor_size_tool import quantify_tumor_size_from_thumbnail


class RealWSILoader:
    """Builds hierarchical semantic memory from real SVS pixels (thumbnail-level analysis)."""

    def __init__(
        self,
        target_regions: int = 5,
        min_patches: int = 5,
        max_patches: int = 10,
        thumbnail_max_side: int = 4096,
        grid_size: int = 8,
        thumbnail_tile_size: int = 384,
        tissue_ratio_thresh: float = 0.20,
        save_tile_pngs: bool = True,
        stage1_use_lowest_level: bool = True,
        use_llm_tile_planner: bool = False,
        tile_llm_backend: str = "local_hf",
        tile_llm_model_name: str = "wisdomik/Quilt-Llava-v1.5-7b",
        tile_llm_model_path: str = "",
        tile_llm_device_map: str = "auto",
        tile_llm_dtype: str = "bf16",
        tile_llm_max_new_tokens: int = 192,
        tile_llm_grid_max_side: int = 1536,
        tile_llm_max_anchor_tiles: int = 32,
        tile_llm_max_anchors_per_tile: int = 2,
        gemini_api_key: str = "",
        gemini_api_env: str = "GEMINI_API_KEY",
        gemini_request_timeout: int = 120,
        api_provider: str = "",
        api_key: str = "",
        api_key_env: str = "",
        api_key_file: str = "",
        api_base_url: str = "",
        api_version: str = "v1beta",
        api_timeout_sec: int = 300,
        use_llm_patch_describer: bool = False,
        patch_description_strict: bool = True,
        patch_llm_max_new_tokens: int = 224,
        patch_llm_batch_size: int = 128,
        anchor_crop_size_level0: int = 4096,
        patch_crop_size_level0: int = 512,
        real_patch_output_size: int = 256,
        verbose: bool = False,
        patch_description_conf_threshold: float = 0.0,
        quilt_llava_repo: str = "",
    ):
        self.target_regions = target_regions
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.thumbnail_max_side = int(thumbnail_max_side)
        self.grid_size = int(grid_size)
        self.thumbnail_tile_size = int(max(32, thumbnail_tile_size))
        self.tissue_ratio_thresh = float(tissue_ratio_thresh)
        self.save_tile_pngs = bool(save_tile_pngs)
        self.stage1_use_lowest_level = bool(stage1_use_lowest_level)
        self.use_llm_tile_planner = bool(use_llm_tile_planner)
        self.tile_llm_backend = str(tile_llm_backend or "local_hf").strip().lower()
        self.tile_llm_model_name = str(tile_llm_model_name)
        self.tile_llm_model_path = str(tile_llm_model_path)
        self.tile_llm_device_map = str(tile_llm_device_map)
        self.tile_llm_dtype = str(tile_llm_dtype).lower()
        self.tile_llm_max_new_tokens = int(tile_llm_max_new_tokens)
        self.tile_llm_grid_max_side = int(max(512, tile_llm_grid_max_side))
        self.tile_llm_max_anchor_tiles = int(max(1, tile_llm_max_anchor_tiles))
        self.tile_llm_max_anchors_per_tile = int(max(1, tile_llm_max_anchors_per_tile))
        self.gemini_api_key = str(gemini_api_key or "").strip()
        self.gemini_api_env = str(gemini_api_env or "GEMINI_API_KEY").strip() or "GEMINI_API_KEY"
        self.gemini_request_timeout = int(max(30, gemini_request_timeout))
        self.api_provider = str(api_provider or "").strip().lower()
        self.api_key = str(api_key or "").strip()
        self.api_key_env = str(api_key_env or "").strip()
        self.api_key_file = str(api_key_file or "").strip()
        self.api_base_url = str(api_base_url or "").strip()
        self.api_version = str(api_version or "v1beta").strip()
        self.api_timeout_sec = int(max(30, api_timeout_sec))
        self.use_llm_patch_describer = bool(use_llm_patch_describer)
        self.patch_description_strict = bool(patch_description_strict)
        self.patch_description_conf_threshold = float(max(0.0, min(1.0, patch_description_conf_threshold)))
        self.patch_llm_max_new_tokens = int(max(64, patch_llm_max_new_tokens))
        self.patch_llm_batch_size = int(max(1, patch_llm_batch_size))
        self.anchor_crop_size_level0 = int(max(512, anchor_crop_size_level0))
        self.patch_crop_size_level0 = int(max(256, patch_crop_size_level0))
        self.real_patch_output_size = int(max(128, real_patch_output_size))
        self.verbose = bool(verbose)
        self.signal_ontology_path = os.path.join(os.path.dirname(__file__), "signal_ontology.json")
        self.signal_ontology = self._load_signal_ontology()
        self._tile_llm_model = None
        self._tile_llm_processor = None
        self._gemini_client = None
        self._gemini_types = None
        self._last_tile_review_order: List[str] = []
        self._last_anchor_tile_order: List[str] = []
        self._last_tile_coord_predictions: List[Dict[str, Any]] = []
        self._tile_llm_runtime = "uninitialized"
        self.quilt_llava_repo_path = str(
            quilt_llava_repo or os.getenv("SLIDEBANK_QUILT_LLAVA_REPO", "")
        ).strip()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[RealWSILoader] {message}")

    @staticmethod
    def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _map_box_to_level0(box: List[int], sx: float, sy: float) -> List[int]:
        x, y, w, h = box
        return [
            max(0, int(round(x * sx))),
            max(0, int(round(y * sy))),
            max(1, int(round(w * sx))),
            max(1, int(round(h * sy))),
        ]

    @staticmethod
    def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        h_img, w_img = img.shape[:2]
        x0 = max(0, min(x, w_img - 1))
        y0 = max(0, min(y, h_img - 1))
        x1 = max(x0 + 1, min(x0 + w, w_img))
        y1 = max(y0 + 1, min(y0 + h, h_img))
        return img[y0:y1, x0:x1]

    def _load_lowest_pyramid_level(self, slide_path: str) -> Dict[str, Any]:
        if tifffile is None:
            raise RuntimeError("tifffile is required to read real SVS files.")
        self._log(f"Reading SVS pyramid metadata: {slide_path}")
        with tifffile.TiffFile(slide_path) as tif:
            if not tif.series:
                raise RuntimeError(f"No image series found in SVS: {slide_path}")
            series = tif.series[0]
            levels = list(series.levels) if getattr(series, "levels", None) else [series]
            level_dims: List[tuple[int, int]] = []
            for lv in levels:
                h, w = int(lv.shape[0]), int(lv.shape[1])
                level_dims.append((w, h))
            self._log(f"Detected {len(level_dims)} pyramid levels (W,H): {level_dims}")

            # A large single-level tiled TIFF has no precomputed low-resolution
            # pyramid. Reading levels[-1] in that case decodes the complete WSI
            # (often several GB in memory). Ask OpenSlide to create bounded
            # thumbnails instead; keep direct tifffile decoding for true pyramid
            # levels and genuinely small images.
            use_openslide_thumbnail = (
                len(levels) == 1
                and max(level_dims[0]) > int(self.thumbnail_max_side)
            )
            try:
                if use_openslide_thumbnail:
                    lowest, medium = self._load_lowres_single_level_with_openslide(slide_path)
                    self._log(
                        "Single-level large TIFF: using chunked OpenSlide thumbnails "
                        f"| lowest={lowest.shape} | medium={medium.shape}"
                    )
                else:
                    lowest = self._as_uint8_rgb(levels[-1].asarray())
                    medium = (
                        self._as_uint8_rgb(levels[-2].asarray())
                        if len(levels) > 1
                        else lowest.copy()
                    )
                    self._log(
                        f"Decoded levels with tifffile directly | lowest={lowest.shape} | medium={medium.shape}"
                    )
            except Exception as exc:
                self._log(f"tifffile decode failed: {type(exc).__name__}: {exc}")
                # Fallback order:
                # 1) OpenSlide thumbnail path (most robust for tiled JPEG SVS)
                # 2) Pillow frame decode
                # This keeps ingestion running even when one backend fails on a slide.
                try:
                    lowest, medium = self._load_lowres_with_openslide(slide_path)
                    self._log(
                        f"Fallback success via OpenSlide | lowest={lowest.shape} | medium={medium.shape}"
                    )
                except Exception as os_exc:
                    self._log(f"OpenSlide fallback failed: {type(os_exc).__name__}: {os_exc}")
                    try:
                        lowest, medium = self._load_lowres_with_pillow(slide_path, level_dims)
                        self._log(
                            f"Fallback success via Pillow | lowest={lowest.shape} | medium={medium.shape}"
                        )
                    except Exception as pil_exc:
                        raise RuntimeError(
                            "Failed to decode low-resolution SVS image. "
                            f"tifffile={type(exc).__name__}: {exc}; "
                            f"openslide={type(os_exc).__name__}: {os_exc}; "
                            f"pillow={type(pil_exc).__name__}: {pil_exc}. "
                            "Install `imagecodecs` to improve tifffile JPEG decoding robustness."
                        ) from pil_exc

            image_desc = ""
            try:
                page0 = tif.pages[0]
                if "ImageDescription" in page0.tags:
                    image_desc = str(page0.tags["ImageDescription"].value)
            except Exception:
                image_desc = ""
            if image_desc:
                self._log(f"ImageDescription head: {image_desc[:160].replace(chr(10), ' ')}")

        return {"lowest": lowest, "medium": medium, "level_dims": level_dims, "image_desc": image_desc}

    @staticmethod
    def _fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
        side = max(1, int(max_side))
        scale = min(1.0, side / float(max(width, height)))
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))
        return out_w, out_h

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)

    @staticmethod
    def _resolve_model_path(model_name_or_path: str) -> str:
        model_path = os.path.expanduser(str(model_name_or_path))
        if not os.path.isdir(model_path):
            return model_name_or_path
        snapshots_dir = os.path.join(model_path, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return model_path
        ref_path = os.path.join(model_path, "refs", "main")
        if os.path.isfile(ref_path):
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    ref_id = f.read().strip()
                candidate = os.path.join(snapshots_dir, ref_id)
                if os.path.isdir(candidate):
                    return candidate
            except Exception:
                pass
        snapshots = [
            os.path.join(snapshots_dir, d)
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if snapshots:
            snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return snapshots[0]
        return model_path

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        s = (text or "").strip().replace("```json", "").replace("```", "").strip()
        if not s:
            return {}

        def _normalize_jsonish(raw: str) -> str:
            # Some VLM outputs escape markdown characters inside JSON keys, e.g. "qc\_flags".
            # That is invalid JSON; normalize these escapes before parsing.
            out = str(raw)
            out = out.replace("\\_", "_")
            # Common malformed agent output: tool_trace list misses one closing bracket
            # before next top-level key (selected_cell/selected_cells/confidence/rationale).
            out = re.sub(
                r'("tool_trace"\s*:\s*\[\s*\[[\s\S]*?\])\s*,\s*("selected_cell"|"selected_cells"|"confidence"|"rationale")',
                r"\1], \2",
                out,
                flags=re.IGNORECASE,
            )
            return out

        s = _normalize_jsonish(s)

        def _loads_obj(candidate: str) -> Dict[str, Any]:
            cand = _normalize_jsonish(str(candidate or ""))
            if not cand:
                return {}
            # Remove trailing commas before closing braces/brackets.
            cand = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                obj = json.loads(cand)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

        obj = _loads_obj(s)
        if obj:
            return obj

        # If model output starts a JSON object but is truncated, try balancing braces.
        if s.startswith("{"):
            depth = 0
            in_str = False
            esc = False
            for ch in s:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth = max(0, depth - 1)
            if depth > 0:
                obj = _loads_obj(s + ("}" * depth))
                if obj:
                    return obj

        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return {}
        obj = _loads_obj(m.group(0))
        if obj:
            return obj
        # Last attempt for partially truncated matched span.
        span = str(m.group(0))
        depth = span.count("{") - span.count("}")
        if depth > 0:
            obj = _loads_obj(span + ("}" * depth))
            if obj:
                return obj
        return {}

    def _recover_obj_from_raw_text(
        self,
        raw_text: str,
        prompt: str,
    ) -> Dict[str, Any]:
        s = str(raw_text or "").strip()
        if not s:
            return {}
        obj = self._extract_json_object(s)
        if isinstance(obj, dict) and obj:
            return obj

        p = str(prompt or "").lower()
        # Agent cell-selection prompts.
        if "selected_cells" in p:
            top_k = 4
            m_top = re.search(r"top[- ]?(\d+)", p, flags=re.IGNORECASE)
            if m_top:
                try:
                    top_k = max(1, min(16, int(m_top.group(1))))
                except Exception:
                    top_k = 4
            grid = 4
            m_grid = re.search(r"(\d+)\s*x\s*(\d+)\s*grid", p, flags=re.IGNORECASE)
            if m_grid and m_grid.group(1) == m_grid.group(2):
                try:
                    grid = max(2, min(16, int(m_grid.group(1))))
                except Exception:
                    grid = 4
            # Regex salvage is prone to copied-template coordinates; disable it for
            # selected_cells and let feature fallback handle malformed outputs.
            allow_regex_fallback = False
            cells = self._try_parse_multi_cells_from_text(
                s,
                top_k=top_k,
                grid=grid,
                allow_regex_fallback=allow_regex_fallback,
            )
            if cells:
                return {
                    "selected_cells": [
                        {
                            "row": int(c.get("row", 0)),
                            "col": int(c.get("col", 0)),
                            "confidence": float(c.get("confidence", 0.5)),
                            "rationale": str(c.get("rationale", "")).strip() or "recovered_from_raw_text",
                        }
                        for c in cells
                    ]
                }
        if "selected_cell" in p:
            one = self._try_parse_single_cell_from_text(s, grid=4)
            if one:
                return {
                    "selected_cell": {
                        "row": int(one.get("row", 0)),
                        "col": int(one.get("col", 0)),
                    },
                    "confidence": float(one.get("confidence", 0.5)),
                    "rationale": str(one.get("rationale", "")).strip() or "recovered_from_raw_text",
                }

        # Patch-description prompt.
        if "caption" in p and "qc_flags" in p:
            patch_obj = self._coerce_patch_desc_obj_from_text(s)
            if patch_obj:
                return patch_obj

        # Tile classification prompt.
        if "label" in p and "confidence" in p:
            m_label = re.search(r'"?label"?\s*[:=]\s*"?(tumor|necrosis|adipose|stroma|background|other)"?', s, flags=re.IGNORECASE)
            if m_label:
                label = self._normalize_tile_label(m_label.group(1))
                conf = 0.5
                m_conf = re.search(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', s, flags=re.IGNORECASE)
                if m_conf:
                    try:
                        conf = float(m_conf.group(1))
                    except Exception:
                        conf = 0.5
                conf = max(0.0, min(1.0, conf))
                return {
                    "label": label,
                    "confidence": conf,
                    "rationale": "recovered_from_raw_text",
                }
        return {}

    @staticmethod
    def _coerce_patch_desc_obj_from_text(text: str) -> Dict[str, Any]:
        s = str(text or "").strip().replace("```json", "").replace("```", "").strip()
        if not s:
            return {}
        s = s.replace("\\_", "_")
        s = s.replace("\u0000", " ").strip()
        obj = RealWSILoader._extract_json_object(s)
        if isinstance(obj, dict) and len(obj) > 0:
            return obj

        cap = ""
        # Accept caption key with or without opening quote, e.g.
        # "caption": "..."  or  caption": "..."
        m = re.search(
            r'["\']?caption["\']?\s*:\s*"((?:[^"\\]|\\.)+)"',
            s,
            flags=re.IGNORECASE,
        )
        if m:
            cap = str(m.group(1)).strip().replace('\\"', '"')
        if not cap:
            lines: List[str] = []
            for ln in s.splitlines():
                t = str(ln).strip().strip(",")
                if not t:
                    continue
                low = t.lower()
                if low.startswith(("confidence", '"confidence"', "'confidence'")):
                    continue
                if low.startswith(("qc_flags", '"qc_flags"', "'qc_flags'")):
                    continue
                if t in {"{", "}"}:
                    continue
                lines.append(t)
            cap = " ".join(lines).strip()
        if not cap:
            cap = s
        cap = re.sub(r"^[\"']?caption[\"']?\s*:\s*[\"']?", "", cap, flags=re.IGNORECASE)
        cap = re.sub(r"\s+", " ", cap).strip().strip('"').strip("'")
        if not cap:
            return {}

        qc_flags: List[str] = ["none"]
        s_low = s.lower()
        if "blur" in s_low:
            qc_flags = ["blur"]
        elif "out_of_focus" in s_low or "out of focus" in s_low:
            qc_flags = ["out_of_focus"]
        elif "artifact" in s_low:
            qc_flags = ["artifact"]
        elif "low_tissue" in s_low or "low tissue" in s_low:
            qc_flags = ["low_tissue"]

        confidence = 0.5
        m_conf = re.search(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', s, flags=re.IGNORECASE)
        if m_conf:
            try:
                confidence = float(m_conf.group(1))
            except Exception:
                confidence = 0.5
        confidence = max(0.01, min(0.99, confidence))
        return {
            "caption": cap,
            "confidence": round(confidence, 4),
            "qc_flags": qc_flags,
        }

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text
        candidates = getattr(response, "candidates", None)
        if not isinstance(candidates, list):
            return ""
        parts_out: List[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                t = str(getattr(part, "text", "") or "").strip()
                if t:
                    parts_out.append(t)
        return "\n".join(parts_out).strip()

    def _is_llava_checkpoint(self, model_name_or_path: str) -> bool:
        cfg = self._load_config_dict(model_name_or_path)
        model_type = str(cfg.get("model_type", "")).strip().lower()
        if "llava" in model_type:
            return True
        archs = " ".join(str(x) for x in cfg.get("architectures", []))
        return "llava" in archs.lower()

    @staticmethod
    def _patch_transformers_registry_conflicts() -> None:
        try:
            from transformers import AutoConfig
        except Exception:
            return
        if getattr(AutoConfig, "_wsichat_safe_register_patch", False):
            return
        original_register = AutoConfig.register

        def safe_register(model_type: str, config: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original_register(model_type, config, *args, **kwargs)
            except ValueError as exc:
                msg = str(exc)
                if "already used by a Transformers config" in msg:
                    return original_register(model_type, config, exist_ok=True)
                raise

        AutoConfig.register = safe_register  # type: ignore[assignment]
        setattr(AutoConfig, "_wsichat_safe_register_patch", True)

    @staticmethod
    def _patch_transformers_legacy_mask_symbols() -> None:
        try:
            import torch
            import transformers.models.bloom.modeling_bloom as bloom_mod

            if not hasattr(bloom_mod, "_expand_mask"):
                def _expand_mask_bloom(mask: Any, tgt_length: Optional[int] = None) -> Any:
                    bsz, src_len = mask.shape
                    tgt_len = tgt_length if tgt_length is not None else src_len
                    return mask[:, None, None, :].to(torch.bool).expand(bsz, 1, tgt_len, src_len)

                bloom_mod._expand_mask = _expand_mask_bloom

            if not hasattr(bloom_mod, "_make_causal_mask"):
                def _make_causal_mask_bloom(
                    input_shape: Tuple[int, int],
                    device: Optional[Any] = None,
                    past_key_values_length: int = 0,
                ) -> Any:
                    bsz, tgt_len = input_shape
                    if device is None:
                        device = torch.device("cpu")
                    src_len = tgt_len + int(past_key_values_length)
                    i = torch.arange(tgt_len, device=device)[:, None]
                    j = torch.arange(src_len, device=device)[None, :]
                    mask = j <= (i + int(past_key_values_length))
                    return mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)

                bloom_mod._make_causal_mask = _make_causal_mask_bloom
        except Exception:
            pass

        try:
            import torch
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            import transformers.models.opt.modeling_opt as opt_mod

            if not hasattr(opt_mod, "_expand_mask"):
                def _expand_mask_opt(mask: Any, dtype: Any, tgt_len: Optional[int] = None) -> Any:
                    return AttentionMaskConverter._expand_mask(mask, dtype, tgt_len=tgt_len)

                opt_mod._expand_mask = _expand_mask_opt

            if not hasattr(opt_mod, "_make_causal_mask"):
                def _make_causal_mask_opt(
                    input_shape: Tuple[int, int],
                    dtype: Any,
                    device: Optional[Any] = None,
                    past_key_values_length: int = 0,
                ) -> Any:
                    if device is None:
                        device = torch.device("cpu")
                    return AttentionMaskConverter._make_causal_mask(
                        input_shape,
                        dtype,
                        device=device,
                        past_key_values_length=past_key_values_length,
                    )

                opt_mod._make_causal_mask = _make_causal_mask_opt
        except Exception:
            pass

    @staticmethod
    def _infer_quilt_model_name(
        model_path_arg: str,
        resolved_model_path: str,
        get_model_name_from_path: Any,
    ) -> str:
        raw_name = str(get_model_name_from_path(model_path_arg))
        if raw_name.startswith("models--") and "--" in raw_name:
            parts = raw_name.split("--")
            if len(parts) >= 3:
                raw_name = parts[-1]
        if "llava" in raw_name.lower():
            return raw_name

        resolved_name = str(get_model_name_from_path(resolved_model_path))
        if "llava" in resolved_name.lower():
            return resolved_name

        try:
            cfg_path = os.path.join(resolved_model_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = str(cfg.get("model_type", "")).lower()
            archs = " ".join(str(x) for x in cfg.get("architectures", []))
            if "llava" in model_type or "llava" in archs.lower():
                return "llava"
        except Exception:
            pass
        return resolved_name

    def _get_quilt_runtime_device(self) -> str:
        device_map = str(self.tile_llm_device_map or "auto").strip().lower()
        if device_map != "auto":
            return str(self.tile_llm_device_map)
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _lazy_load_tile_llm_quilt_repo(self, model_name_or_path: str) -> None:
        repo_path = os.path.expanduser(str(self.quilt_llava_repo_path or "").strip())
        if not repo_path or not os.path.isdir(repo_path):
            raise RuntimeError(
                "Quilt-LLaVA requires its external source checkout. Set "
                "SLIDEBANK_QUILT_LLAVA_REPO or patch_vlm.quilt_llava_repo "
                f"to that directory (configured value: {repo_path!r})."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            import torchvision
            disable_warn = getattr(torchvision, "disable_beta_transforms_warning", None)
            if callable(disable_warn):
                disable_warn()
        except Exception:
            pass
        self._patch_transformers_registry_conflicts()
        self._patch_transformers_legacy_mask_symbols()

        try:
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
            )
            from llava.conversation import conv_templates
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Quilt-LLaVA repo runtime."
            ) from exc

        disable_torch_init()
        model_path_arg = str(model_name_or_path)
        resolved_model_path = self._resolve_model_path(model_name_or_path)
        model_name = self._infer_quilt_model_name(
            model_path_arg=model_path_arg,
            resolved_model_path=resolved_model_path,
            get_model_name_from_path=get_model_name_from_path,
        )
        device = self._get_quilt_runtime_device()
        forced_single_device = False
        if str(self.tile_llm_device_map).strip().lower() == "auto":
            # Quilt-LLaVA + auto-sharding may leave part of weights on CPU and trigger
            # CUDA matmul device mismatch during generate. Prefer single-device load on CUDA.
            if str(device).startswith("cuda"):
                device_map = {"": device}
                forced_single_device = True
            else:
                device_map = "auto"
        else:
            # Force the full model onto the exact requested device (e.g., cuda:4).
            device_map = {"": device}
            forced_single_device = True
        tokenizer, model, image_processor, _ = load_pretrained_model(
            resolved_model_path,
            None,
            model_name,
            False,
            False,
            device_map=device_map,
            device=device,
        )
        if str(self.tile_llm_device_map).strip().lower() != "auto":
            try:
                model = model.to(device)
            except Exception:
                pass
        if image_processor is None and hasattr(model, "get_vision_tower"):
            try:
                import torch
                vision_tower = model.get_vision_tower()
                if vision_tower is not None:
                    if not getattr(vision_tower, "is_loaded", True):
                        vision_tower.load_model()
                    vision_tower.to(device=device, dtype=torch.float16)
                    image_processor = getattr(vision_tower, "image_processor", None)
            except Exception:
                pass
        if image_processor is None:
            raise RuntimeError(
                "image_processor is None after Quilt-LLaVA model load. Vision tower init failed."
            )

        self._tile_llm_model = model
        self._tile_llm_processor = {
            "runtime": "quilt_repo",
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
            "conv_templates": conv_templates,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
            "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
            "model_name": model_name,
        }
        self._tile_llm_runtime = "quilt_repo"
        self._log(
            "Tile LLM planner ready | backend=local_hf(quilt_repo) | "
            f"model={resolved_model_path} | device={device} | device_map={device_map}"
            f" | forced_single_device={forced_single_device}"
        )

    @staticmethod
    def _load_config_dict(model_name_or_path: str) -> Dict[str, Any]:
        model_path = RealWSILoader._resolve_model_path(model_name_or_path)
        cfg_path = os.path.join(str(model_path), "config.json")
        if not os.path.isfile(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _iter_llava_image_processor_sources(model_name_or_path: str) -> List[str]:
        cfg = RealWSILoader._load_config_dict(model_name_or_path)
        sources: List[str] = []
        mm_vision_tower = str(cfg.get("mm_vision_tower", "")).strip()
        if mm_vision_tower:
            sources.append(mm_vision_tower)
            sources.append(RealWSILoader._resolve_model_path(mm_vision_tower))
        sources.append(str(model_name_or_path))
        sources.append(RealWSILoader._resolve_model_path(model_name_or_path))
        out: List[str] = []
        seen = set()
        for s in sources:
            ss = str(s or "").strip()
            if not ss or ss in seen:
                continue
            seen.add(ss)
            out.append(ss)
        return out

    @staticmethod
    def _infer_llava_patch_size(model_name_or_path: str) -> int:
        cfg = RealWSILoader._load_config_dict(model_name_or_path)
        mm_vision_tower = str(cfg.get("mm_vision_tower", "")).strip()
        if mm_vision_tower:
            vt_cfg = RealWSILoader._load_config_dict(mm_vision_tower)
            if isinstance(vt_cfg, dict):
                if isinstance(vt_cfg.get("patch_size", None), int):
                    return int(vt_cfg["patch_size"])
                vvc = vt_cfg.get("vision_config", {})
                if isinstance(vvc, dict) and isinstance(vvc.get("patch_size", None), int):
                    return int(vvc["patch_size"])
            m = re.search(r"patch(\d+)", mm_vision_tower.lower())
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return 14

    @staticmethod
    def _infer_llava_processor_strategy(model_name_or_path: str) -> str:
        cfg = RealWSILoader._load_config_dict(model_name_or_path)
        v = str(cfg.get("vision_feature_select_strategy", "")).strip().lower()
        if v in {"default", "full"}:
            return v
        return "default"

    @staticmethod
    def _infer_llava_num_additional_image_tokens(model_name_or_path: str, strategy: str) -> int:
        cfg = RealWSILoader._load_config_dict(model_name_or_path)
        n = cfg.get("num_additional_image_tokens", None)
        if isinstance(n, int):
            return int(n)
        return 1 if str(strategy) == "default" else 0

    def _build_llava_processor_from_components(self, model_name: str) -> Any:
        try:
            import transformers
        except Exception as exc:
            raise RuntimeError("Failed to import transformers for LLaVA processor fallback.") from exc
        AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
        CLIPImageProcessor = getattr(transformers, "CLIPImageProcessor", None)
        LlavaProcessor = getattr(transformers, "LlavaProcessor", None)
        if AutoTokenizer is None or CLIPImageProcessor is None or LlavaProcessor is None:
            raise RuntimeError(
                "LLaVA processor fallback requires AutoTokenizer, CLIPImageProcessor, LlavaProcessor."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            eos = getattr(tokenizer, "eos_token", None)
            unk = getattr(tokenizer, "unk_token", None)
            tokenizer.pad_token = eos or unk
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)
        # Quilt/LLaVA checkpoints may miss processor metadata; ensure <image> is a
        # real tokenizer special token so LlavaProcessor can expand image tokens.
        image_token = "<image>"
        token_exists = False
        try:
            token_exists = image_token in tokenizer.get_vocab()
        except Exception:
            token_exists = False
        if not token_exists:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

        image_processor = None
        last_exc: Optional[Exception] = None
        for src in self._iter_llava_image_processor_sources(model_name):
            try:
                image_processor = CLIPImageProcessor.from_pretrained(
                    src,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self._log(f"LLaVA processor fallback image source: {src}")
                break
            except Exception as exc:
                last_exc = exc
                continue
        if image_processor is None:
            raise RuntimeError(
                "Failed to load CLIP image processor for LLaVA fallback processor."
            ) from last_exc

        strategy = self._infer_llava_processor_strategy(model_name)
        patch_size = self._infer_llava_patch_size(model_name)
        num_additional = self._infer_llava_num_additional_image_tokens(model_name, strategy)
        processor = LlavaProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            patch_size=int(patch_size),
            vision_feature_select_strategy=str(strategy),
            num_additional_image_tokens=int(num_additional),
            image_token=image_token,
        )
        self._log(
            "LLaVA processor fallback params | "
            f"patch_size={patch_size} | vision_feature_select_strategy={strategy} | "
            f"num_additional_image_tokens={num_additional}"
        )
        return processor

    def _align_model_tokenizer_vocab(self, model: Any, processor: Any) -> None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return
        if not hasattr(model, "get_input_embeddings"):
            return
        try:
            cur_vocab = int(model.get_input_embeddings().weight.shape[0])
            tok_vocab = int(len(tokenizer))
        except Exception:
            return
        if tok_vocab <= cur_vocab:
            return
        try:
            model.resize_token_embeddings(tok_vocab)
            try:
                model.config.vocab_size = int(tok_vocab)
            except Exception:
                pass
            self._log(f"Resized model token embeddings: {cur_vocab} -> {tok_vocab}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resize model token embeddings from {cur_vocab} to {tok_vocab}"
            ) from exc

    def _get_gemini_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.gemini_api_key:
            return self.gemini_api_key
        if self.api_key_env:
            key = str(os.getenv(self.api_key_env, "")).strip()
            if key:
                return key
        key = str(os.getenv(self.gemini_api_env, "")).strip()
        if key:
            return key
        if self.api_key_file:
            try:
                from ..vlm import VLMConfig, _resolve_api_key

                return _resolve_api_key(
                    VLMConfig(
                        api_provider="gemini",
                        api_key_file=self.api_key_file,
                        api_key_env=self.api_key_env or self.gemini_api_env,
                        gemini_api_env=self.gemini_api_env,
                    ),
                    "gemini",
                )
            except Exception:
                return ""
        return str(os.getenv("GEMINI_API_KEY", "")).strip()

    def _lazy_init_gemini_client(self) -> None:
        if self._gemini_client is not None and self._gemini_types is not None:
            return
        api_key = self._get_gemini_api_key()
        if not api_key:
            raise RuntimeError(
                f"Gemini API key is missing. Set {self.gemini_api_env} or pass --gemini_api_key."
            )
        try:
            from google import genai
            from google.genai import types as genai_types
        except Exception as exc:
            raise RuntimeError(
                "Gemini API runtime import failed. Install SDK: pip install -U google-genai"
            ) from exc
        self._gemini_client = genai.Client(api_key=api_key)
        self._gemini_types = genai_types
        self._log(
            f"Tile LLM planner ready | backend=gemini_api | model={self.tile_llm_model_name}"
        )

    def _run_tile_llm_json_gemini(
        self,
        tile_rgb: np.ndarray,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow is required for Gemini tile planner.")
        self._lazy_init_gemini_client()
        if self._gemini_client is None or self._gemini_types is None:
            raise RuntimeError("Gemini client failed to initialize.")

        image = Image.fromarray(self._as_uint8_rgb(tile_rgb))
        last_output = ""
        attempt_suffixes = [
            "",
            "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text.",
            "\n\nReturn one complete valid JSON object only. Ensure keys use '_' and not '\\_'.",
        ]
        for suffix in attempt_suffixes:
            used_prompt = str(prompt) + str(suffix)
            cfg = self._gemini_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                response_mime_type="application/json",
            )
            response = self._gemini_client.models.generate_content(
                model=str(self.tile_llm_model_name or "wisdomik/Quilt-Llava-v1.5-7b"),
                contents=[used_prompt, image],
                config=cfg,
            )
            last_output = self._extract_gemini_text(response)
            obj = self._extract_json_object(last_output)
            if obj:
                return obj
        raise RuntimeError(f"Gemini did not return valid JSON. Raw output: {last_output[:240]}")

    def _run_tile_llm_json_api(
        self,
        tile_rgb: np.ndarray,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow is required for API tile planner.")
        try:
            from ..vlm import VLMClient, VLMConfig
        except Exception as exc:
            raise RuntimeError("SlideBank API VLM runtime is unavailable.") from exc

        image = Image.fromarray(self._as_uint8_rgb(tile_rgb))
        last_output = ""
        attempt_suffixes = [
            "",
            "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text.",
            "\n\nReturn one complete valid JSON object only. Ensure keys use '_' and not '\\_'.",
        ]
        model_name = str(self.tile_llm_model_name or self.tile_llm_model_path)
        for suffix in attempt_suffixes:
            used_prompt = str(prompt) + str(suffix)
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                image.save(tmp_path)
                client = VLMClient(
                    VLMConfig(
                        backend=self.tile_llm_backend,
                        model_path=model_name,
                        max_new_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                        api_provider=self.api_provider,
                        api_key=self.api_key,
                        api_key_env=self.api_key_env,
                        api_key_file=self.api_key_file,
                        api_base_url=self.api_base_url,
                        api_version=self.api_version,
                        api_timeout_sec=self.api_timeout_sec,
                        gemini_api_key=self.gemini_api_key,
                        gemini_api_env=self.gemini_api_env,
                    )
                )
                last_output = client.generate(image_path=tmp_path, prompt=used_prompt)
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            obj = self._extract_json_object(last_output)
            if obj:
                return obj
        raise RuntimeError(f"API backend did not return valid JSON. Raw output: {last_output[:240]}")

    def _lazy_load_tile_llm_local(self) -> None:
        if self._tile_llm_model is not None and self._tile_llm_processor is not None:
            return
        if not (self.use_llm_tile_planner or self.use_llm_patch_describer):
            raise RuntimeError("LLM runtime is disabled for both tile planning and patch description.")
        try:
            import torch
            import transformers
        except Exception as exc:
            raise RuntimeError(
                "LLM runtime import failed. "
                f"torch/transformers unavailable or broken: {type(exc).__name__}: {exc}"
            ) from exc
        AutoProcessor = getattr(transformers, "AutoProcessor", None)
        if AutoProcessor is None:
            raise RuntimeError(
                "LLM tile planner requires transformers.AutoProcessor, but current transformers lacks it. "
                f"version={getattr(transformers, '__version__', 'unknown')}"
            )

        model_name = self.tile_llm_model_path or self.tile_llm_model_name
        model_name = self._resolve_model_path(model_name)
        if self._is_llava_checkpoint(model_name):
            self._lazy_load_tile_llm_quilt_repo(model_name)
            return

        torch_dtype = None
        if self.tile_llm_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.tile_llm_dtype == "fp16":
            torch_dtype = torch.float16
        elif self.tile_llm_dtype == "fp32":
            torch_dtype = torch.float32

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.tile_llm_device_map == "auto":
            model_kwargs["device_map"] = "auto"

        model_cls_candidates = []
        for cls_name in [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]:
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                model_cls_candidates.append((cls_name, cls))
        if not model_cls_candidates:
            raise RuntimeError(
                "No supported AutoModel class found in transformers for tile LLM planner."
            )

        model = None
        last_exc: Optional[Exception] = None
        for _, cls in model_cls_candidates:
            try:
                model = cls.from_pretrained(model_name, **model_kwargs)
                break
            except Exception as exc:
                last_exc = exc
                continue
        if model is None:
            raise RuntimeError(
                f"Failed to load tile LLM model from {model_name} with available classes."
            ) from last_exc
        if self.tile_llm_device_map != "auto":
            model = model.to(self.tile_llm_device_map)
        try:
            model.config.use_cache = True
        except Exception:
            pass
        try:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True
            )
        except Exception as proc_exc:
            processor = self._build_llava_processor_from_components(model_name)
            self._log(
                "AutoProcessor load failed; using LLaVA processor fallback "
                f"(tokenizer+CLIPImageProcessor). reason={type(proc_exc).__name__}: {proc_exc}"
            )
        self._align_model_tokenizer_vocab(model, processor)
        self._tile_llm_model = model
        self._tile_llm_processor = processor
        self._tile_llm_runtime = "hf_auto"
        self._log(f"Tile LLM planner ready | backend=local_hf | model={model_name} | device={self.tile_llm_device_map}")

    def _lazy_load_tile_llm(self) -> None:
        if self.tile_llm_backend in {"gemini", "gemini_api"}:
            self._lazy_init_gemini_client()
            return
        self._lazy_load_tile_llm_local()

    def _prepare_vlm_inputs(self, image: Any, prompt: str) -> Any:
        return self._prepare_vlm_inputs_batch([image], [prompt])

    def _prepare_vlm_inputs_batch(self, images: List[Any], prompts: List[str]) -> Any:
        processor = self._tile_llm_processor
        assert processor is not None
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length for VLM batch inputs.")
        texts: List[str] = []
        chat_templater = processor
        if not hasattr(chat_templater, "apply_chat_template"):
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and hasattr(tok, "apply_chat_template"):
                chat_templater = tok
        template_available = False
        if hasattr(chat_templater, "apply_chat_template"):
            tmpl = getattr(chat_templater, "chat_template", None)
            if isinstance(tmpl, str) and tmpl.strip():
                template_available = True
            elif isinstance(tmpl, dict) and len(tmpl) > 0:
                template_available = True
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = ""
            if template_available and hasattr(chat_templater, "apply_chat_template"):
                try:
                    text = chat_templater.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    text = ""
            if not str(text).strip():
                text = f"<image>\n{prompt}"
            texts.append(text)

        last_exc: Optional[Exception] = None
        image_variants = [images, [[img] for img in images]]
        text_variants = [texts, prompts]
        for txt in text_variants:
            for img in image_variants:
                try:
                    return processor(text=txt, images=img, padding=True, return_tensors="pt")
                except Exception as exc:
                    last_exc = exc
                    continue
        raise RuntimeError(
            "Failed to prepare VLM batch inputs with available image/text layouts."
        ) from last_exc

    @staticmethod
    def _move_inputs_to_model_device(inputs: Dict[str, Any], model: Any, device: str) -> Dict[str, Any]:
        if device == "auto":
            try:
                device = next(model.parameters()).device
            except StopIteration:
                return inputs
        if device is None:
            return inputs

        def _move(x: Any) -> Any:
            if hasattr(x, "to"):
                return x.to(device)
            if isinstance(x, (list, tuple)):
                return type(x)(_move(v) for v in x)
            return x

        return {k: _move(v) for k, v in inputs.items()}

    def _run_tile_llm_json_quilt_repo(
        self,
        tile_rgb: np.ndarray,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow is required for tile LLM planner.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Tile LLM planner requires torch.") from exc

        model = self._tile_llm_model
        cfg = self._tile_llm_processor
        if model is None or not isinstance(cfg, dict):
            raise RuntimeError("Quilt-LLaVA runtime not initialized.")
        tokenizer = cfg.get("tokenizer", None)
        image_processor = cfg.get("image_processor", None)
        process_images = cfg.get("process_images", None)
        tokenizer_image_token = cfg.get("tokenizer_image_token", None)
        conv_templates = cfg.get("conv_templates", None)
        if (
            tokenizer is None
            or image_processor is None
            or process_images is None
            or tokenizer_image_token is None
            or conv_templates is None
        ):
            raise RuntimeError("Quilt-LLaVA runtime dict missing required components.")

        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device(self._get_quilt_runtime_device())

        image = Image.fromarray(self._as_uint8_rgb(tile_rgb))
        last_output = ""
        model_name = str(cfg.get("model_name", "")).lower()
        if "llama-2" in model_name:
            conv_mode = "llava_llama_2"
        elif "v1" in model_name:
            conv_mode = "llava_v1"
        elif "mpt" in model_name:
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        if conv_mode not in conv_templates:
            conv_mode = "llava_v0"

        for attempt in range(2):
            used_prompt = prompt
            if attempt == 1:
                used_prompt = (
                    prompt
                    + "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text."
                )
            if bool(getattr(model.config, "mm_use_im_start_end", False)):
                qs = (
                    str(cfg["DEFAULT_IM_START_TOKEN"])
                    + str(cfg["DEFAULT_IMAGE_TOKEN"])
                    + str(cfg["DEFAULT_IM_END_TOKEN"])
                    + "\n"
                    + used_prompt
                )
            else:
                qs = str(cfg["DEFAULT_IMAGE_TOKEN"]) + "\n" + used_prompt

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    full_prompt,
                    tokenizer,
                    int(cfg["IMAGE_TOKEN_INDEX"]),
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(model_device)
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
            image_tensor = None
            try:
                image_tensor = process_images([image], image_processor, model.config)
            except Exception:
                image_tensor = None
            if image_tensor is None:
                pixel_values = image_processor(images=image, return_tensors="pt").get("pixel_values", None)
                if pixel_values is None:
                    raise RuntimeError("Quilt image_processor returned no pixel_values.")
                image_tensor = pixel_values
            if isinstance(image_tensor, list):
                image_tensor = [x.to(model_device, dtype=torch.float16) for x in image_tensor]
            else:
                image_tensor = image_tensor.to(model_device, dtype=torch.float16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                    use_cache=True,
                )
            output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            last_output = str(output).strip()
            obj = self._extract_json_object(last_output)
            if not obj:
                obj = self._recover_obj_from_raw_text(last_output, used_prompt)
            if obj:
                return obj
        raise RuntimeError(f"Tile LLM did not return valid JSON. Raw output: {last_output}")

    def _run_tile_llm_json_batch_quilt_repo(
        self,
        tile_rgbs: List[np.ndarray],
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if Image is None:
            raise RuntimeError("Pillow is required for tile LLM planner.")
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Tile LLM planner requires torch.") from exc

        model = self._tile_llm_model
        cfg = self._tile_llm_processor
        if model is None or not isinstance(cfg, dict):
            raise RuntimeError("Quilt-LLaVA runtime not initialized.")
        tokenizer = cfg.get("tokenizer", None)
        image_processor = cfg.get("image_processor", None)
        process_images = cfg.get("process_images", None)
        tokenizer_image_token = cfg.get("tokenizer_image_token", None)
        conv_templates = cfg.get("conv_templates", None)
        if (
            tokenizer is None
            or image_processor is None
            or process_images is None
            or tokenizer_image_token is None
            or conv_templates is None
        ):
            raise RuntimeError("Quilt-LLaVA runtime dict missing required components.")

        try:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
        except Exception:
            model_device = torch.device(self._get_quilt_runtime_device())
            model_dtype = torch.float16
        if not isinstance(model_dtype, torch.dtype) or not torch.is_floating_point(
            torch.zeros((), dtype=model_dtype)
        ):
            model_dtype = torch.float16

        model_name = str(cfg.get("model_name", "")).lower()
        if "llama-2" in model_name:
            conv_mode = "llava_llama_2"
        elif "v1" in model_name:
            conv_mode = "llava_v1"
        elif "mpt" in model_name:
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        if conv_mode not in conv_templates:
            conv_mode = "llava_v0"

        outputs_obj: List[Dict[str, Any]] = [{} for _ in range(len(tile_rgbs))]
        last_outputs: List[str] = ["" for _ in range(len(tile_rgbs))]
        pending: List[int] = list(range(len(tile_rgbs)))

        attempt_suffixes = [
            "",
            "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text.",
            "\n\nReturn one complete valid JSON object only. Ensure keys use '_' and not '\\_'.",
        ]
        for attempt, suffix in enumerate(attempt_suffixes):
            if not pending:
                break
            batch_indices = list(pending)
            batch_images = [
                Image.fromarray(self._as_uint8_rgb(tile_rgbs[idx]))
                for idx in batch_indices
            ]
            batch_prompt_ids: List[Any] = []
            for idx in batch_indices:
                used_prompt = str(prompts[idx]) + suffix
                if bool(getattr(model.config, "mm_use_im_start_end", False)):
                    qs = (
                        str(cfg["DEFAULT_IM_START_TOKEN"])
                        + str(cfg["DEFAULT_IMAGE_TOKEN"])
                        + str(cfg["DEFAULT_IM_END_TOKEN"])
                        + "\n"
                        + used_prompt
                    )
                else:
                    qs = str(cfg["DEFAULT_IMAGE_TOKEN"]) + "\n" + used_prompt
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                full_prompt = conv.get_prompt()
                one_ids = tokenizer_image_token(
                    full_prompt,
                    tokenizer,
                    int(cfg["IMAGE_TOKEN_INDEX"]),
                    return_tensors="pt",
                )
                if hasattr(one_ids, "view"):
                    one_ids = one_ids.view(-1)
                batch_prompt_ids.append(one_ids)

            if not batch_prompt_ids:
                break
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None:
                pad_id = 0
            max_len = max(int(x.shape[0]) for x in batch_prompt_ids)
            padded_rows: List[torch.Tensor] = []
            mask_rows: List[torch.Tensor] = []
            for ids in batch_prompt_ids:
                ids = ids.to(device=model_device, dtype=torch.long)
                cur_len = int(ids.shape[0])
                pad_len = int(max_len - cur_len)
                if pad_len > 0:
                    pad_tokens = torch.full(
                        (pad_len,),
                        int(pad_id),
                        dtype=torch.long,
                        device=model_device,
                    )
                    row = torch.cat([pad_tokens, ids], dim=0)
                    row_mask = torch.cat(
                        [
                            torch.zeros((pad_len,), dtype=torch.long, device=model_device),
                            torch.ones((cur_len,), dtype=torch.long, device=model_device),
                        ],
                        dim=0,
                    )
                else:
                    row = ids
                    row_mask = torch.ones((cur_len,), dtype=torch.long, device=model_device)
                padded_rows.append(row)
                mask_rows.append(row_mask)
            input_ids = torch.stack(padded_rows, dim=0)
            attention_mask = torch.stack(mask_rows, dim=0)

            image_tensor = None
            try:
                image_tensor = process_images(batch_images, image_processor, model.config)
            except Exception:
                image_tensor = None
            if image_tensor is None:
                pixel_values = image_processor(images=batch_images, return_tensors="pt").get("pixel_values", None)
                if pixel_values is None:
                    raise RuntimeError("Quilt image_processor returned no pixel_values for batch.")
                image_tensor = pixel_values
            if isinstance(image_tensor, list):
                stacked: List[torch.Tensor] = []
                for t in image_tensor:
                    if not isinstance(t, torch.Tensor):
                        raise RuntimeError("Quilt process_images returned non-tensor item in batch list.")
                    if t.ndim == 3:
                        t = t.unsqueeze(0)
                    stacked.append(t)
                try:
                    image_tensor = torch.cat(stacked, dim=0)
                except Exception:
                    image_tensor = [x.to(model_device, dtype=model_dtype) for x in stacked]
            if isinstance(image_tensor, list):
                image_tensor = [x.to(model_device, dtype=model_dtype) for x in image_tensor]
            else:
                image_tensor = image_tensor.to(model_device, dtype=model_dtype)

            try:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                        use_cache=True,
                    )
            except Exception as exc:
                input_debug = {
                    "input_ids": {
                        "shape": tuple(input_ids.shape),
                        "dtype": str(input_ids.dtype),
                        "device": str(input_ids.device),
                    },
                    "attention_mask": {
                        "shape": tuple(attention_mask.shape),
                        "dtype": str(attention_mask.dtype),
                        "device": str(attention_mask.device),
                    },
                }
                if isinstance(image_tensor, list):
                    input_debug["images"] = [
                        {"shape": tuple(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
                        for x in image_tensor
                    ]
                else:
                    input_debug["images"] = {
                        "shape": tuple(image_tensor.shape),
                        "dtype": str(image_tensor.dtype),
                        "device": str(image_tensor.device),
                    }
                self._log(
                    "Tile LLM quilt_repo batch generate failed; retrying pending items one-by-one | "
                    f"attempt={attempt + 1}/{len(attempt_suffixes)} | pending={len(batch_indices)} | "
                    f"input_debug={input_debug} | {type(exc).__name__}: {exc}"
                )
                next_pending: List[int] = []
                single_suffix = (
                    "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text."
                )
                for idx in batch_indices:
                    try:
                        obj = self._run_tile_llm_json_quilt_repo(
                            tile_rgb=tile_rgbs[idx],
                            prompt=str(prompts[idx]) + single_suffix,
                            max_new_tokens=max_new_tokens,
                        )
                        if obj:
                            outputs_obj[idx] = obj
                            continue
                    except Exception as single_exc:
                        err_text = f"{type(single_exc).__name__}: {single_exc}"
                        last_outputs[idx] = err_text
                        raw = self._extract_raw_output_from_error_text(err_text)
                        rec = self._recover_obj_from_raw_text(raw, prompts[idx]) if raw else {}
                        if rec:
                            outputs_obj[idx] = rec
                            continue
                    next_pending.append(idx)
                pending = next_pending
                continue

            prompt_len = int(input_ids.shape[1])
            next_pending: List[int] = []
            for local_idx, out_ids in enumerate(generated_ids):
                idx = batch_indices[local_idx]
                gen_ids = out_ids[prompt_len:]
                txt = str(tokenizer.decode(gen_ids, skip_special_tokens=True)).strip()
                last_outputs[idx] = txt
                obj = self._extract_json_object(txt)
                if not obj:
                    obj = self._recover_obj_from_raw_text(txt, prompts[idx])
                if obj:
                    outputs_obj[idx] = obj
                else:
                    next_pending.append(idx)
            pending = next_pending

        if pending:
            # Final strict single-item retry for residual non-JSON outputs.
            retry_suffix = (
                "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text."
            )
            next_pending: List[int] = []
            for idx in pending:
                try:
                    obj = self._run_tile_llm_json_quilt_repo(
                        tile_rgb=tile_rgbs[idx],
                        prompt=str(prompts[idx]) + retry_suffix,
                        max_new_tokens=max_new_tokens,
                    )
                    if obj:
                        outputs_obj[idx] = obj
                        continue
                except Exception as exc:
                    err_text = f"{type(exc).__name__}: {exc}"
                    last_outputs[idx] = err_text
                    raw = self._extract_raw_output_from_error_text(err_text)
                    rec = self._recover_obj_from_raw_text(raw, prompts[idx]) if raw else {}
                    if rec:
                        outputs_obj[idx] = rec
                        continue
                next_pending.append(idx)
            pending = next_pending

        if pending:
            # Best-effort raw-text recovery before hard failure.
            recovered_pending: List[int] = []
            for idx in pending:
                raw = self._extract_raw_output_from_error_text(last_outputs[idx])
                rec = self._recover_obj_from_raw_text(raw, prompts[idx]) if raw else {}
                if not rec:
                    rec = self._recover_obj_from_raw_text(last_outputs[idx], prompts[idx])
                if rec:
                    outputs_obj[idx] = rec
                else:
                    recovered_pending.append(idx)
            pending = recovered_pending

        if pending:
            first_idx = pending[0]
            raise RuntimeError(
                "Tile LLM quilt_repo batch returned non-JSON outputs | "
                f"failed={len(pending)}/{len(tile_rgbs)} | "
                f"first_raw={last_outputs[first_idx][:240]}"
            )
        return outputs_obj

    def _run_tile_llm_json(
        self,
        tile_rgb: np.ndarray,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.tile_llm_backend in {"gemini", "gemini_api"}:
            return self._run_tile_llm_json_gemini(
                tile_rgb=tile_rgb,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        if self.tile_llm_backend in {"api", "openai_api", "openai_compatible_api"}:
            return self._run_tile_llm_json_api(
                tile_rgb=tile_rgb,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        if Image is None:
            raise RuntimeError("Pillow is required for tile LLM planner.")
        self._lazy_load_tile_llm()
        model = self._tile_llm_model
        processor = self._tile_llm_processor
        if model is None or processor is None:
            raise RuntimeError("Tile LLM planner failed to initialize.")
        if isinstance(processor, dict) and processor.get("runtime") == "quilt_repo":
            return self._run_tile_llm_json_quilt_repo(
                tile_rgb=tile_rgb,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Tile LLM planner requires torch.") from exc

        image = Image.fromarray(self._as_uint8_rgb(tile_rgb))
        last_output = ""
        attempt_suffixes = [
            "",
            "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text.",
            "\n\nReturn one complete valid JSON object only. Ensure keys use '_' and not '\\_'.",
        ]
        for suffix in attempt_suffixes:
            used_prompt = str(prompt) + str(suffix)
            inputs = self._prepare_vlm_inputs(image, used_prompt)
            inputs = self._move_inputs_to_model_device(inputs, model, self.tile_llm_device_map)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            gen = generated_ids
            if "input_ids" in inputs:
                prompt_len = inputs["input_ids"].shape[1]
                gen = generated_ids[:, prompt_len:]
            output = processor.batch_decode(gen, skip_special_tokens=True)[0]
            last_output = str(output).strip()
            obj = self._extract_json_object(last_output)
            if not obj:
                obj = self._recover_obj_from_raw_text(last_output, used_prompt)
            if obj:
                return obj
        raise RuntimeError(f"Tile LLM did not return valid JSON. Raw output: {last_output}")

    def _run_tile_llm_json_batch(
        self,
        tile_rgbs: List[np.ndarray],
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if len(tile_rgbs) != len(prompts):
            raise ValueError("tile_rgbs and prompts must have the same length for batch JSON generation.")
        if not tile_rgbs:
            return []
        if self.tile_llm_backend in {"gemini", "gemini_api"}:
            outputs: List[Dict[str, Any]] = []
            for idx, (tile_rgb, prompt) in enumerate(zip(tile_rgbs, prompts)):
                try:
                    outputs.append(
                        self._run_tile_llm_json_gemini(
                            tile_rgb=tile_rgb,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                except Exception as exc:
                    self._log(
                        f"Gemini batch item failed | idx={idx} | {type(exc).__name__}: {exc}"
                    )
                    outputs.append({})
            return outputs
        if self.tile_llm_backend in {"api", "openai_api", "openai_compatible_api"}:
            outputs = []
            for idx, (tile_rgb, prompt) in enumerate(zip(tile_rgbs, prompts)):
                try:
                    outputs.append(
                        self._run_tile_llm_json_api(
                            tile_rgb=tile_rgb,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                except Exception as exc:
                    self._log(
                        f"API batch item failed | idx={idx} | {type(exc).__name__}: {exc}"
                    )
                    outputs.append({})
            return outputs
        if Image is None:
            raise RuntimeError("Pillow is required for tile LLM planner.")

        self._lazy_load_tile_llm()
        model = self._tile_llm_model
        processor = self._tile_llm_processor
        if model is None or processor is None:
            raise RuntimeError("Tile LLM planner failed to initialize.")
        if isinstance(processor, dict) and processor.get("runtime") == "quilt_repo":
            return self._run_tile_llm_json_batch_quilt_repo(
                tile_rgbs=tile_rgbs,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Tile LLM planner requires torch.") from exc

        images = [Image.fromarray(self._as_uint8_rgb(img)) for img in tile_rgbs]
        outputs_obj: List[Dict[str, Any]] = [{} for _ in range(len(images))]
        last_outputs: List[str] = ["" for _ in range(len(images))]
        pending: List[int] = list(range(len(images)))

        for attempt in range(2):
            if not pending:
                break
            prompt_suffix = ""
            if attempt == 1:
                prompt_suffix = "\n\nReturn only strict JSON object. No markdown, no explanation, no trailing text."
            batch_indices = list(pending)
            batch_images = [images[i] for i in batch_indices]
            batch_prompts = [prompts[i] + prompt_suffix for i in batch_indices]
            try:
                inputs = self._prepare_vlm_inputs_batch(batch_images, batch_prompts)
                inputs = self._move_inputs_to_model_device(inputs, model, self.tile_llm_device_map)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=int(max_new_tokens or self.tile_llm_max_new_tokens),
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                    )
            except Exception as exc:
                # Some processors/models are fragile for batched image-text alignment.
                # Fallback to single-sample generation keeps correctness while preserving batch fast-path.
                self._log(
                    f"Tile LLM batch generate fallback->single | pending={len(batch_indices)} | "
                    f"{type(exc).__name__}: {exc}"
                )
                next_pending: List[int] = []
                for idx in batch_indices:
                    try:
                        outputs_obj[idx] = self._run_tile_llm_json(
                            tile_rgb=tile_rgbs[idx],
                            prompt=prompts[idx],
                            max_new_tokens=max_new_tokens,
                        )
                    except Exception as single_exc:
                        last_outputs[idx] = str(single_exc)
                        next_pending.append(idx)
                pending = next_pending
                continue

            gen_rows: Any = generated_ids
            if "input_ids" in inputs:
                if "attention_mask" in inputs:
                    try:
                        prompt_lens = [int(v) for v in inputs["attention_mask"].sum(dim=1).tolist()]
                    except Exception:
                        prompt_lens = [int(inputs["input_ids"].shape[1])] * int(generated_ids.shape[0])
                else:
                    prompt_lens = [int(inputs["input_ids"].shape[1])] * int(generated_ids.shape[0])
                gen_rows = [generated_ids[i, prompt_lens[i]:] for i in range(int(generated_ids.shape[0]))]

            decoded = processor.batch_decode(gen_rows, skip_special_tokens=True)
            next_pending: List[int] = []
            for local_idx, out in enumerate(decoded):
                idx = batch_indices[local_idx]
                txt = str(out).strip()
                last_outputs[idx] = txt
                obj = self._extract_json_object(txt)
                if not obj:
                    obj = self._recover_obj_from_raw_text(txt, prompts[idx])
                if obj:
                    outputs_obj[idx] = obj
                else:
                    next_pending.append(idx)
            pending = next_pending

        if pending:
            for idx in pending:
                self._log(f"Tile LLM batch JSON parse failed | idx={idx} | raw={last_outputs[idx][:240]}")
        return outputs_obj

    @staticmethod
    def _patch_request_key(tile_id: str, anchor_id: str, view_id: str) -> str:
        return f"{str(tile_id)}::{str(anchor_id)}::{str(view_id)}"

    @staticmethod
    def _patch_request_text(
        category: str,
        magnification_hint: str,
        tile_id: str,
        anchor_id: str,
        view_id: str,
    ) -> str:
        return (
            f"tile={str(tile_id or '-')}"
            f" anchor={str(anchor_id or '-')}"
            f" view={str(view_id or '-')}"
            f" category={str(category)}"
            f" mag={str(magnification_hint)}"
        )

    def _prepare_patch_for_llm(self, crop: np.ndarray) -> np.ndarray:
        if Image is None:
            raise RuntimeError("Pillow is required for LLM patch description.")
        patch = self._as_uint8_rgb(crop)
        h, w = patch.shape[:2]
        max_side = max(h, w)
        if max_side <= 768:
            return patch
        scale = 768.0 / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if cv2 is not None:
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            patch = np.array(Image.fromarray(patch).resize((new_w, new_h), Image.BILINEAR))
        return self._as_uint8_rgb(patch)

    @staticmethod
    def _build_patch_describe_prompt(
        category: str,
        magnification_hint: str,
        tile_id: str,
        anchor_id: str,
        view_id: str,
    ) -> str:
        return (
f"You are analyzing ONE {str(magnification_hint or 'unknown')} H&E patch.\n"
"Return strict JSON: caption, confidence, qc_flags.\n"
"\n"
"caption:\n"
"- 2-3 sentences of concise pathologist-style morphology description (≤100 words).\n"
"- Start with concrete visible features: color, cellularity, nuclear features, architecture, stroma, necrosis, calcifications, edge.\n"
"- Use only observable morphology (e.g., enlarged irregular nuclei, solid nests, duct-like lumen, central pale debris, bright puncta, tumor at edge).\n"
"- No diagnostic summary. No grading terms. No signal names.\n"
"- Any high-level inference must be explicitly tied to a stated visible feature.\n"
"- If minimal tissue or no clear morphology → brief description + qc_flag='low_tissue'.\n"
"\n"
"confidence:\n"
"- ≥0.8 if multiple clear concrete features.\n"
"- 0.4–0.7 if limited but interpretable morphology.\n"
"- ≤0.3 if sparse, blurred, or uncertain.\n"
"\n"
"qc_flags: [none, blur, out_of_focus, artifact, low_tissue].\n"
"\n"
f"tile_id={tile_id}; anchor_id={anchor_id}; view_id={view_id}."
        )

    @staticmethod
    def _parse_patch_desc_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
        caption = str(obj.get("caption", "")).strip()
        if not caption:
            caption = str(obj.get("description", "")).strip()
        if not caption:
            raise RuntimeError("Patch describer LLM returned JSON without `caption`.")

        try:
            confidence = float(obj.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.01, min(0.99, confidence))

        qc_raw = obj.get("qc_flags", [])
        qc_flags: List[str] = []
        if isinstance(qc_raw, list):
            qc_flags = [str(x).strip() for x in qc_raw if str(x).strip()]
        elif isinstance(qc_raw, str) and qc_raw.strip():
            qc_flags = [qc_raw.strip()]
        if not qc_flags:
            qc_flags = ["none"]
        return {
            "caption": caption,
            "confidence": round(confidence, 4),
            "qc_flags": qc_flags,
        }

    @staticmethod
    def _build_anchor_description(
        coarse_label: str,
        evidence_summary: str,
        observations: List[Dict[str, Any]],
    ) -> str:
        base = re.sub(r"\s+", " ", str(evidence_summary or "")).strip()
        if not base:
            base = f"{str(coarse_label or 'unknown')} pattern."

        def _mag_value(mag: Any) -> float:
            m = re.search(r"[-+]?\d*\.?\d+", str(mag or ""))
            if not m:
                return 0.0
            try:
                return float(m.group(0))
            except Exception:
                return 0.0

        seen_caps = set()
        cues: List[str] = []
        obs_items = [o for o in observations if isinstance(o, dict)]
        low_mag_items: List[Dict[str, Any]] = []
        other_items: List[Dict[str, Any]] = []
        for o in obs_items:
            mag_v = _mag_value(o.get("mag", ""))
            if mag_v > 0 and abs(mag_v - 10.0) <= 2.5:
                low_mag_items.append(o)
            else:
                other_items.append(o)
        low_mag_items.sort(key=lambda o: abs(_mag_value(o.get("mag", "")) - 10.0))
        ordered_obs: List[Dict[str, Any]] = []
        if low_mag_items:
            ordered_obs.append(low_mag_items[0])
        ordered_obs.extend(other_items)
        if len(low_mag_items) > 1:
            ordered_obs.extend(low_mag_items[1:])

        for obs in ordered_obs:
            if not isinstance(obs, dict):
                continue
            view_id = str(obs.get("view_id", "")).strip()
            mag = str(obs.get("mag", "")).strip()
            cap = re.sub(r"\s+", " ", str(obs.get("obs", "")).strip())
            cap = cap.strip().rstrip(".")
            if not cap:
                continue
            cap_key = cap.lower()
            if cap_key in seen_caps:
                continue
            seen_caps.add(cap_key)
            if view_id:
                if mag:
                    cues.append(f"{view_id}({mag}) {cap}.")
                else:
                    cues.append(f"{view_id} {cap}.")
            else:
                cues.append(f"{cap}.")
            if len(cues) >= 2:
                break

        if cues:
            desc = f"{base} Cues: {' '.join(cues)}".strip()
        else:
            desc = base
        return desc

    def _describe_patches_with_llm_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        if not requests:
            return results

        batch_size = max(1, int(self.patch_llm_batch_size))
        max_new_tokens = max(self.patch_llm_max_new_tokens, self.tile_llm_max_new_tokens)
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            batch_id = start // batch_size + 1
            self._log(
                f"LLM patch batch start | batch={batch_id} | size={len(chunk)} | batch_size={self.patch_llm_batch_size}"
            )

            rgbs: List[np.ndarray] = []
            prompts: List[str] = []
            req_keys: List[str] = []
            req_texts: List[str] = []
            for req in chunk:
                req_key = str(req.get("req_key", ""))
                tile_id = str(req.get("tile_id", ""))
                anchor_id = str(req.get("anchor_id", ""))
                view_id = str(req.get("view_id", ""))
                category = str(req.get("category", "other"))
                magnification_hint = str(req.get("magnification_hint", "unknown"))
                req_text = self._patch_request_text(
                    category=category,
                    magnification_hint=magnification_hint,
                    tile_id=tile_id,
                    anchor_id=anchor_id,
                    view_id=view_id,
                )
                self._log(f"LLM patch describe start | {req_text}")
                req_keys.append(req_key)
                req_texts.append(req_text)
                rgbs.append(self._prepare_patch_for_llm(req["crop"]))
                prompts.append(
                    self._build_patch_describe_prompt(
                        category=category,
                        magnification_hint=magnification_hint,
                        tile_id=tile_id,
                        anchor_id=anchor_id,
                        view_id=view_id,
                    )
                )

            batch_error = ""
            try:
                obj_list = self._run_tile_llm_json_batch(
                    tile_rgbs=rgbs,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"
                self._log(f"LLM patch batch failed | batch={batch_id} | {batch_error}")
                obj_list = [{} for _ in chunk]

            for i, req_key in enumerate(req_keys):
                req_text = req_texts[i]
                obj = obj_list[i] if i < len(obj_list) else {}
                if batch_error and not obj:
                    self._log(f"LLM patch describe failed | {req_text} | {batch_error}")
                    results[req_key] = {"error": batch_error}
                    continue
                try:
                    parsed = self._parse_patch_desc_obj(obj)
                    results[req_key] = parsed
                    self._log(
                        "LLM patch describe done | "
                        f"{req_text} | conf={parsed['confidence']:.4f} | "
                        f"qc={','.join(parsed['qc_flags'])} | caption={parsed['caption'][:180]}"
                    )
                except Exception as exc:
                    self._log(f"LLM patch describe failed | {req_text} | {type(exc).__name__}: {exc}")
                    results[req_key] = {"error": f"{type(exc).__name__}: {exc}"}
        return results

    def _describe_patch_with_llm(
        self,
        crop: np.ndarray,
        category: str,
        magnification_hint: str,
        tile_id: str = "",
        anchor_id: str = "",
        view_id: str = "",
    ) -> Dict[str, Any]:
        if Image is None:
            raise RuntimeError("Pillow is required for LLM patch description.")
        req_id = self._patch_request_text(
            category=category,
            magnification_hint=magnification_hint,
            tile_id=tile_id,
            anchor_id=anchor_id,
            view_id=view_id,
        )
        self._log(f"LLM patch describe start | {req_id}")
        patch = self._prepare_patch_for_llm(crop)
        prompt = self._build_patch_describe_prompt(
            category=category,
            magnification_hint=magnification_hint,
            tile_id=tile_id,
            anchor_id=anchor_id,
            view_id=view_id,
        )
        try:
            obj = self._run_tile_llm_json(
                tile_rgb=patch,
                prompt=prompt,
                max_new_tokens=max(self.patch_llm_max_new_tokens, self.tile_llm_max_new_tokens),
            )
        except Exception as exc:
            self._log(f"LLM patch describe failed | {req_id} | {type(exc).__name__}: {exc}")
            raise
        parsed = self._parse_patch_desc_obj(obj)
        self._log(
            "LLM patch describe done | "
            f"{req_id} | conf={parsed['confidence']:.4f} | "
            f"qc={','.join(parsed['qc_flags'])} | caption={parsed['caption'][:180]}"
        )
        return parsed

    @staticmethod
    def _normalize_tile_label(raw_label: Any) -> str:
        label = str(raw_label or "").strip().lower()
        aliases = {
            "tumour": "tumor",
            "carcinoma": "tumor",
            "malignant": "tumor",
            "cancer": "tumor",
            "necrotic": "necrosis",
            "fat": "adipose",
            "stromal": "stroma",
            "fibrous": "stroma",
            "artifact": "background",
            "bg": "background",
        }
        if label in aliases:
            return aliases[label]
        allowed = {"tumor", "necrosis", "adipose", "stroma", "background", "other"}
        return label if label in allowed else "other"

    @staticmethod
    def _normalize_prob_dict(scores: Dict[str, float]) -> Dict[str, float]:
        labels = ["tumor", "necrosis", "adipose", "stroma", "other"]
        raw = {k: max(1e-6, float(scores.get(k, 0.0))) for k in labels}
        total = sum(raw.values())
        if total <= 0:
            raw = {"tumor": 0.2, "necrosis": 0.2, "adipose": 0.2, "stroma": 0.3, "other": 0.1}
            total = sum(raw.values())
        probs = {k: raw[k] / total for k in labels}
        rounded = {k: round(probs[k], 4) for k in labels}
        diff = round(1.0 - sum(rounded.values()), 4)
        max_key = max(rounded.items(), key=lambda kv: kv[1])[0]
        rounded[max_key] = round(max(0.0, min(1.0, rounded[max_key] + diff)), 4)
        return rounded

    def _classify_tile_rule_based(
        self,
        tile_rgb: np.ndarray,
        tissue_ratio: float,
    ) -> Dict[str, Any]:
        feat = self._compute_features(tile_rgb)
        white = float(feat.get("white_ratio", 0.0))
        purple = float(feat.get("purple_ratio", 0.0))
        pink = float(feat.get("pink_ratio", 0.0))
        dark = float(feat.get("dark_ratio", 0.0))
        texture = float(feat.get("texture", 0.0))
        nuclei_ratio = float(feat.get("nuclei_ratio", 0.0))
        eosinophilic = float(feat.get("eosinophilic_ratio", 0.0))
        sat_norm = float(feat.get("sat_mean", 0.0)) / 255.0
        tissue = float(max(0.0, min(1.0, tissue_ratio)))

        score_map = {
            "tumor": (
                0.12
                + 1.10 * purple
                + 0.22 * dark
                + 1.80 * nuclei_ratio
                + 0.00025 * texture
                + 0.10 * tissue
                - 0.30 * white
                - 0.18 * pink
            ),
            "necrosis": (
                0.10
                + 1.25 * pink
                + 0.55 * eosinophilic
                + 0.20 * dark
                + 0.14 * tissue
                - 0.22 * white
                - 0.45 * purple
                - 0.00008 * max(0.0, texture - 500.0)
            ),
            "adipose": (
                0.08
                + 1.45 * white
                + 0.15 * max(0.0, 0.45 - tissue)
                + 0.10 * max(0.0, 0.10 - nuclei_ratio)
                - 0.55 * purple
                - 0.25 * dark
                - 0.10 * pink
            ),
            "stroma": (
                0.20
                + 0.55 * tissue
                + 0.25 * (1.0 - white)
                + 0.10 * sat_norm
                - 0.18 * pink
                - 0.12 * purple
                + 0.00006 * texture
            ),
            "other": (
                0.06
                + 0.04 * (1.0 - tissue)
            ),
        }
        conf_map = self._normalize_prob_dict(score_map)
        label = max(conf_map.items(), key=lambda kv: kv[1])[0]
        desc_map = {
            "tumor": "Cell-dense atypical epithelial region suspicious for carcinoma.",
            "necrosis": "Eosinophilic low-structure region suggestive of necrotic debris.",
            "adipose": "Predominantly adipose-rich region with sparse nuclei.",
            "stroma": "Fibro-stromal region without overt malignant pattern.",
            "other": "Mixed tissue pattern without a dominant morphologic class.",
        }
        return {
            "coarse_label": label,
            "label_score": round(float(conf_map.get(label, 0.0)), 4),
            "label_confidence": conf_map,
            "tile_desc": desc_map.get(label, desc_map["other"]),
            "qc_flags": ["none"],
        }

    def _classify_tile_with_llm(
        self,
        tile_rgb: np.ndarray,
        tile_id: str,
        tissue_ratio: float,
    ) -> Dict[str, Any]:
        prompt = (
            "You are a pathology assistant. Classify this tile into exactly one label from: "
            "tumor, necrosis, adipose, stroma, background, other.\n"
            "Return strict JSON with keys: label, confidence, rationale.\n"
            'Example: {"label":"tumor","confidence":0.87,"rationale":"short reason"}.\n'
            f"tile_id={tile_id}; tissue_ratio_estimate={round(float(tissue_ratio), 4)}."
        )
        obj = self._run_tile_llm_json(tile_rgb=tile_rgb, prompt=prompt)
        label = self._normalize_tile_label(obj.get("label", "other"))
        try:
            score = float(obj.get("confidence", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        rationale = str(obj.get("rationale", "")).strip()
        if not rationale:
            rationale = f"LLM labeled tile as {label}."
        conf_map = {
            "tumor": 0.0,
            "necrosis": 0.0,
            "adipose": 0.0,
            "stroma": 0.0,
            "other": 0.0,
        }
        if label in conf_map:
            conf_map[label] = round(score, 4)
        else:
            conf_map["other"] = round(score, 4)
        return {
            "coarse_label": label,
            "label_score": round(score, 4),
            "label_confidence": conf_map,
            "tile_desc": rationale,
            "qc_flags": ["none"],
        }

    @staticmethod
    def _tile_id_from_rc(r: int, c: int) -> str:
        return f"R{int(r)}C{int(c)}"

    def _predict_grid_with_llm(self, thumb_rgb: np.ndarray, grid: int) -> Dict[str, Any]:
        image = thumb_rgb
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side > self.tile_llm_grid_max_side:
            scale = float(self.tile_llm_grid_max_side) / float(max_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if cv2 is not None:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif Image is not None:
                image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
            image = self._as_uint8_rgb(image)

        prompt = (
            "You are a pathology assistant doing coarse triage on a whole-slide thumbnail.\n"
            f"Partition this image into a {grid}x{grid} grid.\n"
            "For each coordinate (r,c), assign one label from: tumor, necrosis, adipose, stroma, background, other.\n"
            "Return strict JSON with fields:\n"
            "1) cells: list of objects with keys r, c, label, confidence, rationale;\n"
            "2) review_order: list of tile ids like R3C4 indicating where to inspect first.\n"
            "The cells list must contain all grid coordinates exactly once.\n"
            'Example: {"cells":[{"r":0,"c":0,"label":"stroma","confidence":0.71,"rationale":"..."},'
            '{"r":0,"c":1,"label":"background","confidence":0.92,"rationale":"..."}],'
            '"review_order":["R3C4","R2C1","R4C4"]}'
        )
        obj = self._run_tile_llm_json(
            tile_rgb=image,
            prompt=prompt,
            max_new_tokens=max(1536, self.tile_llm_max_new_tokens),
        )

        cells = obj.get("cells", [])
        if not isinstance(cells, list):
            raise RuntimeError("Grid prediction JSON must include list field `cells`.")

        labels = [["background" for _ in range(grid)] for _ in range(grid)]
        scores = [[0.0 for _ in range(grid)] for _ in range(grid)]
        rationales = [["" for _ in range(grid)] for _ in range(grid)]
        cell_preds: List[Dict[str, Any]] = []
        seen = set()
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            try:
                r = int(cell.get("r"))
                c = int(cell.get("c"))
            except Exception:
                continue
            if not (0 <= r < grid and 0 <= c < grid):
                continue
            key = (r, c)
            if key in seen:
                continue
            seen.add(key)
            label = self._normalize_tile_label(cell.get("label", "other"))
            try:
                score = float(cell.get("confidence", 0.0))
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))
            rationale = str(cell.get("rationale", "")).strip()
            labels[r][c] = label
            scores[r][c] = round(score, 4)
            rationales[r][c] = rationale
            cell_preds.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "tile_id": self._tile_id_from_rc(r, c),
                    "label": label,
                    "confidence": round(score, 4),
                    "rationale": rationale,
                }
            )
        if len(seen) != grid * grid:
            raise RuntimeError(
                f"Grid prediction missing coordinates: got={len(seen)} expected={grid * grid}"
            )

        review_order_raw = obj.get("review_order", [])
        review_order: List[str] = []
        if isinstance(review_order_raw, list):
            for tid in review_order_raw:
                s = str(tid).strip()
                m = re.fullmatch(r"R(\d+)C(\d+)", s)
                if not m:
                    continue
                rr = int(m.group(1))
                cc = int(m.group(2))
                if 0 <= rr < grid and 0 <= cc < grid and s not in review_order:
                    review_order.append(s)
        if not review_order:
            scored = sorted(
                cell_preds,
                key=lambda x: (float(x.get("confidence", 0.0)), str(x.get("label", "")) == "background"),
                reverse=True,
            )
            review_order = [str(x["tile_id"]) for x in scored]
        return {
            "grid_labels": labels,
            "grid_scores": scores,
            "grid_rationales": rationales,
            "review_order": review_order,
            "cell_predictions": cell_preds,
        }

    def _find_anchors_in_tile_with_llm(
        self,
        tile_rgb: np.ndarray,
        tile_id: str,
        max_anchors: int,
    ) -> Tuple[List[List[float]], List[float]]:
        prompt = (
            "You are selecting anchor ROIs in one pathology tile.\n"
            "Return strict JSON with keys anchors and view_scales.\n"
            "anchors is a list of 1-3 boxes. Each box has x0,y0,x1,y1 normalized to [0,1], with x1>x0 and y1>y0.\n"
            "view_scales is a list of zoom scales in [0.3,1.0] from coarse to fine.\n"
            'Example: {"anchors":[{"x0":0.12,"y0":0.20,"x1":0.72,"y1":0.85}],"view_scales":[1.0,0.7,0.45]}\n'
            f"tile_id={tile_id}; max_anchors={max_anchors}."
        )
        obj = self._run_tile_llm_json(tile_rgb=tile_rgb, prompt=prompt)
        anchors_raw = obj.get("anchors", [])
        if not isinstance(anchors_raw, list):
            raise RuntimeError(f"anchors must be list for tile={tile_id}")
        anchors_norm: List[List[float]] = []
        for a in anchors_raw[: max(1, max_anchors)]:
            if not isinstance(a, dict):
                continue
            try:
                x0 = float(a.get("x0", 0.0))
                y0 = float(a.get("y0", 0.0))
                x1 = float(a.get("x1", 1.0))
                y1 = float(a.get("y1", 1.0))
            except Exception:
                continue
            x0 = max(0.0, min(1.0, x0))
            y0 = max(0.0, min(1.0, y0))
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            anchors_norm.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)])
        if not anchors_norm:
            raise RuntimeError(f"No valid anchors returned by LLM for tile={tile_id}")

        scales_raw = obj.get("view_scales", [])
        view_scales: List[float] = []
        if isinstance(scales_raw, list):
            for v in scales_raw:
                try:
                    vv = float(v)
                except Exception:
                    continue
                if 0.3 <= vv <= 1.0:
                    view_scales.append(round(vv, 2))
        if not view_scales:
            view_scales = [1.0, 0.7, 0.45]
        return anchors_norm, view_scales

    @staticmethod
    def _safe_float_or_none(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            s = str(v).strip()
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if not m:
                return None
            return float(m.group(0))
        except Exception:
            return None

    def _estimate_base_magnification(
        self,
        objective_power: Optional[float],
        mpp_x: Optional[float],
        mpp_y: Optional[float],
    ) -> float:
        if objective_power is not None and objective_power > 0:
            return float(objective_power)
        mpp = None
        if mpp_x is not None and mpp_y is not None and mpp_x > 0 and mpp_y > 0:
            mpp = 0.5 * (float(mpp_x) + float(mpp_y))
        elif mpp_x is not None and mpp_x > 0:
            mpp = float(mpp_x)
        elif mpp_y is not None and mpp_y > 0:
            mpp = float(mpp_y)

        if mpp is None:
            return 40.0
        if mpp <= 0.30:
            return 40.0
        if mpp <= 0.55:
            return 20.0
        if mpp <= 1.10:
            return 10.0
        if mpp <= 2.20:
            return 5.0
        return 2.5

    def _mpp_from_tiff_resolution(
        self,
        properties: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Derive microns-per-pixel from standard TIFF resolution metadata."""
        unit = str(properties.get("tiff.ResolutionUnit", "") or "").strip().lower()
        if unit in {"inch", "inches", "in", "2"}:
            microns_per_unit = 25400.0
        elif unit in {"centimeter", "centimeters", "centimetre", "centimetres", "cm", "3"}:
            microns_per_unit = 10000.0
        else:
            return None, None

        x_resolution = self._safe_float_or_none(properties.get("tiff.XResolution"))
        y_resolution = self._safe_float_or_none(properties.get("tiff.YResolution"))
        mpp_x = (
            microns_per_unit / float(x_resolution)
            if x_resolution is not None and x_resolution > 0
            else None
        )
        mpp_y = (
            microns_per_unit / float(y_resolution)
            if y_resolution is not None and y_resolution > 0
            else None
        )
        return mpp_x, mpp_y

    def _build_stage1_thumbnail(
        self,
        slide_path: str,
        level0_size: List[int],
        low_rgb: np.ndarray,
        medium_rgb: np.ndarray,
        out_dir: str | None,
        slide_id: str,
    ) -> Dict[str, Any]:
        level0_w, level0_h = int(level0_size[0]), int(level0_size[1])
        thumb_rgb = self._as_uint8_rgb(low_rgb.copy())
        source = "lowest_pyramid_level"
        mpp_x: Optional[float] = None
        mpp_y: Optional[float] = None
        objective_power: Optional[float] = None
        if openslide is not None:
            try:
                with openslide.OpenSlide(slide_path) as os_slide:
                    props = dict(getattr(os_slide, "properties", {}) or {})
                    mpp_x = self._safe_float_or_none(props.get("openslide.mpp-x", None))
                    mpp_y = self._safe_float_or_none(props.get("openslide.mpp-y", None))
                    if mpp_x is None or mpp_y is None:
                        tiff_mpp_x, tiff_mpp_y = self._mpp_from_tiff_resolution(props)
                        if mpp_x is None:
                            mpp_x = tiff_mpp_x
                        if mpp_y is None:
                            mpp_y = tiff_mpp_y
                    objective_power = self._safe_float_or_none(
                        props.get("openslide.objective-power", None)
                    )
                    if objective_power is None:
                        objective_power = self._safe_float_or_none(props.get("aperio.AppMag", None))
            except Exception as exc:
                self._log(f"Read OpenSlide metadata failed: {type(exc).__name__}: {exc}")
        base_magnification = self._estimate_base_magnification(
            objective_power=objective_power,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
        )

        tw, th = int(thumb_rgb.shape[1]), int(thumb_rgb.shape[0])
        scale_x = float(level0_w) / float(max(1, tw))
        scale_y = float(level0_h) / float(max(1, th))

        thumb_path = ""
        if out_dir:
            thumb_dir = os.path.join(out_dir, "thumb")
            os.makedirs(thumb_dir, exist_ok=True)
            stem = self._safe_name(os.path.splitext(slide_id)[0])
            thumb_path = os.path.join(thumb_dir, f"{stem}_thumbnail_{max(tw, th)}.png")
            if Image is not None:
                Image.fromarray(thumb_rgb).save(thumb_path)

        return {
            "thumb_rgb": thumb_rgb,
            "thumbnail_meta": {
                "slide_id": slide_id,
                "thumbnail_path": thumb_path,
                "thumbnail_size": [tw, th],
                "level0_size": [level0_w, level0_h],
                "scale_x": scale_x,
                "scale_y": scale_y,
                "mpp": {"x": mpp_x, "y": mpp_y},
                "objective_power": objective_power,
                "base_magnification": round(float(base_magnification), 4),
                "real_patch_output_size": int(self.real_patch_output_size),
                "grid_size": self.grid_size,
                "thumbnail_tile_size": int(self.thumbnail_tile_size),
                "tissue_ratio_thresh": self.tissue_ratio_thresh,
                "thumbnail_source": source,
                "stage1_use_lowest_level": True,
            },
        }

    def _write_stage1_debug_images(
        self,
        out_dir: str,
        slide_id: str,
        thumb_rgb: np.ndarray,
        tissue_mask: np.ndarray,
    ) -> Dict[str, str]:
        if not out_dir:
            return {
                "tissue_mask_path": "",
                "grid_overlay_path": "",
                "selected_patch_overlay_path": "",
            }
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        stem = self._safe_name(os.path.splitext(slide_id)[0])
        tissue_mask_path = os.path.join(debug_dir, f"{stem}_tissue_mask.png")
        grid_overlay_path = os.path.join(debug_dir, f"{stem}_grid_overlay.png")

        if Image is not None:
            Image.fromarray((tissue_mask.astype(np.uint8) * 255), mode="L").save(tissue_mask_path)

        overlay = thumb_rgb.copy()
        if cv2 is not None:
            tile_sz = max(1, int(self.thumbnail_tile_size))
            th, tw = overlay.shape[:2]
            for x in range(tile_sz, tw, tile_sz):
                cv2.line(overlay, (x, 0), (x, th - 1), (0, 255, 0), 1)
            for y in range(tile_sz, th, tile_sz):
                cv2.line(overlay, (0, y), (tw - 1, y), (0, 255, 0), 1)
            if Image is not None:
                Image.fromarray(overlay).save(grid_overlay_path)

        return {
            "tissue_mask_path": tissue_mask_path,
            "grid_overlay_path": grid_overlay_path,
            "selected_patch_overlay_path": "",
        }

    def _write_selected_patch_overlay_debug_image(
        self,
        out_dir: str,
        slide_id: str,
        thumb_rgb: np.ndarray,
        thumbnail_meta: Dict[str, Any],
        tiles_catalog: List[Dict[str, Any]],
        patch_views_catalog: List[Dict[str, Any]],
        anchors_catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not out_dir or Image is None:
            return ""
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        stem = self._safe_name(os.path.splitext(slide_id)[0])
        out_path_exact = os.path.join(debug_dir, f"{stem}_selected_patches_overlay_exact.png")
        # legacy_out_path = os.path.join(debug_dir, f"{stem}_selected_patches_overlay.png")

        exact_overlay = thumb_rgb.copy()
        pil_overlay = None
        pil_draw = None
        use_cv2 = cv2 is not None
        if not use_cv2:
            if ImageDraw is None:
                return ""
            pil_overlay = Image.fromarray(exact_overlay)
            pil_draw = ImageDraw.Draw(pil_overlay)
        label_items: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []
        has_box = False
        h, w = int(exact_overlay.shape[0]), int(exact_overlay.shape[1])
        scale_x = float(thumbnail_meta.get("scale_x", 1.0) or 1.0)
        scale_y = float(thumbnail_meta.get("scale_y", 1.0) or 1.0)
        scale_x = max(1e-6, scale_x)
        scale_y = max(1e-6, scale_y)
        tiles_by_id = {
            str(t.get("tile_id", "")): t
            for t in (tiles_catalog or [])
            if str(t.get("tile_id", ""))
        }
        anchors_by_id: Dict[str, Dict[str, Any]] = {}
        for a in anchors_catalog or []:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("anchor_id", ""))
            if not aid:
                continue
            anchors_by_id[aid] = a

        # Tissue-level large tiles (non-background tiles) in red.
        for t in tiles_catalog or []:
            tb = t.get("thumb_bbox_xywh", [])
            if not isinstance(tb, list) or len(tb) < 4:
                continue
            try:
                tx, ty, tw, th = [int(v) for v in tb[:4]]
            except Exception:
                continue
            x0t = max(0, min(w - 1, tx))
            y0t = max(0, min(h - 1, ty))
            x1t = max(x0t + 1, min(w - 1, x0t + max(1, tw) - 1))
            y1t = max(y0t + 1, min(h - 1, y0t + max(1, th) - 1))
            if use_cv2:
                cv2.rectangle(exact_overlay, (x0t, y0t), (x1t, y1t), (255, 0, 0), 1)
            else:
                pil_draw.rectangle([x0t, y0t, x1t, y1t], outline=(255, 0, 0), width=1)

        for idx, pv in enumerate(patch_views_catalog or []):
            box = pv.get("patch_xywh_level0", [])
            if not isinstance(box, list) or len(box) < 4:
                continue
            try:
                x0_l0, y0_l0, bw_l0, bh_l0 = [int(v) for v in box[:4]]
            except Exception:
                continue
            tile_id = str(pv.get("tile_id", ""))
            ex0 = ey0 = ex1 = ey1 = 0
            mapped_with_tile = False
            tile = tiles_by_id.get(tile_id, None)
            if isinstance(tile, dict):
                tb = tile.get("thumb_bbox_xywh", [])
                lb = tile.get("bbox_level0_xywh", [])
                if (
                    isinstance(tb, list)
                    and isinstance(lb, list)
                    and len(tb) >= 4
                    and len(lb) >= 4
                ):
                    try:
                        ttx, tty, ttw, tth = [int(v) for v in tb[:4]]
                        llx, lly, llw, llh = [int(v) for v in lb[:4]]
                        if llw > 0 and llh > 0 and ttw > 0 and tth > 0:
                            # Robust visualization: clip patch box into tile-level0 bounds first.
                            # This prevents historical out-of-tile coordinates from drawing across tiles.
                            px0 = max(llx, min(x0_l0, llx + llw - 1))
                            py0 = max(lly, min(y0_l0, lly + llh - 1))
                            px1 = max(px0 + 1, min(x0_l0 + max(1, bw_l0), llx + llw))
                            py1 = max(py0 + 1, min(y0_l0 + max(1, bh_l0), lly + llh))
                            fx0 = float(ttx) + (float(px0 - llx) * float(ttw) / float(llw))
                            fy0 = float(tty) + (float(py0 - lly) * float(tth) / float(llh))
                            fx1 = float(ttx) + (float(px1 - llx) * float(ttw) / float(llw))
                            fy1 = float(tty) + (float(py1 - lly) * float(tth) / float(llh))
                            ex0 = int(np.floor(fx0))
                            ey0 = int(np.floor(fy0))
                            ex1 = int(np.ceil(fx1)) - 1
                            ey1 = int(np.ceil(fy1)) - 1
                            mapped_with_tile = True
                    except Exception:
                        mapped_with_tile = False
            if not mapped_with_tile:
                fx0 = float(x0_l0) / scale_x
                fy0 = float(y0_l0) / scale_y
                fx1 = float(x0_l0 + bw_l0) / scale_x
                fy1 = float(y0_l0 + bh_l0) / scale_y
                ex0 = int(np.floor(fx0))
                ey0 = int(np.floor(fy0))
                ex1 = int(np.ceil(fx1)) - 1
                ey1 = int(np.ceil(fy1)) - 1
            if ex1 < ex0:
                ex1 = ex0
            if ey1 < ey0:
                ey1 = ey0
            # Exact mapped patch box (before display enlargement).
            ex0 = max(0, min(w - 1, ex0))
            ey0 = max(0, min(h - 1, ey0))
            ex1 = max(ex0, min(w - 1, ex1))
            ey1 = max(ey0, min(h - 1, ey1))

            anchor_id = str(pv.get("anchor_id", ""))
            seed_text = f"{tile_id}::{anchor_id}".encode("utf-8")
            color_seed = int(hashlib.sha256(seed_text).hexdigest()[:8], 16) % 255
            color = (
                int(80 + (color_seed * 37) % 176),
                int(80 + (color_seed * 71) % 176),
                int(80 + (color_seed * 17) % 176),
            )
            has_box = True
            if use_cv2:
                cv2.rectangle(exact_overlay, (ex0, ey0), (ex1, ey1), color, 2)
            else:
                pil_draw.rectangle([ex0, ey0, ex1, ey1], outline=color, width=2)
            label = str(pv.get("view_id", f"V{idx+1}"))
            label_items.append((label, (ex0, max(10, ey0 + 12)), color))

            # Draw anchor box on thumbnail (cyan), instead of selected 4x4 cell box.
            anchor_id = str(pv.get("anchor_id", ""))
            if anchor_id:
                anchor = anchors_by_id.get(anchor_id, {})
                abox = anchor.get("anchor_bbox_xywh", []) if isinstance(anchor, dict) else []
                if isinstance(abox, list) and len(abox) >= 4:
                    try:
                        ax, ay, aw, ah = [int(v) for v in abox[:4]]
                        if aw > 0 and ah > 0:
                            # Map anchor box to thumbnail (prefer tile-relative mapping).
                            tile = tiles_by_id.get(tile_id, None)
                            anchor_x0 = anchor_y0 = anchor_x1 = anchor_y1 = 0
                            mapped_anchor_with_tile = False
                            if isinstance(tile, dict):
                                tb = tile.get("thumb_bbox_xywh", [])
                                lb = tile.get("bbox_level0_xywh", [])
                                if (
                                    isinstance(tb, list)
                                    and isinstance(lb, list)
                                    and len(tb) >= 4
                                    and len(lb) >= 4
                                ):
                                    try:
                                        ttx, tty, ttw, tth = [int(v) for v in tb[:4]]
                                        llx, lly, llw, llh = [int(v) for v in lb[:4]]
                                        if llw > 0 and llh > 0 and ttw > 0 and tth > 0:
                                            px0 = max(llx, min(ax, llx + llw - 1))
                                            py0 = max(lly, min(ay, lly + llh - 1))
                                            px1 = max(px0 + 1, min(ax + max(1, aw), llx + llw))
                                            py1 = max(py0 + 1, min(ay + max(1, ah), lly + llh))
                                            fax0 = float(ttx) + (float(px0 - llx) * float(ttw) / float(llw))
                                            fay0 = float(tty) + (float(py0 - lly) * float(tth) / float(llh))
                                            fax1 = float(ttx) + (float(px1 - llx) * float(ttw) / float(llw))
                                            fay1 = float(tty) + (float(py1 - lly) * float(tth) / float(llh))
                                            anchor_x0 = int(np.floor(fax0))
                                            anchor_y0 = int(np.floor(fay0))
                                            anchor_x1 = int(np.ceil(fax1)) - 1
                                            anchor_y1 = int(np.ceil(fay1)) - 1
                                            mapped_anchor_with_tile = True
                                    except Exception:
                                        mapped_anchor_with_tile = False
                            if not mapped_anchor_with_tile:
                                fax0 = float(ax) / scale_x
                                fay0 = float(ay) / scale_y
                                fax1 = float(ax + aw) / scale_x
                                fay1 = float(ay + ah) / scale_y
                                anchor_x0 = int(np.floor(fax0))
                                anchor_y0 = int(np.floor(fay0))
                                anchor_x1 = int(np.ceil(fax1)) - 1
                                anchor_y1 = int(np.ceil(fay1)) - 1

                            anchor_x0 = max(0, min(w - 1, anchor_x0))
                            anchor_y0 = max(0, min(h - 1, anchor_y0))
                            anchor_x1 = max(anchor_x0, min(w - 1, anchor_x1))
                            anchor_y1 = max(anchor_y0, min(h - 1, anchor_y1))
                            if use_cv2:
                                cv2.rectangle(exact_overlay, (anchor_x0, anchor_y0), (anchor_x1, anchor_y1), (0, 255, 255), 1)
                            else:
                                pil_draw.rectangle([anchor_x0, anchor_y0, anchor_x1, anchor_y1], outline=(0, 255, 255), width=1)
                    except Exception:
                        pass

        if has_box:
            for label, origin, color in label_items:
                if use_cv2:
                    cv2.putText(
                        exact_overlay,
                        label,
                        origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    pil_draw.text(origin, label, fill=color)

        if use_cv2:
            Image.fromarray(exact_overlay).save(out_path_exact)
            # Keep a compatibility copy with legacy filename.
            # Image.fromarray(exact_overlay).save(legacy_out_path)
        else:
            pil_overlay.save(out_path_exact)
            # Keep a compatibility copy with legacy filename.
            # pil_overlay.save(legacy_out_path)
        return out_path_exact

    def _build_tiles_catalog(
        self,
        slide_id: str,
        thumb_rgb: np.ndarray,
        tissue_mask: np.ndarray,
        thumbnail_meta: Dict[str, Any],
        out_dir: str | None,
    ) -> Dict[str, Any]:
        th, tw = int(thumb_rgb.shape[0]), int(thumb_rgb.shape[1])
        tile_sz = max(1, int(self.thumbnail_tile_size))
        rows = max(1, int(np.ceil(float(th) / float(tile_sz))))
        cols = max(1, int(np.ceil(float(tw) / float(tile_sz))))
        scale_x = float(thumbnail_meta["scale_x"])
        scale_y = float(thumbnail_meta["scale_y"])

        tiles_survey: List[Dict[str, Any]] = []
        tiles_catalog: List[Dict[str, Any]] = []
        tiles_dir = ""
        tiles_catalog_path = ""
        if out_dir and self.save_tile_pngs:
            tiles_dir = os.path.join(out_dir, "tiles")
            os.makedirs(tiles_dir, exist_ok=True)

        for r in range(rows):
            y0 = int(r * tile_sz)
            y1 = int(min(th, y0 + tile_sz))
            gh = max(1, y1 - y0)
            for c in range(cols):
                x0 = int(c * tile_sz)
                x1 = int(min(tw, x0 + tile_sz))
                gw = max(1, x1 - x0)
                tile_id = f"R{r}C{c}"
                mask_crop = tissue_mask[y0:y1, x0:x1]
                tissue_ratio = float(np.mean(mask_crop > 0)) if mask_crop.size else 0.0
                bbox_level0_xywh = [
                    int(round(x0 * scale_x)),
                    int(round(y0 * scale_y)),
                    max(1, int(round(gw * scale_x))),
                    max(1, int(round(gh * scale_y))),
                ]
                is_background = tissue_ratio < self.tissue_ratio_thresh
                survey = {
                    "tile_id": tile_id,
                    "grid_rc": [r, c],
                    "thumb_bbox_xywh": [x0, y0, gw, gh],
                    "bbox_level0_xywh": bbox_level0_xywh,
                    "tissue_ratio": round(tissue_ratio, 4),
                    "is_background": is_background,
                }
                tiles_survey.append(survey)
                if is_background:
                    continue

                tile_crop = thumb_rgb[y0:y1, x0:x1]
                cls = self._classify_tile_rule_based(
                    tile_rgb=tile_crop,
                    tissue_ratio=tissue_ratio,
                )
                tile_png_path = ""
                if tiles_dir and Image is not None:
                    stem = self._safe_name(os.path.splitext(slide_id)[0])
                    tile_png_path = os.path.join(tiles_dir, f"{stem}_{tile_id}.png")
                    Image.fromarray(tile_crop).save(tile_png_path)

                tiles_catalog.append(
                    {
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "grid_rc": [r, c],
                        "tissue_ratio": round(tissue_ratio, 4),
                        "tile_png_path": tile_png_path,
                        "thumb_bbox_xywh": [x0, y0, gw, gh],
                        "bbox_level0_xywh": bbox_level0_xywh,
                        "coarse_label": cls["coarse_label"],
                        "label_score": cls["label_score"],
                        "label_confidence": cls.get("label_confidence", {}),
                        "tile_desc": cls["tile_desc"],
                        "qc_flags": cls["qc_flags"],
                    }
                )

        self._last_tile_review_order = [str(t.get("tile_id", "")) for t in tiles_catalog if str(t.get("tile_id", ""))]
        self._last_tile_coord_predictions = []

        tiles_survey_path = ""
        if out_dir:
            survey_dir = os.path.join(out_dir, "survey")
            os.makedirs(survey_dir, exist_ok=True)
            stem = self._safe_name(os.path.splitext(slide_id)[0])
            tiles_survey_path = os.path.join(survey_dir, f"{stem}_tiles_survey.json")
            tiles_catalog_path = os.path.join(survey_dir, f"{stem}_tiles_catalog.json")
            with open(tiles_survey_path, "w", encoding="utf-8") as f:
                json.dump(tiles_survey, f, indent=2, ensure_ascii=False)
            with open(tiles_catalog_path, "w", encoding="utf-8") as f:
                json.dump(tiles_catalog, f, indent=2, ensure_ascii=False)

        return {
            "tiles_survey": tiles_survey,
            "tiles_catalog": tiles_catalog,
            "tiles_survey_path": tiles_survey_path,
            "tiles_catalog_path": tiles_catalog_path,
        }

    @staticmethod
    def _default_signal_ontology() -> Dict[str, Any]:
        return {
            "version": "v1.0-fallback",
            "signals": [
                {
                    "signal_id": "SIG101__NUCLEAR_PLEOMORPHISM_HIGH",
                    "signal_name": "nuclear_pleomorphism_high",
                    "description": "Marked nuclear size and shape variation suggestive of aggressive morphology.",
                    "weight": 0.90,
                    "applicable_labels": ["tumor", "other"],
                },
                {
                    "signal_id": "SIG102__MITOTIC_ACTIVITY_HIGH",
                    "signal_name": "mitotic_activity_high",
                    "description": "Frequent mitotic figures indicating increased proliferative activity.",
                    "weight": 0.88,
                    "applicable_labels": ["tumor", "other"],
                },
                {
                    "signal_id": "SIG103__TUBULE_FORMATION_LOW",
                    "signal_name": "tubule_formation_low",
                    "description": "Reduced tubular differentiation with predominance of solid growth pattern.",
                    "weight": 0.86,
                    "applicable_labels": ["tumor", "other"],
                },
                {
                    "signal_id": "SIG104__TUMOR_GRADE_HIGH_SUPPORT",
                    "signal_name": "tumor_grade_high_support",
                    "description": "Composite visual cues supporting higher tumor grade impression.",
                    "weight": 0.84,
                    "applicable_labels": ["tumor", "other"],
                },
                {
                    "signal_id": "SIG111__DCIS_PRESENT",
                    "signal_name": "dcis_present",
                    "description": "In-situ ductal epithelial proliferation without clear stromal invasion in sampled field.",
                    "weight": 0.78,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
                {
                    "signal_id": "SIG112__DCIS_COMEDO_NECROSIS",
                    "signal_name": "dcis_comedo_necrosis",
                    "description": "Intraluminal necrotic debris compatible with comedo-type DCIS morphology.",
                    "weight": 0.82,
                    "applicable_labels": ["tumor", "necrosis", "other"],
                },
                {
                    "signal_id": "SIG113__DCIS_ARCH_SOLID_CRIBRIFORM",
                    "signal_name": "dcis_arch_solid_cribriform",
                    "description": "Solid or cribriform in-situ architecture in ductal units.",
                    "weight": 0.76,
                    "applicable_labels": ["tumor", "other"],
                },
                {
                    "signal_id": "SIG121__LYMPHOVASCULAR_INVASION_PRESENT",
                    "signal_name": "lymphovascular_invasion_present",
                    "description": "Suspicious tumor emboli within endothelial-lined spaces.",
                    "weight": 0.74,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
                {
                    "signal_id": "SIG131__MICROCALCIFICATION_PRESENT",
                    "signal_name": "microcalcification_present",
                    "description": "Fine basophilic calcific deposits within tumor-associated tissue.",
                    "weight": 0.72,
                    "applicable_labels": ["tumor", "stroma", "necrosis", "other"],
                },
                {
                    "signal_id": "SIG132__TUMOR_NECROSIS_PRESENT",
                    "signal_name": "tumor_necrosis_present",
                    "description": "Necrotic tumor-associated debris with reduced viable cellular architecture.",
                    "weight": 0.80,
                    "applicable_labels": ["tumor", "necrosis", "other"],
                },
                {
                    "signal_id": "SIG141__MARGIN_TUMOR_CLOSE",
                    "signal_name": "margin_tumor_close",
                    "description": "Tumor component appears close to a specimen edge in sampled context.",
                    "weight": 0.66,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
                {
                    "signal_id": "SIG142__MARGIN_TUMOR_INVOLVED",
                    "signal_name": "margin_tumor_involved",
                    "description": "Tumor appears to involve sampled edge or margin-like boundary.",
                    "weight": 0.64,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
                {
                    "signal_id": "SIG151__LYMPH_NODE_METASTASIS_PRESENT",
                    "signal_name": "lymph_node_metastasis_present",
                    "description": "Metastatic epithelial population in lymphoid background context.",
                    "weight": 0.70,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
                {
                    "signal_id": "SIG152__MICROMETASTASIS_PRESENT",
                    "signal_name": "micrometastasis_present",
                    "description": "Small metastatic focus compatible with micrometastatic burden.",
                    "weight": 0.68,
                    "applicable_labels": ["tumor", "stroma", "other"],
                },
            ],
        }

    def _load_signal_ontology(self) -> Dict[str, Any]:
        fallback = self._default_signal_ontology()
        try:
            with open(self.signal_ontology_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        except Exception as exc:
            self._log(
                f"Load signal ontology failed, fallback used | path={self.signal_ontology_path} | "
                f"{type(exc).__name__}: {exc}"
            )
            return fallback

        if not isinstance(parsed, dict):
            return fallback
        signals = parsed.get("signals", [])
        if not isinstance(signals, list) or not signals:
            return fallback
        clean = []
        for s in signals:
            if not isinstance(s, dict):
                continue
            signal_id = str(s.get("signal_id", "")).strip()
            signal_name = str(s.get("signal_name", "")).strip()
            if not signal_id or not signal_name:
                continue
            labels = s.get("applicable_labels", ["other"])
            if not isinstance(labels, list) or not labels:
                labels = ["other"]
            clean.append(
                {
                    "signal_id": signal_id,
                    "signal_name": signal_name,
                    "description": str(s.get("description", "")).strip(),
                    "weight": float(s.get("weight", 0.70)),
                    "applicable_labels": [str(x).strip().lower() for x in labels if str(x).strip()],
                }
            )
        if not clean:
            return fallback
        out: Dict[str, Any] = {"version": str(parsed.get("version", "v1.0")), "signals": clean}
        for extra_key in ("concept_set", "router_concepts", "concept_signal_relations_v2"):
            extra_val = parsed.get(extra_key, None)
            if isinstance(extra_val, (dict, list)):
                out[extra_key] = extra_val
        return out

    def _signal_templates_for_label(self, label: str) -> List[Dict[str, Any]]:
        label_key = str(label).strip().lower() or "other"
        candidates: List[Dict[str, Any]] = []
        for s in self.signal_ontology.get("signals", []):
            labels = [str(x).strip().lower() for x in s.get("applicable_labels", [])]
            if label_key in labels or "any" in labels or "all" in labels:
                candidates.append(s)
        if candidates:
            return candidates
        # Fallback to non-specific label bucket.
        for s in self.signal_ontology.get("signals", []):
            labels = [str(x).strip().lower() for x in s.get("applicable_labels", [])]
            if "other" in labels:
                candidates.append(s)
        return candidates

    @staticmethod
    def _coerce_grid_index(v: Any, grid: int = 4) -> Optional[int]:
        try:
            i = int(v)
        except Exception:
            return None
        if 0 <= i < grid:
            return i
        if 1 <= i <= grid:
            return i - 1
        return None

    @staticmethod
    def _parse_grid_cell_token(token: Any, grid: int = 4) -> Tuple[Optional[int], Optional[int]]:
        s = str(token or "").strip().upper()
        if not s:
            return None, None
        m = re.search(r"R\s*([0-9]+)\s*C\s*([0-9]+)", s)
        if m:
            r = RealWSILoader._coerce_grid_index(m.group(1), grid)
            c = RealWSILoader._coerce_grid_index(m.group(2), grid)
            return r, c
        nums = re.findall(r"[0-9]+", s)
        if len(nums) >= 2:
            r = RealWSILoader._coerce_grid_index(nums[0], grid)
            c = RealWSILoader._coerce_grid_index(nums[1], grid)
            return r, c
        return None, None

    @staticmethod
    def _center_box_with_size(
        cx: int,
        cy: int,
        bw: int,
        bh: int,
        max_w: int,
        max_h: int,
    ) -> List[int]:
        bw = max(1, min(int(bw), int(max_w)))
        bh = max(1, min(int(bh), int(max_h)))
        x = int(round(float(cx) - float(bw) / 2.0))
        y = int(round(float(cy) - float(bh) / 2.0))
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + bw > max_w:
            x = max(0, int(max_w - bw))
        if y + bh > max_h:
            y = max(0, int(max_h - bh))
        return [int(x), int(y), int(bw), int(bh)]

    @staticmethod
    def _center_box_with_size_in_outer_bbox(
        cx: int,
        cy: int,
        bw: int,
        bh: int,
        outer_xywh: List[int],
        max_w: int,
        max_h: int,
    ) -> List[int]:
        ox, oy, ow, oh = RealWSILoader._clip_bbox_xywh(
            [int(v) for v in outer_xywh],
            max_w=int(max_w),
            max_h=int(max_h),
        )
        bw = max(1, min(int(bw), int(ow)))
        bh = max(1, min(int(bh), int(oh)))
        x = int(round(float(cx) - float(bw) / 2.0))
        y = int(round(float(cy) - float(bh) / 2.0))
        x_min = int(ox)
        y_min = int(oy)
        x_max = int(ox + ow - bw)
        y_max = int(oy + oh - bh)
        x = min(max(x, x_min), max(x_min, x_max))
        y = min(max(y, y_min), max(y_min, y_max))
        return [int(x), int(y), int(bw), int(bh)]

    @staticmethod
    def _cell_bounds_xywh(w: int, h: int, row: int, col: int, grid: int = 4) -> List[int]:
        x0 = int(round(float(col) * float(w) / float(grid)))
        x1 = int(round(float(col + 1) * float(w) / float(grid)))
        y0 = int(round(float(row) * float(h) / float(grid)))
        y1 = int(round(float(row + 1) * float(h) / float(grid)))
        return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]

    def _fallback_rank_cells(self, image_rgb: np.ndarray, top_k: int, grid: int = 4) -> List[Dict[str, Any]]:
        h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        ranked: List[Dict[str, Any]] = []
        for r in range(grid):
            for c in range(grid):
                x0, y0, cw, ch = self._cell_bounds_xywh(w, h, r, c, grid=grid)
                crop = image_rgb[y0:y0 + ch, x0:x0 + cw]
                try:
                    feat = self._compute_features(crop)
                    score = (
                        0.45 * float(feat.get("nuclei_ratio", 0.0))
                        + 0.25 * float(feat.get("purple_ratio", 0.0))
                        + 0.20 * min(1.0, float(feat.get("texture", 0.0)) / 400.0)
                        + 0.10 * float(feat.get("tissue_ratio", 0.0))
                    )
                except Exception:
                    score = 0.5
                ranked.append(
                    {
                        "row": int(r),
                        "col": int(c),
                        "confidence": round(max(0.01, min(0.99, score)), 4),
                        "rationale": "fallback_feature_rank",
                        "tool_trace": [{"tool": "fallback_rank_cells", "cell": f"R{r}C{c}", "score": round(float(score), 4)}],
                    }
                )
        ranked.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return ranked[: max(1, int(top_k))]

    @staticmethod
    def _build_agent_tile_focus_prompt(
        tile_id: str,
        coarse_label: str,
        tissue_ratio: float,
        focus_count: int,
    ) -> str:
        return (
            "You are a pathology diagnostic agent performing focused anchor planning.\n"
            "Input image is one low-resolution tile (256x256) from a WSI.\n"
            "Split it into a 4x4 grid (rows 0..3, cols 0..3).\n"
            f"Select TOP-{int(focus_count)} cells for anchor extraction.\n"
            "Prioritize diagnostically rich structure: tumor-like atypia, hypercellularity, necrosis interface, and gland/duct architecture.\n"
            "Evaluate all 16 cells before choosing.\n"
            "Use ZERO-BASED indices only: row,col must be in [0,1,2,3].\n"
            "Do NOT default to corner cells (especially R0C0) unless there is clear strongest evidence there.\n"
            "If multiple cells are close in quality, prefer the one closer to the tile center over corner cells.\n"
            "Return EXACTLY one JSON object and nothing else.\n"
            "Output format requirements:\n"
            "1) Key: selected_cells\n"
            f"2) selected_cells length MUST be exactly {int(focus_count)}\n"
            "3) Each item must include: row (int 0..3), col (int 0..3), confidence (float 0..1), rationale (short text)\n"
            "4) selected_cells must contain unique (row,col) only\n"
            "Do NOT copy template text from this prompt.\n"
            "Do NOT output placeholder values such as \"...\", \"TBD\", or dummy coordinates.\n"
            "Coordinates must come from visual evidence in the image.\n"
            "Rationale must mention at least one concrete visual cue (e.g., pleomorphism, nuclear crowding, necrosis, tubule/gland structure).\n"
            "No tool_trace key. No markdown. No explanation text.\n"
            f"tile_id={tile_id}; coarse_label_hint={coarse_label}; tissue_ratio={round(float(tissue_ratio),4)}; target_count={int(focus_count)}.\n"
            "Keep rationale short."
        )

    @staticmethod
    def _build_agent_patch_focus_prompt(
        tile_id: str,
        anchor_id: str,
        coarse_label: str,
        patch_count: int,
    ) -> str:
        return (
"You are a pathology diagnostic agent selecting evidence-driven patches.\n"
"Input: 4096x4096 level-0 context resized to 256x256. Split into 4x4 grid (row/col 0..3).\n"
f"Select TOP-{int(patch_count)} cells for extraction.\n"
"\n"
"Priority is morphology evidence, not geometry. Do NOT optimize for diagonals/symmetry.\n"
"Spatial diversity is ONLY a tie-breaker when evidence is similar.\n"
"\n"
"Focus signals (summarized):\n"
"- High-grade invasive tumor cues: marked nuclear pleomorphism, hypercellular dark hotspots (mitoses proxy), solid growth / low tubules.\n"
"- DCIS cues: duct-centered intraductal proliferation; solid/cribriform patterns; comedo-type central necrosis.\n"
"- Necrosis and calcifications: pale granular debris/ghost areas; tiny bright punctate clusters.\n"
"- LVI / node mets: tumor clusters within vessel-like spaces or in nodal tissue (select only if clearly suggested).\n"
"- Margin involvement/close: tumor abutting/near tissue edge (only if tumor is visibly near boundary).\n"
"\n"
"Return EXACTLY one JSON object.\n"
"Keys:\n"
"1) cell_scores (len 16): each {row,col,qc:\"ok\"|\"blur\"|\"artifact\",score:0..1,tags:[...],note}\n"
"   - tags choose from: pleomorphism, mitosis_hotspot, low_tubules_solid, dcis, dcis_comedo, dcis_cribriform_solid,\n"
"     lvi_suspect, microcalc, necrosis, margin_close, margin_involved, node_met, micro_met\n"
"   - note: <=12 words, visible cue only (no diagnosis).\n"
f"2) selected_cells (len {int(patch_count)}): each {{row,col,conf:0..1,primary_tag,rationale}}\n"
"   - rationale: <=12 words, include primary_tag + one visible cue.\n"
"\n"
"Rules:\n"
"- unique (row,col). Avoid qc=artifact unless insufficient.\n"
"- do not select score<0.35 unless not enough cells >=0.35.\n"
"- rank by score. tie (diff<0.08): prefer qc ok, then cover different rows/cols.\n"
"- never pick solely for spatial spread.\n"
"\n"
f"tile_id={tile_id}; anchor_id={anchor_id}; coarse_label_hint={coarse_label}; target_count={int(patch_count)}."
        )

    @staticmethod
    def _build_agent_patch_focus_prompt_10x(
        tile_id: str,
        anchor_id: str,
        coarse_label: str,
        patch_count: int,
    ) -> str:
        return (
"You are a pathology diagnostic agent selecting low-magnification context patches.\n"
"Input: anchor context resized to 256x256 and split into 4x4 grid (row/col 0..3).\n"
f"Select TOP-{int(patch_count)} cells for ~10x context extraction.\n"
"\n"
"Goal:\n"
"- Capture broader tissue architecture and lesion context, independently from high-magnification picks.\n"
"- Prioritize representative context zones (transition fronts, heterogeneous structure, necrosis interface, ductal layout).\n"
"- Avoid selecting solely by corners/diagonals/symmetry.\n"
"\n"
"Return EXACTLY one JSON object.\n"
"Keys:\n"
f"1) selected_cells (len {int(patch_count)}): each {{row,col,confidence,rationale}}\n"
"2) rationale must mention low-mag context cue (architecture/interface/distribution) in <=12 words.\n"
"\n"
"Rules:\n"
"- Use unique (row,col).\n"
"- Evaluate all 16 cells before selecting.\n"
"- Do NOT copy text from prompt.\n"
"- No markdown, no tool_trace, no extra keys.\n"
f"tile_id={tile_id}; anchor_id={anchor_id}; coarse_label_hint={coarse_label}; target_count={int(patch_count)}."
        )

    @staticmethod
    def _target_tile_count_for_label(label: Any) -> int:
        key = str(label or "").strip().lower()
        if key in {"tumor", "carcinoma", "high_grade"}:
            return 3
        if key in {"background", ""}:
            return 0
        return 2

    @staticmethod
    def _target_anchor_count_for_label(label: Any) -> int:
        key = str(label or "").strip().lower()
        if key in {"tumor", "carcinoma", "high_grade"}:
            return 3
        return 2

    @staticmethod
    def _target_patch_count_for_label(label: Any) -> int:
        key = str(label or "").strip().lower()
        if key in {"tumor", "carcinoma", "high_grade"}:
            return 2
        return 1

    @staticmethod
    def _extract_raw_output_from_error_text(err_text: str) -> str:
        s = str(err_text or "")
        m = re.search(r"Raw output:\s*(.+)$", s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        return str(m.group(1)).strip()

    def _try_parse_single_cell_from_text(self, text: str, grid: int = 4) -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return None
        s = s.replace("\\_", "_")
        obj = self._extract_json_object(s)
        if isinstance(obj, dict) and obj:
            row = None
            col = None
            sel = obj.get("selected_cell", {})
            if isinstance(sel, dict):
                row = self._coerce_grid_index(sel.get("row", None), grid)
                col = self._coerce_grid_index(sel.get("col", None), grid)
            if row is None:
                row = self._coerce_grid_index(obj.get("row", None), grid)
            if col is None:
                col = self._coerce_grid_index(obj.get("col", None), grid)
            if row is not None and col is not None:
                try:
                    conf = float(obj.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                return {
                    "row": int(row),
                    "col": int(col),
                    "confidence": round(max(0.01, min(0.99, conf)), 4),
                    "rationale": str(obj.get("rationale", "")).strip() or "agent_focus_cell",
                    "tool_trace": obj.get("tool_trace", []) if isinstance(obj.get("tool_trace", []), list) else [],
                }

        # Regex fallback for malformed JSON bodies.
        m_rc = re.search(
            r'"selected_cell"\s*:\s*\{[\s\S]*?"row"\s*:\s*([0-9]+)[\s\S]*?"col"\s*:\s*([0-9]+)',
            s,
            flags=re.IGNORECASE,
        )
        if not m_rc:
            m_rc = re.search(r'"row"\s*:\s*([0-9]+)[\s\S]*?"col"\s*:\s*([0-9]+)', s, flags=re.IGNORECASE)
        if not m_rc:
            return None
        row = self._coerce_grid_index(m_rc.group(1), grid)
        col = self._coerce_grid_index(m_rc.group(2), grid)
        if row is None or col is None:
            return None
        m_conf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', s, flags=re.IGNORECASE)
        try:
            conf = float(m_conf.group(1)) if m_conf else 0.5
        except Exception:
            conf = 0.5
        m_rat = re.search(r'"rationale"\s*:\s*"([^"]+)"', s, flags=re.IGNORECASE)
        rationale = str(m_rat.group(1)).strip() if m_rat else "agent_focus_cell"
        return {
            "row": int(row),
            "col": int(col),
            "confidence": round(max(0.01, min(0.99, conf)), 4),
            "rationale": rationale,
            "tool_trace": [],
        }

    def _try_parse_multi_cells_from_text(
        self,
        text: str,
        top_k: int,
        grid: int = 4,
        allow_regex_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return []
        s = s.replace("\\_", "_")
        obj = self._extract_json_object(s)
        out: List[Dict[str, Any]] = []
        seen = set()
        if isinstance(obj, dict) and obj:
            cells = obj.get("selected_cells", [])
            if isinstance(cells, list):
                for item in cells:
                    if not isinstance(item, dict):
                        continue
                    row = self._coerce_grid_index(item.get("row", None), grid)
                    col = self._coerce_grid_index(item.get("col", None), grid)
                    if row is None or col is None:
                        continue
                    key = (int(row), int(col))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        conf = float(item.get("confidence", 0.5))
                    except Exception:
                        conf = 0.5
                    out.append(
                        {
                            "row": int(row),
                            "col": int(col),
                            "confidence": round(max(0.01, min(0.99, conf)), 4),
                            "rationale": str(item.get("rationale", "")).strip() or "agent_patch_cell",
                            "tool_trace": obj.get("tool_trace", []) if isinstance(obj.get("tool_trace", []), list) else [],
                        }
                    )
                    if len(out) >= int(top_k):
                        return out
        if allow_regex_fallback:
            # Regex fallback: collect row/col pairs in order.
            for m in re.finditer(r'"row"\s*:\s*([0-9]+)[\s\S]*?"col"\s*:\s*([0-9]+)', s, flags=re.IGNORECASE):
                row = self._coerce_grid_index(m.group(1), grid)
                col = self._coerce_grid_index(m.group(2), grid)
                if row is None or col is None:
                    continue
                key = (int(row), int(col))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "confidence": 0.5,
                        "rationale": "agent_patch_cell_regex",
                        "tool_trace": [],
                    }
                )
                if len(out) >= int(top_k):
                    break
        return out

    @staticmethod
    def _is_weak_cell_rationale(text: Any) -> bool:
        s = str(text or "").strip().lower()
        return s in {
            "",
            "...",
            "agent_focus_cell",
            "agent_patch_cell",
            "agent_patch_cell_regex",
            "recovered_from_raw_text",
        }

    def _looks_template_cell_set(self, cells: List[Dict[str, Any]]) -> bool:
        if not isinstance(cells, list) or not cells:
            return True
        coords: List[Tuple[int, int]] = []
        confs: List[float] = []
        weak_count = 0
        for c in cells:
            if not isinstance(c, dict):
                continue
            try:
                rr = int(c.get("row", -1))
                cc = int(c.get("col", -1))
            except Exception:
                rr, cc = -1, -1
            if rr < 0 or cc < 0:
                continue
            coords.append((rr, cc))
            try:
                confs.append(float(c.get("confidence", 0.5)))
            except Exception:
                confs.append(0.5)
            if self._is_weak_cell_rationale(c.get("rationale", "")):
                weak_count += 1
        if not coords:
            return True
        weak_ratio = float(weak_count) / float(max(1, len(coords)))
        conf_spread = (max(confs) - min(confs)) if confs else 0.0
        in_top_left_quadrant = all((r in {0, 1} and c in {0, 1}) for r, c in coords)
        has_top_left = any((r == 0 and c == 0) for r, c in coords)
        if weak_ratio >= 0.8 and conf_spread <= 0.12:
            return True
        if has_top_left and in_top_left_quadrant and weak_ratio >= 0.5:
            return True
        return False

    def _parse_single_cell_from_obj(self, obj: Any, grid: int = 4) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        row: Optional[int] = None
        col: Optional[int] = None
        sel = obj.get("selected_cell", {})
        if isinstance(sel, dict):
            row = self._coerce_grid_index(sel.get("row", None), grid)
            col = self._coerce_grid_index(sel.get("col", None), grid)
        if row is None:
            row = self._coerce_grid_index(obj.get("row", None), grid)
        if col is None:
            col = self._coerce_grid_index(obj.get("col", None), grid)
        if row is None or col is None:
            rr, cc = self._parse_grid_cell_token(obj.get("selected_cell", None), grid)
            row = row if row is not None else rr
            col = col if col is not None else cc
        if row is None or col is None:
            rr, cc = self._parse_grid_cell_token(obj.get("cell", None), grid)
            row = row if row is not None else rr
            col = col if col is not None else cc
        if row is None or col is None:
            return None
        try:
            confidence = float(obj.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.01, min(0.99, confidence))
        rationale = str(obj.get("rationale", "")).strip() or "agent_focus_cell"
        tool_trace = obj.get("tool_trace", [])
        if not isinstance(tool_trace, list):
            tool_trace = []
        return {
            "row": int(row),
            "col": int(col),
            "confidence": round(confidence, 4),
            "rationale": rationale,
            "tool_trace": tool_trace,
        }

    def _parse_multi_cells_from_obj(
        self,
        obj: Any,
        top_k: int,
        grid: int = 4,
    ) -> List[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return []
        out: List[Dict[str, Any]] = []
        seen = set()
        sel_cells = obj.get("selected_cells", [])
        if isinstance(sel_cells, list):
            for item in sel_cells:
                row: Optional[int] = None
                col: Optional[int] = None
                conf = 0.5
                rationale = "agent_patch_cell"
                if isinstance(item, dict):
                    row = self._coerce_grid_index(item.get("row", None), grid)
                    col = self._coerce_grid_index(item.get("col", None), grid)
                    try:
                        conf = float(item.get("confidence", 0.5))
                    except Exception:
                        conf = 0.5
                    rationale = str(item.get("rationale", "")).strip() or rationale
                else:
                    row, col = self._parse_grid_cell_token(item, grid)
                if row is None or col is None:
                    continue
                key = (int(row), int(col))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "confidence": round(max(0.01, min(0.99, conf)), 4),
                        "rationale": rationale,
                        "tool_trace": obj.get("tool_trace", []) if isinstance(obj.get("tool_trace", []), list) else [],
                    }
                )
                if len(out) >= int(top_k):
                    break
        return out[: max(1, int(top_k))]

    def _select_single_cells_with_agent_batch(
        self,
        image_rgbs: List[np.ndarray],
        prompts: List[str],
        grid: int = 4,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if len(image_rgbs) != len(prompts):
            raise ValueError("image_rgbs and prompts must have the same length.")
        n = len(image_rgbs)
        if n == 0:
            return []
        bs = int(batch_size) if batch_size is not None else int(self.patch_llm_batch_size)
        bs = max(1, bs)
        out: List[Dict[str, Any]] = []
        for start in range(0, n, bs):
            end = min(n, start + bs)
            chunk_imgs = image_rgbs[start:end]
            chunk_prompts = prompts[start:end]
            try:
                obj_list = self._run_tile_llm_json_batch(
                    tile_rgbs=chunk_imgs,
                    prompts=chunk_prompts,
                    max_new_tokens=max(128, self.tile_llm_max_new_tokens),
                )
            except Exception as exc:
                self._log(
                    f"Agent single-cell batch failed | start={start} size={len(chunk_imgs)} | "
                    f"{type(exc).__name__}: {exc}"
                )
                obj_list = [{} for _ in chunk_imgs]
            for i, img in enumerate(chunk_imgs):
                obj = obj_list[i] if i < len(obj_list) else {}
                parsed = self._parse_single_cell_from_obj(obj, grid=grid)
                if parsed is None:
                    parsed = self._fallback_rank_cells(image_rgb=img, top_k=1, grid=grid)[0]
                out.append(parsed)
        return out

    def _select_multi_cells_with_agent_batch(
        self,
        image_rgbs: List[np.ndarray],
        prompts: List[str],
        top_ks: List[int],
        grids: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        if len(image_rgbs) != len(prompts) or len(image_rgbs) != len(top_ks):
            raise ValueError("image_rgbs/prompts/top_ks must have the same length.")
        n = len(image_rgbs)
        if n == 0:
            return []
        if grids is None:
            grids = [4 for _ in range(n)]
        if len(grids) != n:
            raise ValueError("grids must have the same length as image_rgbs.")
        bs = int(batch_size) if batch_size is not None else int(self.patch_llm_batch_size)
        bs = max(1, bs)
        out: List[List[Dict[str, Any]]] = []
        for start in range(0, n, bs):
            end = min(n, start + bs)
            chunk_imgs = image_rgbs[start:end]
            chunk_prompts = prompts[start:end]
            chunk_top_ks = top_ks[start:end]
            chunk_grids = grids[start:end]
            try:
                obj_list = self._run_tile_llm_json_batch(
                    tile_rgbs=chunk_imgs,
                    prompts=chunk_prompts,
                    max_new_tokens=max(128, self.tile_llm_max_new_tokens),
                )
            except Exception as exc:
                self._log(
                    f"Agent multi-cell batch failed | start={start} size={len(chunk_imgs)} | "
                    f"{type(exc).__name__}: {exc}"
                )
                obj_list = [{} for _ in chunk_imgs]
            for i, img in enumerate(chunk_imgs):
                top_k = max(1, int(chunk_top_ks[i]))
                grid_i = max(2, int(chunk_grids[i]))
                obj = obj_list[i] if i < len(obj_list) else {}
                parsed = self._parse_multi_cells_from_obj(obj, top_k=top_k, grid=grid_i)
                if self._looks_template_cell_set(parsed):
                    parsed = []
                if len(parsed) < top_k:
                    fallback = self._fallback_rank_cells(image_rgb=img, top_k=top_k, grid=grid_i)
                    seen = {(int(p["row"]), int(p["col"])) for p in parsed}
                    for fb in fallback:
                        key = (int(fb["row"]), int(fb["col"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        parsed.append(fb)
                        if len(parsed) >= top_k:
                            break
                out.append(parsed[:top_k])
        return out

    def _select_single_cell_with_agent(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        grid: int = 4,
    ) -> Dict[str, Any]:
        try:
            obj = self._run_tile_llm_json(
                tile_rgb=image_rgb,
                prompt=prompt,
                max_new_tokens=max(128, self.tile_llm_max_new_tokens),
            )
        except Exception as exc:
            err_text = str(exc)
            raw = self._extract_raw_output_from_error_text(err_text)
            parsed = self._try_parse_single_cell_from_text(raw, grid=grid)
            if parsed is None:
                parsed = self._try_parse_single_cell_from_text(err_text, grid=grid)
            if parsed is not None:
                self._log("Agent single-cell parse recovered from non-strict JSON raw output.")
                return parsed
            self._log(
                f"Agent single-cell select failed; fallback rank used | "
                f"{type(exc).__name__}: {exc}"
            )
            return self._fallback_rank_cells(image_rgb=image_rgb, top_k=1, grid=grid)[0]
        row: Optional[int] = None
        col: Optional[int] = None
        sel = obj.get("selected_cell", {})
        if isinstance(sel, dict):
            row = self._coerce_grid_index(sel.get("row", None), grid)
            col = self._coerce_grid_index(sel.get("col", None), grid)
        if row is None:
            row = self._coerce_grid_index(obj.get("row", None), grid)
        if col is None:
            col = self._coerce_grid_index(obj.get("col", None), grid)
        if row is None or col is None:
            rr, cc = self._parse_grid_cell_token(obj.get("selected_cell", None), grid)
            row = row if row is not None else rr
            col = col if col is not None else cc
        if row is None or col is None:
            rr, cc = self._parse_grid_cell_token(obj.get("cell", None), grid)
            row = row if row is not None else rr
            col = col if col is not None else cc

        if row is None or col is None:
            fallback = self._fallback_rank_cells(image_rgb=image_rgb, top_k=1, grid=grid)[0]
            return fallback

        confidence = 0.5
        try:
            confidence = float(obj.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.01, min(0.99, confidence))
        rationale = str(obj.get("rationale", "")).strip() or "agent_focus_cell"
        tool_trace = obj.get("tool_trace", [])
        if not isinstance(tool_trace, list):
            tool_trace = []
        return {
            "row": int(row),
            "col": int(col),
            "confidence": round(confidence, 4),
            "rationale": rationale,
            "tool_trace": tool_trace,
        }

    def _select_multi_cells_with_agent(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        top_k: int,
        grid: int = 4,
    ) -> List[Dict[str, Any]]:
        try:
            obj = self._run_tile_llm_json(
                tile_rgb=image_rgb,
                prompt=prompt,
                max_new_tokens=max(128, self.tile_llm_max_new_tokens),
            )
        except Exception as exc:
            err_text = str(exc)
            raw = self._extract_raw_output_from_error_text(err_text)
            recovered = self._try_parse_multi_cells_from_text(raw, top_k=int(top_k), grid=grid)
            if not recovered:
                recovered = self._try_parse_multi_cells_from_text(
                    err_text, top_k=int(top_k), grid=grid
                )
            if recovered:
                self._log(
                    f"Agent multi-cell parse recovered from non-strict JSON raw output | n={len(recovered)}"
                )
                return recovered[: max(1, int(top_k))]
            self._log(
                f"Agent multi-cell select failed; fallback rank used | "
                f"{type(exc).__name__}: {exc}"
            )
            return self._fallback_rank_cells(image_rgb=image_rgb, top_k=int(top_k), grid=grid)
        out: List[Dict[str, Any]] = []
        seen = set()
        sel_cells = obj.get("selected_cells", [])
        if isinstance(sel_cells, list):
            for item in sel_cells:
                row: Optional[int] = None
                col: Optional[int] = None
                conf = 0.5
                rationale = "agent_patch_cell"
                if isinstance(item, dict):
                    row = self._coerce_grid_index(item.get("row", None), grid)
                    col = self._coerce_grid_index(item.get("col", None), grid)
                    try:
                        conf = float(item.get("confidence", 0.5))
                    except Exception:
                        conf = 0.5
                    rationale = str(item.get("rationale", "")).strip() or rationale
                else:
                    row, col = self._parse_grid_cell_token(item, grid)
                if row is None or col is None:
                    continue
                key = (int(row), int(col))
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "confidence": round(max(0.01, min(0.99, conf)), 4),
                        "rationale": rationale,
                        "tool_trace": obj.get("tool_trace", []) if isinstance(obj.get("tool_trace", []), list) else [],
                    }
                )
                if len(out) >= int(top_k):
                    break

        if self._looks_template_cell_set(out):
            out = []
            seen = set()
        if len(out) < int(top_k):
            fallback = self._fallback_rank_cells(image_rgb=image_rgb, top_k=int(top_k), grid=grid)
            for fb in fallback:
                key = (int(fb["row"]), int(fb["col"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(fb)
                if len(out) >= int(top_k):
                    break
        return out[: max(1, int(top_k))]

    def _build_tile_anchor_packages(
        self,
        slide_id: str,
        tiles_catalog: List[Dict[str, Any]],
        thumb_rgb: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not tiles_catalog:
            return []
        tiles_by_id = {str(t.get("tile_id", "")): t for t in tiles_catalog}
        label_priority = {
            "tumor": 3,
            "carcinoma": 3,
            "high_grade": 3,
            "necrosis": 2,
            "stroma": 1,
            "adipose": 1,
            "other": 0,
            "background": -1,
        }
        ranked_tiles = sorted(
            [t for t in tiles_catalog if str(t.get("tile_id", ""))],
            key=lambda t: (
                int(label_priority.get(str(t.get("coarse_label", "other")).strip().lower(), 0)),
                float(t.get("label_score", 0.0)),
                float(t.get("tissue_ratio", 0.0)),
            ),
            reverse=True,
        )

        # Stage-1 tile quota:
        # tumor -> 2 tiles; non-background other labels -> 1 tile each.
        selected_tiles: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        label_counts: Dict[str, int] = {}
        for t in ranked_tiles:
            tile_id = str(t.get("tile_id", ""))
            if not tile_id or tile_id in selected_ids:
                continue
            raw_label = str(t.get("coarse_label", "other")).strip().lower()
            if raw_label in {"carcinoma", "high_grade"}:
                label_key = "tumor"
            else:
                label_key = raw_label
            quota = int(self._target_tile_count_for_label(label_key))
            if quota <= 0:
                continue
            cur = int(label_counts.get(label_key, 0))
            if cur >= quota:
                continue
            selected_tiles.append(t)
            selected_ids.add(tile_id)
            label_counts[label_key] = cur + 1

        if not selected_tiles:
            selected_tiles = ranked_tiles[:1]
            selected_ids = {str(t.get("tile_id", "")) for t in selected_tiles if str(t.get("tile_id", ""))}

        # Keep one anchor per tile and compensate by using 2x anchor tiles.
        base_selected_count = max(1, len(selected_tiles))
        target_selected_count = min(len(ranked_tiles), int(base_selected_count * 2))
        if len(selected_tiles) < target_selected_count:
            for t in ranked_tiles:
                tile_id = str(t.get("tile_id", ""))
                if not tile_id or tile_id in selected_ids:
                    continue
                selected_tiles.append(t)
                selected_ids.add(tile_id)
                if len(selected_tiles) >= target_selected_count:
                    break

        if self.tile_llm_max_anchor_tiles > 0:
            selected_tiles = selected_tiles[: int(self.tile_llm_max_anchor_tiles)]

        ranked_tile_ids = [str(t.get("tile_id", "")) for t in selected_tiles]
        self._last_tile_review_order = list(ranked_tile_ids)

        packages: List[Dict[str, Any]] = []
        anchor_tiles = ranked_tile_ids
        self._last_anchor_tile_order = list(anchor_tiles)
        self._log(f"Agent anchor selection start | candidate_tiles={len(anchor_tiles)}")

        tile_jobs: List[Dict[str, Any]] = []
        for tile_id in anchor_tiles:
            tile = tiles_by_id.get(tile_id)
            if not tile:
                continue
            label = str(tile.get("coarse_label", "other"))
            label_score = float(tile.get("label_score", 0.0))
            tx, ty, tw, th = [int(v) for v in tile["bbox_level0_xywh"]]
            tx_thumb, ty_thumb, tw_thumb, th_thumb = [int(v) for v in tile.get("thumb_bbox_xywh", [0, 0, 1, 1])]
            tile_crop = thumb_rgb[ty_thumb:ty_thumb + th_thumb, tx_thumb:tx_thumb + tw_thumb]
            if tile_crop.size == 0:
                tile_crop = np.zeros((self.real_patch_output_size, self.real_patch_output_size, 3), dtype=np.uint8)
            elif tile_crop.shape[0] != self.real_patch_output_size or tile_crop.shape[1] != self.real_patch_output_size:
                if cv2 is not None:
                    tile_crop = cv2.resize(
                        tile_crop,
                        (self.real_patch_output_size, self.real_patch_output_size),
                        interpolation=cv2.INTER_AREA,
                    )
                elif Image is not None:
                    tile_crop = np.array(
                        Image.fromarray(tile_crop).resize(
                            (self.real_patch_output_size, self.real_patch_output_size),
                            Image.BILINEAR,
                        )
                    )
                tile_crop = self._as_uint8_rgb(tile_crop)
            tile_jobs.append(
                {
                    "tile_id": tile_id,
                    "tile": tile,
                    "label": label,
                    "label_score": label_score,
                    "tx": tx,
                    "ty": ty,
                    "tw": tw,
                    "th": th,
                    "tile_crop": tile_crop,
                    "focus_top_k": int(self._target_anchor_count_for_label(label)),
                    "focus_prompt": self._build_agent_tile_focus_prompt(
                        tile_id=tile_id,
                        coarse_label=label,
                        tissue_ratio=float(tile.get("tissue_ratio", 0.0)),
                        focus_count=int(self._target_anchor_count_for_label(label)),
                    ),
                }
            )

        focus_cells: List[List[Dict[str, Any]]] = []
        if tile_jobs:
            batch_size = max(1, int(self.patch_llm_batch_size))
            self._log(
                f"Agent tile focus batch select | tiles={len(tile_jobs)} | batch_size={batch_size}"
            )
            focus_cells = self._select_multi_cells_with_agent_batch(
                image_rgbs=[j["tile_crop"] for j in tile_jobs],
                prompts=[str(j["focus_prompt"]) for j in tile_jobs],
                top_ks=[int(j.get("focus_top_k", 2)) for j in tile_jobs],
                grids=[4 for _ in tile_jobs],
                batch_size=batch_size,
            )

        for idx, job in enumerate(tile_jobs):
            tile_id = str(job["tile_id"])
            tile = dict(job["tile"])
            label = str(job["label"])
            label_score = float(job["label_score"])
            tx = int(job["tx"])
            ty = int(job["ty"])
            tw = int(job["tw"])
            th = int(job["th"])
            focus_top_k = max(1, int(job.get("focus_top_k", 2)))
            focus_cell_candidates = (
                list(focus_cells[idx])
                if idx < len(focus_cells) and isinstance(focus_cells[idx], list)
                else []
            )
            if len(focus_cell_candidates) < focus_top_k:
                fallback = self._fallback_rank_cells(
                    image_rgb=np.asarray(job["tile_crop"]),
                    top_k=focus_top_k,
                    grid=4,
                )
                seen = {
                    (int(c.get("row", -1)), int(c.get("col", -1)))
                    for c in focus_cell_candidates
                    if isinstance(c, dict)
                }
                for fb in fallback:
                    key = (int(fb.get("row", -1)), int(fb.get("col", -1)))
                    if key in seen:
                        continue
                    seen.add(key)
                    focus_cell_candidates.append(fb)
                    if len(focus_cell_candidates) >= focus_top_k:
                        break
            focus_cell_candidates = focus_cell_candidates[:focus_top_k]
            # Additional guard: if TOP-1 lands on R0C0, keep it only when it is
            # clearly stronger than non-corner options by fallback evidence.
            if focus_top_k == 1 and focus_cell_candidates:
                try:
                    c0 = focus_cell_candidates[0]
                    r0 = int(c0.get("row", 0))
                    c0_col = int(c0.get("col", 0))
                    if r0 == 0 and c0_col == 0:
                        fallback_full = self._fallback_rank_cells(
                            image_rgb=np.asarray(job["tile_crop"]),
                            top_k=16,
                            grid=4,
                        )
                        score_map = {
                            (int(fc.get("row", -1)), int(fc.get("col", -1))): float(fc.get("confidence", 0.0))
                            for fc in fallback_full
                            if isinstance(fc, dict)
                        }
                        top_left_score = float(score_map.get((0, 0), 0.0))
                        best_non_corner: Optional[Dict[str, Any]] = None
                        for fc in fallback_full:
                            rr = int(fc.get("row", -1))
                            cc = int(fc.get("col", -1))
                            if rr < 0 or cc < 0:
                                continue
                            if rr in {0, 3} and cc in {0, 3}:
                                continue
                            best_non_corner = fc
                            break
                        if best_non_corner is not None:
                            non_corner_score = float(best_non_corner.get("confidence", 0.0))
                            c0_conf = float(c0.get("confidence", 0.5))
                            c0_rat = str(c0.get("rationale", "")).strip().lower()
                            weak_rationale = c0_rat in {
                                "",
                                "...",
                                "agent_focus_cell",
                                "agent_patch_cell",
                                "agent_patch_cell_regex",
                                "recovered_from_raw_text",
                            }
                            if weak_rationale or c0_conf < 0.75 or non_corner_score >= (top_left_score - 0.03):
                                focus_cell_candidates = [best_non_corner]
                except Exception:
                    pass
            patch_count = self._target_patch_count_for_label(label)

            anchors: List[Dict[str, Any]] = []
            for a_idx, focus_cell in enumerate(focus_cell_candidates, start=1):
                fr = int(focus_cell.get("row", 0))
                fc = int(focus_cell.get("col", 0))
                fx0 = tx + int(round(float(fc) * float(tw) / 4.0))
                fx1 = tx + int(round(float(fc + 1) * float(tw) / 4.0))
                fy0 = ty + int(round(float(fr) * float(th) / 4.0))
                fy1 = ty + int(round(float(fr + 1) * float(th) / 4.0))
                focus_box = [fx0, fy0, max(1, fx1 - fx0), max(1, fy1 - fy0)]
                fcx = int(focus_box[0] + focus_box[2] // 2)
                fcy = int(focus_box[1] + focus_box[3] // 2)
                anchor_id = f"A_{tile_id}_{a_idx:02d}"
                anchors.append(
                    {
                        "anchor_id": anchor_id,
                        "tile_id": tile_id,
                        "anchor_bbox_xywh": focus_box,
                        "bbox_norm": [
                            round(float(fc) / 4.0, 4),
                            round(float(fr) / 4.0, 4),
                            round(float(fc + 1) / 4.0, 4),
                            round(float(fr + 1) / 4.0, 4),
                        ],
                        "focus_center_level0": [int(fcx), int(fcy)],
                        "focus_cell_rc": [int(fr), int(fc)],
                        "focus_confidence": round(float(focus_cell.get("confidence", 0.5)), 4),
                        "focus_rationale": str(focus_cell.get("rationale", "")),
                        "focus_tool_trace": list(focus_cell.get("tool_trace", []))
                        if isinstance(focus_cell.get("tool_trace", []), list)
                        else [],
                        "evidence_id": f"E_{anchor_id}",
                        "patches_per_anchor": int(patch_count),
                    }
                )

            if not anchors:
                continue

            candidate_signals: List[Dict[str, Any]] = []
            for tpl in self._signal_templates_for_label(label):
                candidate_signals.append(
                    {
                        "signal_id": tpl["signal_id"],
                        "signal_name": tpl["signal_name"],
                        "description": tpl["description"],
                        "confidence": round(min(0.999, float(label_score) * float(tpl.get("weight", 1.0))), 4),
                    }
                )

            # Keep anchor crop proportional to tile size to avoid oversized anchor boxes
            # on smaller tiles, while respecting configured global upper bound.
            tile_min_side = max(1, min(int(tw), int(th)))
            adaptive_anchor_crop = int(max(512, round(float(tile_min_side) * 0.25)))
            adaptive_anchor_crop = int(min(int(self.anchor_crop_size_level0), adaptive_anchor_crop))

            sampling_plan = {
                "policy": "agent_tool_three_stage",
                "tile_subgrid": [4, 4],
                "anchor_subgrid": [4, 4],
                "anchor_crop_size_level0": int(adaptive_anchor_crop),
                "patch_crop_size_level0": int(self.patch_crop_size_level0),
                "selected_anchor_count": int(len(anchors)),
                "planned_view_count_per_anchor": int(patch_count + 1),
                "max_patches_per_anchor": int(patch_count),
                "planned_low_mag_views_per_anchor": 1,
                "anchor_order": [a["anchor_id"] for a in anchors],
            }

            packages.append(
                {
                    "slide_id": slide_id,
                    "tile_id": tile_id,
                    "tile_bbox_xywh": [tx, ty, tw, th],
                    "coarse_label": label,
                    "label_score": round(float(label_score), 4),
                    "tile_desc": str(tile.get("tile_desc", "")),
                    "anchors": anchors,
                    "sampling_plan": sampling_plan,
                    "candidate_signals": candidate_signals,
                    "evidence_seed_ids": [a["evidence_id"] for a in anchors],
                }
            )

        return packages

    @staticmethod
    def _clip_bbox_xywh(box: List[int], max_w: int, max_h: int) -> List[int]:
        x, y, w, h = [int(v) for v in box]
        x = max(0, min(x, max(0, max_w - 1)))
        y = max(0, min(y, max(0, max_h - 1)))
        w = max(1, min(w, max_w - x))
        h = max(1, min(h, max_h - y))
        return [x, y, w, h]

    @staticmethod
    def _mag_from_scale(scale: float) -> str:
        s = float(scale)
        if s >= 0.85:
            return "5x"
        if s >= 0.55:
            return "10x"
        if s >= 0.35:
            return "20x"
        return "40x"

    @staticmethod
    def _target_mag_from_scale(scale: float) -> float:
        s = float(scale)
        if s >= 0.85:
            return 5.0
        if s >= 0.55:
            return 10.0
        if s >= 0.35:
            return 20.0
        return 40.0

    @staticmethod
    def _format_mag_label(mag: float) -> str:
        m = float(mag)
        if abs(m - round(m)) < 1e-4:
            return f"{int(round(m))}x"
        return f"{m:.1f}x"

    def _build_stage1_evidence_objects(
        self,
        slide_id: str,
        slide_path: str,
        tile_anchor_packages: List[Dict[str, Any]],
        level0_size: List[int],
        base_magnification: Optional[float],
        out_dir: str | None,
        thumb_rgb: Optional[np.ndarray] = None,
        thumbnail_meta: Optional[Dict[str, Any]] = None,
        tiles_catalog: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        level0_w, level0_h = int(level0_size[0]), int(level0_size[1])
        base_mag = float(base_magnification) if base_magnification and base_magnification > 0 else 40.0
        anchors_catalog: List[Dict[str, Any]] = []
        patch_views_catalog: List[Dict[str, Any]] = []
        evidence_items: List[Dict[str, Any]] = []
        signal_screenings: List[Dict[str, Any]] = []
        signal_evidence_edges: List[Dict[str, Any]] = []
        patch_sampler_backend = "none"

        patch_dir = ""
        stem = self._safe_name(os.path.splitext(slide_id)[0])
        if out_dir:
            patch_dir = os.path.join(out_dir, "patches")
            os.makedirs(patch_dir, exist_ok=True)

        os_slide = None
        if openslide is not None and Image is not None:
            try:
                os_slide = openslide.OpenSlide(slide_path)
                patch_sampler_backend = "openslide_level0_true_magnification"
            except Exception as exc:
                self._log(f"Patch sampler OpenSlide init failed: {type(exc).__name__}: {exc}")
                patch_sampler_backend = "none"

        anchor_contexts: List[Dict[str, Any]] = []
        all_llm_requests: List[Dict[str, Any]] = []
        try:

            anchor_jobs: List[Dict[str, Any]] = []
            for pkg in tile_anchor_packages:
                tile_id = str(pkg.get("tile_id", ""))
                tile_bbox_xywh = [int(v) for v in pkg.get("tile_bbox_xywh", [0, 0, level0_w, level0_h])]
                tile_bbox_xywh = self._clip_bbox_xywh(tile_bbox_xywh, level0_w, level0_h)
                coarse_label = str(pkg.get("coarse_label", "other"))
                label_score = float(pkg.get("label_score", 0.0))
                candidate_signals = list(pkg.get("candidate_signals", []))
                sampling_plan = dict(pkg.get("sampling_plan", {}))
                tile_min_side = max(1, min(int(tile_bbox_xywh[2]), int(tile_bbox_xywh[3])))
                default_anchor_crop = int(max(512, round(float(tile_min_side) * 0.25)))
                default_anchor_crop = int(min(int(self.anchor_crop_size_level0), default_anchor_crop))
                anchor_crop_size = int(max(512, sampling_plan.get("anchor_crop_size_level0", default_anchor_crop)))
                patch_crop_size = int(
                    max(256, sampling_plan.get("patch_crop_size_level0", int(self.patch_crop_size_level0)))
                )
                patch_grid = 4
                try:
                    anchor_grid_cfg = sampling_plan.get("anchor_subgrid", [4, 4])
                    if isinstance(anchor_grid_cfg, list) and anchor_grid_cfg:
                        patch_grid = max(2, int(anchor_grid_cfg[0]))
                except Exception:
                    patch_grid = 4

                for anchor in pkg.get("anchors", []):
                    anchor_id = str(anchor.get("anchor_id", ""))
                    evidence_id = str(anchor.get("evidence_id", f"E_{anchor_id}"))
                    abox_raw = [int(v) for v in anchor.get("anchor_bbox_xywh", [0, 0, 1, 1])]
                    focus_center = anchor.get("focus_center_level0", [])
                    if isinstance(focus_center, list) and len(focus_center) >= 2:
                        try:
                            cx = int(focus_center[0])
                            cy = int(focus_center[1])
                        except Exception:
                            cx = int(abox_raw[0] + abox_raw[2] // 2)
                            cy = int(abox_raw[1] + abox_raw[3] // 2)
                    else:
                        cx = int(abox_raw[0] + abox_raw[2] // 2)
                        cy = int(abox_raw[1] + abox_raw[3] // 2)
                    anchor_box = self._center_box_with_size(
                        cx=cx,
                        cy=cy,
                        bw=anchor_crop_size,
                        bh=anchor_crop_size,
                        max_w=level0_w,
                        max_h=level0_h,
                    )
                    anchor_box = self._center_box_with_size_in_outer_bbox(
                        cx=int(anchor_box[0] + anchor_box[2] // 2),
                        cy=int(anchor_box[1] + anchor_box[3] // 2),
                        bw=int(anchor_box[2]),
                        bh=int(anchor_box[3]),
                        outer_xywh=tile_bbox_xywh,
                        max_w=level0_w,
                        max_h=level0_h,
                    )
                    ax, ay, aw, ah = [int(v) for v in anchor_box]
                    patches_per_anchor = int(self._target_patch_count_for_label(coarse_label))

                    anchor_preview = None
                    if os_slide is not None and Image is not None:
                        try:
                            arx, ary, arw, arh = anchor_box
                            anchor_img = os_slide.read_region((arx, ary), 0, (arw, arh)).convert("RGB")
                            anchor_preview_img = anchor_img.resize(
                                (self.real_patch_output_size, self.real_patch_output_size),
                                Image.BILINEAR,
                            )
                            anchor_preview = self._as_uint8_rgb(np.array(anchor_preview_img))
                        except Exception as exc:
                            self._log(
                                f"Anchor preview sampling failed | anchor={anchor_id}: "
                                f"{type(exc).__name__}: {exc}"
                            )

                    patch_prompt = ""
                    patch_prompt_10x = ""
                    low_mag_views_per_anchor = 1
                    if anchor_preview is not None:
                        patch_prompt = self._build_agent_patch_focus_prompt(
                            tile_id=tile_id,
                            anchor_id=anchor_id,
                            coarse_label=coarse_label,
                            patch_count=patches_per_anchor,
                        )
                        patch_prompt_10x = self._build_agent_patch_focus_prompt_10x(
                            tile_id=tile_id,
                            anchor_id=anchor_id,
                            coarse_label=coarse_label,
                            patch_count=low_mag_views_per_anchor,
                        )
                    anchor_jobs.append(
                        {
                            "tile_id": tile_id,
                            "tile_bbox_xywh": list(tile_bbox_xywh),
                            "coarse_label": coarse_label,
                            "label_score": label_score,
                            "candidate_signals": candidate_signals,
                            "sampling_plan": sampling_plan,
                            "anchor_id": anchor_id,
                            "evidence_id": evidence_id,
                            "bbox_norm": list(anchor.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])),
                            "anchor_bbox_xywh": [ax, ay, aw, ah],
                            "patch_crop_size": int(patch_crop_size),
                            "patch_grid": int(patch_grid),
                            "patches_per_anchor": int(patches_per_anchor),
                            "low_mag_views_per_anchor": int(low_mag_views_per_anchor),
                            "anchor_preview": anchor_preview,
                            "patch_prompt": patch_prompt,
                            "patch_prompt_10x": patch_prompt_10x,
                        }
                    )

            sel_job_indices: List[int] = []
            sel_images: List[np.ndarray] = []
            sel_prompts: List[str] = []
            sel_top_ks: List[int] = []
            sel_grids: List[int] = []
            sel10_job_indices: List[int] = []
            sel10_images: List[np.ndarray] = []
            sel10_prompts: List[str] = []
            sel10_top_ks: List[int] = []
            sel10_grids: List[int] = []
            for j, job in enumerate(anchor_jobs):
                if job.get("anchor_preview", None) is None:
                    continue
                p = str(job.get("patch_prompt", "")).strip()
                if not p:
                    continue
                sel_job_indices.append(j)
                sel_images.append(job["anchor_preview"])
                sel_prompts.append(p)
                sel_top_ks.append(int(job.get("patches_per_anchor", 2)))
                sel_grids.append(int(job.get("patch_grid", 4)))
                p10 = str(job.get("patch_prompt_10x", "")).strip()
                if p10:
                    sel10_job_indices.append(j)
                    sel10_images.append(job["anchor_preview"])
                    sel10_prompts.append(p10)
                    sel10_top_ks.append(int(job.get("low_mag_views_per_anchor", 1)))
                    sel10_grids.append(int(job.get("patch_grid", 4)))
            if sel_images:
                batch_size = max(1, int(self.patch_llm_batch_size))
                self._log(
                    f"Agent patch-cell batch select | anchors={len(sel_images)} | batch_size={batch_size}"
                )
                sel_outputs = self._select_multi_cells_with_agent_batch(
                    image_rgbs=sel_images,
                    prompts=sel_prompts,
                    top_ks=sel_top_ks,
                    grids=sel_grids,
                    batch_size=batch_size,
                )
                for i, job_idx in enumerate(sel_job_indices):
                    if i < len(sel_outputs):
                        anchor_jobs[job_idx]["patch_choices"] = sel_outputs[i]
            if sel10_images:
                batch_size = max(1, int(self.patch_llm_batch_size))
                self._log(
                    f"Agent patch-cell(10x) batch select | anchors={len(sel10_images)} | batch_size={batch_size}"
                )
                sel10_outputs = self._select_multi_cells_with_agent_batch(
                    image_rgbs=sel10_images,
                    prompts=sel10_prompts,
                    top_ks=sel10_top_ks,
                    grids=sel10_grids,
                    batch_size=batch_size,
                )
                for i, job_idx in enumerate(sel10_job_indices):
                    if i < len(sel10_outputs):
                        anchor_jobs[job_idx]["patch_choices_10x"] = sel10_outputs[i]

            for job in anchor_jobs:
                tile_id = str(job["tile_id"])
                tile_bbox_xywh = [int(v) for v in job.get("tile_bbox_xywh", [0, 0, level0_w, level0_h])]
                coarse_label = str(job["coarse_label"])
                label_score = float(job["label_score"])
                candidate_signals = list(job["candidate_signals"])
                sampling_plan = dict(job["sampling_plan"])
                anchor_id = str(job["anchor_id"])
                evidence_id = str(job["evidence_id"])
                ax, ay, aw, ah = [int(v) for v in job["anchor_bbox_xywh"]]
                patch_crop_size = int(job["patch_crop_size"])
                patch_grid = int(job["patch_grid"])
                patches_per_anchor = int(job["patches_per_anchor"])
                low_mag_views_per_anchor = int(max(1, job.get("low_mag_views_per_anchor", 1)))
                anchor_preview = job.get("anchor_preview", None)

                patch_choices = list(job.get("patch_choices", []))
                patch_choices_10x = list(job.get("patch_choices_10x", []))
                fallback_img = (
                    anchor_preview
                    if anchor_preview is not None
                    else np.zeros((256, 256, 3), dtype=np.uint8)
                )
                fallback_all = self._fallback_rank_cells(
                    image_rgb=fallback_img,
                    top_k=max(int(patches_per_anchor), int(patch_grid * patch_grid)),
                    grid=patch_grid,
                )

                def _normalize_patch_choices(candidates: Any, target_count: int) -> List[Dict[str, Any]]:
                    choices = list(candidates) if isinstance(candidates, list) else []
                    if (not choices) or self._looks_template_cell_set(choices):
                        return list(fallback_all[: int(target_count)])
                    normalized_choices: List[Dict[str, Any]] = []
                    seen_cells = set()
                    for cell in choices:
                        if not isinstance(cell, dict):
                            continue
                        rr = int(cell.get("row", -1))
                        cc = int(cell.get("col", -1))
                        if rr < 0 or cc < 0 or rr >= patch_grid or cc >= patch_grid:
                            continue
                        key = (rr, cc)
                        if key in seen_cells:
                            continue
                        seen_cells.add(key)
                        normalized_choices.append(cell)
                        if len(normalized_choices) >= int(target_count):
                            break
                    if len(normalized_choices) < int(target_count):
                        seen_cells = {(int(c["row"]), int(c["col"])) for c in normalized_choices}
                        for fc in fallback_all:
                            rr = int(fc.get("row", -1))
                            cc = int(fc.get("col", -1))
                            key = (rr, cc)
                            if rr < 0 or cc < 0 or rr >= patch_grid or cc >= patch_grid or key in seen_cells:
                                continue
                            seen_cells.add(key)
                            normalized_choices.append(fc)
                            if len(normalized_choices) >= int(target_count):
                                break
                    return list(normalized_choices[: int(target_count)])

                patch_choices = _normalize_patch_choices(patch_choices, int(patches_per_anchor))
                patch_choices_10x = _normalize_patch_choices(
                    patch_choices_10x, int(low_mag_views_per_anchor)
                )
                coords_20 = [(int(c.get("row", -1)), int(c.get("col", -1))) for c in patch_choices]
                coords_10 = [(int(c.get("row", -1)), int(c.get("col", -1))) for c in patch_choices_10x]
                if coords_10 == coords_20 and fallback_all:
                    offset = max(1, int(len(fallback_all) // 2))
                    alt_order = list(fallback_all[offset:]) + list(fallback_all[:offset])
                    alt_choices: List[Dict[str, Any]] = []
                    seen_alt = set()
                    for fc in alt_order:
                        if not isinstance(fc, dict):
                            continue
                        rr = int(fc.get("row", -1))
                        cc = int(fc.get("col", -1))
                        key = (rr, cc)
                        if rr < 0 or cc < 0 or rr >= patch_grid or cc >= patch_grid or key in seen_alt:
                            continue
                        seen_alt.add(key)
                        alt_choices.append(fc)
                        if len(alt_choices) >= int(low_mag_views_per_anchor):
                            break
                    if alt_choices:
                        patch_choices_10x = alt_choices

                sampled_views: List[Dict[str, Any]] = []
                view_specs: List[Dict[str, Any]] = []
                for i, cell in enumerate(patch_choices, start=1):
                    view_specs.append(
                        {
                            "view_id": f"V{i}",
                            "cell": cell,
                            "crop_size": int(patch_crop_size),
                            "source_suffix": "",
                        }
                    )
                # Add independently selected low-mag context views (around 10x).
                if patch_choices_10x:
                    low_mag_target = 10.0
                    low_mag_crop_size = int(
                        round(
                            float(base_mag) * float(self.real_patch_output_size) / float(max(1.0, low_mag_target))
                        )
                    )
                    low_mag_crop_size = max(int(patch_crop_size), low_mag_crop_size)
                    tile_min_side = max(1, min(int(tile_bbox_xywh[2]), int(tile_bbox_xywh[3])))
                    low_mag_crop_size = int(max(256, min(low_mag_crop_size, tile_min_side)))
                    if low_mag_crop_size > int(patch_crop_size):
                        for i, cell in enumerate(patch_choices_10x, start=1):
                            low_mag_view_id = f"V{i}_10x"
                            view_specs.append(
                                {
                                    "view_id": low_mag_view_id,
                                    "cell": cell,
                                    "crop_size": int(low_mag_crop_size),
                                    "source_suffix": "_10x_independent",
                                }
                            )

                for view_rank, spec in enumerate(view_specs, start=1):
                    cell = spec["cell"]
                    row = int(cell.get("row", 0))
                    col = int(cell.get("col", 0))
                    crop_size = int(spec.get("crop_size", patch_crop_size))
                    effective_target_mag = max(
                        1.0,
                        min(
                            float(base_mag),
                            float(base_mag) * float(self.real_patch_output_size) / float(max(1, crop_size)),
                        ),
                    )
                    mag = self._format_mag_label(effective_target_mag)
                    cell_cx = int(ax + round((float(col) + 0.5) * float(aw) / float(max(1, patch_grid))))
                    cell_cy = int(ay + round((float(row) + 0.5) * float(ah) / float(max(1, patch_grid))))
                    patch_box = self._center_box_with_size(
                        cx=cell_cx,
                        cy=cell_cy,
                        bw=crop_size,
                        bh=crop_size,
                        max_w=level0_w,
                        max_h=level0_h,
                    )
                    patch_box = self._center_box_with_size_in_outer_bbox(
                        cx=int(patch_box[0] + patch_box[2] // 2),
                        cy=int(patch_box[1] + patch_box[3] // 2),
                        bw=int(patch_box[2]),
                        bh=int(patch_box[3]),
                        outer_xywh=tile_bbox_xywh,
                        max_w=level0_w,
                        max_h=level0_h,
                    )
                    view_id = str(spec.get("view_id", f"V{view_rank}"))
                    source = f"agent_patch_cell_r{row}c{col}{str(spec.get('source_suffix', ''))}"
                    patch_path = ""
                    patch_arr = None
                    if os_slide is not None and Image is not None:
                        try:
                            rx, ry, rw, rh = patch_box
                            patch_img = os_slide.read_region((rx, ry), 0, (rw, rh)).convert("RGB")
                            patch_img_out = patch_img.resize(
                                (self.real_patch_output_size, self.real_patch_output_size),
                                Image.BILINEAR,
                            )
                            patch_arr = self._as_uint8_rgb(np.array(patch_img_out))
                            if patch_dir:
                                patch_name = f"{stem}_{anchor_id}_{view_id}_{mag}.png"
                                patch_path = os.path.join(patch_dir, patch_name)
                                patch_img_out.save(patch_path)
                        except Exception as exc:
                            self._log(
                                f"Patch sampling failed | anchor={anchor_id} view={view_id}: "
                                f"{type(exc).__name__}: {exc}"
                            )

                    req_key = self._patch_request_key(tile_id, anchor_id, view_id)
                    sampled_view = {
                        "view_rank": view_rank,
                        "view_id": view_id,
                        "mag": mag,
                        "effective_target_mag": effective_target_mag,
                        "patch_xywh_level0": patch_box,
                        "patch_path": patch_path,
                        "source": source,
                        "patch_arr": patch_arr,
                        "req_key": req_key,
                        "selected_cell_rc": [row, col],
                        "selection_confidence": float(cell.get("confidence", 0.5)),
                        "selection_rationale": str(cell.get("rationale", "")),
                        "selection_tool_trace": list(cell.get("tool_trace", []))
                        if isinstance(cell.get("tool_trace", []), list)
                        else [],
                    }
                    sampled_views.append(sampled_view)

                    if self.use_llm_patch_describer:
                        if patch_arr is None:
                            if self.patch_description_strict:
                                raise RuntimeError(
                                    "LLM patch description requires a sampled patch image | "
                                    f"anchor={anchor_id} view={view_id}"
                                )
                        else:
                            all_llm_requests.append(
                                {
                                    "req_key": req_key,
                                    "crop": patch_arr,
                                    "category": coarse_label,
                                    "magnification_hint": str(mag),
                                    "tile_id": tile_id,
                                    "anchor_id": anchor_id,
                                    "view_id": view_id,
                                }
                            )

                anchor_contexts.append(
                    {
                        "tile_id": tile_id,
                        "coarse_label": coarse_label,
                        "label_score": label_score,
                        "candidate_signals": candidate_signals,
                        "sampling_plan": sampling_plan,
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "anchor_bbox_xywh": [ax, ay, aw, ah],
                        "bbox_norm": list(job.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])),
                        "sampled_views": sampled_views,
                    }
                )

            llm_desc_map: Dict[str, Dict[str, Any]] = {}
            if self.use_llm_patch_describer and all_llm_requests:
                self._log(
                    f"LLM patch global queue | requests={len(all_llm_requests)} | "
                    f"batch_size={self.patch_llm_batch_size}"
                )
                llm_desc_map = self._describe_patches_with_llm_batch(all_llm_requests)

            for ctx in anchor_contexts:
                tile_id = str(ctx["tile_id"])
                coarse_label = str(ctx["coarse_label"])
                label_score = float(ctx["label_score"])
                candidate_signals = list(ctx["candidate_signals"])
                anchor_id = str(ctx["anchor_id"])
                evidence_id = str(ctx["evidence_id"])
                ax, ay, aw, ah = [int(v) for v in ctx["anchor_bbox_xywh"]]
                sampling_plan_ctx = dict(ctx.get("sampling_plan", {}))
                sampled_views = list(ctx["sampled_views"])

                observations: List[Dict[str, Any]] = []
                anchor_views: List[Dict[str, Any]] = []
                for sv in sampled_views:
                    view_rank = int(sv["view_rank"])
                    view_id = str(sv["view_id"])
                    mag = str(sv["mag"])
                    patch_box = list(sv["patch_xywh_level0"])
                    patch_path = str(sv["patch_path"])
                    source = str(sv["source"])
                    effective_target_mag = float(sv["effective_target_mag"])
                    patch_arr = sv.get("patch_arr")
                    req_key = str(sv.get("req_key", ""))
                    obs_text = ""
                    obs_qc_flags: List[str] = ["none"]
                    llm_obs_conf: Optional[float] = None

                    if self.use_llm_patch_describer:
                        if patch_arr is None:
                            obs_text = (
                                f"{coarse_label} context at {mag} shows tissue morphology requiring "
                                f"anchor-level verification."
                            )
                            obs_qc_flags = ["patch_sampling_failed"]
                        else:
                            llm_desc = dict(llm_desc_map.get(req_key, {}))
                            llm_err = str(llm_desc.get("error", "")).strip()
                            if not llm_err:
                                obs_text = str(llm_desc.get("caption", "")).strip()
                                obs_qc_flags = list(llm_desc.get("qc_flags", ["none"]))
                                try:
                                    llm_obs_conf = float(llm_desc.get("confidence", 0.5))
                                except Exception:
                                    llm_obs_conf = 0.5
                            if not obs_text:
                                if self.patch_description_strict:
                                    raise RuntimeError(
                                        "LLM patch description is empty | "
                                        f"anchor={anchor_id} view={view_id} mag={mag} err={llm_err or 'unknown'}"
                                    )
                                self._log(
                                    f"LLM patch describe fallback->rule | anchor={anchor_id} view={view_id} | "
                                    f"{llm_err or 'empty_caption'}"
                                )
                                obs_text = self._describe_patch(patch_arr, coarse_label)
                                obs_qc_flags = ["llm_failed_fallback_rule"]
                    else:
                        if patch_arr is not None:
                            obs_text = self._describe_patch(patch_arr, coarse_label)
                        else:
                            obs_text = (
                                f"{coarse_label} context at {mag} shows tissue morphology requiring "
                                f"anchor-level verification."
                            )
                        obs_qc_flags = ["none"]

                    if not obs_text:
                        raise RuntimeError(
                            f"Patch description is empty | anchor={anchor_id} view={view_id} mag={mag}"
                        )
                    if llm_obs_conf is not None:
                        obs_conf = round(max(0.01, min(0.99, llm_obs_conf)), 4)
                    else:
                        obs_conf = round(min(0.99, max(0.05, label_score * (0.90 + 0.04 * view_rank))), 4)
                    if (
                        float(self.patch_description_conf_threshold) > 0.0
                        and float(obs_conf) <= float(self.patch_description_conf_threshold)
                    ):
                        self._log(
                            "Drop low-confidence patch description | "
                            f"tile={tile_id} anchor={anchor_id} view={view_id} conf={obs_conf:.4f} "
                            f"threshold={self.patch_description_conf_threshold:.2f}"
                        )
                        continue
                    observation = {
                        "view_id": view_id,
                        "mag": mag,
                        "obs": obs_text,
                        "qc_flags": obs_qc_flags,
                        "conf": obs_conf,
                        "caption_source": "llm_patch_describer" if self.use_llm_patch_describer else "rule_based",
                    }
                    patch_view = {
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "view_id": view_id,
                        "mag": mag,
                        "patch_xywh_level0": patch_box,
                        "patch_path": patch_path,
                        "source": source,
                        "target_magnification": round(float(effective_target_mag), 2),
                        "base_magnification": round(float(base_mag), 2),
                        "output_size_px": int(self.real_patch_output_size),
                        "selected_cell_rc": list(sv.get("selected_cell_rc", [])),
                        "selection_confidence": float(sv.get("selection_confidence", 0.5)),
                        "selection_rationale": str(sv.get("selection_rationale", "")),
                        "selection_tool_trace": list(sv.get("selection_tool_trace", []))
                        if isinstance(sv.get("selection_tool_trace", []), list)
                        else [],
                    }
                    observations.append(observation)
                    anchor_views.append(patch_view)
                    patch_views_catalog.append(patch_view)

                if not anchor_views:
                    self._log(
                        "Skip anchor evidence: no patch description above threshold | "
                        f"tile={tile_id} anchor={anchor_id} threshold={self.patch_description_conf_threshold:.2f}"
                    )
                    continue

                screening_items: List[Dict[str, Any]] = []
                present_signal_names: List[str] = []
                missing_info: List[str] = []
                signal_present_threshold = 0.50
                for sig in candidate_signals:
                    signal_id = str(sig.get("signal_id", "SIG999__OTHER_PATTERN"))
                    signal_name = str(sig.get("signal_name", "other_pattern"))
                    base_conf = float(sig.get("confidence", 0.0))
                    if base_conf >= signal_present_threshold:
                        status = "present"
                        confidence = round(min(0.99, base_conf), 4)
                        supporting_views = [v["view_id"] for v in anchor_views[:2]]
                        rationale = (
                            f"Candidate signal '{signal_name}' aligns with anchor morphology at sampled views."
                        )
                        present_signal_names.append(signal_name)
                    else:
                        status = "absent"
                        confidence = round(max(0.01, base_conf), 4)
                        supporting_views = []
                        rationale = (
                            f"No strong visual support for candidate signal '{signal_name}' in sampled views."
                        )

                    screening_items.append(
                        {
                            "signal_id": signal_id,
                            "status": status,
                            "confidence": confidence,
                            "supporting_views": supporting_views,
                            "rationale": rationale,
                        }
                    )
                    if status == "present":
                        polarity = "supports"
                    else:
                        polarity = "contradicts"
                    signal_evidence_edges.append(
                        {
                            "signal_id": signal_id,
                            "signal_name": signal_name,
                            "evidence_id": evidence_id,
                            "anchor_id": anchor_id,
                            "tile_id": tile_id,
                            "polarity": polarity,
                            "strength": round(float(confidence), 4),
                            "status": status,
                            "supporting_views": list(supporting_views),
                        }
                    )

                if present_signal_names:
                    evidence_summary = "Anchor supports: " + ", ".join(sorted(set(present_signal_names))) + "."
                else:
                    evidence_summary = (
                        f"Anchor-level review is {coarse_label}-leaning but lacks definitive positive signal support."
                    )
                anchor_description = self._build_anchor_description(
                    coarse_label=coarse_label,
                    evidence_summary=evidence_summary,
                    observations=observations,
                )
                anchor_qc_flags: List[str] = []
                for o in observations:
                    for q in o.get("qc_flags", []):
                        qn = str(q).strip().lower()
                        if not qn or qn == "none" or qn in anchor_qc_flags:
                            continue
                        anchor_qc_flags.append(qn)
                if not anchor_qc_flags:
                    anchor_qc_flags = ["none"]

                signal_screenings.append(
                    {
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "signal_screening": screening_items,
                        "evidence_summary": evidence_summary,
                        "missing_info": missing_info,
                    }
                )

                obs_by_view = {o["view_id"]: o for o in observations}
                evidence_views = []
                for v in anchor_views:
                    o = obs_by_view.get(v["view_id"], {})
                    evidence_views.append(
                        {
                            "view_id": v["view_id"],
                            "mag": v["mag"],
                            "patch_xywh_level0": v["patch_xywh_level0"],
                            "patch_path": v["patch_path"],
                            "caption": str(o.get("obs", "")),
                            "qc_flags": list(o.get("qc_flags", ["none"])),
                            "selected_cell_rc": list(v.get("selected_cell_rc", [])),
                            "selection_confidence": float(v.get("selection_confidence", 0.5)),
                            "selection_rationale": str(v.get("selection_rationale", "")),
                        }
                    )

                signal_conf = [float(s.get("confidence", 0.0)) for s in screening_items]
                evidence_confidence = round(max(signal_conf), 4) if signal_conf else round(label_score, 4)
                evidence_items.append(
                    {
                        "evidence_id": evidence_id,
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "anchor_id": anchor_id,
                        "target_type": f"{coarse_label.upper()}_PATTERN",
                        "views": evidence_views,
                        "summary": evidence_summary,
                        "anchor_description": anchor_description,
                        "anchor_qc_flags": anchor_qc_flags,
                        "confidence": evidence_confidence,
                    }
                )

                anchors_catalog.append(
                    {
                        "slide_id": slide_id,
                        "tile_id": tile_id,
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "coarse_label": coarse_label,
                        "label_score": round(label_score, 4),
                        "anchor_bbox_xywh": [ax, ay, aw, ah],
                        "bbox_norm": list(ctx.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])),
                        "anchor_description": anchor_description,
                        "anchor_qc_flags": anchor_qc_flags,
                        "candidate_signals": candidate_signals,
                        "sampling_plan": {
                            "policy": "agent_tool_three_stage",
                            "anchor_bbox_level0": [ax, ay, aw, ah],
                            "tile_subgrid": [4, 4],
                            "anchor_subgrid": [4, 4],
                            "anchor_crop_size_level0": int(
                                sampling_plan_ctx.get("anchor_crop_size_level0", int(self.anchor_crop_size_level0))
                            ),
                            "patch_crop_size_level0": int(
                                sampling_plan_ctx.get("patch_crop_size_level0", int(self.patch_crop_size_level0))
                            ),
                            "views": [
                                {
                                    "view_id": v["view_id"],
                                    "mag": v["mag"],
                                    "patch_xywh_level0": v["patch_xywh_level0"],
                                    "source": v["source"],
                                    "selected_cell_rc": list(v.get("selected_cell_rc", [])),
                                }
                                for v in anchor_views
                            ],
                        },
                    }
                )
        except Exception:
            # Save a best-effort overlay for debugging even when strict stage-1 build fails.
            if (
                out_dir
                and thumb_rgb is not None
                and isinstance(thumbnail_meta, dict)
                and isinstance(tiles_catalog, list)
                and Image is not None
            ):
                provisional_views: List[Dict[str, Any]] = list(patch_views_catalog)
                if not provisional_views:
                    for ctx in anchor_contexts:
                        tile_id = str(ctx.get("tile_id", ""))
                        anchor_id = str(ctx.get("anchor_id", ""))
                        for sv in ctx.get("sampled_views", []):
                            if not isinstance(sv, dict):
                                continue
                            box = sv.get("patch_xywh_level0", [])
                            if not isinstance(box, list) or len(box) < 4:
                                continue
                            provisional_views.append(
                                {
                                    "tile_id": tile_id,
                                    "anchor_id": anchor_id,
                                    "view_id": str(sv.get("view_id", "")),
                                    "patch_xywh_level0": [int(v) for v in box[:4]],
                                }
                            )
                if provisional_views:
                    try:
                        self._write_selected_patch_overlay_debug_image(
                            out_dir=out_dir,
                            slide_id=slide_id,
                            thumb_rgb=thumb_rgb,
                            thumbnail_meta=thumbnail_meta,
                            tiles_catalog=tiles_catalog,
                            patch_views_catalog=provisional_views,
                            anchors_catalog=anchors_catalog,
                        )
                    except Exception as overlay_exc:
                        self._log(
                            f"Overlay write on failure skipped | {type(overlay_exc).__name__}: {overlay_exc}"
                        )
            raise
        finally:
            if os_slide is not None:
                os_slide.close()

        return {
            "anchors_catalog": anchors_catalog,
            "patch_views_catalog": patch_views_catalog,
            "evidence_items": evidence_items,
            "signal_screenings": signal_screenings,
            "signal_evidence_edges": signal_evidence_edges,
            "patch_sampler_backend": patch_sampler_backend,
        }

    def _load_lowres_with_openslide(self, slide_path: str) -> tuple[np.ndarray, np.ndarray]:
        if openslide is None:
            raise RuntimeError("openslide-python is required for OpenSlide fallback decoding.")
        if Image is None:
            raise RuntimeError("Pillow is required to convert OpenSlide thumbnail outputs.")

        self._log("Loading low-resolution SVS views via OpenSlide thumbnail fallback.")
        with openslide.OpenSlide(slide_path) as os_slide:
            w0, h0 = int(os_slide.dimensions[0]), int(os_slide.dimensions[1])
            low_wh = self._fit_size(w0, h0, max_side=1024)
            med_wh = self._fit_size(w0, h0, max_side=2048)

            low_img = os_slide.get_thumbnail(low_wh).convert("RGB")
            med_img = os_slide.get_thumbnail(med_wh).convert("RGB")
            lowest = self._as_uint8_rgb(np.array(low_img))
            medium = self._as_uint8_rgb(np.array(med_img))
            self._log(
                f"OpenSlide thumbnails | level0=({w0},{h0}) | low_wh={low_wh} | med_wh={med_wh}"
            )
            return lowest, medium

    def _load_lowres_single_level_with_openslide(
        self,
        slide_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create bounded thumbnails without materializing a single-level WSI."""
        if openslide is None:
            raise RuntimeError("openslide-python is required for chunked TIFF decoding.")
        if Image is None:
            raise RuntimeError("Pillow is required for chunked TIFF thumbnail generation.")

        self._log("Loading single-level TIFF as chunked OpenSlide regions.")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        chunk_side = 2048
        with openslide.OpenSlide(slide_path) as os_slide:
            w0, h0 = int(os_slide.dimensions[0]), int(os_slide.dimensions[1])
            low_wh = self._fit_size(w0, h0, max_side=1024)
            med_wh = self._fit_size(w0, h0, max_side=2048)
            med_w, med_h = int(med_wh[0]), int(med_wh[1])
            medium_img = Image.new("RGB", (med_w, med_h), (255, 255, 255))

            for y0 in range(0, h0, chunk_side):
                y1 = min(h0, y0 + chunk_side)
                out_y0 = int(round(float(y0) * med_h / max(1, h0)))
                out_y1 = int(round(float(y1) * med_h / max(1, h0)))
                for x0 in range(0, w0, chunk_side):
                    x1 = min(w0, x0 + chunk_side)
                    out_x0 = int(round(float(x0) * med_w / max(1, w0)))
                    out_x1 = int(round(float(x1) * med_w / max(1, w0)))
                    out_size = (max(1, out_x1 - out_x0), max(1, out_y1 - out_y0))
                    region = os_slide.read_region(
                        (x0, y0),
                        0,
                        (x1 - x0, y1 - y0),
                    ).convert("RGB")
                    if region.size != out_size:
                        region = region.resize(out_size, resampling)
                    medium_img.paste(region, (out_x0, out_y0))

            low_img = medium_img.resize(low_wh, resampling)
            lowest = self._as_uint8_rgb(np.asarray(low_img))
            medium = self._as_uint8_rgb(np.asarray(medium_img))
            self._log(
                "Chunked OpenSlide thumbnails "
                f"| level0=({w0},{h0}) | chunks={chunk_side} | low_wh={low_wh} | med_wh={med_wh}"
            )
            return lowest, medium

    def _load_lowres_with_pillow(
        self,
        slide_path: str,
        level_dims: List[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if Image is None:
            raise RuntimeError("Pillow is required for low-resolution fallback decoding.")

        Image.MAX_IMAGE_PIXELS = None
        self._log("Loading low-resolution SVS frames via Pillow.")
        with Image.open(slide_path) as im:
            n_frames = int(getattr(im, "n_frames", 1))
            frame_sizes: List[tuple[int, int]] = []
            for fi in range(n_frames):
                im.seek(fi)
                frame_sizes.append((int(im.size[0]), int(im.size[1])))
            self._log(f"Pillow frames: {n_frames} | sizes={frame_sizes}")

            def _find_best_frame(target_wh: tuple[int, int]) -> int:
                tw, th = target_wh
                best_idx = 0
                best_score = float("inf")
                for idx, (w, h) in enumerate(frame_sizes):
                    score = abs(w - tw) + abs(h - th)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                return best_idx

            low_target = level_dims[-1]
            med_target = level_dims[-2] if len(level_dims) > 1 else low_target
            low_idx = _find_best_frame(low_target)
            med_idx = _find_best_frame(med_target)
            self._log(
                f"Selected Pillow frames | low_idx={low_idx} target={low_target} | med_idx={med_idx} target={med_target}"
            )

            im.seek(low_idx)
            lowest = self._as_uint8_rgb(np.array(im.convert("RGB")))
            im.seek(med_idx)
            medium = self._as_uint8_rgb(np.array(im.convert("RGB")))
        return lowest, medium

    def _segment_tissue(self, low_rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for tissue segmentation.")
        hsv = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        val = hsv[..., 2]
        mask = ((sat > 15) | (val < 225)).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _extract_region_boxes(self, mask: np.ndarray) -> List[Dict[str, int]]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for region extraction.")
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total_px = int(mask.shape[0] * mask.shape[1])
        min_area = max(400, int(total_px * 0.0025))
        regions: List[Dict[str, int]] = []

        for label in range(1, n_labels):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < min_area:
                continue
            regions.append({"x": x, "y": y, "w": w, "h": h, "area": area})

        if not regions:
            h, w = mask.shape[:2]
            regions = [{"x": 0, "y": 0, "w": w, "h": h, "area": w * h}]

        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions

    @staticmethod
    def _split_box(box: Dict[str, int]) -> List[Dict[str, int]]:
        x, y, w, h, area = box["x"], box["y"], box["w"], box["h"], box["area"]
        if w >= h and w >= 64:
            w1 = w // 2
            w2 = w - w1
            return [
                {"x": x, "y": y, "w": w1, "h": h, "area": w1 * h},
                {"x": x + w1, "y": y, "w": w2, "h": h, "area": w2 * h},
            ]
        if h >= 64:
            h1 = h // 2
            h2 = h - h1
            return [
                {"x": x, "y": y, "w": w, "h": h1, "area": w * h1},
                {"x": x, "y": y + h1, "w": w, "h": h2, "area": w * h2},
            ]
        return [{"x": x, "y": y, "w": w, "h": h, "area": area}]

    def _ensure_target_region_count(self, regions: List[Dict[str, int]]) -> List[Dict[str, int]]:
        out = list(regions)
        while len(out) < self.target_regions:
            out.sort(key=lambda r: r["area"], reverse=True)
            seed = out[0]
            split = self._split_box(seed)
            if len(split) < 2:
                break
            out = split + out[1:]
        out.sort(key=lambda r: r["area"], reverse=True)
        return out[: self.target_regions]

    def _compute_features(self, crop: np.ndarray) -> Dict[str, float]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for feature extraction.")

        work = crop
        h0, w0 = int(crop.shape[0]), int(crop.shape[1])
        max_side = max(h0, w0)
        if max_side > 768:
            scale = 768.0 / float(max_side)
            new_w = max(1, int(round(w0 * scale)))
            new_h = max(1, int(round(h0 * scale)))
            work = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)

        white_ratio = float(np.mean(np.all(work > 230, axis=2)))
        purple_ratio = float(np.mean((work[..., 2] > work[..., 1] + 8) & (work[..., 2] > work[..., 0] + 8)))
        pink_ratio = float(np.mean((work[..., 0] > work[..., 1]) & (work[..., 0] > work[..., 2])))
        dark_ratio = float(np.mean(gray < 90))
        texture = float(np.var(lap))
        sat_mean = float(np.mean(hsv[..., 1]))
        tissue_ratio = float(np.mean(gray < 235))

        # Heuristic nuclei candidates from H&E-like basophilic/purple signal.
        r = work[..., 0].astype(np.int16)
        g = work[..., 1].astype(np.int16)
        b = work[..., 2].astype(np.int16)
        hue = hsv[..., 0].astype(np.int16)
        sat = hsv[..., 1].astype(np.int16)
        val = hsv[..., 2].astype(np.int16)
        nuclei_hsv = (hue >= 100) & (hue <= 170) & (sat >= 35) & (val <= 210)
        nuclei_rgb = (b > g + 15) & (b > r + 8) & (gray < 175)
        nuclei_mask = (nuclei_hsv | nuclei_rgb).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
        nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas: List[float] = []
        circularities: List[float] = []
        small_round_count = 0
        nuclei_area_sum = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 8.0 or area > 3000.0:
                continue
            peri = float(cv2.arcLength(cnt, True))
            circ = (4.0 * np.pi * area / (peri * peri)) if peri > 1e-6 else 0.0
            circ = float(max(0.0, min(1.0, circ)))
            areas.append(area)
            nuclei_area_sum += area
            circularities.append(circ)
            if area <= 48.0 and circ >= 0.70:
                small_round_count += 1

        nuclei_component_count = int(len(areas))
        if nuclei_component_count > 0:
            area_mean = float(np.mean(areas))
            area_cv = float(np.std(areas) / max(1e-6, area_mean))
            circularity_mean = float(np.mean(circularities))
            irregularity_mean = float(np.mean([1.0 - c for c in circularities]))
            small_round_ratio = float(small_round_count / nuclei_component_count)
        else:
            area_mean = 0.0
            area_cv = 0.0
            circularity_mean = 0.0
            irregularity_mean = 0.0
            small_round_ratio = 0.0

        nuclei_ratio = float(nuclei_area_sum / max(1.0, float(work.shape[0] * work.shape[1])))
        tissue_px = float(max(1.0, np.sum(gray < 235)))
        nuclei_density_per_mpx = float(nuclei_component_count / (tissue_px / 1e6))

        eosinophilic_ratio = float(np.mean((r > g + 10) & (g > b + 3) & (gray > 85)))

        return {
            "white_ratio": white_ratio,
            "purple_ratio": purple_ratio,
            "pink_ratio": pink_ratio,
            "dark_ratio": dark_ratio,
            "texture": texture,
            "sat_mean": sat_mean,
            "tissue_ratio": tissue_ratio,
            "nuclei_ratio": nuclei_ratio,
            "nuclei_component_count": float(nuclei_component_count),
            "nuclei_density_per_mpx": nuclei_density_per_mpx,
            "nuclei_area_mean": area_mean,
            "nuclei_area_cv": area_cv,
            "nuclei_circularity_mean": circularity_mean,
            "nuclei_irregularity_mean": irregularity_mean,
            "small_round_nuclei_ratio": small_round_ratio,
            "eosinophilic_ratio": eosinophilic_ratio,
        }

    def _classify_region(self, feat: Dict[str, float]) -> Dict[str, str]:
        white = feat["white_ratio"]
        purple = feat["purple_ratio"]
        pink = feat["pink_ratio"]
        dark = feat["dark_ratio"]
        texture = feat["texture"]

        if white > 0.62 and texture < 150:
            return {
                "category": "adipose",
                "semantic_tag": "Adipose-Rich Tissue",
                "description": "Large pale spaces with sparse nuclei, compatible with adipose-rich stroma.",
            }
        if pink > 0.45 and texture < 220:
            return {
                "category": "necrosis",
                "semantic_tag": "Necrotic/Eosinophilic Debris Candidate",
                "description": "Eosinophilic low-structure area suggestive of necrotic or proteinaceous debris.",
            }
        if purple > 0.22 or dark > 0.28:
            if texture > 260:
                return {
                    "category": "high_grade",
                    "semantic_tag": "High-Cellularity Carcinoma Candidate",
                    "description": "Densely basophilic hypercellular field suspicious for high-grade carcinoma.",
                }
            return {
                "category": "carcinoma",
                "semantic_tag": "Invasive Carcinoma Candidate",
                "description": "Hyperchromatic epithelial-rich region suspicious for invasive carcinoma.",
            }
        return {
            "category": "stroma",
            "semantic_tag": "Fibro-Stromal Region",
            "description": "Moderate cellular fibrous tissue, favoring reactive stromal background.",
        }

    def _describe_patch(self, crop: np.ndarray, category: str) -> str:
        feat = self._compute_features(crop)
        purple = float(feat["purple_ratio"])
        texture = float(feat["texture"])
        white = float(feat["white_ratio"])
        pink = float(feat["pink_ratio"])
        nuclei_ratio = float(feat["nuclei_ratio"])
        nuclei_count = int(round(float(feat["nuclei_component_count"])))
        nuclei_density = float(feat["nuclei_density_per_mpx"])
        nuclei_cv = float(feat["nuclei_area_cv"])
        irregularity = float(feat["nuclei_irregularity_mean"])
        circularity = float(feat["nuclei_circularity_mean"])
        small_round = float(feat["small_round_nuclei_ratio"])
        eosinophilic = float(feat["eosinophilic_ratio"])

        if nuclei_count < 40 and nuclei_ratio < 0.08:
            nuc_density = "mild nuclear crowding"
        elif nuclei_ratio >= 0.16 or nuclei_density >= 2800:
            nuc_density = "marked nuclear crowding"
        elif nuclei_ratio >= 0.09 or nuclei_density >= 1500:
            nuc_density = "moderate nuclear crowding"
        else:
            nuc_density = "mild nuclear crowding"

        if texture >= 320:
            arch_complexity = "high architectural complexity"
        elif texture >= 220:
            arch_complexity = "intermediate architectural complexity"
        else:
            arch_complexity = "relatively simple architecture"

        pleomorphism_score = 0
        sufficient_nuclear_yield = nuclei_ratio >= 0.06 and nuclei_count >= 120
        if sufficient_nuclear_yield:
            if nuclei_cv >= 1.10:
                pleomorphism_score += 1
            if irregularity >= 0.45:
                pleomorphism_score += 1
            if category in {"high_grade", "carcinoma"} and purple >= 0.23:
                pleomorphism_score += 1
        if not sufficient_nuclear_yield:
            pleomorphism = "pleomorphism not reliably assessable (low nuclear yield)"
        elif pleomorphism_score >= 2:
            pleomorphism = "high pleomorphism proxy"
        elif pleomorphism_score == 1:
            pleomorphism = "moderate pleomorphism proxy"
        else:
            pleomorphism = "low pleomorphism proxy"

        findings: List[str] = []
        if category in {"high_grade", "carcinoma"}:
            if (purple > 0.24 and texture > 200) or nuclei_ratio > 0.16:
                findings.append(
                    "Epithelial-dominant hyperchromatic field with crowded atypical nuclei and infiltrative proliferative pattern."
                )
            else:
                findings.append(
                    "Atypical epithelial proliferation is present with focal nuclear hyperchromasia."
                )
        elif category == "adipose" or white > 0.65:
            findings.append(
                "Adipose-rich field with broad clear adipocytic spaces and sparse bland nuclei."
            )
        elif category == "necrosis" or (pink > 0.42 and texture < 220):
            findings.append(
                "Eosinophilic acellular debris with architectural loss, compatible with necrotic change."
            )
        else:
            findings.append(
                "Stromal-predominant patch with fibrocollagenous background and no overt high-grade atypia."
            )

        if small_round >= 0.55 and nuclei_ratio >= 0.05:
            findings.append("Background is enriched with small round nuclei, suggestive of lymphocyte-rich inflammation.")
        if white > 0.55:
            findings.append("Prominent clear spaces are present.")
        if pink > 0.48 or eosinophilic > 0.30:
            findings.append("Eosinophilic stromal matrix/debris is prominent.")

        findings.append(f"Morphologic impression: {nuc_density}, {pleomorphism}, and {arch_complexity}.")
        findings.append(
            "Quantified cues: "
            f"nuclei_ratio={nuclei_ratio:.3f}, nuclei_density={nuclei_density:.0f}/MPix, "
            f"nuclei_count={nuclei_count}, "
            f"nuclei_area_cv={nuclei_cv:.2f}, nuclei_circularity={circularity:.2f}, "
            f"nuclei_irregularity={irregularity:.2f}, texture={texture:.1f}."
        )
        return " ".join(findings)

    def _build_global_metrics(
        self,
        regions: List[Dict[str, Any]],
        tissue_ratio: float,
        slide_dims: List[int],
        raw_region_count: int,
        selected_region_count: int,
    ) -> Dict[str, Any]:
        area_by_category: Dict[str, int] = {}
        for r in regions:
            cat = r.get("inferred_category", "stroma")
            area = int(r["coordinates"][2] * r["coordinates"][3])
            area_by_category[cat] = area_by_category.get(cat, 0) + area

        carcinoma_area = area_by_category.get("carcinoma", 0) + area_by_category.get("high_grade", 0)
        total_area = max(1, sum(area_by_category.values()))
        carcinoma_ratio = carcinoma_area / total_area
        high_grade_ratio = area_by_category.get("high_grade", 0) / total_area
        necrosis_ratio = area_by_category.get("necrosis", 0) / total_area
        adipose_ratio = area_by_category.get("adipose", 0) / total_area
        stroma_ratio = area_by_category.get("stroma", 0) / total_area
        slide_area = max(1, int(slide_dims[0]) * int(slide_dims[1]))
        region_coverage_ratio = min(1.0, float(total_area) / float(slide_area))
        malignant_region_count = sum(1 for r in regions if r.get("inferred_category") in {"carcinoma", "high_grade"})
        high_grade_present = area_by_category.get("high_grade", 0) > 0
        dominant_category = max(area_by_category.items(), key=lambda x: x[1])[0] if area_by_category else "unknown"
        suspicion_score = (
            0.45 * carcinoma_ratio
            + 0.35 * high_grade_ratio
            + 0.10 * necrosis_ratio
            + 0.10 * min(1.0, malignant_region_count / 3.0)
        )
        suspicion_level = self._score_to_suspicion_level(suspicion_score)

        area_ratio_by_category = {k: round(v / total_area, 4) for k, v in area_by_category.items()}
        return {
            "tissue_ratio": round(float(tissue_ratio), 4),
            "raw_region_count": int(raw_region_count),
            "selected_region_count": int(selected_region_count),
            "region_coverage_ratio": round(region_coverage_ratio, 4),
            "dominant_category": dominant_category,
            "malignant_region_count": int(malignant_region_count),
            "high_grade_present": bool(high_grade_present),
            "carcinoma_ratio": round(carcinoma_ratio, 4),
            "high_grade_ratio": round(high_grade_ratio, 4),
            "necrosis_ratio": round(necrosis_ratio, 4),
            "adipose_ratio": round(adipose_ratio, 4),
            "stroma_ratio": round(stroma_ratio, 4),
            "suspicion_score": round(float(suspicion_score), 4),
            "suspicion_level": suspicion_level,
            "area_by_category": area_by_category,
            "area_ratio_by_category": area_ratio_by_category,
        }

    @staticmethod
    def _score_to_suspicion_level(score: float) -> str:
        s = float(score)
        if s >= 0.55:
            return "high"
        if s >= 0.28:
            return "moderate"
        return "low"

    @staticmethod
    def _fuse_carcinoma_ratio(region_ratio: float, tile_ratio: float) -> float:
        rr = max(0.0, min(1.0, float(region_ratio)))
        tr = max(0.0, min(1.0, float(tile_ratio)))
        # Region ratio is more specific; tile ratio is more sensitive.
        blended = 0.55 * rr + 0.45 * tr
        return max(rr, min(1.0, blended))

    def _apply_hybrid_global_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(metrics)
        region_ratio = float(out.get("carcinoma_ratio", 0.0))
        tile_ratio = float(out.get("tile_tumor_ratio", 0.0))
        high_grade_ratio = float(out.get("high_grade_ratio", 0.0))
        necrosis_ratio = float(out.get("necrosis_ratio", 0.0))
        malignant_region_count = float(out.get("malignant_region_count", 0.0))

        hybrid_ratio = self._fuse_carcinoma_ratio(region_ratio, tile_ratio)
        out["carcinoma_ratio_region"] = round(region_ratio, 4)
        out["carcinoma_ratio_tile"] = round(tile_ratio, 4)
        out["carcinoma_ratio"] = round(hybrid_ratio, 4)
        out["ratio_fusion_policy"] = "hybrid_region_tile_v1"

        region_score = float(out.get("suspicion_score", 0.0))
        hybrid_score = (
            0.45 * hybrid_ratio
            + 0.35 * high_grade_ratio
            + 0.10 * necrosis_ratio
            + 0.10 * min(1.0, malignant_region_count / 3.0)
        )
        out["suspicion_score_region"] = round(region_score, 4)
        out["suspicion_score"] = round(float(hybrid_score), 4)
        out["suspicion_level"] = self._score_to_suspicion_level(float(hybrid_score))
        return out

    def _build_global_summary(self, metrics: Dict[str, Any], region_count: int) -> str:
        tissue_ratio = float(metrics.get("tissue_ratio", 0.0))
        carcinoma_ratio = float(metrics.get("carcinoma_ratio", 0.0))
        carcinoma_ratio_region = float(metrics.get("carcinoma_ratio_region", carcinoma_ratio))
        carcinoma_ratio_tile = float(metrics.get("carcinoma_ratio_tile", metrics.get("tile_tumor_ratio", 0.0)))
        high_grade_present = bool(metrics.get("high_grade_present", False))
        dominant_category = str(metrics.get("dominant_category", "unknown"))
        suspicion_level = str(metrics.get("suspicion_level", "low"))
        high_grade_note = " High-grade-like regions are present." if high_grade_present else ""
        tumor_size_est = metrics.get("tumor_size_estimation", {})
        size_note = ""
        if isinstance(tumor_size_est, dict) and bool(tumor_size_est.get("available", False)):
            feret_mm = tumor_size_est.get("max_feret_mm", None)
            principal_mm = tumor_size_est.get("principal_axis_mm", None)
            feret_l0 = tumor_size_est.get("max_feret_level0_px", None)
            principal_l0 = tumor_size_est.get("principal_axis_level0_px", None)
            if feret_mm is not None and principal_mm is not None:
                size_note = (
                    f" Quantified tumor size estimate (thumbnail tumor mask): "
                    f"max Feret diameter≈{float(feret_mm):.2f} mm; principal axis≈{float(principal_mm):.2f} mm."
                )
            elif feret_l0 is not None and principal_l0 is not None:
                size_note = (
                    f" Quantified tumor size estimate (mpp unavailable): "
                    f"max Feret≈{float(feret_l0):.0f} px@L0; principal axis≈{float(principal_l0):.0f} px@L0."
                )

        if suspicion_level == "high":
            impression = "Overall pattern is suspicious for invasive carcinoma with proliferative activity."
        elif suspicion_level == "moderate":
            impression = "Focal malignant-appearing epithelial proliferation is present; invasive carcinoma remains likely."
        else:
            impression = "No dominant malignant morphology at global scale; correlation with higher-magnification review is required."

        return (
            f"Real SVS-derived global summary: tissue coverage={tissue_ratio:.2%}; "
            f"detected {region_count} tissue ROIs; dominant category={dominant_category}; "
            f"carcinoma-like area ratio={carcinoma_ratio:.2%} "
            f"(region={carcinoma_ratio_region:.2%}, tile={carcinoma_ratio_tile:.2%}). "
            f"{impression}{high_grade_note}{size_note}"
        )

    def _build_regions_from_stage1_outputs(
        self,
        slide_id: str,
        tiles_catalog: List[Dict[str, Any]],
        anchors_catalog: List[Dict[str, Any]],
        patch_views_catalog: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        def _coarse_to_region_category(label: Any) -> str:
            key = str(label or "").strip().lower()
            if key in {"tumor", "carcinoma"}:
                return "carcinoma"
            if key in {"high_grade"}:
                return "high_grade"
            if key in {"necrosis", "adipose", "stroma"}:
                return key
            return "stroma"

        def _view_rank(view_id: str) -> int:
            m = re.search(r"(\d+)$", str(view_id or ""))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return 10_000
            return 10_000

        tiles_by_id = {str(t.get("tile_id", "")): t for t in tiles_catalog if str(t.get("tile_id", ""))}
        views_by_anchor: Dict[str, List[Dict[str, Any]]] = {}
        for pv in patch_views_catalog or []:
            if not isinstance(pv, dict):
                continue
            aid = str(pv.get("anchor_id", ""))
            if not aid:
                continue
            views_by_anchor.setdefault(aid, []).append(pv)
        for aid, views in views_by_anchor.items():
            views.sort(key=lambda v: _view_rank(str(v.get("view_id", ""))))

        caption_by_anchor_view: Dict[Tuple[str, str], str] = {}
        anchor_desc_by_anchor: Dict[str, str] = {}
        summary_by_anchor: Dict[str, str] = {}
        for ev in evidence_items or []:
            if not isinstance(ev, dict):
                continue
            aid = str(ev.get("anchor_id", ""))
            if not aid:
                continue
            anchor_desc = str(ev.get("anchor_description", "")).strip()
            if anchor_desc and aid not in anchor_desc_by_anchor:
                anchor_desc_by_anchor[aid] = anchor_desc
            summary = str(ev.get("summary", "")).strip()
            if summary and aid not in summary_by_anchor:
                summary_by_anchor[aid] = summary
            for v in ev.get("views", []):
                if not isinstance(v, dict):
                    continue
                vid = str(v.get("view_id", ""))
                cap = str(v.get("caption", "")).strip()
                if vid and cap:
                    caption_by_anchor_view[(aid, vid)] = cap

        regions: List[Dict[str, Any]] = []
        for anchor in anchors_catalog or []:
            if not isinstance(anchor, dict):
                continue
            anchor_id = str(anchor.get("anchor_id", "")).strip()
            if not anchor_id:
                continue
            tile_id = str(anchor.get("tile_id", "")).strip()
            tile = tiles_by_id.get(tile_id, {})
            coarse_label = str(anchor.get("coarse_label", tile.get("coarse_label", "stroma")))
            inferred_category = _coarse_to_region_category(coarse_label)
            abox = anchor.get("anchor_bbox_xywh", [])
            if not (isinstance(abox, list) and len(abox) >= 4):
                continue
            ax, ay, aw, ah = [int(v) for v in abox[:4]]
            if aw <= 0 or ah <= 0:
                continue
            region_coords = [ax, ay, aw, ah]
            semantic_tag = f"Anchor Region ({coarse_label})"
            region_desc = anchor_desc_by_anchor.get(anchor_id, "").strip()
            if not region_desc:
                region_desc = summary_by_anchor.get(anchor_id, "").strip()
            if not region_desc:
                region_desc = (
                    f"Anchor-centered region from tile {tile_id or 'unknown'} with "
                    f"{coarse_label}-leaning morphology."
                )

            patches: List[Dict[str, Any]] = []
            for idx, pv in enumerate(views_by_anchor.get(anchor_id, []), start=1):
                view_id = str(pv.get("view_id", f"V{idx}"))
                pbox = pv.get("patch_xywh_level0", [])
                if not (isinstance(pbox, list) and len(pbox) >= 4):
                    continue
                px, py, pw, ph = [int(v) for v in pbox[:4]]
                if pw <= 0 or ph <= 0:
                    continue
                patch_desc = caption_by_anchor_view.get((anchor_id, view_id), "").strip()
                if not patch_desc:
                    patch_desc = str(pv.get("selection_rationale", "")).strip()
                if not patch_desc:
                    patch_desc = f"Patch view {view_id} sampled from anchor {anchor_id}."
                patches.append(
                    {
                        "patch_id": f"{anchor_id}_{view_id}",
                        "coordinates": [px, py, pw, ph],
                        "description": patch_desc,
                        "semantic_tag": f"{coarse_label}_patch",
                    }
                )

            regions.append(
                {
                    "region_id": anchor_id,
                    "coordinates": region_coords,
                    "semantic_tag": semantic_tag,
                    "description": region_desc,
                    "patches": patches,
                    "inferred_category": inferred_category,
                }
            )

        if regions:
            return regions

        # Fallback: if no anchors are available, use tile regions to keep output schema stable.
        for tile in tiles_catalog or []:
            if not isinstance(tile, dict):
                continue
            tile_id = str(tile.get("tile_id", "")).strip()
            bbox = tile.get("bbox_level0_xywh", [])
            if not tile_id or not (isinstance(bbox, list) and len(bbox) >= 4):
                continue
            tx, ty, tw, th = [int(v) for v in bbox[:4]]
            if tw <= 0 or th <= 0:
                continue
            coarse_label = str(tile.get("coarse_label", "stroma"))
            regions.append(
                {
                    "region_id": f"TILE_{tile_id}",
                    "coordinates": [tx, ty, tw, th],
                    "semantic_tag": f"Tile Region ({coarse_label})",
                    "description": str(tile.get("tile_desc", "")).strip()
                    or f"Tile-centered region with {coarse_label} morphology.",
                    "patches": [],
                    "inferred_category": _coarse_to_region_category(coarse_label),
                }
            )
        return regions

    def load(self, slide_path: str, out_dir: str | None = None) -> Dict[str, Any]:
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"SVS not found: {slide_path}")

        slide_id = os.path.basename(slide_path)
        seed = int(hashlib.sha256(slide_id.encode("utf-8")).hexdigest()[:8], 16)
        self._log(f"Start building slide hierarchy | slide_id={slide_id} | rng_seed={seed}")

        pyramids = self._load_lowest_pyramid_level(slide_path)
        low_rgb = pyramids["lowest"]
        medium_rgb = pyramids["medium"]
        level_dims = pyramids["level_dims"]

        level0_w, level0_h = level_dims[0]

        # Stage 1 global survey on thumbnail space.
        thumb_pack = self._build_stage1_thumbnail(
            slide_path=slide_path,
            level0_size=[level0_w, level0_h],
            low_rgb=low_rgb,
            medium_rgb=medium_rgb,
            out_dir=out_dir,
            slide_id=slide_id,
        )
        thumb_rgb = thumb_pack["thumb_rgb"]
        thumbnail_meta = thumb_pack["thumbnail_meta"]
        tissue_mask_thumb = self._segment_tissue(thumb_rgb)
        stage1_debug = self._write_stage1_debug_images(
            out_dir=out_dir or "",
            slide_id=slide_id,
            thumb_rgb=thumb_rgb,
            tissue_mask=tissue_mask_thumb,
        )
        tile_pack = self._build_tiles_catalog(
            slide_id=slide_id,
            thumb_rgb=thumb_rgb,
            tissue_mask=tissue_mask_thumb,
            thumbnail_meta=thumbnail_meta,
            out_dir=out_dir,
        )
        tiles_survey = tile_pack["tiles_survey"]
        tiles_catalog = tile_pack["tiles_catalog"]
        tiles_survey_path = tile_pack["tiles_survey_path"]
        tiles_catalog_path = tile_pack["tiles_catalog_path"]
        tumor_size_estimation = quantify_tumor_size_from_thumbnail(
            thumb_rgb=thumb_rgb,
            tissue_mask=tissue_mask_thumb,
            tiles_catalog=tiles_catalog,
            thumbnail_meta=thumbnail_meta,
            out_dir=out_dir or "",
            slide_id=slide_id,
        )
        if bool(tumor_size_estimation.get("available", False)):
            self._log(
                "Tumor size quantification | "
                f"feret_mm={tumor_size_estimation.get('max_feret_mm', None)} | "
                f"principal_mm={tumor_size_estimation.get('principal_axis_mm', None)} | "
                f"mask_ratio={tumor_size_estimation.get('tumor_mask_area_ratio_in_tissue', None)}"
            )
        else:
            self._log(
                "Tumor size quantification unavailable | "
                f"reason={tumor_size_estimation.get('error', 'unknown')}"
            )
        tile_anchor_packages = self._build_tile_anchor_packages(
            slide_id=slide_id,
            tiles_catalog=tiles_catalog,
            thumb_rgb=thumb_rgb,
        )

        stage1_exec = self._build_stage1_evidence_objects(
            slide_id=slide_id,
            slide_path=slide_path,
            tile_anchor_packages=tile_anchor_packages,
            level0_size=[level0_w, level0_h],
            base_magnification=thumbnail_meta.get("base_magnification", None),
            out_dir=out_dir,
            thumb_rgb=thumb_rgb,
            thumbnail_meta=thumbnail_meta,
            tiles_catalog=tiles_catalog,
        )
        anchors_catalog = stage1_exec["anchors_catalog"]
        patch_views_catalog = stage1_exec["patch_views_catalog"]
        evidence_items = stage1_exec["evidence_items"]
        signal_screenings = stage1_exec["signal_screenings"]
        signal_evidence_edges = stage1_exec["signal_evidence_edges"]
        patch_sampler_backend = stage1_exec["patch_sampler_backend"]
        selected_patch_overlay_path = self._write_selected_patch_overlay_debug_image(
            out_dir=out_dir or "",
            slide_id=slide_id,
            thumb_rgb=thumb_rgb,
            thumbnail_meta=thumbnail_meta,
            tiles_catalog=tiles_catalog,
            patch_views_catalog=patch_views_catalog,
            anchors_catalog=anchors_catalog,
        )
        stage1_debug["selected_patch_overlay_path"] = str(selected_patch_overlay_path or "")

        tile_anchor_packages_path = ""
        anchors_catalog_path = ""
        patch_views_catalog_path = ""
        evidence_items_path = ""
        signal_screenings_path = ""
        signal_evidence_edges_path = ""
        signal_ontology_path = ""
        if out_dir:
            survey_dir = os.path.join(out_dir, "survey")
            os.makedirs(survey_dir, exist_ok=True)
            stem = self._safe_name(os.path.splitext(slide_id)[0])
            tile_anchor_packages_path = os.path.join(survey_dir, f"{stem}_tile_anchor_packages.json")
            with open(tile_anchor_packages_path, "w", encoding="utf-8") as f:
                json.dump(tile_anchor_packages, f, indent=2, ensure_ascii=False)
            anchors_catalog_path = os.path.join(survey_dir, f"{stem}_anchors_catalog.json")
            with open(anchors_catalog_path, "w", encoding="utf-8") as f:
                json.dump(anchors_catalog, f, indent=2, ensure_ascii=False)
            patch_views_catalog_path = os.path.join(survey_dir, f"{stem}_patch_views_catalog.json")
            with open(patch_views_catalog_path, "w", encoding="utf-8") as f:
                json.dump(patch_views_catalog, f, indent=2, ensure_ascii=False)
            evidence_items_path = os.path.join(survey_dir, f"{stem}_evidence_items.json")
            with open(evidence_items_path, "w", encoding="utf-8") as f:
                json.dump(evidence_items, f, indent=2, ensure_ascii=False)
            signal_screenings_path = os.path.join(survey_dir, f"{stem}_signal_screenings.json")
            with open(signal_screenings_path, "w", encoding="utf-8") as f:
                json.dump(signal_screenings, f, indent=2, ensure_ascii=False)
            signal_evidence_edges_path = os.path.join(survey_dir, f"{stem}_signal_evidence_edges.json")
            with open(signal_evidence_edges_path, "w", encoding="utf-8") as f:
                json.dump(signal_evidence_edges, f, indent=2, ensure_ascii=False)
            signal_ontology_path = os.path.join(survey_dir, f"{stem}_signal_ontology.json")
            with open(signal_ontology_path, "w", encoding="utf-8") as f:
                json.dump(self.signal_ontology, f, indent=2, ensure_ascii=False)
        tissue_ratio_thumb = float(np.mean(tissue_mask_thumb > 0))
        thumbnail_meta["non_background_tile_count"] = int(len(tiles_catalog))
        thumbnail_meta["total_tile_count"] = int(len(tiles_survey))
        thumbnail_meta["anchor_package_count"] = int(len(tile_anchor_packages))
        thumbnail_meta["anchor_count"] = int(sum(len(p.get("anchors", [])) for p in tile_anchor_packages))
        thumbnail_meta["anchors_catalog_count"] = int(len(anchors_catalog))
        thumbnail_meta["patch_view_count"] = int(len(patch_views_catalog))
        thumbnail_meta["evidence_count"] = int(len(evidence_items))
        thumbnail_meta["signal_screening_count"] = int(len(signal_screenings))
        thumbnail_meta["signal_evidence_edge_count"] = int(len(signal_evidence_edges))
        thumbnail_meta["patch_sampler_backend"] = patch_sampler_backend
        thumbnail_meta["signal_ontology_version"] = str(self.signal_ontology.get("version", "unknown"))
        thumbnail_meta["signal_ontology_signal_count"] = int(len(self.signal_ontology.get("signals", [])))
        thumbnail_meta["stage1_tissue_ratio"] = round(tissue_ratio_thumb, 4)
        thumbnail_meta["debug_selected_patch_overlay_path"] = stage1_debug.get("selected_patch_overlay_path", "")
        self._log(
            f"Stage1 survey done | thumb={thumbnail_meta['thumbnail_size']} | non_bg_tiles={len(tiles_catalog)}/{len(tiles_survey)}"
        )

        low_h, low_w = low_rgb.shape[:2]
        sx = level0_w / max(1, low_w)
        sy = level0_h / max(1, low_h)
        self._log(
            f"Scale map (low->level0) | level0=({level0_w},{level0_h}) | low=({low_w},{low_h}) | sx={sx:.4f} sy={sy:.4f}"
        )

        tissue_mask_low = self._segment_tissue(low_rgb)
        tissue_ratio_low = float(np.mean(tissue_mask_low > 0))
        regions = self._build_regions_from_stage1_outputs(
            slide_id=slide_id,
            tiles_catalog=tiles_catalog,
            anchors_catalog=anchors_catalog,
            patch_views_catalog=patch_views_catalog,
            evidence_items=evidence_items,
        )
        region_areas = [max(1, int(r["coordinates"][2]) * int(r["coordinates"][3])) for r in regions]
        region_area_total = max(1, sum(region_areas))
        largest_region_ratio = round(float(max(region_areas, default=0)) / float(region_area_total), 4)
        segmentation_qc = {
            "tissue_ratio_thumb": round(tissue_ratio_thumb, 4),
            "tissue_ratio_low": round(tissue_ratio_low, 4),
            "raw_region_count": int(len(regions)),
            "selected_region_count": int(len(regions)),
            "largest_raw_region_ratio": largest_region_ratio,
            "grid_non_background_ratio": round(float(len(tiles_catalog)) / float(max(1, len(tiles_survey))), 4),
            "debug_tissue_mask_path": stage1_debug.get("tissue_mask_path", ""),
            "debug_grid_overlay_path": stage1_debug.get("grid_overlay_path", ""),
            "debug_selected_patch_overlay_path": stage1_debug.get("selected_patch_overlay_path", ""),
            "tumor_size_estimation": tumor_size_estimation,
        }
        self._log(
            f"Stage2 region assembly (tile-anchor-patch only) | tissue_ratio_low={tissue_ratio_low:.2%} | "
            f"regions={len(regions)} | total_views={len(patch_views_catalog)}"
        )

        global_metrics = self._build_global_metrics(
            regions=regions,
            tissue_ratio=tissue_ratio_thumb,
            slide_dims=[level0_w, level0_h],
            raw_region_count=len(regions),
            selected_region_count=len(regions),
        )
        tile_label_counts: Dict[str, int] = {}
        for t in tiles_catalog:
            label = str(t.get("coarse_label", "unknown"))
            tile_label_counts[label] = tile_label_counts.get(label, 0) + 1
        tumor_tiles = tile_label_counts.get("tumor", 0)
        non_bg_tiles = max(1, len(tiles_catalog))
        global_metrics.update(
            {
                "tile_non_background_count": int(len(tiles_catalog)),
                "tile_total_count": int(len(tiles_survey)),
                "tile_label_counts": tile_label_counts,
                "tile_tumor_ratio": round(float(tumor_tiles) / float(non_bg_tiles), 4),
                "tumor_size_estimation": tumor_size_estimation,
            }
        )
        global_metrics = self._apply_hybrid_global_metrics(global_metrics)
        global_summary = self._build_global_summary(global_metrics, region_count=len(regions))
        question_scope = {
            "likely_answerable": [
                "overall_diagnosis_pattern",
                "coarse_histologic_pattern",
                "tumor_presence_and_distribution",
                "coarse_tumor_localization_by_tile",
                "tumor_maximum_diameter_estimation",
            ],
            "needs_local_evidence": [
                "nuclear_grade",
                "mitotic_activity",
                "lymphovascular_invasion",
                "microcalcification",
            ],
            "likely_not_answerable_from_h_e_image_only": [
                "vital_status",
                "survival_time",
                "er_pr_her2_status",
            ],
        }
        self._log(f"Global summary: {global_summary}")
        self._log(
            f"Finished slide hierarchy | regions={len(regions)} | total_patches={sum(len(r['patches']) for r in regions)}"
        )

        return {
            "slide_id": slide_id,
            "slide_path": slide_path,
            "virtual_dimensions": [level0_w, level0_h],
            "pyramid_arrays": {
                "L3": low_rgb,
                "L2": medium_rgb,
                "L1": medium_rgb[::2, ::2],
            },
            "level3_global": {
                "summary": global_summary,
                "coordinates": [0, 0, level0_w, level0_h],
                "metrics": global_metrics,
                "segmentation_qc": segmentation_qc,
                "tumor_size_estimation": tumor_size_estimation,
                "question_scope": question_scope,
                "thumbnail_meta": thumbnail_meta,
                "tiles_catalog": tiles_catalog,
                "tiles_survey": tiles_survey,
                "tiles_survey_path": tiles_survey_path,
                "tiles_catalog_path": tiles_catalog_path,
                "tile_anchor_packages": tile_anchor_packages,
                "tile_anchor_packages_path": tile_anchor_packages_path,
                "anchors_catalog": anchors_catalog,
                "anchors_catalog_path": anchors_catalog_path,
                "patch_views_catalog": patch_views_catalog,
                "patch_views_catalog_path": patch_views_catalog_path,
                "evidence_items": evidence_items,
                "evidence_items_path": evidence_items_path,
                "signal_screenings": signal_screenings,
                "signal_screenings_path": signal_screenings_path,
                "signal_evidence_edges": signal_evidence_edges,
                "signal_evidence_edges_path": signal_evidence_edges_path,
                "signal_ontology": self.signal_ontology,
                "signal_ontology_path": signal_ontology_path,
            },
            "level2_regions": regions,
        }
