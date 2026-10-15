from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _build_tumor_prior(
    shape_hw: Tuple[int, int],
    tiles_catalog: Sequence[Dict[str, Any]],
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    prior = np.zeros((h, w), dtype=np.float32)
    for tile in tiles_catalog:
        bbox = tile.get("thumb_bbox_xywh", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, tw, th = [int(v) for v in bbox]
        x0 = max(0, min(w - 1, x))
        y0 = max(0, min(h - 1, y))
        x1 = max(x0 + 1, min(w, x0 + max(1, tw)))
        y1 = max(y0 + 1, min(h, y0 + max(1, th)))
        conf_map = tile.get("label_confidence", {})
        tumor_conf = 0.0
        if isinstance(conf_map, dict):
            try:
                tumor_conf = float(conf_map.get("tumor", 0.0))
            except Exception:
                tumor_conf = 0.0
        if tumor_conf <= 0.0:
            label = str(tile.get("coarse_label", ""))
            score = _float_or_none(tile.get("label_score"))
            if label == "tumor":
                tumor_conf = float(score if score is not None else 0.7)
        if tumor_conf <= 0.0:
            continue
        prior[y0:y1, x0:x1] = np.maximum(prior[y0:y1, x0:x1], float(tumor_conf))
    return np.clip(prior, 0.0, 1.0)


def _max_pairwise_distance(points_xy: np.ndarray) -> Tuple[float, List[float], List[float]]:
    n = int(points_xy.shape[0])
    if n <= 1:
        p = points_xy[0].tolist() if n == 1 else [0.0, 0.0]
        return 0.0, [float(p[0]), float(p[1])], [float(p[0]), float(p[1])]
    if n > 1400:
        keep = np.linspace(0, n - 1, 1400, dtype=np.int32)
        points_xy = points_xy[keep]
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    d = float(np.sqrt(max(0.0, float(dist2[i, j]))))
    p1 = [float(points_xy[i, 0]), float(points_xy[i, 1])]
    p2 = [float(points_xy[j, 0]), float(points_xy[j, 1])]
    return d, p1, p2


def _principal_axis_extent(points_xy: np.ndarray) -> float:
    n = int(points_xy.shape[0])
    if n <= 1:
        return 0.0
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    proj = centered @ axis
    return float(np.max(proj) - np.min(proj))


def quantify_tumor_size_from_thumbnail(
    thumb_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    tiles_catalog: Sequence[Dict[str, Any]],
    thumbnail_meta: Dict[str, Any],
    out_dir: str = "",
    slide_id: str = "",
) -> Dict[str, Any]:
    if cv2 is None:
        return {
            "available": False,
            "error": "opencv-python is required for tumor size quantification.",
            "method": "thumbnail_tumor_mask_feret_principal_axis_v1",
        }

    h, w = int(thumb_rgb.shape[0]), int(thumb_rgb.shape[1])
    tissue = (tissue_mask > 0).astype(np.uint8)
    prior = _build_tumor_prior((h, w), tiles_catalog=tiles_catalog)

    if float(np.max(prior)) <= 1e-6:
        return {
            "available": False,
            "error": "No tumor-prior tiles found in stage1 tiles_catalog.",
            "method": "thumbnail_tumor_mask_feret_principal_axis_v1",
        }

    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    r = thumb_rgb[..., 0].astype(np.int16)
    g = thumb_rgb[..., 1].astype(np.int16)
    b = thumb_rgb[..., 2].astype(np.int16)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    nuclei_like = ((hue >= 100) & (hue <= 170) & (sat >= 25) & (val <= 230)) | (
        (b > g + 10) & (b > r + 6) & (gray < 210)
    )
    prior_mid = prior >= 0.25
    prior_high = prior >= 0.55
    mask = (((nuclei_like & prior_mid) | prior_high) & (tissue > 0)).astype(np.uint8)

    k = max(3, int(round(min(h, w) / 320.0)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(80, int(0.00006 * h * w))
    keep = np.zeros_like(mask, dtype=np.uint8)
    for lid in range(1, n_labels):
        x, y, bw, bh, area = [int(v) for v in stats[lid]]
        if area < min_area:
            continue
        comp = labels[y : y + bh, x : x + bw] == lid
        comp_prior = prior[y : y + bh, x : x + bw][comp]
        if comp_prior.size == 0:
            continue
        if float(np.mean(comp_prior)) < 0.28:
            continue
        keep[y : y + bh, x : x + bw][comp] = 1
    mask = keep

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_source = "strict"

    # Relaxed retry for stain/domain shift.
    if not contours:
        relaxed = ((((sat >= 18) & (gray < 235) & (prior >= 0.15)) | (prior >= 0.45)) & (tissue > 0)).astype(np.uint8)
        k2 = max(3, k - 2)
        if k2 % 2 == 0:
            k2 += 1
        kernel2 = np.ones((k2, k2), dtype=np.uint8)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, kernel2)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, kernel2)

        n2, l2, st2, _ = cv2.connectedComponentsWithStats(relaxed, connectivity=8)
        min_area2 = max(40, int(0.00003 * h * w))
        keep2 = np.zeros_like(relaxed, dtype=np.uint8)
        for lid in range(1, n2):
            x, y, bw, bh, area = [int(v) for v in st2[lid]]
            if area < min_area2:
                continue
            comp = l2[y : y + bh, x : x + bw] == lid
            comp_prior = prior[y : y + bh, x : x + bw][comp]
            if comp_prior.size == 0:
                continue
            if float(np.mean(comp_prior)) < 0.12:
                continue
            keep2[y : y + bh, x : x + bw][comp] = 1
        mask = keep2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_source = "relaxed"

    # Prior-only fallback if morphology gate still fails.
    if not contours:
        prior_mask = (((prior >= 0.20) & (tissue > 0))).astype(np.uint8)
        kernel3 = np.ones((max(3, k // 2), max(3, k // 2)), dtype=np.uint8)
        prior_mask = cv2.morphologyEx(prior_mask, cv2.MORPH_CLOSE, kernel3)
        mask = prior_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_source = "prior_only"

    contour = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
    else:
        # Last fallback: top tumor tile bbox as coarse contour.
        best_bbox = None
        best_score = -1.0
        for tile in tiles_catalog:
            bbox = tile.get("thumb_bbox_xywh", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            conf_map = tile.get("label_confidence", {})
            score = 0.0
            if isinstance(conf_map, dict):
                score = float(conf_map.get("tumor", 0.0))
            if score <= 0.0 and str(tile.get("coarse_label", "")) == "tumor":
                score = float(_float_or_none(tile.get("label_score")) or 0.5)
            if score > best_score:
                best_score = score
                best_bbox = [int(v) for v in bbox]
        if best_bbox is None:
            return {
                "available": False,
                "error": "Tumor mask has no valid contour after refinement.",
                "method": "thumbnail_tumor_mask_feret_principal_axis_v1",
            }
        x, y, bw, bh = best_bbox
        x0 = max(0, min(w - 1, x))
        y0 = max(0, min(h - 1, y))
        x1 = max(x0 + 1, min(w, x0 + max(1, bw)))
        y1 = max(y0 + 1, min(h, y0 + max(1, bh)))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        contour = np.array(
            [[[x0, y0]], [[x1 - 1, y0]], [[x1 - 1, y1 - 1]], [[x0, y1 - 1]]],
            dtype=np.int32,
        )
        mask_source = "tile_bbox_fallback"

    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0.0:
        return {
            "available": False,
            "error": "Largest tumor contour area is zero.",
            "method": "thumbnail_tumor_mask_feret_principal_axis_v1",
        }

    points_thumb = contour.reshape(-1, 2).astype(np.float32)
    feret_thumb_px, p1_thumb, p2_thumb = _max_pairwise_distance(points_thumb)
    principal_thumb_px = _principal_axis_extent(points_thumb)
    bx, by, bw, bh = [int(v) for v in cv2.boundingRect(contour)]

    scale_x = float(thumbnail_meta.get("scale_x", 1.0))
    scale_y = float(thumbnail_meta.get("scale_y", 1.0))
    points_l0 = np.empty_like(points_thumb, dtype=np.float32)
    points_l0[:, 0] = points_thumb[:, 0] * scale_x
    points_l0[:, 1] = points_thumb[:, 1] * scale_y
    feret_l0_px, _, _ = _max_pairwise_distance(points_l0)
    principal_l0_px = _principal_axis_extent(points_l0)

    mpp_info = thumbnail_meta.get("mpp", {})
    mpp_x = _float_or_none(mpp_info.get("x") if isinstance(mpp_info, dict) else None)
    mpp_y = _float_or_none(mpp_info.get("y") if isinstance(mpp_info, dict) else None)
    if mpp_x is None and mpp_y is not None:
        mpp_x = mpp_y
    if mpp_y is None and mpp_x is not None:
        mpp_y = mpp_x
    has_mpp = (mpp_x is not None) and (mpp_y is not None)

    feret_mm = None
    principal_mm = None
    if has_mpp:
        points_mm = np.empty_like(points_l0, dtype=np.float32)
        points_mm[:, 0] = points_l0[:, 0] * float(mpp_x) / 1000.0
        points_mm[:, 1] = points_l0[:, 1] * float(mpp_y) / 1000.0
        feret_mm, _, _ = _max_pairwise_distance(points_mm)
        principal_mm = _principal_axis_extent(points_mm)

    mask_path = ""
    if out_dir:
        debug_dir = os.path.join(out_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        stem = _safe_name(os.path.splitext(slide_id)[0] if slide_id else "slide")
        mask_path = os.path.join(debug_dir, f"{stem}_tumor_mask.png")
        if Image is not None:
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
        else:
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

    tissue_px = int(np.sum(tissue > 0))
    tumor_px = int(np.sum(mask > 0))
    out: Dict[str, Any] = {
        "available": True,
        "method": "thumbnail_tumor_mask_feret_principal_axis_v1",
        "mask_source": mask_source,
        "mask_path": mask_path,
        "tumor_mask_area_px": tumor_px,
        "tumor_mask_area_ratio_in_tissue": round(float(tumor_px) / float(max(1, tissue_px)), 4),
        "largest_contour_area_px": round(contour_area, 2),
        "largest_contour_bbox_thumb_xywh": [int(bx), int(by), int(bw), int(bh)],
        "max_feret_thumb_px": round(float(feret_thumb_px), 2),
        "principal_axis_thumb_px": round(float(principal_thumb_px), 2),
        "max_feret_level0_px": round(float(feret_l0_px), 2),
        "principal_axis_level0_px": round(float(principal_l0_px), 2),
        "feret_endpoints_thumb_xy": [p1_thumb, p2_thumb],
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "max_feret_mm": round(float(feret_mm), 3) if feret_mm is not None else None,
        "principal_axis_mm": round(float(principal_mm), 3) if principal_mm is not None else None,
        "mpp_available": bool(has_mpp),
    }
    if not has_mpp:
        out["note"] = "mpp unavailable; mm conversion is not available."
    if mask_source in {"relaxed", "prior_only", "tile_bbox_fallback"}:
        out["note"] = (
            (str(out.get("note", "")).strip() + " ").strip()
            + f"fallback_mode={mask_source}; interpret as coarse estimate."
        ).strip()
    return out
