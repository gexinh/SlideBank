from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .vector_store import LocalInMemoryCollection


class HierarchicalMemoryBank:
    """Stores L3/L2/L1 slide memory with local lexical retrieval."""

    def __init__(
        self,
        collection_name: str = "wsi_hierarchical_memory",
        persist_directory: Optional[str] = None,
        reset_collection: bool = False,
        ingest_legacy_region_patches: bool = False,
        verbose: bool = False,
    ):
        self.active_slide_id: Optional[str] = None
        self.collection_name = collection_name
        self.backend = "local-lexical"
        self.persist_directory = persist_directory
        self.ingest_legacy_region_patches = bool(ingest_legacy_region_patches)
        self.verbose = bool(verbose)

        self._log(
            f"Init collection='{collection_name}' | persist_dir='{persist_directory}' | reset={reset_collection}"
        )

        fallback_path = None
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            fallback_path = os.path.join(persist_directory, f"{collection_name}_fallback.json")
        self.collection = LocalInMemoryCollection(persist_path=fallback_path)
        if reset_collection and hasattr(self.collection, "clear"):
            self.collection.clear()
        self.backend = "local-lexical-persistent" if persist_directory else "local-lexical"
        self._log(f"Backend ready: {self.backend}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[MemoryBank] {message}")

    @staticmethod
    def _coords_to_meta(coords: List[int]) -> str:
        return json.dumps([int(v) for v in coords])

    @staticmethod
    def _coords_from_meta(meta_val: str) -> List[int]:
        try:
            return [int(v) for v in json.loads(meta_val)]
        except Exception:
            return [0, 0, 0, 0]

    @staticmethod
    def _safe_filename(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)

    @staticmethod
    def _safe_json_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        try:
            parsed = json.loads(str(value))
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _extract_json_dict_from_text(text: str) -> Dict[str, Any]:
        s = (text or "").strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_json_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        try:
            parsed = json.loads(str(value))
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _extract_json_list_from_text(text: str) -> List[Any]:
        s = (text or "").strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        m = re.search(r"\[[\s\S]*\]", s)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _drop_empty_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, str) and v == "":
                continue
            if isinstance(v, (list, dict)) and len(v) == 0:
                continue
            cleaned[k] = v
        return cleaned

    def _write_human_readable_snapshot(
        self,
        slide_id: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not self.persist_directory:
            return
        out_dir = os.path.join(self.persist_directory, "readable")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self._safe_filename(slide_id)}.jsonl")

        with open(out_path, "w", encoding="utf-8") as f:
            for row_id, doc, meta in zip(ids, documents, metadatas):
                row = self._drop_empty_fields(
                    {
                        "id": row_id,
                        "slide_id": meta.get("slide_id", ""),
                        "level": meta.get("level", ""),
                        "region_id": meta.get("region_id", ""),
                        "patch_id": meta.get("patch_id", ""),
                    "parent_region_id": meta.get("parent_region_id", ""),
                        "semantic_tag": meta.get("semantic_tag", ""),
                        "coordinates": self._coords_from_meta(meta.get("coordinates", "[0,0,0,0]")),
                        "text": doc,
                    }
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._log(f"Readable snapshot written: {out_path}")

    def ingest_slide(self, slide_data: Dict[str, Any]) -> None:
        slide_id = slide_data["slide_id"]
        self.active_slide_id = slide_id
        self._log(f"Begin ingest slide_id={slide_id}")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        global_block = slide_data["level3_global"]
        global_metrics = self._safe_json_dict(global_block.get("metrics", {}))
        question_scope = self._safe_json_dict(global_block.get("question_scope", {}))
        segmentation_qc = self._safe_json_dict(global_block.get("segmentation_qc", {}))
        thumbnail_meta = self._safe_json_dict(global_block.get("thumbnail_meta", {}))
        tiles_catalog = self._safe_json_list(global_block.get("tiles_catalog", []))
        tiles_survey = self._safe_json_list(global_block.get("tiles_survey", []))
        tile_anchor_packages = self._safe_json_list(global_block.get("tile_anchor_packages", []))
        anchors_catalog = self._safe_json_list(global_block.get("anchors_catalog", []))
        patch_views_catalog = self._safe_json_list(global_block.get("patch_views_catalog", []))
        evidence_items = self._safe_json_list(global_block.get("evidence_items", []))
        signal_screenings = self._safe_json_list(global_block.get("signal_screenings", []))
        signal_evidence_edges = self._safe_json_list(global_block.get("signal_evidence_edges", []))
        signal_ontology = self._safe_json_dict(global_block.get("signal_ontology", {}))
        metrics_json = json.dumps(global_metrics, ensure_ascii=False, sort_keys=True)
        question_scope_json = json.dumps(question_scope, ensure_ascii=False, sort_keys=True)
        segmentation_qc_json = json.dumps(segmentation_qc, ensure_ascii=False, sort_keys=True)
        thumbnail_meta_json = json.dumps(thumbnail_meta, ensure_ascii=False, sort_keys=True)
        signal_ontology_json = json.dumps(signal_ontology, ensure_ascii=False, sort_keys=True)

        def _to_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return default

        global_doc_total = 0
        ids.append(f"{slide_id}::global")
        documents.append(global_block["summary"])
        metadatas.append(
            {
                "slide_id": slide_id,
                "level": "global",
                "global_subtype": "summary",
                "coordinates": self._coords_to_meta(global_block["coordinates"]),
                "parent_region_id": "",
                "semantic_tag": "Global slide context",
                "region_id": "",
                "patch_id": "",
                "carcinoma_ratio": _to_float(global_metrics.get("carcinoma_ratio", 0.0)),
                "high_grade_ratio": _to_float(global_metrics.get("high_grade_ratio", 0.0)),
                "suspicion_score": _to_float(global_metrics.get("suspicion_score", 0.0)),
                "suspicion_level": str(global_metrics.get("suspicion_level", "unknown")),
                "metrics_json": metrics_json,
                "question_scope_json": question_scope_json,
                "segmentation_qc_json": segmentation_qc_json,
                "thumbnail_meta_json": thumbnail_meta_json,
                "tile_non_background_count": int(len(tiles_catalog)),
                "tile_total_count": int(len(tiles_survey)),
                "anchor_package_count": int(len(tile_anchor_packages)),
                "anchor_count": int(sum(len(p.get("anchors", [])) for p in tile_anchor_packages if isinstance(p, dict))),
                "anchors_catalog_count": int(len(anchors_catalog)),
                "patch_view_count": int(len(patch_views_catalog)),
                "evidence_count": int(len(evidence_items)),
                "signal_screening_count": int(len(signal_screenings)),
                "signal_evidence_edge_count": int(len(signal_evidence_edges)),
                "signal_ontology_json": signal_ontology_json,
                "signal_ontology_signal_count": int(len(signal_ontology.get("signals", [])))
                if isinstance(signal_ontology, dict)
                else 0,
            }
        )
        global_doc_total += 1
        self._log(
            f"Add GLOBAL doc | id={ids[-1]} | coordinates={global_block['coordinates']} | text_head={documents[-1][:120]}"
        )

        if global_metrics:
            ids.append(f"{slide_id}::global::metrics")
            documents.append(f"Structured global metrics: {metrics_json}")
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "metrics",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global structured metrics",
                    "region_id": "",
                    "patch_id": "",
                    "metrics_json": metrics_json,
                }
            )
            global_doc_total += 1

        if question_scope:
            ids.append(f"{slide_id}::global::scope")
            documents.append(
                "Global question scope guidance: "
                f"likely_answerable={question_scope.get('likely_answerable', [])}; "
                f"needs_local_evidence={question_scope.get('needs_local_evidence', [])}; "
                f"likely_not_answerable_from_h_e_image_only={question_scope.get('likely_not_answerable_from_h_e_image_only', [])}"
            )
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "scope",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global question scope",
                    "region_id": "",
                    "patch_id": "",
                    "question_scope_json": question_scope_json,
                }
            )
            global_doc_total += 1

        if segmentation_qc:
            ids.append(f"{slide_id}::global::qc")
            documents.append(f"Global segmentation QC: {segmentation_qc_json}")
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "qc",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global segmentation quality",
                    "region_id": "",
                    "patch_id": "",
                    "segmentation_qc_json": segmentation_qc_json,
                }
            )
            global_doc_total += 1

        if thumbnail_meta:
            ids.append(f"{slide_id}::global::thumbnail_meta")
            documents.append(thumbnail_meta_json)
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "thumbnail_meta",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global thumbnail metadata",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if tiles_catalog:
            ids.append(f"{slide_id}::global::tiles_catalog")
            documents.append(json.dumps(tiles_catalog, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "tiles_catalog",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global non-background tile catalog",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if tiles_survey:
            ids.append(f"{slide_id}::global::tiles_survey")
            documents.append(json.dumps(tiles_survey, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "tiles_survey",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global 8x8 full tile survey",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if tile_anchor_packages:
            ids.append(f"{slide_id}::global::tile_anchor_packages")
            documents.append(json.dumps(tile_anchor_packages, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "tile_anchor_packages",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global tile anchor packages",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if anchors_catalog:
            ids.append(f"{slide_id}::global::anchors_catalog")
            documents.append(json.dumps(anchors_catalog, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "anchors_catalog",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global anchors catalog",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if patch_views_catalog:
            ids.append(f"{slide_id}::global::patch_views_catalog")
            documents.append(json.dumps(patch_views_catalog, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "patch_views_catalog",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global patch views catalog",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if evidence_items:
            ids.append(f"{slide_id}::global::evidence_items")
            documents.append(json.dumps(evidence_items, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "evidence_items",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global evidence items",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if signal_screenings:
            ids.append(f"{slide_id}::global::signal_screenings")
            documents.append(json.dumps(signal_screenings, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "signal_screenings",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global signal screenings",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if signal_evidence_edges:
            ids.append(f"{slide_id}::global::signal_evidence_edges")
            documents.append(json.dumps(signal_evidence_edges, ensure_ascii=False))
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "signal_evidence_edges",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global signal-to-evidence edges",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        if signal_ontology:
            ids.append(f"{slide_id}::global::signal_ontology")
            documents.append(signal_ontology_json)
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "global",
                    "global_subtype": "signal_ontology",
                    "coordinates": self._coords_to_meta(global_block["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": "Global signal ontology snapshot",
                    "region_id": "",
                    "patch_id": "",
                }
            )
            global_doc_total += 1

        region_total = 0
        patch_total = 0
        evidence_total = 0
        signal_total = 0
        view_total = 0

        for item in evidence_items:
            evidence_id = str(item.get("evidence_id", ""))
            anchor_id = str(item.get("anchor_id", ""))
            target_type = str(item.get("target_type", ""))
            views = self._safe_json_list(item.get("views", []))
            coords = [0, 0, 1, 1]
            if views and isinstance(views[0], dict):
                coords = self._coords_from_meta(self._coords_to_meta(views[0].get("patch_xywh_level0", [0, 0, 1, 1])))
            summary = str(item.get("summary", ""))
            anchor_desc = str(item.get("anchor_description", "")).strip()
            summary_short = self._compact_text(summary, max_len=180).strip()
            anchor_short = self._compact_text(anchor_desc, max_len=220).strip()
            if summary_short:
                evidence_doc = summary_short
            elif anchor_short:
                evidence_doc = anchor_short
            else:
                evidence_doc = "Anchor evidence available."
            ids.append(f"{slide_id}::evidence::{evidence_id}")
            documents.append(evidence_doc)
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "evidence",
                    "coordinates": self._coords_to_meta(coords),
                    "parent_region_id": "",
                    "semantic_tag": "Anchor evidence item",
                    "region_id": "",
                    "patch_id": "",
                    "anchor_id": anchor_id,
                    "evidence_id": evidence_id,
                    "target_type": target_type,
                    "evidence_confidence": _to_float(item.get("confidence", 0.0)),
                    "anchor_description": anchor_desc,
                }
            )
            evidence_total += 1

            for v in views:
                view_id = str(v.get("view_id", ""))
                ids.append(f"{slide_id}::view::{evidence_id}::{view_id}")
                documents.append(str(v.get("caption", "")))
                metadatas.append(
                    {
                        "slide_id": slide_id,
                        "level": "view",
                        "coordinates": self._coords_to_meta(v.get("patch_xywh_level0", [0, 0, 1, 1])),
                        "parent_region_id": "",
                        "semantic_tag": "Anchor patch view",
                        "region_id": "",
                        "patch_id": "",
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "view_id": view_id,
                        "mag": str(v.get("mag", "")),
                    }
                )
                view_total += 1

        for screening in signal_screenings:
            anchor_id = str(screening.get("anchor_id", ""))
            evidence_id = str(screening.get("evidence_id", ""))
            for sig in self._safe_json_list(screening.get("signal_screening", [])):
                signal_id = str(sig.get("signal_id", ""))
                status = str(sig.get("status", "unknown"))
                rationale = str(sig.get("rationale", ""))
                conf = _to_float(sig.get("confidence", 0.0))
                ids.append(f"{slide_id}::signal::{evidence_id}::{signal_id}")
                documents.append(f"{status}: {rationale}")
                metadatas.append(
                    {
                        "slide_id": slide_id,
                        "level": "signal",
                        "coordinates": self._coords_to_meta([0, 0, 1, 1]),
                        "parent_region_id": "",
                        "semantic_tag": "Anchor signal screening",
                        "region_id": "",
                        "patch_id": "",
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "signal_id": signal_id,
                        "signal_status": status,
                        "signal_confidence": conf,
                        "supporting_views_json": json.dumps(sig.get("supporting_views", []), ensure_ascii=False),
                    }
                )
                signal_total += 1

        for region in slide_data["level2_regions"]:
            region_id = region["region_id"]
            region_doc = f"{region['semantic_tag']}: {region['description']}"
            ids.append(f"{slide_id}::region::{region_id}")
            documents.append(region_doc)
            metadatas.append(
                {
                    "slide_id": slide_id,
                    "level": "region",
                    "coordinates": self._coords_to_meta(region["coordinates"]),
                    "parent_region_id": "",
                    "semantic_tag": region["semantic_tag"],
                    "region_id": region_id,
                    "patch_id": "",
                }
            )
            region_total += 1
            self._log(
                f"Add REGION doc | region_id={region_id} | coordinates={region['coordinates']} | semantic={region['semantic_tag']}"
            )
            if self.ingest_legacy_region_patches:
                for patch in region["patches"]:
                    patch_id = patch["patch_id"]
                    ids.append(f"{slide_id}::patch::{patch_id}")
                    documents.append(patch["description"])
                    metadatas.append(
                        {
                            "slide_id": slide_id,
                            "level": "patch",
                            "coordinates": self._coords_to_meta(patch["coordinates"]),
                            "parent_region_id": region_id,
                            "semantic_tag": patch["semantic_tag"],
                            "region_id": region_id,
                            "patch_id": patch_id,
                        }
                    )
                    patch_total += 1
                    if self.verbose:
                        self._log(
                            f"  Add PATCH doc | patch_id={patch_id} | parent={region_id} | coordinates={patch['coordinates']} | "
                            f"desc_head={patch['description'][:96]}"
                        )

        metadatas = [self._drop_empty_fields(m) for m in metadatas]
        if hasattr(self.collection, "upsert"):
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._write_human_readable_snapshot(slide_id, ids, documents, metadatas)
        self._log(
            f"Ingest complete | total_docs={len(ids)} "
            f"(global={global_doc_total}, evidence={evidence_total}, view={view_total}, signal={signal_total}, "
            f"region={region_total}, patch={patch_total}) | "
            f"slide_id={slide_id}"
        )

    def has_slide(self, slide_id: str) -> bool:
        result = self.collection.query(
            query_texts=["global tissue composition and diagnosis context"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "summary"},
            n_results=1,
        )
        if result.get("ids") and result["ids"][0]:
            return True
        result = self.collection.query(
            query_texts=["global tissue composition and diagnosis context"],
            where={"slide_id": slide_id, "level": "global"},
            n_results=1,
        )
        return bool(result.get("ids") and result["ids"][0])

    def consult_global_view(self, slide_id: str) -> Dict[str, Any]:
        self.active_slide_id = slide_id
        result = self.collection.query(
            query_texts=["global tissue composition and diagnosis context"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "summary"},
            n_results=1,
        )
        if not result["ids"] or not result["ids"][0]:
            result = self.collection.query(
                query_texts=["global tissue composition and diagnosis context"],
                where={"slide_id": slide_id, "level": "global"},
                n_results=1,
            )
        if not result["ids"] or not result["ids"][0]:
            return {"error": f"Slide '{slide_id}' not found in memory bank."}

        doc = result["documents"][0][0]
        meta = result["metadatas"][0][0]
        metrics = self._safe_json_dict(meta.get("metrics_json", {}))
        question_scope = self._safe_json_dict(meta.get("question_scope_json", {}))
        segmentation_qc = self._safe_json_dict(meta.get("segmentation_qc_json", {}))
        thumbnail_meta = self._safe_json_dict(meta.get("thumbnail_meta_json", {}))
        signal_ontology = self._safe_json_dict(meta.get("signal_ontology_json", {}))
        tiles_catalog: List[Any] = []
        tiles_survey: List[Any] = []
        tile_anchor_packages: List[Any] = []
        anchors_catalog: List[Any] = []
        patch_views_catalog: List[Any] = []
        evidence_items: List[Any] = []
        signal_screenings: List[Any] = []
        signal_evidence_edges: List[Any] = []

        if not metrics:
            metrics_result = self.collection.query(
                query_texts=["global structured metrics"],
                where={"slide_id": slide_id, "level": "global", "global_subtype": "metrics"},
                n_results=1,
            )
            if metrics_result.get("documents") and metrics_result["documents"][0]:
                metrics = self._extract_json_dict_from_text(metrics_result["documents"][0][0])

        if not question_scope:
            scope_result = self.collection.query(
                query_texts=["global question scope guidance"],
                where={"slide_id": slide_id, "level": "global", "global_subtype": "scope"},
                n_results=1,
            )
            if scope_result.get("metadatas") and scope_result["metadatas"][0]:
                question_scope = self._safe_json_dict(scope_result["metadatas"][0][0].get("question_scope_json", {}))

        if not segmentation_qc:
            qc_result = self.collection.query(
                query_texts=["global segmentation quality"],
                where={"slide_id": slide_id, "level": "global", "global_subtype": "qc"},
                n_results=1,
            )
            if qc_result.get("metadatas") and qc_result["metadatas"][0]:
                segmentation_qc = self._safe_json_dict(qc_result["metadatas"][0][0].get("segmentation_qc_json", {}))

        if not thumbnail_meta:
            thumbnail_result = self.collection.query(
                query_texts=["global thumbnail metadata"],
                where={"slide_id": slide_id, "level": "global", "global_subtype": "thumbnail_meta"},
                n_results=1,
            )
            if thumbnail_result.get("documents") and thumbnail_result["documents"][0]:
                thumbnail_meta = self._extract_json_dict_from_text(thumbnail_result["documents"][0][0])

        if not signal_ontology:
            ontology_result = self.collection.query(
                query_texts=["global signal ontology snapshot"],
                where={"slide_id": slide_id, "level": "global", "global_subtype": "signal_ontology"},
                n_results=1,
            )
            if ontology_result.get("documents") and ontology_result["documents"][0]:
                signal_ontology = self._extract_json_dict_from_text(ontology_result["documents"][0][0])

        tiles_result = self.collection.query(
            query_texts=["global non-background tile catalog"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "tiles_catalog"},
            n_results=1,
        )
        if tiles_result.get("documents") and tiles_result["documents"][0]:
            tiles_catalog = self._extract_json_list_from_text(tiles_result["documents"][0][0])

        survey_result = self.collection.query(
            query_texts=["global 8x8 full tile survey"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "tiles_survey"},
            n_results=1,
        )
        if survey_result.get("documents") and survey_result["documents"][0]:
            tiles_survey = self._extract_json_list_from_text(survey_result["documents"][0][0])

        anchor_result = self.collection.query(
            query_texts=["global tile anchor packages"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "tile_anchor_packages"},
            n_results=1,
        )
        if anchor_result.get("documents") and anchor_result["documents"][0]:
            tile_anchor_packages = self._extract_json_list_from_text(anchor_result["documents"][0][0])

        anchors_result = self.collection.query(
            query_texts=["global anchors catalog"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "anchors_catalog"},
            n_results=1,
        )
        if anchors_result.get("documents") and anchors_result["documents"][0]:
            anchors_catalog = self._extract_json_list_from_text(anchors_result["documents"][0][0])

        views_result = self.collection.query(
            query_texts=["global patch views catalog"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "patch_views_catalog"},
            n_results=1,
        )
        if views_result.get("documents") and views_result["documents"][0]:
            patch_views_catalog = self._extract_json_list_from_text(views_result["documents"][0][0])

        evidence_result = self.collection.query(
            query_texts=["global evidence items"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "evidence_items"},
            n_results=1,
        )
        if evidence_result.get("documents") and evidence_result["documents"][0]:
            evidence_items = self._extract_json_list_from_text(evidence_result["documents"][0][0])

        screening_result = self.collection.query(
            query_texts=["global signal screenings"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "signal_screenings"},
            n_results=1,
        )
        if screening_result.get("documents") and screening_result["documents"][0]:
            signal_screenings = self._extract_json_list_from_text(screening_result["documents"][0][0])

        edges_result = self.collection.query(
            query_texts=["global signal-to-evidence edges"],
            where={"slide_id": slide_id, "level": "global", "global_subtype": "signal_evidence_edges"},
            n_results=1,
        )
        if edges_result.get("documents") and edges_result["documents"][0]:
            signal_evidence_edges = self._extract_json_list_from_text(edges_result["documents"][0][0])

        text = doc.lower()
        if metrics:
            def _as_float(v: Any, default: float = 0.0) -> float:
                try:
                    return float(v)
                except Exception:
                    return default

            suspicion_level = str(metrics.get("suspicion_level", "unknown")).lower()
            suspicion_score = _as_float(metrics.get("suspicion_score", 0.0))
            carcinoma_ratio = _as_float(metrics.get("carcinoma_ratio", 0.0))
            high_grade_present = bool(metrics.get("high_grade_present", False))
            if suspicion_level == "high" or suspicion_score >= 0.55:
                impression = "Global morphology is suspicious for invasive carcinoma with proliferative activity."
            elif suspicion_level == "moderate" or suspicion_score >= 0.28:
                impression = "Global view suggests focal malignant epithelial proliferation; local confirmation is needed."
            elif carcinoma_ratio > 0.10:
                impression = "Global view shows limited malignant-like areas, but confidence is low without local evidence."
            else:
                impression = "No definitive malignant pattern from global view alone."
            if high_grade_present:
                impression += " High-grade-like regions are present."
        elif "no dominant malignant morphology" in text:
            impression = "Indeterminate on global view; malignancy cannot be confirmed without finer-scale evidence."
        elif "suspicious for invasive carcinoma" in text or "invasive carcinoma remains likely" in text:
            impression = "Global morphology is suspicious for invasive carcinoma."
        elif "carcinoma" in text or "malignant" in text:
            impression = "Malignant features are present on global assessment, requiring region/patch confirmation."
        else:
            impression = "No definitive malignant pattern from global view alone."

        return {
            "slide_id": slide_id,
            "level": "global",
            "summary": doc,
            "overall_impression": impression,
            "coordinates": self._coords_from_meta(meta["coordinates"]),
            "metrics": metrics,
            "question_scope": question_scope,
            "segmentation_qc": segmentation_qc,
            "thumbnail_meta": thumbnail_meta,
            "tiles_catalog": tiles_catalog,
            "tiles_survey": tiles_survey,
            "tile_anchor_packages": tile_anchor_packages,
            "anchors_catalog": anchors_catalog,
            "patch_views_catalog": patch_views_catalog,
            "evidence_items": evidence_items,
            "signal_screenings": signal_screenings,
            "signal_evidence_edges": signal_evidence_edges,
            "signal_ontology": signal_ontology,
        }

    def scan_regions(self, query: str) -> Dict[str, Any]:
        if not self.active_slide_id:
            return {"error": "No active slide. Run consult_global_view(slide_id) first."}

        result = self.collection.query(
            query_texts=[query],
            where={"slide_id": self.active_slide_id, "level": "region"},
            n_results=5,
        )

        matches: List[Dict[str, Any]] = []
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        dists = result["distances"][0] if result["distances"] else []
        for doc, meta, dist in zip(docs, metas, dists):
            matches.append(
                {
                    "region_id": meta["region_id"],
                    "coordinates": self._coords_from_meta(meta["coordinates"]),
                    "semantic_tag": meta["semantic_tag"],
                    "description": doc,
                    "distance": round(float(dist), 4),
                }
            )
        return {"slide_id": self.active_slide_id, "query": query, "matches": matches}

    def inspect_details(self, region_id: str, feature_query: str) -> Dict[str, Any]:
        if not self.active_slide_id:
            return {"error": "No active slide. Run consult_global_view(slide_id) first."}

        result = self.collection.query(
            query_texts=[feature_query],
            where={
                "slide_id": self.active_slide_id,
                "level": "patch",
                "parent_region_id": region_id,
            },
            n_results=8,
        )
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        source_level = "patch"
        if not docs:
            # Default ingestion can skip legacy low-res region patches.
            # Fall back to anchor view captions to keep this tool usable.
            alt = self.collection.query(
                query_texts=[feature_query],
                where={
                    "slide_id": self.active_slide_id,
                    "level": "view",
                },
                n_results=8,
            )
            docs = alt["documents"][0] if alt.get("documents") else []
            metas = alt["metadatas"][0] if alt.get("metadatas") else []
            source_level = "view"

        hit_terms = ["mitotic", "mitosis", "pleomorphism", "high n:c", "atypical mitoses", "proliferation"]
        findings: List[Dict[str, Any]] = []
        feature_hits = 0
        for doc, meta in zip(docs, metas):
            lc = doc.lower()
            has_hit = any(term in lc for term in hit_terms)
            feature_hits += int(has_hit)
            patch_id = str(meta.get("patch_id", "")).strip()
            if not patch_id:
                evidence_id = str(meta.get("evidence_id", "")).strip()
                view_id = str(meta.get("view_id", "")).strip()
                patch_id = f"view:{evidence_id}:{view_id}".strip(":")
            findings.append(
                {
                    "patch_id": patch_id,
                    "coordinates": self._coords_from_meta(meta["coordinates"]),
                    "semantic_tag": meta["semantic_tag"],
                    "description": doc,
                    "feature_hit": has_hit,
                }
            )

        if feature_hits >= 3:
            interpretation = "Patch-level review supports high-grade proliferative activity."
        elif feature_hits > 0:
            interpretation = "Some proliferative activity is present, but grade is not definitive."
        else:
            interpretation = "No strong patch-level evidence of high-grade proliferation."

        return {
            "slide_id": self.active_slide_id,
            "region_id": region_id,
            "feature_query": feature_query,
            "source_level": source_level,
            "feature_hits": feature_hits,
            "interpretation": interpretation,
            "findings": findings,
        }

    def inspect_evidence(self, anchor_id: str, signal_query: str = "") -> Dict[str, Any]:
        if not self.active_slide_id:
            return {"error": "No active slide. Run consult_global_view(slide_id) first."}

        query_text = signal_query.strip() or "anchor evidence summary"
        evidence_result = self.collection.query(
            query_texts=[query_text],
            where={"slide_id": self.active_slide_id, "level": "evidence", "anchor_id": anchor_id},
            n_results=5,
        )
        evidence_docs = evidence_result["documents"][0] if evidence_result.get("documents") else []
        evidence_metas = evidence_result["metadatas"][0] if evidence_result.get("metadatas") else []

        evidences: List[Dict[str, Any]] = []
        evidence_ids: List[str] = []
        for doc, meta in zip(evidence_docs, evidence_metas):
            evidence_id = str(meta.get("evidence_id", ""))
            evidence_ids.append(evidence_id)
            evidences.append(
                {
                    "evidence_id": evidence_id,
                    "anchor_id": str(meta.get("anchor_id", "")),
                    "target_type": str(meta.get("target_type", "")),
                    "confidence": float(meta.get("evidence_confidence", 0.0)),
                    "summary": doc,
                    "coordinates": self._coords_from_meta(meta.get("coordinates", "[0,0,0,0]")),
                }
            )

        signal_result = self.collection.query(
            query_texts=[query_text],
            where={"slide_id": self.active_slide_id, "level": "signal", "anchor_id": anchor_id},
            n_results=20,
        )
        signal_docs = signal_result["documents"][0] if signal_result.get("documents") else []
        signal_metas = signal_result["metadatas"][0] if signal_result.get("metadatas") else []

        screenings: List[Dict[str, Any]] = []
        signal_query_lc = signal_query.strip().lower()
        for doc, meta in zip(signal_docs, signal_metas):
            signal_id = str(meta.get("signal_id", ""))
            if signal_query_lc and signal_query_lc not in signal_id.lower() and signal_query_lc not in doc.lower():
                continue
            evidence_id = str(meta.get("evidence_id", ""))
            if evidence_ids and evidence_id not in evidence_ids:
                continue
            screenings.append(
                {
                    "evidence_id": evidence_id,
                    "signal_id": signal_id,
                    "status": str(meta.get("signal_status", "unknown")),
                    "confidence": float(meta.get("signal_confidence", 0.0)),
                    "supporting_views": self._safe_json_list(meta.get("supporting_views_json", [])),
                    "rationale": doc,
                }
            )

        view_result = self.collection.query(
            query_texts=["anchor patch view"],
            where={"slide_id": self.active_slide_id, "level": "view", "anchor_id": anchor_id},
            n_results=32,
        )
        view_docs = view_result["documents"][0] if view_result.get("documents") else []
        view_metas = view_result["metadatas"][0] if view_result.get("metadatas") else []
        views: List[Dict[str, Any]] = []
        for doc, meta in zip(view_docs, view_metas):
            views.append(
                {
                    "evidence_id": str(meta.get("evidence_id", "")),
                    "view_id": str(meta.get("view_id", "")),
                    "mag": str(meta.get("mag", "")),
                    "caption": doc,
                    "coordinates": self._coords_from_meta(meta.get("coordinates", "[0,0,0,0]")),
                }
            )

        return {
            "slide_id": self.active_slide_id,
            "anchor_id": anchor_id,
            "signal_query": signal_query,
            "evidences": evidences,
            "signal_screening": screenings,
            "views": views,
        }

    @staticmethod
    def route_question(question: str) -> Dict[str, Any]:
        q = (question or "").lower()
        concept_by_route: Dict[str, str] = {
            "wsi_global_structured": "MD02_Tumor_Size_Extent_Stage_Metadata_plus_WSIProxy",
            "grade_mitosis_pleomorphism": "M03_Tumor_Grade_Generic",
            "subtype_visual": "M02_Histological_Tumor_Type_Coarse",
            "tumor_visual": "M02_Histological_Tumor_Type_Coarse",
            "biomarker_proxy": "P01_IHC_Biomarker_Outcome_Unified_Proxy_Prediction",
            "non_visual": "MD01_Specimen_Procedure_Site_Sampling_Metadata",
            "dcis": "M06_InSitu_Component_Assessment",
            "lvi": "M08_LVI_Assessment",
            "calc_necrosis": "M04_Necrosis_Assessment",
            "margin": "M07_Margin_Status_Assessment",
            "lymph_node_metastasis": "MD02_Tumor_Size_Extent_Stage_Metadata_plus_WSIProxy",
            "coarse_global": "M01_Tumor_Presence_Distribution_Focality",
            "spatial_quant": "M01_Tumor_Presence_Distribution_Focality",
        }
        routes: List[Tuple[str, str, List[str], List[str], str]] = [
            (
                "wsi_global_structured",
                r"\btnm\b|\bpt\b|\bpn\b|\bpm\b|stage|maximum diameter|greatest dimension|largest dimension|largest focus|maximum size|tumor size|size of the .*carcinoma|laterality|left breast|right breast|left-sided|right-sided",
                [],
                ["global"],
                "WSI_GLOBAL_ONLY",
            ),
            (
                "grade_mitosis_pleomorphism",
                r"grade|mito|pleomorph|nuclear|tubule|nottingham|prolifer",
                [
                    "SIG101__NUCLEAR_PLEOMORPHISM_HIGH",
                    "SIG102__MITOTIC_ACTIVITY_HIGH",
                    "SIG103__TUBULE_FORMATION_LOW",
                    "SIG104__TUMOR_GRADE_HIGH_SUPPORT",
                ],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "subtype_visual",
                r"molecular subtype|histological subtype|subtype of|tumor subtype|type of (the )?(breast )?tumou?r|classification",
                [],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "tumor_visual",
                r"histological type|type of breast cancer|diagnosis|overall type|invasive ductal|invasive lobular",
                [],
                ["evidence", "view", "global"],
                "WSI_SPATIAL_MULTI_SCALE",
            ),
            (
                "biomarker_proxy",
                r"\ber\b|\bpr\b|her2|hormone receptor|receptor status|biomarker|ihc",
                [],
                ["signal", "evidence", "global"],
                "WSI_PROXY_SIGNAL_EVIDENCE",
            ),
            (
                "non_visual",
                r"survival|vital status|\bage\b|sex|history|patient info|patient information|clinical history|genomic|transcriptomic|rna[\s-]?seq|whole exome|mutation status",
                [],
                ["global"],
                "WSI_GLOBAL_ONLY",
            ),
            (
                "dcis",
                r"dcis|in\s*situ|comedo|cribriform|solid pattern",
                [
                    "SIG111__DCIS_PRESENT",
                    "SIG112__DCIS_COMEDO_NECROSIS",
                    "SIG113__DCIS_ARCH_SOLID_CRIBRIFORM",
                ],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "lvi",
                r"lymphovascular|vascular invasion|lvi|embol",
                ["SIG121__LYMPHOVASCULAR_INVASION_PRESENT"],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "calc_necrosis",
                r"calcification|microcalcification|necrosis",
                ["SIG131__MICROCALCIFICATION_PRESENT", "SIG132__TUMOR_NECROSIS_PRESENT"],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "margin",
                r"margin|closest margin|involved margin|surgical margin",
                ["SIG141__MARGIN_TUMOR_CLOSE", "SIG142__MARGIN_TUMOR_INVOLVED"],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "lymph_node_metastasis",
                r"lymph node|sentinel node|metasta|micromet",
                ["SIG151__LYMPH_NODE_METASTASIS_PRESENT", "SIG152__MICROMETASTASIS_PRESENT"],
                ["signal", "evidence", "view", "global"],
                "WSI_HIERARCHICAL",
            ),
            (
                "coarse_global",
                r"diagnosis|overall|type of|subtype|classification|stage|tumor size|size of the tumor|size of the infiltrating|size approximately|mass size|largest focus|multifocal|presence of multifocal",
                [],
                ["global"],
                "WSI_GLOBAL_ONLY",
            ),
            (
                "spatial_quant",
                r"where|location|site|found|distance|percentage|proportion|how many|count|ratio",
                [],
                ["evidence", "view", "global"],
                "WSI_SPATIAL_MULTI_SCALE",
            ),
        ]
        for name, pattern, sigs, levels, task_layer in routes:
            if re.search(pattern, q):
                out = {
                    "route": name,
                    "target_signals": sigs,
                    "target_levels": levels,
                    "task_layer": task_layer,
                    "source_policy": "WSI_ONLY",
                }
                concept_id = str(concept_by_route.get(name, "")).strip()
                if concept_id:
                    out["target_concept"] = concept_id
                return out
        return {
            "route": "coarse_global_default",
            "target_signals": [],
            "target_levels": ["global"],
            "task_layer": "WSI_GLOBAL_ONLY",
            "source_policy": "WSI_ONLY",
            "target_concept": "M01_Tumor_Presence_Distribution_Focality",
        }

    @staticmethod
    def _status_rank(status: str) -> int:
        s = (status or "").strip().lower()
        if s == "present":
            return 3
        if s == "unknown":
            return 2
        if s == "absent":
            return 1
        return 0

    @staticmethod
    def _compact_text(text: str, max_len: int = 220) -> str:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def build_signal_patch_context(
        self,
        question: str,
        route: Optional[Dict[str, Any]] = None,
        max_signals: int = 4,
        max_views_per_signal: int = 2,
        max_chars: int = 3200,
    ) -> Dict[str, Any]:
        if not self.active_slide_id:
            return {"error": "No active slide. Run consult_global_view(slide_id) first."}

        slide_id = str(self.active_slide_id)
        global_view = self.consult_global_view(slide_id)
        if not isinstance(global_view, dict) or global_view.get("error"):
            return {
                "error": f"Slide '{slide_id}' not found in memory bank.",
                "slide_id": slide_id,
                "context": "",
                "evidence": [],
            }

        active_route = dict(route or self.route_question(question))
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _signal_priority(
            status: str,
            confidence: float,
            present_count: int,
            absent_count: int,
            match_count: int,
        ) -> float:
            # In WSI QA, sparse positive findings are often more informative than numerous negatives.
            rank = self._status_rank(status)
            if rank >= 3:  # present
                status_bonus = 0.50
            elif rank == 2:  # unknown
                status_bonus = 0.10
            elif rank == 1:  # absent
                status_bonus = -0.30
            else:
                status_bonus = -0.15
            support_balance = float(present_count - absent_count) / float(max(1, match_count))
            rarity_boost = min(0.12, 0.04 * float(max(0, present_count)))
            return float(confidence) + status_bonus + (0.20 * support_balance) + rarity_boost

        target_signals_raw = active_route.get("target_signals", [])
        target_signals: List[str] = []
        if isinstance(target_signals_raw, list):
            for sid in target_signals_raw:
                ss = str(sid).strip()
                if ss and ss not in target_signals:
                    target_signals.append(ss)
        elif isinstance(target_signals_raw, str):
            for sid in target_signals_raw.split(","):
                ss = sid.strip()
                if ss and ss not in target_signals:
                    target_signals.append(ss)
        active_route["target_signals"] = target_signals

        valid_levels = {"global", "signal", "evidence", "view"}
        target_levels_raw = active_route.get("target_levels", [])
        target_levels: List[str] = []
        if isinstance(target_levels_raw, list):
            for lv in target_levels_raw:
                ll = str(lv).strip().lower()
                if ll in valid_levels and ll not in target_levels:
                    target_levels.append(ll)
        elif isinstance(target_levels_raw, str):
            for lv in re.split(r"[,\s]+", target_levels_raw):
                ll = str(lv).strip().lower()
                if ll in valid_levels and ll not in target_levels:
                    target_levels.append(ll)
        if not target_levels:
            target_levels = ["global"]
        active_route["target_levels"] = target_levels

        allow_levels = set(target_levels)
        allow_global = "global" in allow_levels
        allow_signal = "signal" in allow_levels
        allow_evidence = "evidence" in allow_levels
        allow_view = "view" in allow_levels

        summary = str(global_view.get("summary", "")).strip()
        impression = str(global_view.get("overall_impression", "")).strip()
        global_hint = " ".join([x for x in [summary, impression] if x]).strip()
        evidence_items_raw = global_view.get("evidence_items", [])
        evidence_count_total = len(evidence_items_raw) if isinstance(evidence_items_raw, list) else 0

        needs_fine_grained = bool(target_signals) or allow_signal or allow_evidence or allow_view
        if not needs_fine_grained:
            coverage_flags = {
                "coverage_level": "global_only",
                "has_conflict": False,
                "needs_refine": False,
                "refine_reason": "",
                "unknown_ratio": 0.0,
                "signal_rows_total": 0,
                "signal_rows_shown": 0,
                "evidence_count_total": int(evidence_count_total),
                "evidence_count_shown": 0,
                "view_count_total": 0,
                "view_count_shown": 0,
            }
            context = (
                "Structured signal evidence from memory:\n"
                + f"Route: {str(active_route.get('route', 'global_only'))}\n"
                + "Level policy: global\n"
                + "No target signal matching required for this question. Use global context only."
            )
            if global_hint:
                context += f"\nGlobal hint: {self._compact_text(global_hint, max_len=700)}"
            evidence = []
            if global_hint:
                evidence.append(
                    {
                        "level": "global",
                        "anchor_id": "",
                        "evidence_id": "",
                        "signal_id": "",
                        "signal_status": "",
                        "view_id": "",
                        "mag": "",
                        "patch_xywh_level0": [],
                        "text": global_hint,
                        "score": 0.0,
                        "score_semantic": 0.0,
                        "score_lexical": 0.0,
                        "score_confidence": 0.0,
                        "score_signal": 0.0,
                    }
                )
            return {
                "slide_id": slide_id,
                "route": active_route,
                "needs_fine_grained": False,
                "target_signals": target_signals,
                "matched_signals": [],
                "coverage_flags": coverage_flags,
                "context": context[:max_chars],
                "evidence": evidence,
            }

        ontology = global_view.get("signal_ontology", {})
        signal_name_map: Dict[str, str] = {}
        if isinstance(ontology, dict):
            signals = ontology.get("signals", [])
            if isinstance(signals, list):
                for sig in signals:
                    if not isinstance(sig, dict):
                        continue
                    sid = str(sig.get("signal_id", "")).strip()
                    sname = str(sig.get("signal_name", "")).strip()
                    if sid:
                        signal_name_map[sid] = sname

        evidence_items = evidence_items_raw
        evidence_by_id: Dict[str, Dict[str, Any]] = {}
        anchor_to_evidence_ids: Dict[str, List[str]] = {}
        total_views_in_memory = 0
        if isinstance(evidence_items, list):
            for ev in evidence_items:
                if not isinstance(ev, dict):
                    continue
                evidence_id = str(ev.get("evidence_id", "")).strip()
                anchor_id = str(ev.get("anchor_id", "")).strip()
                ev_views = ev.get("views", [])
                if isinstance(ev_views, list):
                    total_views_in_memory += len([v for v in ev_views if isinstance(v, dict)])
                if evidence_id:
                    evidence_by_id[evidence_id] = ev
                    if anchor_id:
                        anchor_to_evidence_ids.setdefault(anchor_id, []).append(evidence_id)

        signal_edges = global_view.get("signal_evidence_edges", [])
        edge_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        edge_by_signal: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(signal_edges, list):
            for edge in signal_edges:
                if not isinstance(edge, dict):
                    continue
                sid = str(edge.get("signal_id", "")).strip()
                eid = str(edge.get("evidence_id", "")).strip()
                if not sid or not eid:
                    continue
                try:
                    strength = float(edge.get("strength", 0.0))
                except Exception:
                    strength = 0.0
                strength = max(0.0, min(1.0, strength))
                polarity = str(edge.get("polarity", "")).strip().lower()
                if polarity not in {"supports", "contradicts", "uncertain"}:
                    polarity = "uncertain"
                normalized_edge = {
                    "signal_id": sid,
                    "evidence_id": eid,
                    "anchor_id": str(edge.get("anchor_id", "")).strip(),
                    "polarity": polarity,
                    "strength": strength,
                    "status": str(edge.get("status", "")).strip().lower(),
                    "supporting_views": self._safe_json_list(edge.get("supporting_views", [])),
                }
                edge_by_pair.setdefault((sid, eid), []).append(normalized_edge)
                edge_by_signal.setdefault(sid, []).append(normalized_edge)

        signal_screenings = global_view.get("signal_screenings", [])
        matches_by_signal: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in target_signals}
        target_set = set(target_signals)

        if isinstance(signal_screenings, list):
            for block in signal_screenings:
                if not isinstance(block, dict):
                    continue
                anchor_id = str(block.get("anchor_id", "")).strip()
                evidence_id = str(block.get("evidence_id", "")).strip()
                if not evidence_id and anchor_id and anchor_to_evidence_ids.get(anchor_id):
                    evidence_id = anchor_to_evidence_ids[anchor_id][0]
                ev = evidence_by_id.get(evidence_id, {})
                ev_views = ev.get("views", []) if isinstance(ev, dict) else []
                if not isinstance(ev_views, list):
                    ev_views = []

                items = block.get("signal_screening", [])
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    signal_id = str(it.get("signal_id", "")).strip()
                    if not signal_id:
                        continue
                    if target_set and signal_id not in target_set:
                        continue
                    status = str(it.get("status", "unknown")).strip().lower()
                    try:
                        confidence = float(it.get("confidence", 0.0))
                    except Exception:
                        confidence = 0.0
                    confidence = max(0.0, min(1.0, confidence))
                    rationale = str(it.get("rationale", "")).strip()
                    supporting = it.get("supporting_views", [])
                    supporting_ids = {
                        str(v).strip()
                        for v in supporting
                        if str(v).strip()
                    } if isinstance(supporting, list) else set()

                    pair_edges = edge_by_pair.get((signal_id, evidence_id), [])
                    if pair_edges:
                        best_edge = max(pair_edges, key=lambda x: float(x.get("strength", 0.0)))
                        edge_polarity = str(best_edge.get("polarity", "uncertain"))
                        edge_strength = float(best_edge.get("strength", 0.0))
                        if status == "unknown":
                            if edge_polarity == "supports":
                                status = "present"
                            elif edge_polarity == "contradicts":
                                status = "absent"
                        confidence = max(confidence, edge_strength)
                        if not supporting_ids:
                            supporting_from_edge = best_edge.get("supporting_views", [])
                            if isinstance(supporting_from_edge, list):
                                supporting_ids = {str(v).strip() for v in supporting_from_edge if str(v).strip()}

                    selected_views: List[Dict[str, Any]] = []
                    if allow_view:
                        for v in ev_views:
                            if not isinstance(v, dict):
                                continue
                            vid = str(v.get("view_id", "")).strip()
                            if supporting_ids and vid not in supporting_ids:
                                continue
                            xywh = v.get("patch_xywh_level0", [])
                            if isinstance(xywh, list) and len(xywh) == 4:
                                try:
                                    xywh = [int(x) for x in xywh]
                                except Exception:
                                    xywh = []
                            else:
                                xywh = []
                            selected_views.append(
                                {
                                    "view_id": vid,
                                    "mag": str(v.get("mag", "")).strip(),
                                    "patch_xywh_level0": xywh,
                                    "patch_path": str(v.get("patch_path", "")).strip(),
                                    "caption": str(v.get("caption", "")).strip(),
                                }
                            )
                        if not selected_views and ev_views:
                            for v in ev_views[: max(1, int(max_views_per_signal))]:
                                if not isinstance(v, dict):
                                    continue
                                xywh = v.get("patch_xywh_level0", [])
                                if isinstance(xywh, list) and len(xywh) == 4:
                                    try:
                                        xywh = [int(x) for x in xywh]
                                    except Exception:
                                        xywh = []
                                else:
                                    xywh = []
                                selected_views.append(
                                    {
                                        "view_id": str(v.get("view_id", "")).strip(),
                                        "mag": str(v.get("mag", "")).strip(),
                                        "patch_xywh_level0": xywh,
                                        "patch_path": str(v.get("patch_path", "")).strip(),
                                        "caption": str(v.get("caption", "")).strip(),
                                    }
                                )

                    matches_by_signal.setdefault(signal_id, []).append(
                        {
                            "signal_id": signal_id,
                            "status": status,
                            "confidence": confidence,
                            "rationale": rationale,
                            "anchor_id": anchor_id,
                            "evidence_id": evidence_id,
                            "views": selected_views,
                            "edge_count": int(len(pair_edges)),
                        }
                    )

        if target_signals:
            for signal_id in target_signals:
                if matches_by_signal.get(signal_id):
                    continue
                for edge in sorted(
                    edge_by_signal.get(signal_id, []),
                    key=lambda x: float(x.get("strength", 0.0)),
                    reverse=True,
                ):
                    evidence_id = str(edge.get("evidence_id", "")).strip()
                    if not evidence_id:
                        continue
                    ev = evidence_by_id.get(evidence_id, {})
                    if not isinstance(ev, dict):
                        continue
                    edge_polarity = str(edge.get("polarity", "uncertain")).lower()
                    if edge_polarity == "supports":
                        status = "present"
                    elif edge_polarity == "contradicts":
                        status = "absent"
                    else:
                        status = "unknown"
                    matches_by_signal.setdefault(signal_id, []).append(
                        {
                            "signal_id": signal_id,
                            "status": status,
                            "confidence": float(edge.get("strength", 0.0)),
                            "rationale": f"Edge-derived {edge_polarity} relation from signal_evidence_edges.",
                            "anchor_id": str(edge.get("anchor_id", "")).strip(),
                            "evidence_id": evidence_id,
                            "views": [],
                            "edge_count": 1,
                        }
                    )
                    break

        signal_rows: List[Dict[str, Any]] = []
        order_map = {sid: i for i, sid in enumerate(target_signals)}
        for signal_id, matches in matches_by_signal.items():
            if not matches:
                continue
            best_status = "unknown"
            best_conf = 0.0
            best_rationale = ""
            best_anchor_id = ""
            best_evidence_id = ""
            present_cnt = 0
            unknown_cnt = 0
            absent_cnt = 0
            edge_cnt = 0
            gathered_views: List[Dict[str, Any]] = []
            used_view_keys = set()
            evidence_id_set = set()

            for m in sorted(matches, key=lambda x: float(x.get("confidence", 0.0)), reverse=True):
                status = str(m.get("status", "unknown")).lower()
                conf = float(m.get("confidence", 0.0))
                if status == "present":
                    present_cnt += 1
                elif status == "absent":
                    absent_cnt += 1
                else:
                    unknown_cnt += 1
                edge_cnt += int(m.get("edge_count", 0))
                eid = str(m.get("evidence_id", "")).strip()
                if eid:
                    evidence_id_set.add(eid)

                if (
                    self._status_rank(status) > self._status_rank(best_status)
                    or (
                        self._status_rank(status) == self._status_rank(best_status)
                        and conf > best_conf
                    )
                ):
                    best_status = status
                    best_conf = conf
                    best_rationale = str(m.get("rationale", "")).strip()
                    best_anchor_id = str(m.get("anchor_id", "")).strip()
                    best_evidence_id = str(m.get("evidence_id", "")).strip()

                for v in m.get("views", []):
                    if not isinstance(v, dict):
                        continue
                    key = (
                        eid,
                        str(v.get("view_id", "")).strip(),
                    )
                    if key in used_view_keys:
                        continue
                    used_view_keys.add(key)
                    gathered_views.append(v)
                    if len(gathered_views) >= max(1, int(max_views_per_signal)):
                        break
                if len(gathered_views) >= max(1, int(max_views_per_signal)):
                    break

            priority_score = _signal_priority(
                status=best_status,
                confidence=float(best_conf),
                present_count=int(present_cnt),
                absent_count=int(absent_cnt),
                match_count=int(len(matches)),
            )
            signal_rows.append(
                {
                    "signal_id": signal_id,
                    "signal_name": signal_name_map.get(signal_id, signal_id.split("__", 1)[-1].lower()),
                    "status": best_status,
                    "confidence": round(float(best_conf), 4),
                    "priority_score": round(float(priority_score), 4),
                    "rationale": best_rationale,
                    "anchor_id": best_anchor_id,
                    "evidence_id": best_evidence_id,
                    "present_count": int(present_cnt),
                    "unknown_count": int(unknown_cnt),
                    "absent_count": int(absent_cnt),
                    "match_count": int(len(matches)),
                    "edge_count": int(edge_cnt),
                    "match_evidence_ids": sorted(list(evidence_id_set)),
                    "views": gathered_views,
                }
            )

        if target_signals:
            signal_rows.sort(
                key=lambda x: (
                    -float(x.get("priority_score", 0.0)),
                    -self._status_rank(str(x.get("status", "unknown"))),
                    -float(x.get("confidence", 0.0)),
                    order_map.get(str(x.get("signal_id", "")), 10**9),
                    -int(x.get("match_count", 0)),
                )
            )
        else:
            signal_rows.sort(
                key=lambda x: (
                    -float(x.get("priority_score", 0.0)),
                    -self._status_rank(str(x.get("status", "unknown"))),
                    -float(x.get("confidence", 0.0)),
                    -int(x.get("match_count", 0)),
                )
            )
        signal_rows_all = list(signal_rows)
        present_rows_all = [row for row in signal_rows_all if str(row.get("status", "")).lower() == "present"]
        present_gated_mode = bool(present_rows_all)
        if present_gated_mode:
            signal_rows_for_selection = present_rows_all[: max(1, int(max_signals))]
        else:
            signal_rows_for_selection = signal_rows_all[: max(1, int(max_signals))]
        signal_rows_for_context = signal_rows_for_selection if allow_signal else []

        evidence_picks: List[Dict[str, Any]] = []
        seen_evidence_ids = set()
        signal_rows_for_evidence = sorted(
            signal_rows_for_selection,
            key=lambda r: (
                0 if str(r.get("status", "")).lower() == "present" else (1 if str(r.get("status", "")).lower() == "unknown" else 2),
                -float(r.get("priority_score", 0.0)),
                -float(r.get("confidence", 0.0)),
            ),
        )

        def _append_one_evidence_from_row(row: Dict[str, Any]) -> bool:
            if len(evidence_picks) >= max(1, int(max_signals)):
                return False
            evidence_candidates = []
            best_eid = str(row.get("evidence_id", "")).strip()
            if best_eid:
                evidence_candidates.append(best_eid)
            for eid in row.get("match_evidence_ids", []):
                ee = str(eid).strip()
                if ee and ee not in evidence_candidates:
                    evidence_candidates.append(ee)
            for eid in evidence_candidates:
                if eid in seen_evidence_ids:
                    continue
                ev = evidence_by_id.get(eid, {})
                if not isinstance(ev, dict):
                    continue
                ev_views = ev.get("views", [])
                if not isinstance(ev_views, list):
                    ev_views = []
                selected_views: List[Dict[str, Any]] = []
                if allow_view:
                    if eid == best_eid and isinstance(row.get("views", []), list) and row.get("views", []):
                        selected_views = [v for v in row.get("views", []) if isinstance(v, dict)]
                    if not selected_views:
                        selected_views = [v for v in ev_views if isinstance(v, dict)]
                    selected_views = selected_views[: max(1, int(max_views_per_signal))]
                evidence_picks.append(
                    {
                        "evidence_id": eid,
                        "signal_id": str(row.get("signal_id", "")).strip(),
                        "signal_status": str(row.get("status", "")).strip().lower(),
                        "priority_score": float(row.get("priority_score", 0.0)),
                        "anchor_id": str(ev.get("anchor_id", "")).strip() or str(row.get("anchor_id", "")).strip(),
                        "summary": str(ev.get("summary", "")).strip(),
                        "anchor_description": str(ev.get("anchor_description", "")).strip(),
                        "confidence": max(
                            float(row.get("confidence", 0.0)),
                            _to_float(ev.get("confidence", 0.0), 0.0),
                        ),
                        "target_type": str(ev.get("target_type", "")).strip(),
                        "views": selected_views,
                    }
                )
                seen_evidence_ids.add(eid)
                return True
            return False

        evidence_status_order = ("present",) if present_gated_mode else ("unknown", "absent")
        for status_bucket in evidence_status_order:
            for row in signal_rows_for_evidence:
                if len(evidence_picks) >= max(1, int(max_signals)):
                    break
                if str(row.get("status", "")).lower() != status_bucket:
                    continue
                _append_one_evidence_from_row(row)
            if len(evidence_picks) >= max(1, int(max_signals)):
                break

        if not present_gated_mode:
            for row in signal_rows_for_evidence:
                if len(evidence_picks) >= max(1, int(max_signals)):
                    break
                _append_one_evidence_from_row(row)

        if (allow_evidence or allow_view) and not evidence_picks:
            fallback_pool: List[Dict[str, Any]] = []
            if present_gated_mode:
                present_evidence_ids = set()
                for row in signal_rows_for_evidence:
                    if str(row.get("status", "")).lower() != "present":
                        continue
                    eid_best = str(row.get("evidence_id", "")).strip()
                    if eid_best:
                        present_evidence_ids.add(eid_best)
                    for eid in row.get("match_evidence_ids", []):
                        ee = str(eid).strip()
                        if ee:
                            present_evidence_ids.add(ee)
                for eid in present_evidence_ids:
                    ev = evidence_by_id.get(eid, {})
                    if isinstance(ev, dict):
                        fallback_pool.append(ev)
            else:
                fallback_pool = [ev for ev in evidence_by_id.values() if isinstance(ev, dict)]

            fallback_evidence = sorted(
                fallback_pool,
                key=lambda ev: _to_float(ev.get("confidence", 0.0), 0.0),
                reverse=True,
            )[: max(1, int(max_signals))]
            for ev in fallback_evidence:
                eid = str(ev.get("evidence_id", "")).strip()
                if not eid:
                    continue
                ev_views = ev.get("views", [])
                if not isinstance(ev_views, list):
                    ev_views = []
                selected_views = [v for v in ev_views if isinstance(v, dict)][: max(1, int(max_views_per_signal))] if allow_view else []
                evidence_picks.append(
                    {
                        "evidence_id": eid,
                        "signal_id": "",
                        "signal_status": "",
                        "priority_score": 0.0,
                        "anchor_id": str(ev.get("anchor_id", "")).strip(),
                        "summary": str(ev.get("summary", "")).strip(),
                        "anchor_description": str(ev.get("anchor_description", "")).strip(),
                        "confidence": _to_float(ev.get("confidence", 0.0), 0.0),
                        "target_type": str(ev.get("target_type", "")).strip(),
                        "views": selected_views,
                    }
                )

        status_total = 0
        unknown_total = 0
        has_any_present = False
        has_conflict = False
        for row in signal_rows_all:
            present_cnt = int(row.get("present_count", 0))
            unknown_cnt = int(row.get("unknown_count", 0))
            absent_cnt = int(row.get("absent_count", 0))
            status_total += present_cnt + unknown_cnt + absent_cnt
            unknown_total += unknown_cnt
            has_any_present = has_any_present or str(row.get("status", "unknown")).lower() == "present"
            has_conflict = has_conflict or (present_cnt > 0 and absent_cnt > 0)
        unknown_ratio = float(unknown_total) / float(max(1, status_total))

        shown_view_count = sum(
            len([v for v in item.get("views", []) if isinstance(v, dict)])
            for item in evidence_picks
        ) if allow_view else 0
        shown_signal_count = len(signal_rows_for_context) if allow_signal else 0
        shown_evidence_count = len(evidence_picks) if allow_evidence else 0
        shown_present_signal_count = len(
            [row for row in signal_rows_for_context if str(row.get("status", "")).lower() == "present"]
        ) if allow_signal else 0
        shown_present_evidence_count = len(
            [item for item in evidence_picks if str(item.get("signal_status", "")).lower() == "present"]
        ) if allow_evidence else 0
        coverage_units = shown_signal_count + shown_evidence_count + shown_view_count
        if coverage_units >= max(8, int(max_signals) * 2):
            coverage_level = "high"
        elif coverage_units >= max(3, int(max_signals)):
            coverage_level = "moderate"
        else:
            coverage_level = "low"

        needs_refine = False
        refine_reason = ""
        if (allow_signal or bool(target_signals)) and not signal_rows_all:
            needs_refine = True
            refine_reason = "no_signal_match"
        elif not has_any_present and (allow_signal or bool(target_signals)):
            needs_refine = True
            refine_reason = "no_present_signal"
        elif unknown_ratio > 0.60:
            needs_refine = True
            refine_reason = "unknown_ratio_high"
        elif coverage_level == "low":
            needs_refine = True
            refine_reason = "coverage_low"

        coverage_flags = {
            "coverage_level": coverage_level,
            "has_conflict": bool(has_conflict),
            "needs_refine": bool(needs_refine),
            "refine_reason": refine_reason,
            "present_gated_mode": bool(present_gated_mode),
            "unknown_ratio": round(float(unknown_ratio), 4),
            "signal_rows_total": int(len(signal_rows_all)),
            "signal_rows_shown": int(shown_signal_count),
            "present_signal_rows_shown": int(shown_present_signal_count),
            "evidence_count_total": int(len(evidence_by_id)),
            "evidence_count_shown": int(shown_evidence_count),
            "present_evidence_shown": int(shown_present_evidence_count),
            "view_count_total": int(total_views_in_memory),
            "view_count_shown": int(shown_view_count),
        }

        evidence: List[Dict[str, Any]] = []
        lines: List[str] = []
        lines.append("Structured signal evidence from memory:")
        lines.append(f"Route: {str(active_route.get('route', 'signal_router'))}")
        lines.append("Level policy: " + ", ".join(target_levels))
        lines.append(
            "Evidence selection policy: "
            + ("present_only" if present_gated_mode else "fallback_no_present")
        )
        if target_signals:
            lines.append("Target signals: " + ", ".join(target_signals))
        if global_hint and allow_global:
            lines.append("Global hint: " + self._compact_text(global_hint, max_len=420))
        if signal_edges and (allow_signal or allow_evidence or allow_view):
            lines.append(f"Signal-evidence edges available: {len(signal_edges)}")

        if allow_signal:
            if not signal_rows_for_context:
                lines.append("No signal entries found in memory for current target signals.")
            for idx, row in enumerate(signal_rows_for_context, start=1):
                sid = str(row.get("signal_id", ""))
                sname = str(row.get("signal_name", ""))
                status = str(row.get("status", "unknown"))
                conf = float(row.get("confidence", 0.0))
                priority = float(row.get("priority_score", 0.0))
                rationale = self._compact_text(str(row.get("rationale", "")), max_len=260)
                anchor_id = str(row.get("anchor_id", ""))
                evidence_id = str(row.get("evidence_id", ""))
                lines.append(
                    f"[{idx}] {sid} ({sname}) status={status} conf={conf:.2f} "
                    f"priority={priority:.2f} matches={int(row.get('match_count', 0))} "
                    f"edges={int(row.get('edge_count', 0))}"
                )
                if rationale:
                    lines.append(f"    rationale: {rationale}")
                evidence.append(
                    {
                        "level": "signal",
                        "anchor_id": anchor_id,
                        "evidence_id": evidence_id,
                        "signal_id": sid,
                        "signal_status": status,
                        "view_id": "",
                        "mag": "",
                        "patch_xywh_level0": [],
                        "text": f"{sid} status={status}. {rationale}".strip(),
                        "score": conf,
                        "score_semantic": 0.0,
                        "score_lexical": 0.0,
                        "score_confidence": conf,
                        "score_signal": conf,
                    }
                )

        if allow_evidence or allow_view:
            lines.append("Anchor evidence summary:")
            if not evidence_picks:
                lines.append("No anchor evidence entries available.")
            for eidx, item in enumerate(evidence_picks, start=1):
                anchor_id = str(item.get("anchor_id", ""))
                evidence_id = str(item.get("evidence_id", ""))
                target_type = str(item.get("target_type", ""))
                conf = float(item.get("confidence", 0.0))
                signal_status = str(item.get("signal_status", "")).strip().lower()
                signal_id = str(item.get("signal_id", "")).strip()
                priority = float(item.get("priority_score", 0.0))
                summary_text_raw = str(item.get("summary", "")).strip()
                anchor_detail_raw = str(item.get("anchor_description", "")).strip()
                summary_text = self._compact_text(summary_text_raw, max_len=260)
                anchor_detail_text = self._compact_text(anchor_detail_raw, max_len=560)
                evidence_text = summary_text
                if anchor_detail_text:
                    if not evidence_text:
                        evidence_text = anchor_detail_text
                    elif anchor_detail_text.lower() != evidence_text.lower():
                        evidence_text = f"{evidence_text} Anchor detail: {anchor_detail_text}"
                lines.append(
                    f"[E{eidx}] evidence_id={evidence_id or 'N/A'} anchor={anchor_id or 'N/A'} "
                    f"type={target_type or 'N/A'} conf={conf:.2f} "
                    f"signal={signal_id or 'N/A'} status={signal_status or 'N/A'} priority={priority:.2f}"
                )
                if evidence_text:
                    lines.append(f"    summary: {evidence_text}")
                if allow_evidence:
                    evidence.append(
                        {
                            "level": "evidence",
                            "anchor_id": anchor_id,
                            "evidence_id": evidence_id,
                            "signal_id": signal_id,
                            "signal_status": signal_status,
                            "view_id": "",
                            "mag": "",
                            "patch_xywh_level0": [],
                            "text": evidence_text,
                            "score": conf,
                            "score_semantic": 0.0,
                            "score_lexical": 0.0,
                            "score_confidence": conf,
                            "score_signal": 0.0,
                        }
                    )
                if allow_view:
                    for vidx, v in enumerate(item.get("views", []), start=1):
                        if not isinstance(v, dict):
                            continue
                        view_id = str(v.get("view_id", ""))
                        mag = str(v.get("mag", ""))
                        caption = self._compact_text(str(v.get("caption", "")), max_len=200)
                        xywh = v.get("patch_xywh_level0", [])
                        if not (isinstance(xywh, list) and len(xywh) == 4):
                            xywh = []
                        lines.append(
                            f"    patch{vidx}: view={view_id or 'N/A'} mag={mag or 'N/A'} "
                            f"xywh={xywh if xywh else '[]'} caption={caption or 'N/A'}"
                        )
                        evidence.append(
                            {
                                "level": "view",
                                "anchor_id": anchor_id,
                                "evidence_id": evidence_id,
                                "signal_id": "",
                                "signal_status": "",
                                "view_id": view_id,
                                "mag": mag,
                                "patch_xywh_level0": xywh,
                                "text": caption,
                                "score": conf,
                                "score_semantic": 0.0,
                                "score_lexical": 0.0,
                                "score_confidence": conf,
                                "score_signal": 0.0,
                            }
                        )

        lines.append(
            "Coverage flags: "
            f"level={coverage_flags['coverage_level']} "
            f"unknown_ratio={coverage_flags['unknown_ratio']:.2f} "
            f"has_conflict={coverage_flags['has_conflict']} "
            f"needs_refine={coverage_flags['needs_refine']} "
            f"reason={coverage_flags['refine_reason'] or 'none'} "
            f"present_signal_rows={coverage_flags['present_signal_rows_shown']} "
            f"present_evidence={coverage_flags['present_evidence_shown']}"
        )

        context = "\n".join(lines).strip()
        if len(context) > max_chars:
            context = context[:max_chars] + "\n..."

        return {
            "slide_id": slide_id,
            "route": active_route,
            "needs_fine_grained": True,
            "target_signals": target_signals,
            "matched_signals": signal_rows_for_context,
            "signal_evidence_edges": signal_edges,
            "coverage_flags": coverage_flags,
            "context": context,
            "evidence": evidence,
        }
