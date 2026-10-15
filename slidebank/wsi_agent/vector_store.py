from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Set


class LocalInMemoryCollection:
    """Local persistent collection with metadata filtering + lexical ranking.

    Persistence layout:
    - Legacy single-file: <persist_path> (json with {"rows":[...]})
    - New sharded mode:
      - <persist_path>.shards/index.json
      - <persist_path>.shards/<safe_slide>__<hash>.jsonl
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.shard_dir = f"{persist_path}.shards" if persist_path else ""
        self.shard_index_path = os.path.join(self.shard_dir, "index.json") if self.shard_dir else ""

        self._rows_by_id: Dict[str, Dict[str, Any]] = {}
        self._ids_by_slide: Dict[str, List[str]] = {}
        self._loaded_slides: Set[str] = set()
        self._dirty_slides: Set[str] = set()
        self._all_loaded = False
        self._shard_index: Dict[str, Dict[str, Any]] = {}

        if self.persist_path:
            self._load_from_disk()

    @staticmethod
    def _safe_name(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", text or "")

    @staticmethod
    def _matches_where(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
        if where is None:
            return True
        return all(meta.get(k) == v for k, v in where.items())

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {tok for tok in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(tok) >= 2}

    @staticmethod
    def _row_slide_id(row: Dict[str, Any]) -> str:
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            return "__global__"
        sid = str(meta.get("slide_id", "")).strip()
        return sid if sid else "__global__"

    @staticmethod
    def _meta_slide_id(meta: Dict[str, Any]) -> str:
        sid = str((meta or {}).get("slide_id", "")).strip()
        return sid if sid else "__global__"

    def _shard_filename(self, slide_id: str) -> str:
        safe = self._safe_name(slide_id) or "global"
        h = hashlib.md5(slide_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe}__{h}.jsonl"

    def _register_row_loaded(self, row: Dict[str, Any]) -> None:
        if not isinstance(row, dict):
            return
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            return
        doc = str(row.get("document", ""))
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        normalized = {"id": row_id, "document": doc, "metadata": meta}
        slide_id = self._row_slide_id(normalized)
        self._rows_by_id[row_id] = normalized
        ids = self._ids_by_slide.setdefault(slide_id, [])
        if row_id not in ids:
            ids.append(row_id)

    def _write_shard_index(self) -> None:
        if not self.shard_index_path:
            return
        os.makedirs(self.shard_dir, exist_ok=True)
        total_rows = 0
        payload = {"version": 2, "slides": {}}
        for slide_id, rec in sorted(self._shard_index.items(), key=lambda x: x[0]):
            if not isinstance(rec, dict):
                continue
            row_count = int(rec.get("row_count", 0))
            if row_count <= 0:
                continue
            file_name = str(rec.get("file", "")).strip()
            if not file_name:
                continue
            payload["slides"][slide_id] = {"file": file_name, "row_count": row_count}
            total_rows += row_count
        with open(self.shard_index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self.persist_path:
            pointer = {
                "storage": "sharded_jsonl",
                "version": 2,
                "shard_dir": self.shard_dir,
                "slide_count": len(payload["slides"]),
                "total_rows": total_rows,
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(pointer, f, ensure_ascii=False, indent=2)
            with open(f"{self.persist_path}.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(pointer, ensure_ascii=False) + "\n")

    def _load_shard_index(self) -> None:
        self._shard_index = {}
        if not self.shard_index_path or not os.path.exists(self.shard_index_path):
            return
        try:
            with open(self.shard_index_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            slides = obj.get("slides", {}) if isinstance(obj, dict) else {}
            if not isinstance(slides, dict):
                return
            for slide_id, rec in slides.items():
                if not isinstance(rec, dict):
                    continue
                file_name = str(rec.get("file", "")).strip()
                row_count = int(rec.get("row_count", 0))
                if not file_name or row_count < 0:
                    continue
                self._shard_index[str(slide_id)] = {"file": file_name, "row_count": row_count}
        except Exception:
            self._shard_index = {}

    def _iter_legacy_rows(self) -> Iterable[Dict[str, Any]]:
        if not self.persist_path:
            return []
        jsonl_path = f"{self.persist_path}.jsonl"
        if os.path.exists(jsonl_path):
            def _iter_jsonl() -> Iterable[Dict[str, Any]]:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            row = json.loads(s)
                        except Exception:
                            continue
                        if isinstance(row, dict) and {"id", "document", "metadata"}.issubset(set(row.keys())):
                            yield row
            return _iter_jsonl()

        if not os.path.exists(self.persist_path):
            return []

        def _iter_json() -> Iterable[Dict[str, Any]]:
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                return
            rows = payload.get("rows", []) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                return
            for row in rows:
                if isinstance(row, dict) and {"id", "document", "metadata"}.issubset(set(row.keys())):
                    yield row
        return _iter_json()

    def _migrate_legacy_to_shards(self) -> None:
        if not self.persist_path or not self.shard_dir:
            return

        rows_iter = self._iter_legacy_rows()
        # If iterator is empty, keep state untouched.
        has_any = False
        tmp_dir = f"{self.shard_dir}.tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        index: Dict[str, Dict[str, Any]] = {}
        for row in rows_iter:
            has_any = True
            slide_id = self._row_slide_id(row)
            rec = index.get(slide_id)
            if rec is None:
                rec = {"file": self._shard_filename(slide_id), "row_count": 0}
                index[slide_id] = rec
            out_path = os.path.join(tmp_dir, rec["file"])
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rec["row_count"] = int(rec["row_count"]) + 1

        if not has_any:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        if os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir, ignore_errors=True)
        os.makedirs(self.shard_dir, exist_ok=True)
        for name in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, name)
            dst = os.path.join(self.shard_dir, name)
            if os.path.isfile(src):
                shutil.move(src, dst)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        self._shard_index = index
        self._write_shard_index()

    def _load_from_disk(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        if os.path.exists(self.shard_index_path):
            self._load_shard_index()
            return
        # Try one-time migration from legacy monolithic fallback file.
        if os.path.exists(self.persist_path) or os.path.exists(f"{self.persist_path}.jsonl"):
            self._migrate_legacy_to_shards()
            self._load_shard_index()

    def _ensure_slide_loaded(self, slide_id: str) -> None:
        sid = str(slide_id or "__global__")
        if sid in self._loaded_slides:
            return
        rec = self._shard_index.get(sid)
        if not isinstance(rec, dict):
            self._loaded_slides.add(sid)
            self._ids_by_slide.setdefault(sid, [])
            return
        file_name = str(rec.get("file", "")).strip()
        if not file_name:
            self._loaded_slides.add(sid)
            self._ids_by_slide.setdefault(sid, [])
            return
        path = os.path.join(self.shard_dir, file_name)
        ids = self._ids_by_slide.setdefault(sid, [])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    row_id = str(row.get("id", "")).strip()
                    if not row_id:
                        continue
                    doc = str(row.get("document", ""))
                    meta = row.get("metadata", {})
                    if not isinstance(meta, dict):
                        meta = {}
                    normalized = {"id": row_id, "document": doc, "metadata": meta}
                    self._rows_by_id[row_id] = normalized
                    if row_id not in ids:
                        ids.append(row_id)
        self._loaded_slides.add(sid)

    def _ensure_all_loaded(self) -> None:
        if self._all_loaded:
            return
        for slide_id in list(self._shard_index.keys()):
            self._ensure_slide_loaded(slide_id)
        self._all_loaded = True

    def _remove_id_from_slide(self, slide_id: str, row_id: str) -> None:
        ids = self._ids_by_slide.get(slide_id, [])
        if row_id in ids:
            ids.remove(row_id)
        if not ids:
            self._ids_by_slide[slide_id] = []

    def _upsert_row(self, row_id: str, document: str, metadata: Dict[str, Any]) -> None:
        rid_slide = row_id.split("::", 1)[0] if "::" in row_id else ""
        if rid_slide and rid_slide in self._shard_index:
            self._ensure_slide_loaded(rid_slide)
        new_slide = self._meta_slide_id(metadata)
        if new_slide in self._shard_index:
            self._ensure_slide_loaded(new_slide)

        old = self._rows_by_id.get(row_id)
        if old is not None:
            old_slide = self._row_slide_id(old)
            if old_slide != new_slide:
                self._remove_id_from_slide(old_slide, row_id)
                self._dirty_slides.add(old_slide)

        self._rows_by_id[row_id] = {
            "id": row_id,
            "document": document,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        ids = self._ids_by_slide.setdefault(new_slide, [])
        if row_id not in ids:
            ids.append(row_id)
        self._dirty_slides.add(new_slide)

    def _persist_dirty_slides(self) -> None:
        if not self.persist_path or not self._dirty_slides:
            return
        os.makedirs(self.shard_dir, exist_ok=True)
        for slide_id in sorted(self._dirty_slides):
            ids = self._ids_by_slide.get(slide_id, [])
            rows = [self._rows_by_id[rid] for rid in ids if rid in self._rows_by_id]
            existing = self._shard_index.get(slide_id, {})
            file_name = str(existing.get("file", "")).strip() if isinstance(existing, dict) else ""
            if not file_name:
                file_name = self._shard_filename(slide_id)
            shard_path = os.path.join(self.shard_dir, file_name)

            if not rows:
                if os.path.exists(shard_path):
                    os.remove(shard_path)
                if slide_id in self._shard_index:
                    del self._shard_index[slide_id]
                continue

            with open(shard_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._shard_index[slide_id] = {"file": file_name, "row_count": len(rows)}
            self._loaded_slides.add(slide_id)

        self._dirty_slides.clear()
        self._write_shard_index()

    def clear(self) -> None:
        self._rows_by_id = {}
        self._ids_by_slide = {}
        self._loaded_slides = set()
        self._dirty_slides = set()
        self._all_loaded = False
        self._shard_index = {}

        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
        if self.persist_path and os.path.exists(f"{self.persist_path}.jsonl"):
            os.remove(f"{self.persist_path}.jsonl")
        if self.shard_dir and os.path.isdir(self.shard_dir):
            shutil.rmtree(self.shard_dir, ignore_errors=True)

    @staticmethod
    def _build_search_text(row: Dict[str, Any]) -> str:
        doc = str(row.get("document", "")).strip()
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        meta_keys = (
            "slide_id",
            "level",
            "global_subtype",
            "semantic_tag",
            "region_id",
            "patch_id",
            "parent_region_id",
            "anchor_id",
            "evidence_id",
            "signal_id",
            "target_type",
        )
        meta_text = " ".join(str(meta.get(k, "")).strip() for k in meta_keys if str(meta.get(k, "")).strip())
        return f"{doc} {meta_text}".strip()

    @classmethod
    def _score_query(cls, query_text: str, row: Dict[str, Any]) -> float:
        q = (query_text or "").strip().lower()
        if not q:
            return 0.0
        q_tokens = cls._tokenize(q)
        if not q_tokens:
            return 0.0
        search_text = cls._build_search_text(row).lower()
        doc_tokens = cls._tokenize(search_text)
        if not doc_tokens:
            return 0.0
        overlap = q_tokens.intersection(doc_tokens)
        if not overlap:
            return 0.0
        coverage = float(len(overlap)) / float(max(1, len(q_tokens)))
        density = float(len(overlap)) / float(max(1, len(doc_tokens)))
        phrase_bonus = 0.15 if len(q) >= 4 and q in search_text else 0.0
        return max(0.0, coverage + (0.25 * density) + phrase_bonus)

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        # Keep original add() behavior: raise on duplicate.
        self._ensure_all_loaded()
        for i, row_id in enumerate(ids):
            if row_id in self._rows_by_id:
                raise ValueError(f"Duplicate id in collection add(): {row_id}")
            meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            doc = documents[i] if i < len(documents) else ""
            self._upsert_row(row_id=row_id, document=doc, metadata=meta)
        self._persist_dirty_slides()

    def upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        for i, row_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            doc = documents[i] if i < len(documents) else ""
            self._upsert_row(row_id=row_id, document=doc, metadata=meta)
        self._persist_dirty_slides()

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[List[Any]]]:
        slide_id = None
        if isinstance(where, dict):
            sid = where.get("slide_id")
            if sid is not None:
                slide_id = str(sid).strip()

        if slide_id:
            self._ensure_slide_loaded(slide_id)
            candidate_ids = list(self._ids_by_slide.get(slide_id, []))
            candidates = [self._rows_by_id[rid] for rid in candidate_ids if rid in self._rows_by_id]
        else:
            self._ensure_all_loaded()
            candidates = list(self._rows_by_id.values())

        candidates = [r for r in candidates if self._matches_where(r.get("metadata", {}), where)]

        out_ids: List[List[str]] = []
        out_docs: List[List[str]] = []
        out_metas: List[List[Dict[str, Any]]] = []
        out_dists: List[List[float]] = []

        limit = max(1, int(n_results))
        for query_text in query_texts:
            ranked: List[Any] = []
            for idx, row in enumerate(candidates):
                score = self._score_query(query_text, row)
                # Keep legacy key name "distances" for call-site compatibility.
                distance = max(0.0, 1.0 - min(score, 1.0))
                ranked.append((-(score), distance, idx, row))
            ranked.sort(key=lambda x: (x[0], x[2]))
            top = ranked[:limit]
            out_ids.append([item[3]["id"] for item in top])
            out_docs.append([item[3]["document"] for item in top])
            out_metas.append([item[3]["metadata"] for item in top])
            out_dists.append([float(item[1]) for item in top])

        return {"ids": out_ids, "documents": out_docs, "metadatas": out_metas, "distances": out_dists}
