"""Check the shipped public evidence bundle without WSI/model/API execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from scripts.prepare_demo import (COLLECTION, EXCLUDED_FIELDS, SCHEMA, embedded_json,
                                  image_references, sync_web_bundle, validate_bundle)
from slidebank.bank import open_memory_bank
from slidebank.interactive_agent import infer_bank_paths, thumbnail_path_for_bank
from slidebank.ui_v3_backend import load_bank_json, render_global_overlay


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "demos"


class DemoBundleTests(unittest.TestCase):
    def test_manifest_checksums_and_every_image_reference_are_valid(self):
        self.assertEqual(validate_bundle(DEMOS), {"samples": 1, "images": 67, "rows": 356})
        manifest = json.loads((DEMOS / "manifest.json").read_text())
        self.assertEqual(manifest["default_demo"], manifest["samples"][0]["id"])
        for name in manifest["samples"][0]["images"]:
            with Image.open(DEMOS / name) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.info, {})
                image.verify()

    def test_real_bank_supports_global_and_question_guided_retrieval(self):
        manifest = json.loads((DEMOS / "manifest.json").read_text())
        sample = manifest["samples"][0]
        bank = open_memory_bank(str(DEMOS / manifest["bank_dir"]), quiet=True)
        self.assertTrue(bank.has_slide(sample["slide_id"]))
        global_view = bank.consult_global_view(sample["slide_id"])
        self.assertNotIn("error", global_view)
        self.assertTrue(global_view["summary"])
        for question in sample["questions"]:
            result = bank.build_signal_patch_context(question, max_chars=3200)
            self.assertNotIn("error", result)
            self.assertTrue(result["context"])
            self.assertTrue(result["evidence"])
        tree = load_bank_json(DEMOS / sample["tree"])
        persist, artifacts = infer_bank_paths(tree, {})
        self.assertEqual(Path(persist).resolve(), (DEMOS / "bank").resolve())
        self.assertEqual(Path(artifacts).resolve(), (DEMOS / "artifacts").resolve())
        self.assertTrue(Path(thumbnail_path_for_bank(tree)).is_file())
        self.assertGreater(render_global_overlay(tree).width, 0)

    def test_bundle_still_resolves_after_moving_to_a_different_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            moved = Path(tmp) / "renamed_demo"
            shutil.copytree(DEMOS, moved)
            self.assertEqual(validate_bundle(moved)["rows"], 356)
            manifest = json.loads((moved / "manifest.json").read_text())
            sample = manifest["samples"][0]
            tree = load_bank_json(moved / sample["tree"])
            self.assertTrue(Path(thumbnail_path_for_bank(tree)).resolve().is_relative_to(moved))
            bank = open_memory_bank(str(moved / "bank"), quiet=True)
            self.assertNotIn("error", bank.consult_global_view(sample["slide_id"]))

    def test_no_clinical_fields_gold_answers_or_raw_slides_are_packaged(self):
        manifest = json.loads((DEMOS / "manifest.json").read_text())
        sample = manifest["samples"][0]
        shard = DEMOS / "bank" / f"{COLLECTION}.shards" / f"{sample['id']}.jsonl"
        rows = [json.loads(line) for line in shard.read_text().splitlines()]
        found = set()

        def inspect(value):
            if isinstance(value, dict):
                found.update(name.lower() for name in value if name.lower() in EXCLUDED_FIELDS)
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
            elif isinstance(value, str):
                parsed = embedded_json(value)
                if parsed:
                    inspect(parsed[1])

        inspect(rows)
        self.assertFalse(found)
        self.assertEqual({row["metadata"]["slide_id"] for row in rows}, {sample["slide_id"]})
        for path in DEMOS.rglob("*"):
            if path.is_file():
                self.assertIn(path.suffix, {".json", ".jsonl", ".png", ".md"})
        for name in image_references(rows):
            self.assertFalse(Path(name).is_absolute())

    @staticmethod
    def tiny_bundle(root, image_name):
        image_file = root / "artifacts" / image_name
        image_file.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), "white").save(image_file)
        tree_file = root / "ui_tree" / "fixture.ui_tree.json"
        tree_file.parent.mkdir(parents=True, exist_ok=True)
        tree_file.write_text(json.dumps({"schema_version": "slidebank-ui-tree-v1", "slide_id": "fixture.svs",
                                         "global": {"thumbnail": {"thumbnail_path": "../artifacts/" + image_name}}, "tiles": []}))
        shards = root / "bank" / f"{COLLECTION}.shards"
        shards.mkdir(parents=True, exist_ok=True)
        (shards / "fixture.jsonl").write_text("")
        (shards / "index.json").write_text(json.dumps({"version": 2, "slides": {"fixture.svs": {"file": "fixture.jsonl", "row_count": 0}}}))
        files = [image_file, tree_file]
        manifest = {"schema_version": SCHEMA, "default_demo": "fixture", "bank_dir": "bank",
                    "samples": [{"id": "fixture", "slide_id": "fixture.svs", "tree": "ui_tree/fixture.ui_tree.json",
                                 "images": ["artifacts/" + image_name], "row_count": 0}],
                    "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_web_sync_removes_only_previously_managed_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "web"
            self.tiny_bundle(source, "old.png")
            sync_web_bundle(source, target)
            unrelated = target / "artifacts" / "user-not-in-manifest.txt"
            unrelated.write_text("keep this user file")
            self.tiny_bundle(source, "new.png")
            result = sync_web_bundle(source, target)
            self.assertEqual(result["stale_managed_files_removed"], 1)
            self.assertFalse((target / "artifacts/old.png").exists())
            self.assertTrue((target / "artifacts/new.png").is_file())
            self.assertEqual(unrelated.read_text(), "keep this user file")
            self.assertFalse((target / "bank").exists())
            self.assertEqual(sync_web_bundle(source, target)["stale_managed_files_removed"], 0)

    def test_web_sync_rejects_unsafe_previous_manifest_without_deleting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "web"
            self.tiny_bundle(source, "fixture.png")
            target.mkdir()
            marker = Path(tmp) / "keep.txt"
            marker.write_text("keep")
            manifest = json.loads((source / "manifest.json").read_text())
            manifest["samples"][0]["images"] = ["../keep.txt"]
            (target / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                sync_web_bundle(source, target)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
