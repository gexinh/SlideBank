"""Portable release checks; no WSI files, GPU models, or network calls required."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.export_ui_tree import build_ui_tree
from slidebank.bank import BankBuildConfig, PatchVLMConfig, build_bank, make_loader
from slidebank.config import load_config
from slidebank.interactive_agent import make_vlm_config, normalize_artifact_path
from slidebank.ui_v3_backend import load_bank_json
from slidebank.vlm import VLMConfig, _load_quilt_runtime, _read_api_key_file
from slidebank.wsi_agent.loader import RealWSILoader


ROOT = Path(__file__).resolve().parents[1]


class ReleasePortabilityTests(unittest.TestCase):
    def test_relative_build_export_and_load_keep_the_same_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                Path("data/slides").mkdir(parents=True)
                Path("data/slides/demo.svs").touch()
                Path("data/questions.csv").write_text(
                    "Slide,Question,A,B,C,D\ndemo.svs,Fixture question?,a,b,c,d\n"
                )
                config = BankBuildConfig(
                    "data/questions.csv", "data/slides", "outputs/memory_bank",
                    "outputs/memory_bank/artifacts", svs_pattern="**/*.svs", quiet=True,
                )

                def structural_slide(slide_path, out_dir):
                    # Replace only WSI/model processing; exercise the real
                    # persistence, export CLI, and tree loader together.
                    thumbnail = Path(out_dir) / "thumb" / "demo_thumbnail.png"
                    thumbnail.parent.mkdir(parents=True)
                    Image.new("RGB", (8, 8), "white").save(thumbnail)
                    return {
                        "slide_id": Path(slide_path).name,
                        "level3_global": {
                            "summary": "Structural fixture, no pathology observations.",
                            "coordinates": [0, 0, 8, 8],
                            "thumbnail_meta": {"thumbnail_path": str(thumbnail)},
                        },
                        "level2_regions": [],
                    }

                with patch("slidebank.bank.make_loader") as loader_factory:
                    loader_factory.return_value.load.side_effect = structural_slide
                    summary = build_bank(config)
                self.assertEqual(summary["slides_ingested"], 1, summary)
                self.assertEqual(summary["errors"], [])
                self.assertEqual(config.artifact_dir, "outputs/memory_bank/artifacts")
                self.assertEqual(config.csv_path, "data/questions.csv")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/export_ui_tree.py"),
                     "--bank-dir", "outputs/memory_bank", "--out-dir", "outputs/ui_tree"],
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                tree_file = root / "outputs/ui_tree/demo.ui_tree.json"
                bank = load_bank_json(tree_file)
                thumbnail = Path(bank["global"]["thumbnail"]["thumbnail_path"])
                self.assertTrue(thumbnail.is_absolute())
                self.assertTrue(thumbnail.is_file())
                with Image.open(thumbnail) as image:
                    self.assertEqual(image.size, (8, 8))
            finally:
                os.chdir(previous_cwd)

    def test_export_handles_legacy_cwd_paths_and_preserves_portable_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                for source_path, tree_dir in (
                    ("outputs/bank/artifacts/thumbnail.png", root / "outputs/ui_tree"),
                    ("../artifacts/thumbnail.png", root / "portable/ui_tree"),
                ):
                    with self.subTest(source_path=source_path):
                        image_file = (root / source_path) if source_path.startswith("outputs/") else (tree_dir / source_path)
                        image_file.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (8, 8), "white").save(image_file)
                        records = [{
                            "id": "demo.svs::global::thumbnail_meta", "slide_id": "demo.svs",
                            "text": json.dumps({"thumbnail_path": source_path}),
                        }]
                        tree = build_ui_tree(records, root / "readable/demo.jsonl")
                        if source_path.startswith("../"):
                            self.assertEqual(tree["global"]["thumbnail"]["thumbnail_path"], source_path)
                        tree_dir.mkdir(parents=True, exist_ok=True)
                        tree_file = tree_dir / "demo.ui_tree.json"
                        tree_file.write_text(json.dumps(tree))
                        loaded = load_bank_json(tree_file)
                        loaded_image = Path(loaded["global"]["thumbnail"]["thumbnail_path"])
                        self.assertEqual(loaded_image.resolve(), image_file.resolve())
                        self.assertTrue(loaded_image.is_file())
            finally:
                os.chdir(previous_cwd)

    def test_cli_help_runs_without_site_packages(self):
        # -S removes third-party packages, catching accidental model imports.
        for script in ("build_bank.py", "infer.py", "export_ui_tree.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, "-S", str(ROOT / "scripts" / script), "--help"],
                    cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_environment_references_expand_and_missing_names_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"bank": {"persist_dir": "${SLIDEBANK_TEST_ROOT}/bank"}}))
            with patch.dict(os.environ, {"SLIDEBANK_TEST_ROOT": tmp}):
                self.assertEqual(load_config(str(config_path))["bank"]["persist_dir"], f"{tmp}/bank")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "SLIDEBANK_TEST_ROOT"):
                    load_config(str(config_path))

    def test_quilt_requires_an_explicit_external_checkout_before_model_loading(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLIDEBANK_QUILT_LLAVA_REPO"):
                _load_quilt_runtime("wisdomik/Quilt-Llava-v1.5-7b", "auto", "")
            loader = RealWSILoader()
            with self.assertRaisesRegex(RuntimeError, "SLIDEBANK_QUILT_LLAVA_REPO"):
                loader._lazy_load_tile_llm_quilt_repo("unused-checkpoint")

    def test_bank_builder_forwards_the_explicit_quilt_checkout(self):
        config = BankBuildConfig("questions.csv", "slides", "bank", "artifacts",
                                patch_vlm=PatchVLMConfig(quilt_llava_repo="/example/quilt"))
        with patch("slidebank.bank.RealWSILoader") as loader_cls:
            make_loader(config)
        self.assertEqual(loader_cls.call_args.kwargs["quilt_llava_repo"], "/example/quilt")

    def test_local_workbench_preset_does_not_use_the_api_model_identifier(self):
        with patch.dict(os.environ, {"SLIDEBANK_QUILT_LLAVA_REPO": "/example/quilt"}):
            config = make_vlm_config(
                {"model_backend": "Local Model", "local_model": "Quilt-LLaVA"},
                {"vlm": {"model": "your-vision-model", "local_model_path": "local/checkpoint"}},
            )
        self.assertEqual(config.model_path, "local/checkpoint")
        self.assertEqual(config.quilt_llava_repo, "/example/quilt")
        qwen = make_vlm_config({"model_backend": "Local Model", "local_model": "Qwen2.5-VL-7B-Instruct"}, {})
        self.assertEqual(qwen.model_path, "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_runtime_path_expansion_is_portable(self):
        with patch.dict(os.environ, {"SLIDEBANK_TEST_ROOT": "/example/artifacts"}):
            self.assertEqual(normalize_artifact_path("${SLIDEBANK_TEST_ROOT}/slide.png"),
                             "/example/artifacts/slide.png")
        self.assertEqual(VLMConfig().model_path, "wisdomik/Quilt-Llava-v1.5-7b")

    def test_key_file_is_parsed_as_data_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "must-not-exist"
            key_file = Path(tmp) / "credentials.env"
            literal = f"$(touch {marker})"
            key_file.write_text(f"export OPENAI_API_KEY='{literal}'\n")
            self.assertEqual(_read_api_key_file(str(key_file), ["OPENAI_API_KEY"]), literal)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
