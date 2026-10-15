"""Exercise the public UI without patient assets, model loading, or API calls."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slidebank.ui_v3_backend import build_demo_registry, load_bank_json
from slidebank.config import load_config
from slidebank.interactive_agent import make_vlm_config


class WorkbenchReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        scratch = Path(cls.scratch.name)
        cls.environment = mock.patch.dict(os.environ, {
            "SLIDEBANK_UI_TREE_DIR": str(scratch / "trees"),
            "SLIDEBANK_BUILD_CONFIG": str(scratch / "missing-build.yaml"),
            "SLIDEBANK_INFER_CONFIG": str(scratch / "missing-infer.yaml"),
            "GRADIO_TEMP_DIR": str(scratch / "gradio"),
            "SLIDEBANK_BANK_DIR": "",
            "SLIDEBANK_ARTIFACT_DIR": "",
        })
        cls.environment.start()
        spec = importlib.util.spec_from_file_location("slidebank_release_app", ROOT / "app.py")
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.scratch.cleanup()

    def test_fresh_checkout_builds_an_empty_workbench(self):
        self.assertEqual(self.app.INITIAL_BANK, {})
        self.assertEqual(self.app.DEMO_REGISTRY, {})
        self.assertIn("export_ui_tree.py", self.app.INITIAL_STATUS)
        self.assertEqual(self.app.demo.title, "SlideBank Workbench")

    def test_tree_paths_are_relative_to_json_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "example.ui_tree.json"
            tree.write_text(json.dumps({
                "schema_version": "slidebank-ui-tree-v1",
                "slide_id": "example.svs",
                "global": {"thumbnail": {"thumbnail_path": "../artifacts/thumbnail.png"}},
                "tiles": [],
                "_agent": {"persist_dir": "..", "artifact_dir": "../artifacts"},
            }))
            bank = load_bank_json(tree)
            self.assertEqual(Path(bank["global"]["thumbnail"]["thumbnail_path"]).resolve(), (root.parent / "artifacts/thumbnail.png").resolve())
            self.assertEqual(Path(bank["_agent"]["persist_dir"]).resolve(), root.parent.resolve())
            self.assertEqual(len(build_demo_registry(root)), 1)

    def test_invalid_tree_schema_fails_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / "not-a-tree.json"
            tree.write_text(json.dumps({"global": {}, "tiles": []}))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_bank_json(tree)

    def test_send_without_bank_never_calls_inference(self):
        with mock.patch.object(self.app, "answer_question_v5") as answer:
            result = self.app.submit_chat("Question", [], {}, "", "", "Local Model", "Quilt-LLaVA", "Google", "Custom", "", "", "")
        answer.assert_not_called()
        self.assertIn("Load an exported evidence tree", result[-1])

    def test_missing_api_key_blocks_inference(self):
        with mock.patch.object(self.app, "answer_question_v5") as answer:
            result = self.app.submit_chat("Question", [], {"tiles": [{"tile_id": "R0C0"}]}, "", "", "API Model", "Quilt-LLaVA", "Google", "Custom", "provider-model-id", "", "")
        answer.assert_not_called()
        self.assertIn("API key is required", result[-1])

    def test_raw_slide_upload_has_actionable_error(self):
        result = self.app.load_prebuilt_bank("Upload UI-tree JSON", "example.svs", "", "Local Model", "Quilt-LLaVA", "Google", "Custom", "", "", "", [])
        self.assertIn("Build raw WSI inputs with scripts/build_bank.py", result[12])

    def test_quilt_preset_ignores_api_model_ids_in_default_build_config(self):
        build_defaults = load_config(str(ROOT / "configs/build.yaml"))
        infer_defaults = load_config(str(ROOT / "configs/infer.yaml"))
        runtime = self.app._runtime_config("Local Model", "Quilt-LLaVA", "Google", "Custom", "", "", "")
        with mock.patch.object(self.app, "BUILD_DEFAULTS", build_defaults):
            build_model = self.app._patch_vlm_config("Local Model", "Quilt-LLaVA", "Google", "Custom", "", "", "")
        send_model = make_vlm_config(runtime, infer_defaults)
        self.assertEqual(build_model.model_name, "wisdomik/Quilt-Llava-v1.5-7b")
        self.assertEqual(build_model.model_path, "wisdomik/Quilt-Llava-v1.5-7b")
        self.assertEqual(build_model.model_path, send_model.model_path)

    def test_explicit_quilt_checkpoint_overrides_remain_effective(self):
        configs = [
            {"backend": "local_hf", "model_name": "custom-quilt", "model_path": "/tmp/local-quilt"},
            {"backend": "api", "model_name": "api-model", "model_path": "api-model", "local_model_name": "custom-quilt", "local_model_path": "/tmp/local-quilt"},
        ]
        for patch_config in configs:
            with self.subTest(config=patch_config), mock.patch.object(self.app, "BUILD_DEFAULTS", {"patch_vlm": patch_config}):
                model = self.app._patch_vlm_config("Local Model", "Quilt-LLaVA", "Google", "Custom", "", "", "")
            self.assertEqual(model.model_name, "custom-quilt")
            self.assertEqual(model.model_path, "/tmp/local-quilt")

    def test_qwen_preset_uses_its_own_checkpoint_for_build_and_send(self):
        defaults = {"patch_vlm": {"backend": "api", "model_name": "api-model", "model_path": "api-model", "local_model_path": "custom-quilt"}}
        with mock.patch.object(self.app, "BUILD_DEFAULTS", defaults):
            build_model = self.app._patch_vlm_config("Local Model", "Qwen2.5-VL-7B-Instruct", "Google", "Custom", "", "", "")
        runtime = self.app._runtime_config("Local Model", "Qwen2.5-VL-7B-Instruct", "Google", "Custom", "", "", "")
        send_model = make_vlm_config(runtime, {"vlm": {"local_model_path": "custom-quilt"}})
        self.assertEqual(build_model.model_path, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(build_model.model_path, send_model.model_path)


if __name__ == "__main__":
    unittest.main()
