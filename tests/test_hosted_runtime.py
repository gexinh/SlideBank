"""Hosting contracts without model downloads, CUDA initialization, or API calls."""
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slidebank.config import load_config
from slidebank.interactive_agent import get_vlm_client, make_vlm_config
from slidebank.vlm import VLMConfig, _load_hf_runtime, _run_hf_runtime


def load_runtime(zero_gpu=False):
    sdk = types.ModuleType("spaces")
    config = types.ModuleType("spaces.config")
    config.Config = types.SimpleNamespace(zero_gpu=zero_gpu)
    sdk.decorated_calls = []

    def gpu(**options):
        sdk.options = options

        def decorate(fn):
            def allocated(**kwargs):
                sdk.decorated_calls.append(kwargs)
                sdk.in_gpu = True
                try:
                    return fn(**kwargs)
                finally:
                    sdk.in_gpu = False
            return allocated
        return decorate

    sdk.GPU = gpu
    spec = importlib.util.spec_from_file_location("hosted_runtime_under_test", ROOT / "hf_space/runtime.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "true" if zero_gpu else "false"}), mock.patch.dict(sys.modules, {"spaces": sdk, "spaces.config": config}):
        spec.loader.exec_module(module)
    return module, sdk


class HostedRuntimeTests(unittest.TestCase):
    def test_launcher_import_is_model_free_on_cpu(self):
        code = """
import importlib.abc, os, sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'spaces'}:
            raise AssertionError('Model library imported during CPU launcher import: ' + fullname)
sys.meta_path.insert(0, RejectModels())
import space_app
assert os.environ['SLIDEBANK_HOSTED'] == '1'
assert os.environ['SLIDEBANK_UI_TREE_DIR'].endswith('/demos/ui_tree')
assert os.environ['SLIDEBANK_ARTIFACT_DIR'].endswith('/demos/artifacts')
assert os.environ['SLIDEBANK_HOST'] == '0.0.0.0'
assert os.environ['SLIDEBANK_PORT'] == '7860'
assert 'app' not in sys.modules
"""
        environment = dict(os.environ)
        for key in list(environment):
            if key.startswith("SLIDEBANK_"):
                environment.pop(key)
        environment["SPACES_ZERO_GPU"] = "false"
        completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cpu_preparation_skips_model_loading(self):
        runtime, _ = load_runtime(False)
        with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "false"}), mock.patch("slidebank.interactive_agent.get_vlm_client") as client:
            self.assertFalse(runtime.prepare_hosted_model({}))
        client.assert_not_called()

    def test_hosted_configuration_cannot_select_visitor_api(self):
        runtime, _ = load_runtime(False)
        defaults = load_config(str(ROOT / "configs/hosted.yaml"))
        state = runtime.hosted_runtime(defaults)
        config = make_vlm_config(state, defaults)
        self.assertEqual(config.backend, "local_hf")
        self.assertEqual(config.model_path, "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(config.device_map, "cuda:0")
        self.assertTrue(config.allow_remote_download)
        self.assertEqual(config.max_image_pixels, 401408)
        self.assertEqual(state["api_key"], "")
        self.assertEqual(state["api_base_url"], "")
        self.assertFalse(VLMConfig().allow_remote_download)

    def test_zero_gpu_covers_the_complete_shared_answer_function(self):
        runtime, sdk = load_runtime(True)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        supplied = dict(message="What morphology is present?", bank={"slide_id": "fixture"}, runtime={"hosted": True}, defaults={}, previous_messages=[])

        def actual_answer(**kwargs):
            self.assertTrue(sdk.in_gpu)
            self.assertEqual(kwargs, supplied)
            return "real-flow-result"

        with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "true"}), mock.patch.dict(sys.modules, {"torch": torch}), mock.patch("slidebank.agent_inference_v5.answer_question_v5", side_effect=actual_answer) as answer:
            result = runtime.answer_on_gpu(**supplied)
        self.assertEqual(result, "real-flow-result")
        self.assertEqual(sdk.options["duration"], 120)
        answer.assert_called_once()

    def test_failed_gpu_allocation_never_runs_v5_on_cpu(self):
        runtime, _ = load_runtime(True)
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "true"}), mock.patch.dict(sys.modules, {"torch": torch}), mock.patch("slidebank.agent_inference_v5.answer_question_v5") as answer:
            with self.assertRaisesRegex(RuntimeError, "did not allocate CUDA"):
                runtime.answer_on_gpu(message="Question", bank={}, runtime={}, defaults={}, previous_messages=[])
        answer.assert_not_called()

    def test_model_preparation_occurs_only_when_explicitly_requested(self):
        with mock.patch("slidebank.interactive_agent.get_vlm_client") as client:
            runtime, _ = load_runtime(True)
            client.assert_not_called()
            with mock.patch.dict(os.environ, {"SPACES_ZERO_GPU": "true"}):
                self.assertTrue(runtime.prepare_hosted_model(load_config(str(ROOT / "configs/hosted.yaml"))))
        client.return_value.load.assert_called_once()

    def test_remote_model_fetch_requires_explicit_opt_in(self):
        model = mock.Mock()
        model.to.return_value = model
        tf = types.ModuleType("transformers")
        tf.AutoConfig = mock.Mock()
        tf.AutoConfig.from_pretrained.return_value = types.SimpleNamespace(model_type="qwen2_5_vl")
        tf.AutoProcessor = mock.Mock()
        tf.AutoModelForImageTextToText = mock.Mock()
        tf.AutoModelForImageTextToText.from_pretrained.return_value = model
        torch = types.SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": tf}):
            _load_hf_runtime("fixture/model", "cuda:0", "bf16")
            for loader in [tf.AutoConfig, tf.AutoProcessor, tf.AutoModelForImageTextToText]:
                self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])
            _load_hf_runtime("fixture/model", "cuda:0", "bf16", allow_remote_download=True, max_image_pixels=401408)
            for loader in [tf.AutoConfig, tf.AutoProcessor, tf.AutoModelForImageTextToText]:
                self.assertFalse(loader.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertEqual(tf.AutoProcessor.from_pretrained.call_args.kwargs["max_pixels"], 401408)

    def test_decoder_only_output_excludes_router_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture.png"
            Image.new("RGB", (8, 8)).save(image)
            input_ids = mock.Mock()
            input_ids.shape = (1, 3)
            input_ids.to.return_value = input_ids
            processor = mock.Mock(return_value={"input_ids": input_ids})
            processor.apply_chat_template.return_value = "router prompt"
            processor.batch_decode.return_value = ['{"target_levels":["signal"]}']
            model = mock.Mock()
            model.config.is_encoder_decoder = False
            model.generate.return_value = [[1, 2, 3, 7, 8]]
            torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
            with mock.patch.dict(sys.modules, {"torch": torch}):
                result = _run_hf_runtime({"model": model, "processor": processor, "device_map": "cuda:0"}, str(image), "Question", 32)
        self.assertIn("target_levels", result)
        self.assertEqual(processor.batch_decode.call_args.args[0], [[7, 8]])

    def test_client_cache_separates_download_permissions(self):
        with mock.patch("slidebank.interactive_agent._VLM_CACHE", {}), mock.patch("slidebank.interactive_agent.VLMClient") as client:
            client.side_effect = lambda cfg: object()
            offline = get_vlm_client(VLMConfig(backend="local_hf", model_path="fixture/model"))
            hosted = get_vlm_client(VLMConfig(backend="local_hf", model_path="fixture/model", allow_remote_download=True))
        self.assertIsNot(offline, hosted)


class HostedInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        cls.environment = mock.patch.dict(os.environ, {
            "SLIDEBANK_HOSTED": "true",
            "SPACES_ZERO_GPU": "false",
            "SLIDEBANK_UI_TREE_DIR": str(Path(cls.scratch.name) / "missing"),
            "SLIDEBANK_INFER_CONFIG": str(ROOT / "configs/hosted.yaml"),
            "GRADIO_TEMP_DIR": str(Path(cls.scratch.name) / "gradio"),
            "GRADIO_ANALYTICS_ENABLED": "False",
        })
        cls.environment.start()
        spec = importlib.util.spec_from_file_location("slidebank_hosted_app", ROOT / "app.py")
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        cls.scratch.cleanup()

    def test_cpu_workbench_hides_local_controls_and_disables_qa(self):
        self.assertFalse(self.app.image_source.visible)
        self.assertFalse(self.app.model_backend.visible)
        self.assertFalse(self.app.local_model_group.visible)
        self.assertFalse(self.app.run_bank_btn.visible)
        self.assertFalse(self.app.send_btn.interactive)
        registered = {getattr(fn.fn, "__name__", "") for fn in self.app.demo.fns.values()}
        self.assertNotIn("run_bank_extraction", registered)
        self.assertIn("submit_chat", registered)

    def test_cpu_send_never_falls_through_to_api_or_model(self):
        bank = {"slide_id": "fixture", "global": {}, "tiles": [{"tile_id": "R0", "anchors": []}]}
        with mock.patch.object(self.app.hosted_runtime, "answer_on_gpu") as gpu, mock.patch.object(self.app, "answer_question_v5") as local:
            result = self.app.submit_chat("Question", [], bank, "", "", "API Model", "Quilt-LLaVA", "OpenAI", "visitor-model", "", "visitor-key", "https://untrusted.example")
        gpu.assert_not_called()
        local.assert_not_called()
        self.assertIn("requires ZeroGPU", result[-1])
        self.assertEqual(result[-2]["api_key"], "")


if __name__ == "__main__":
    unittest.main()
