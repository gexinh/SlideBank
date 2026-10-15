"""Launch the preset evidence workbench on Hugging Face Spaces."""
from __future__ import annotations

import os
from pathlib import Path


def configure_hosted_environment() -> None:
    root = Path(__file__).resolve().parent
    os.environ["SLIDEBANK_HOSTED"] = "1"
    defaults = {
        "SLIDEBANK_UI_TREE_DIR": str(root / "demos/ui_tree"),
        "SLIDEBANK_BANK_DIR": str(root / "demos/bank"),
        "SLIDEBANK_ARTIFACT_DIR": str(root / "demos/artifacts"),
        "SLIDEBANK_INFER_CONFIG": str(root / "configs/hosted.yaml"),
        "SLIDEBANK_HOST": "0.0.0.0",
        "SLIDEBANK_PORT": "7860",
        "GRADIO_ANALYTICS_ENABLED": "False",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


configure_hosted_environment()

# Load the Spaces SDK before model libraries when ZeroGPU is active. Importing
# this launcher is model-free; preparation is an explicit startup phase below.
from hf_space import runtime as hosted  # noqa: E402


if __name__ == "__main__":
    from slidebank.config import load_config

    hosted.prepare_hosted_model(load_config(os.environ["SLIDEBANK_INFER_CONFIG"]))
    from app import demo, DEFAULT_HOST, DEFAULT_PORT

    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=DEFAULT_HOST,
        server_port=DEFAULT_PORT,
        share=False,
        show_error=False,
    )
