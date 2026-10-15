"""ZeroGPU allocation around the complete, unchanged V5 answer flow.

Importing this module never downloads a model or initializes CUDA. CPU Spaces
retain evidence browsing, and cannot fall through to local CPU inference.
"""
from __future__ import annotations

import os
from typing import Any, Dict


def _enabled(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


HOSTED_MODE = _enabled(os.environ.get("SLIDEBANK_HOSTED", ""))
GPU_DURATION = max(15, min(120, int(os.environ.get("SLIDEBANK_GPU_DURATION", "60"))))
_spaces = None
_sdk_error = ""
_model_error = ""
if _enabled(os.environ.get("SPACES_ZERO_GPU", "")):
    try:
        import spaces as _spaces
        from spaces.config import Config as _SpacesConfig

        if not _SpacesConfig.zero_gpu:
            _sdk_error = "The Spaces SDK has not enabled ZeroGPU."
    except Exception as exc:
        _sdk_error = f"The ZeroGPU SDK could not start ({type(exc).__name__})."


def availability_error() -> str:
    if not _enabled(os.environ.get("SPACES_ZERO_GPU", "")):
        return "Live QA requires ZeroGPU. Evidence browsing remains available on CPU."
    if _sdk_error:
        return _sdk_error + " Evidence browsing remains available."
    if _spaces is None:
        return "The Spaces SDK is unavailable. Evidence browsing remains available."
    if _model_error:
        return _model_error
    return ""


def hosted_runtime(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Server-owned model selection; visitors cannot choose an API or endpoint."""
    cfg = defaults.get("vlm", {})
    return {
        "hosted": True,
        "model_backend": "Local Model",
        "local_model": str(cfg.get("model_path", "Qwen/Qwen2.5-VL-3B-Instruct")),
        "api_provider": "",
        "api_model": "",
        "custom_api_model": "",
        "api_key": "",
        "api_base_url": "",
    }


def prepare_hosted_model(defaults: Dict[str, Any]) -> bool:
    """Explicit launcher startup phase using ZeroGPU's CUDA emulation.

    This is never called during module import, and returns without importing a
    model library on CPU hardware. Preloading lets the SDK preserve CUDA state
    for its GPU workers instead of loading weights inside every allocation.
    """
    global _model_error
    if availability_error():
        return False
    try:
        from slidebank.interactive_agent import get_vlm_client, make_vlm_config

        config = make_vlm_config(hosted_runtime(defaults), defaults)
        get_vlm_client(config).load()
        print(f"Hosted demonstration model ready: {config.model_path}", flush=True)
        return True
    except Exception as exc:
        _model_error = "The hosted model could not be prepared. Evidence browsing remains available."
        print(f"Hosted model preparation failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def _answer_on_gpu(*, message, bank, runtime, defaults, previous_messages):
    unavailable = availability_error()
    if unavailable:
        raise RuntimeError(unavailable)
    # This check runs after GPU allocation, never during CPU startup.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ZeroGPU did not allocate CUDA. Please retry when GPU capacity is available.")
    from slidebank.agent_inference_v5 import answer_question_v5

    return answer_question_v5(
        message=message,
        bank=bank,
        runtime=runtime,
        defaults=defaults,
        previous_messages=previous_messages,
    )


# The SDK decorator is otherwise a no-op on cpu-basic; an explicit availability
# gate prevents that no-op from accidentally running a multi-billion model on CPU.
answer_on_gpu = (
    _spaces.GPU(duration=GPU_DURATION)(_answer_on_gpu)
    if _spaces is not None and not _sdk_error
    else _answer_on_gpu
)
