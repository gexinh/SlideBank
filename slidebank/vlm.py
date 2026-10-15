from __future__ import annotations

import inspect
import json
import mimetypes
import os
import re
import shlex
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest
import base64

from PIL import Image


def resolve_model_path(model_name: str) -> str:
    path = os.path.expanduser(str(model_name))
    if not os.path.isdir(path):
        return str(model_name)
    snapshots_dir = os.path.join(path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return path
    ref_path = os.path.join(path, "refs", "main")
    if os.path.isfile(ref_path):
        ref_id = Path(ref_path).read_text(encoding="utf-8").strip()
        candidate = os.path.join(snapshots_dir, ref_id)
        if os.path.isdir(candidate):
            return candidate
    snapshots = [
        os.path.join(snapshots_dir, name)
        for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return snapshots[0] if snapshots else path


def _patch_transformers_for_legacy_llava() -> None:
    try:
        from transformers import AutoConfig

        original_register = AutoConfig.register

        def safe_register(model_type, config, *args, **kwargs):
            try:
                return original_register(model_type, config, *args, **kwargs)
            except ValueError as exc:
                if "already used by a Transformers config" in str(exc):
                    return original_register(model_type, config, exist_ok=True)
                raise

        AutoConfig.register = safe_register
    except Exception:
        pass

    try:
        import torch
        import transformers.models.bloom.modeling_bloom as bloom_mod

        if not hasattr(bloom_mod, "_expand_mask"):
            def _expand_mask_bloom(mask, tgt_length=None):
                bsz, src_len = mask.shape
                tgt_len = tgt_length if tgt_length is not None else src_len
                return mask[:, None, None, :].to(torch.bool).expand(bsz, 1, tgt_len, src_len)

            bloom_mod._expand_mask = _expand_mask_bloom
        if not hasattr(bloom_mod, "_make_causal_mask"):
            def _make_causal_mask_bloom(input_shape, device=None, past_key_values_length=0):
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
            def _expand_mask_opt(mask, dtype, tgt_len=None):
                return AttentionMaskConverter._expand_mask(mask, dtype, tgt_len=tgt_len)

            opt_mod._expand_mask = _expand_mask_opt
        if not hasattr(opt_mod, "_make_causal_mask"):
            def _make_causal_mask_opt(input_shape, dtype, device=None, past_key_values_length=0):
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


def _runtime_device(device_map: str) -> str:
    if str(device_map).strip().lower() != "auto":
        return str(device_map)
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _infer_quilt_name(model_path_arg: str, resolved_path: str, get_model_name_from_path: Any) -> str:
    raw_name = str(get_model_name_from_path(model_path_arg))
    if raw_name.startswith("models--") and "--" in raw_name:
        parts = raw_name.split("--")
        if len(parts) >= 3:
            raw_name = parts[-1]
    if "llava" in raw_name.lower():
        return raw_name
    resolved_name = str(get_model_name_from_path(resolved_path))
    if "llava" in resolved_name.lower():
        return resolved_name
    try:
        with open(os.path.join(resolved_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "llava" in str(cfg.get("model_type", "")).lower():
            return "llava"
    except Exception:
        pass
    return resolved_name


def _load_quilt_runtime(model_path: str, device_map: str, quilt_llava_repo: str) -> Dict[str, Any]:
    repo_path = os.path.expanduser(os.path.expandvars(str(
        quilt_llava_repo or os.getenv("SLIDEBANK_QUILT_LLAVA_REPO", "")
    )))
    if not repo_path or not os.path.isdir(repo_path):
        raise RuntimeError(
            "Quilt-LLaVA requires its external source checkout. Set "
            "SLIDEBANK_QUILT_LLAVA_REPO or vlm.quilt_llava_repo "
            f"to that directory (configured value: {repo_path!r})."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    for mod_name in list(sys.modules.keys()):
        if mod_name == "llava" or mod_name.startswith("llava."):
            sys.modules.pop(mod_name, None)

    warnings.filterwarnings("ignore", message=r"Some weights of the model checkpoint at .* were not used.*")
    _patch_transformers_for_legacy_llava()

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    resolved_path = resolve_model_path(model_path)
    model_name = _infer_quilt_name(model_path, resolved_path, get_model_name_from_path)
    runtime_device = _runtime_device(device_map)
    load_device_map: Any = {"": runtime_device} if runtime_device.startswith("cuda") else "auto"
    if str(device_map).strip().lower() != "auto":
        load_device_map = {"": runtime_device}
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            resolved_path,
            None,
            model_name,
            False,
            False,
            device_map=load_device_map,
            device=runtime_device,
        )
    except TypeError:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            resolved_path,
            None,
            model_name,
            False,
            False,
            device=runtime_device,
        )
    return {
        "runtime": "quilt_llava",
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "conv_templates": conv_templates,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "model_name": model_name,
        "runtime_device": runtime_device,
    }


def _run_quilt_runtime(runtime: Dict[str, Any], image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    model_name = str(runtime.get("model_name", "")).lower()
    conv_mode = "llava_v1" if "v1" in model_name else "llava_v0"
    if conv_mode not in runtime["conv_templates"]:
        conv_mode = "llava_v0"
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(str(runtime.get("runtime_device", "cpu")))

    if bool(getattr(model.config, "mm_use_im_start_end", False)):
        query = (
            str(runtime["DEFAULT_IM_START_TOKEN"])
            + str(runtime["DEFAULT_IMAGE_TOKEN"])
            + str(runtime["DEFAULT_IM_END_TOKEN"])
            + "\n"
            + prompt
        )
    else:
        query = str(runtime["DEFAULT_IMAGE_TOKEN"]) + "\n" + prompt

    conv = runtime["conv_templates"][conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = (
        runtime["tokenizer_image_token"](
            full_prompt,
            tokenizer,
            int(runtime["IMAGE_TOKEN_INDEX"]),
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(model_device)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    image_tensor = runtime["process_images"]([image], runtime["image_processor"], model.config)
    if isinstance(image_tensor, list):
        image_tensor = [x.to(model_device, dtype=torch.float16) for x in image_tensor]
    else:
        image_tensor = image_tensor.to(model_device, dtype=torch.float16)

    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": image_tensor,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
    }
    try:
        if "return_legacy_cache" in inspect.signature(model.generate).parameters:
            kwargs["return_legacy_cache"] = True
    except Exception:
        pass
    with torch.inference_mode():
        try:
            output_ids = model.generate(**kwargs)
        except Exception as exc:
            if "DynamicCache" not in str(exc):
                raise
            kwargs["use_cache"] = False
            kwargs.pop("return_legacy_cache", None)
            output_ids = model.generate(**kwargs)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def _load_hf_runtime(model_path: str, device_map: str, dtype: str) -> Dict[str, Any]:
    import torch
    import transformers as tf
    from transformers import AutoConfig, AutoProcessor

    model_path = resolve_model_path(model_path)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype)
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map == "auto":
        kwargs["device_map"] = "auto"

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    errors = []
    model = None
    for name in candidates:
        cls = getattr(tf, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_path, **kwargs)
            break
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError(f"Failed to load HF VLM model_type={model_type}; recent errors: {' | '.join(errors[-3:])}")
    if device_map != "auto":
        model = model.to(device_map)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return {"runtime": "local_hf", "model": model, "processor": processor, "device_map": device_map}


def _run_hf_runtime(runtime: Dict[str, Any], image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    model = runtime["model"]
    processor = runtime["processor"]
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    try:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    except Exception:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    if runtime.get("device_map") != "auto":
        inputs = {k: v.to(runtime["device_map"]) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


_API_PROVIDER_ALIASES = {
    "gemini_api": "gemini",
    "google": "gemini",
    "google_gemini": "gemini",
    "openai_api": "openai",
    "openai_compatible_api": "openai_compatible",
    "openai-compatible": "openai_compatible",
}

_API_DEFAULT_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
}


def _canonical_api_provider(provider: str) -> str:
    raw = str(provider or "gemini").strip().lower().replace("-", "_")
    provider = _API_PROVIDER_ALIASES.get(raw, raw)
    if provider not in {"gemini", "openai", "openai_compatible"}:
        raise ValueError(f"Unsupported API provider: {provider}")
    return provider


def _provider_for_backend(backend: str, configured_provider: str) -> str:
    raw_backend = str(backend or "").strip().lower()
    if raw_backend == "gemini_api":
        return "gemini"
    if raw_backend == "openai_api":
        return "openai"
    if raw_backend == "openai_compatible_api":
        return "openai_compatible"
    return _canonical_api_provider(configured_provider or "gemini")


def _parse_shell_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        parts = shlex.split(value, posix=True)
    except ValueError:
        parts = []
    if len(parts) == 1:
        return parts[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value.strip()


def _read_api_key_file(path: str, names: list[str]) -> str:
    if not path:
        return ""
    key_path = Path(path).expanduser()
    if not key_path.exists():
        raise RuntimeError(f"API key file not found: {key_path}")
    wanted = {name for name in names if name}
    assignment = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = assignment.match(line)
        if not match:
            continue
        name, value = match.groups()
        if name not in wanted:
            continue
        parsed = _parse_shell_value(value)
        if parsed:
            return parsed
    return ""


def _resolve_api_key(config: "VLMConfig", provider: str) -> str:
    provider = _canonical_api_provider(provider)
    if config.api_key:
        return config.api_key.strip()
    if provider == "gemini" and config.gemini_api_key:
        return config.gemini_api_key.strip()

    default_env = _API_DEFAULT_ENV[provider]
    env_name = (config.api_key_env or "").strip()
    if not env_name and provider == "gemini":
        env_name = (config.gemini_api_env or "").strip()
    env_name = env_name or default_env
    if env_name:
        env_key = os.getenv(env_name, "").strip()
        if env_key:
            return env_key

    file_key = _read_api_key_file(config.api_key_file, [env_name, default_env])
    if file_key:
        return file_key
    source_hint = f"--api-key, {env_name}, or --api-key-file" if env_name else "--api-key or --api-key-file"
    raise RuntimeError(f"API key missing for provider={provider}. Provide {source_hint}.")


def _response_to_json_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "to_json_dict"):
        data = response.to_json_dict()
        if isinstance(data, dict):
            return data
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        if isinstance(data, dict):
            return data
    if hasattr(response, "to_json"):
        raw = response.to_json()
        if isinstance(raw, str):
            return json.loads(raw)
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    return {}


def _extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    data = _response_to_json_dict(response)
    candidates = data.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts).strip()
    return ""


def _run_gemini_api(config: "VLMConfig", image_path: str, prompt: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("Gemini API backend requires google-genai.") from exc

    api_key = _resolve_api_key(config, "gemini")
    http_options = None
    if config.api_version:
        http_options = types.HttpOptions(api_version=config.api_version)
    client = genai.Client(api_key=api_key, http_options=http_options)

    generation_config = None
    if config.max_new_tokens > 0:
        generation_config = types.GenerateContentConfig(
            max_output_tokens=int(config.max_new_tokens),
            temperature=0.0,
        )
    with Image.open(image_path) as img:
        response = client.models.generate_content(
            model=config.model_path,
            contents=[prompt, img.convert("RGB")],
            config=generation_config,
        )
    return _extract_gemini_text(response)


def _image_data_url(image_path: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _openai_chat_url(base_url: str) -> str:
    base = str(base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _extract_openai_compatible_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _run_openai_compatible_api(config: "VLMConfig", provider: str, image_path: str, prompt: str) -> str:
    provider = _canonical_api_provider(provider)
    api_key = _resolve_api_key(config, provider)
    base_url = config.api_base_url or ("https://api.openai.com/v1" if provider == "openai" else "")
    if not base_url:
        raise RuntimeError("openai_compatible provider requires --api-base-url.")

    payload = {
        "model": config.model_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            }
        ],
        "max_tokens": int(config.max_new_tokens),
        "temperature": 0,
    }
    request = urlrequest.Request(
        _openai_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=int(config.api_timeout_sec)) as response:
            body = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {body[:500]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc
    data = json.loads(body)
    return _extract_openai_compatible_text(data)


@dataclass
class VLMConfig:
    backend: str = "quilt_llava"
    model_path: str = "wisdomik/Quilt-Llava-v1.5-7b"
    quilt_llava_repo: str = ""
    device_map: str = "auto"
    dtype: str = "fp16"
    max_new_tokens: int = 64
    api_provider: str = ""
    api_key: str = ""
    api_key_env: str = ""
    api_key_file: str = ""
    api_base_url: str = ""
    api_version: str = "v1beta"
    api_timeout_sec: int = 300
    gemini_api_key: str = ""
    gemini_api_env: str = "GEMINI_API_KEY"


class VLMClient:
    def __init__(self, config: VLMConfig):
        self.config = config
        self.runtime: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        backend = self.config.backend.lower()
        if backend == "quilt_llava":
            self.runtime = _load_quilt_runtime(
                self.config.model_path,
                self.config.device_map,
                self.config.quilt_llava_repo,
            )
        elif backend == "local_hf":
            self.runtime = _load_hf_runtime(self.config.model_path, self.config.device_map, self.config.dtype)
        elif backend in {"api", "gemini_api", "openai_api", "openai_compatible_api"}:
            provider = _provider_for_backend(backend, self.config.api_provider)
            self.runtime = {"runtime": "api", "provider": provider}
        else:
            raise ValueError(f"Unknown VLM backend: {self.config.backend}")

    def generate(self, image_path: str, prompt: str) -> str:
        if self.runtime is None:
            self.load()
        assert self.runtime is not None
        backend = self.runtime["runtime"]
        if backend == "quilt_llava":
            return _run_quilt_runtime(self.runtime, image_path, prompt, self.config.max_new_tokens)
        if backend == "local_hf":
            return _run_hf_runtime(self.runtime, image_path, prompt, self.config.max_new_tokens)
        if backend == "api":
            provider = str(self.runtime.get("provider", "gemini"))
            if provider == "gemini":
                return _run_gemini_api(self.config, image_path, prompt)
            if provider in {"openai", "openai_compatible"}:
                return _run_openai_compatible_api(self.config, provider, image_path, prompt)
            raise RuntimeError(f"Unsupported API provider: {provider}")
        raise RuntimeError(f"Unsupported runtime: {backend}")
