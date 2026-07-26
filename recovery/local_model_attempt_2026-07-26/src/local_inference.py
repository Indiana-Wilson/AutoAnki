"""Archived Ollama discovery and local client construction."""

import json
import os
import re
import hashlib
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from source_generation.model_catalog import (
    is_local_source_model,
    source_model_profile,
)


LOCAL_INFERENCE_POLICY_VERSION = 4
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_EXPECTED_LOCAL_QUANTIZATION = "Q4_K_M"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_AUTOMATIC_REPAIR_TEMPERATURES = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
)


class LocalInferenceUnavailableError(RuntimeError):
    """The selected local runtime or exact frozen model is unavailable."""


class LocalInferenceRequestError(RuntimeError):
    def __init__(self, message, *, status_code):
        super().__init__(message)
        self.status_code = status_code


def ollama_base_url():
    configured = os.environ.get(
        "AUTOANKI_OLLAMA_BASE_URL",
        DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    parsed = urlparse(configured)
    if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment):
        raise ValueError(
            "AUTOANKI_OLLAMA_BASE_URL must be a plain HTTP(S) server URL.")
    return configured


def _read_json(url, *, method="GET", body=None, timeout=10):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=data,
        headers=headers,
        method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(
                response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise LocalInferenceRequestError(
            "Ollama rejected the local inference request"
            + (f": {detail}" if detail else "."),
            status_code=error.code) from error
    except (
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError) as error:
        raise LocalInferenceUnavailableError(
            "The shared Ollama service is unavailable at "
            f"{ollama_base_url()}. Start it with "
            "`systemctl --user start ollama`.") from error


def local_inference_policy(model_id):
    profile = source_model_profile(model_id)
    if not profile.local:
        return None
    return {
        "schema_version": LOCAL_INFERENCE_POLICY_VERSION,
        "num_ctx": profile.max_context_tokens,
        "temperature": 0.2,
        "automatic_repair_temperature_schedule": list(
            _AUTOMATIC_REPAIR_TEMPERATURES[
                :profile.max_automatic_repairs]),
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
        "keep_alive": "10m",
        "seed_policy": "sha256(job_id,chunk_id,attempt,model_id)",
        "gpu_offload_guard": "api_ps_positive_size_vram",
    }


def local_automatic_repair_temperature(model_id, dispatch_ordinal):
    profile = source_model_profile(model_id)
    if not profile.local:
        raise ValueError(
            "Automatic local temperature requires a local model.")
    if (
            isinstance(dispatch_ordinal, bool)
            or not isinstance(dispatch_ordinal, int)
            or not 1 <= dispatch_ordinal
            <= profile.max_automatic_repairs):
        raise ValueError(
            "Automatic local repair ordinal is outside the model limit.")
    return _AUTOMATIC_REPAIR_TEMPERATURES[dispatch_ordinal - 1]


def ollama_model_snapshot(model_id):
    """Return the exact shared Ollama artifact identity used for authorization."""
    if not is_local_source_model(model_id):
        return None
    profile = source_model_profile(model_id)
    base_url = ollama_base_url()
    tags = _read_json(f"{base_url}/api/tags")
    models = tags.get("models", ()) if isinstance(tags, dict) else ()
    matching = tuple(
        model
        for model in models
        if (
            isinstance(model, dict)
            and model.get("name") == profile.provider_model
            and (
                model.get("model") is None
                or model.get("model") == profile.provider_model)))
    if len(matching) != 1:
        raise LocalInferenceUnavailableError(
            f"The shared local model {profile.provider_model!r} is not "
            "installed. Run `ollama pull "
            f"{profile.provider_model}` once; every repository for this "
            "user will then reuse the same model.")
    model = matching[0]
    digest = model.get("digest")
    if (
            not isinstance(digest, str)
            or _DIGEST_PATTERN.fullmatch(digest) is None):
        raise LocalInferenceUnavailableError(
            "Ollama returned an invalid model digest.")
    details = model.get("details")
    if not isinstance(details, dict):
        details = {}
    quantization_level = details.get("quantization_level")
    if quantization_level != _EXPECTED_LOCAL_QUANTIZATION:
        raise LocalInferenceUnavailableError(
            f"The shared local model {profile.provider_model!r} is "
            f"{quantization_level!r}, not the required "
            f"{_EXPECTED_LOCAL_QUANTIZATION}. Pull the exact supported "
            "Ollama tag before authorizing a job.")
    version = _read_json(f"{base_url}/api/version")
    return {
        "schema_version": 1,
        "provider": "ollama",
        "model_id": model_id,
        "provider_model": profile.provider_model,
        "digest": digest,
        "size": model.get("size"),
        "modified_at": model.get("modified_at"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": quantization_level,
        "ollama_version": (
            version.get("version")
            if isinstance(version, dict)
            else None),
        "base_url": base_url,
        "inference_policy": local_inference_policy(model_id),
    }


def ensure_matching_ollama_model(model_id, expected_snapshot=None):
    current = ollama_model_snapshot(model_id)
    if expected_snapshot is not None and current != expected_snapshot:
        raise LocalInferenceUnavailableError(
            "The installed local model or inference runtime changed after "
            "this job was authorized. Recalculate the estimate and create "
            "a new job; AutoAnki will not silently resume with different "
            "weights or settings.")
    return current


def _ollama_gpu_runtime(
        provider_model,
        *,
        expected_context_length=None):
    """Verify and describe GPU offload for one just-completed request."""
    value = _read_json(f"{ollama_base_url()}/api/ps")
    models = value.get("models", ()) if isinstance(value, dict) else ()
    matching = tuple(
        model
        for model in models
        if (
            isinstance(model, dict)
            and model.get("name") == provider_model
            and (
                model.get("model") is None
                or model.get("model") == provider_model)))
    if len(matching) != 1:
        raise LocalInferenceUnavailableError(
            "Ollama completed the request, but /api/ps did not report the "
            f"exact running model {provider_model!r}. AutoAnki cannot "
            "verify GPU acceleration and will not accept this response.")
    runtime = matching[0]
    digest = runtime.get("digest")
    size = runtime.get("size")
    size_vram = runtime.get("size_vram")
    context_length = runtime.get("context_length")
    if (
            not isinstance(digest, str)
            or _DIGEST_PATTERN.fullmatch(digest) is None):
        raise LocalInferenceUnavailableError(
            "Ollama returned an invalid running-model digest for "
            f"{provider_model!r}; model identity cannot be audited.")
    if (
            isinstance(size_vram, bool)
            or not isinstance(size_vram, int)
            or size_vram <= 0):
        raise LocalInferenceUnavailableError(
            f"Ollama reported no GPU offload for {provider_model!r} "
            "(size_vram is zero, missing, or invalid). Check `ollama ps` "
            "and the NVIDIA runtime before retrying.")
    if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size_vram > size
            or isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length <= 0):
        raise LocalInferenceUnavailableError(
            "Ollama returned incomplete /api/ps allocation metadata for "
            f"{provider_model!r}; GPU acceleration cannot be audited.")
    if (
            expected_context_length is not None
            and context_length != expected_context_length):
        raise LocalInferenceUnavailableError(
            "Ollama reported a different running context for "
            f"{provider_model!r}: requested {expected_context_length}, "
            f"reported {context_length}. Another local request may have "
            "replaced the runner before GPU acceleration was audited.")
    return {
        "digest": digest,
        "size": size,
        "size_vram": size_vram,
        "context_length": context_length,
        "gpu_fraction": size_vram / size,
    }


class _OllamaResponses:
    def create(self, **request_options):
        model = request_options.get("model")
        prompt = request_options.get("input")
        if not isinstance(model, str) or not model:
            raise ValueError("A local Ollama model tag is required.")
        if not isinstance(prompt, str):
            raise ValueError(
                "Local Ollama source requests require one text prompt.")
        text_options = request_options.get("text", {})
        response_format = (
            text_options.get("format")
            if isinstance(text_options, dict)
            else None)
        if (
                not isinstance(response_format, dict)
                or response_format.get("type") != "json_schema"
                or not isinstance(response_format.get("schema"), dict)):
            raise ValueError(
                "Local source requests require a strict JSON schema.")
        reasoning = request_options.get("reasoning", {})
        effort = (
            reasoning.get("effort")
            if isinstance(reasoning, dict)
            else None)
        if effort not in {"none", "low"}:
            raise ValueError(
                "Local reasoning effort must be none or low.")
        extra_body = request_options.get("extra_body", {})
        options = (
            dict(extra_body.get("options", {}))
            if isinstance(extra_body, dict)
            and isinstance(extra_body.get("options"), dict)
            else {})
        max_output_tokens = request_options.get(
            "max_output_tokens")
        if (
                isinstance(max_output_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or max_output_tokens < 1):
            raise ValueError(
                "A positive local output limit is required.")
        options["num_predict"] = max_output_tokens
        body = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt,
            }],
            "format": response_format["schema"],
            "stream": False,
            "think": effort == "low",
            "options": options,
            "keep_alive": (
                extra_body.get("keep_alive", "10m")
                if isinstance(extra_body, dict)
                else "10m"),
        }
        try:
            value = _read_json(
                f"{ollama_base_url()}/api/chat",
                method="POST",
                body=body,
                timeout=1_800)
        except LocalInferenceUnavailableError as error:
            raise ConnectionError(str(error)) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                "Ollama returned an unreadable local response.")
        if value.get("model") != model:
            raise LocalInferenceUnavailableError(
                "Ollama returned a response for a different model than "
                f"requested: expected {model!r}, received "
                f"{value.get('model')!r}.")
        gpu_runtime = _ollama_gpu_runtime(
            model,
            expected_context_length=options.get("num_ctx"))
        message = value.get("message")
        content = (
            message.get("content")
            if isinstance(message, dict)
            else None)
        if not isinstance(content, str):
            content = ""
        thinking = (
            message.get("thinking")
            if isinstance(message, dict)
            and isinstance(message.get("thinking"), str)
            else "")
        input_tokens = int(value.get("prompt_eval_count") or 0)
        output_tokens = int(value.get("eval_count") or 0)
        response_id = "ollama-" + hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return {
            "id": response_id,
            "model": model,
            "status": (
                "incomplete"
                if value.get("done_reason") == "length"
                else "completed"),
            "service_tier": "local",
            "output_text": content,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens": output_tokens,
                "output_tokens_details": {
                    # Ollama does not currently partition eval_count between
                    # visible and thinking tokens.
                    "reasoning_tokens": 0,
                },
                "total_tokens": input_tokens + output_tokens,
            },
            "local_runtime": {
                "done": value.get("done"),
                "done_reason": value.get("done_reason"),
                "created_at": value.get("created_at"),
                "total_duration": value.get("total_duration"),
                "load_duration": value.get("load_duration"),
                "prompt_eval_duration": value.get(
                    "prompt_eval_duration"),
                "eval_duration": value.get("eval_duration"),
                "thinking": thinking,
                **gpu_runtime,
            },
        }


class OllamaResponsesClient:
    """Small facade matching the subset of ``OpenAI.responses`` we use."""

    def __init__(self):
        self.responses = _OllamaResponses()


def create_ollama_client():
    return OllamaResponsesClient()
