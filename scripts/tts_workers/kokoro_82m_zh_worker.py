#!/usr/bin/env python3
"""CUDA-only Kokoro 82M Chinese JSON-lines worker."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping

from _autoanki_tts_worker_common import (
    KOKORO_BACKEND,
    KOKORO_MODEL_ID,
    RequestError,
    clean_batch_outputs_on_error,
    cuda_inventory,
    gpu_synthesis_lease,
    installed_path,
    leased_cuda_inventory,
    load_installation,
    revision_for_assets,
    run_worker,
    seed_synthesis,
    status_response,
    success_artifact,
    validate_synthesis_request,
    validate_synthesis_items,
    write_tensor_wav,
)


VOICE_ID = "zm_010"
NATIVE_SAMPLE_RATE = 24000


def _installation_and_voice() -> tuple[dict, dict[str, str]]:
    installation = load_installation(KOKORO_BACKEND, KOKORO_MODEL_ID)
    voice_metadata = installation.get("voice")
    if (
            not isinstance(voice_metadata, dict)
            or voice_metadata.get("id") != VOICE_ID):
        raise RequestError("The installed Kokoro voice is not zm_010.")
    config = installed_path(
        installation, "model_config", regular_file=True)
    checkpoint = installed_path(
        installation, "model_checkpoint", regular_file=True)
    voice = installed_path(
        installation, "voice", regular_file=True)
    revision = revision_for_assets((
        ("model-config", config),
        ("model-checkpoint", checkpoint),
        ("voice", voice),
    ), text_assets=(("voice-id", VOICE_ID),))
    return installation, {VOICE_ID: revision}


def _status() -> dict:
    model_revision = None
    voices: dict[str, str] = {}
    installation_ok = False
    try:
        installation, voices = _installation_and_voice()
        model_revision = installation["model"]["revision"]
        installation_ok = True
    except Exception:
        pass
    gpu_available, device_name, _problem = leased_cuda_inventory(KOKORO_BACKEND)
    return status_response(
        backend=KOKORO_BACKEND,
        model_id=KOKORO_MODEL_ID,
        model_revision=model_revision,
        runtime_available=installation_ok,
        gpu_available=gpu_available,
        device_name=device_name,
        voices=voices if installation_ok else {})


def _validate_kokoro_item(data: Mapping[str, Any]) -> None:
    if data["pitch"] != 0.0:
        raise RequestError(
            "The installed Kokoro path does not implement pitch "
            "adjustment; use 0.0.")


def _load_pipeline(installation: Mapping[str, Any]):
    gpu_available, _device_name, problem = cuda_inventory()
    if not gpu_available:
        raise RequestError(problem or "CUDA is unavailable.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from kokoro import KModel, KPipeline

    model = KModel(
        repo_id=KOKORO_MODEL_ID,
        config=str(installed_path(
            installation, "model_config", regular_file=True)),
        model=str(installed_path(
            installation, "model_checkpoint", regular_file=True)))
    model = model.to("cuda").eval()
    if model.device.type != "cuda":
        raise RequestError("Kokoro did not load its model on CUDA.")
    pipeline = KPipeline(
        lang_code="z",
        repo_id=KOKORO_MODEL_ID,
        model=model,
        device="cuda")
    return pipeline


def _synthesize_item(
        pipeline,
        installation: Mapping[str, Any],
        data: Mapping[str, Any]) -> dict:
    import torch

    seed_synthesis(data["seed"])
    voice_path = installed_path(
        installation, "voice", regular_file=True)
    chunks = []
    for result in pipeline(
            data["text"],
            voice=str(voice_path),
            speed=data["speed"]):
        audio = getattr(result, "audio", None)
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            raise RequestError("Kokoro returned an invalid audio chunk.")
        chunks.append(audio.detach().cpu().reshape(-1))
    if not chunks:
        raise RequestError("Kokoro returned no audio.")
    if len(chunks) == 1:
        audio = chunks[0]
    else:
        silence = torch.zeros(
            int(NATIVE_SAMPLE_RATE * 0.08), dtype=torch.float32)
        joined = []
        for index, chunk in enumerate(chunks):
            if index:
                joined.append(silence)
            joined.append(chunk)
        audio = torch.cat(joined)
    torch.cuda.synchronize()
    write_tensor_wav(
        audio,
        NATIVE_SAMPLE_RATE,
        data["sample_rate_hz"],
        data["output_path"])
    return success_artifact(data)


def _synthesize(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voice()
    data = validate_synthesis_request(
        request,
        installation,
        supported_languages={"chinese"},
        voice_revisions=voices)
    _validate_kokoro_item(data)
    with gpu_synthesis_lease(KOKORO_BACKEND):
        pipeline = _load_pipeline(installation)
        return _synthesize_item(pipeline, installation, data)


def _synthesize_many(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voice()
    items = validate_synthesis_items(
        request,
        installation,
        supported_languages={"chinese"},
        voice_revisions=voices)
    for item in items:
        _validate_kokoro_item(item)
    with clean_batch_outputs_on_error(items):
        with gpu_synthesis_lease(KOKORO_BACKEND):
            pipeline = _load_pipeline(installation)
            artifacts = [
                _synthesize_item(pipeline, installation, item)["artifact"]
                for item in items
            ]
    return {"artifacts": artifacts}


def handle(
        request: Mapping[str, Any],
        operation: str) -> Mapping[str, Any]:
    if operation == "status":
        return _status()
    if operation == "synthesize_many":
        return _synthesize_many(request)
    return _synthesize(request)


if __name__ == "__main__":
    sys.exit(run_worker(
        backend=KOKORO_BACKEND,
        model_id=KOKORO_MODEL_ID,
        handler=handle))
