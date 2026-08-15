#!/usr/bin/env python3
"""CUDA-only Fun-CosyVoice3-0.5B-2512 JSON-lines worker."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

from _autoanki_tts_worker_common import (
    COSY_BACKEND,
    COSY_MODEL_ID,
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


VOICE_IDS = (
    "neutral-english",
    "neutral-french",
    "neutral-mandarin",
)


def _installation_and_voices() -> tuple[dict, dict[str, str]]:
    installation = load_installation(COSY_BACKEND, COSY_MODEL_ID)
    installed_path(installation, "model", regular_file=False)
    installed_path(installation, "source", regular_file=False)
    prompt_audio = installed_path(
        installation, "prompt_audio", regular_file=True)
    voice = installation.get("voice")
    if not isinstance(voice, dict):
        raise RequestError("The installed CosyVoice voice is invalid.")
    prompt_text = voice.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise RequestError("The installed CosyVoice prompt text is missing.")
    revision = revision_for_assets(
        (("prompt-audio", prompt_audio),),
        text_assets=(("prompt-text", prompt_text),))
    return installation, {
        voice_id: revision
        for voice_id in VOICE_IDS
    }


def _status() -> dict:
    model_revision = None
    voices: dict[str, str] = {}
    installation_ok = False
    try:
        installation, voices = _installation_and_voices()
        model_revision = installation["model"]["revision"]
        installation_ok = True
    except Exception:
        pass
    gpu_available, device_name, _problem = leased_cuda_inventory(COSY_BACKEND)
    return status_response(
        backend=COSY_BACKEND,
        model_id=COSY_MODEL_ID,
        model_revision=model_revision,
        runtime_available=installation_ok,
        gpu_available=gpu_available,
        device_name=device_name,
        voices=voices if installation_ok else {})


def _validate_cosy_item(data: Mapping[str, Any]) -> None:
    if data["pitch"] != 0.0:
        raise RequestError(
            "The installed CosyVoice path does not implement pitch "
            "adjustment; use 0.0.")


def _load_model(installation: Mapping[str, Any]):
    source_path = installed_path(
        installation, "source", regular_file=False)
    matcha_path = source_path / "third_party" / "Matcha-TTS"
    if not matcha_path.is_dir():
        raise RequestError("The CosyVoice Matcha-TTS submodule is missing.")
    sys.path.insert(0, str(matcha_path))
    sys.path.insert(0, str(source_path))

    import onnxruntime

    # ORT uses CUDA 12 side-by-side with PyTorch's CUDA 13 runtime.  Loading
    # the wheel's explicitly installed NVIDIA libraries before CosyVoice
    # constructs its speech-tokenizer session avoids resolving incompatible
    # CUDA symbols from PyTorch's already-loaded process image.
    preload_dlls = getattr(onnxruntime, "preload_dlls", None)
    if not callable(preload_dlls):
        raise RequestError(
            "The installed ONNX Runtime cannot preload its CUDA libraries.")
    preload_dlls(directory="")
    if "CUDAExecutionProvider" not in \
            onnxruntime.get_available_providers():
        raise RequestError(
            "ONNX Runtime's CUDA execution provider is unavailable.")

    gpu_available, _device_name, problem = cuda_inventory()
    if not gpu_available:
        raise RequestError(problem or "CUDA is unavailable.")

    import numpy as np
    import soundfile
    import torch
    import torchaudio.functional

    def load_wav(path: str, target_sr: int, min_sr: int = 16000):
        samples, sample_rate = soundfile.read(
            path,
            dtype="float32",
            always_2d=True)
        if sample_rate < min_sr:
            raise RequestError(
                f"Reference audio sample rate {sample_rate} is below "
                f"{min_sr} Hz.")
        speech = torch.from_numpy(
            np.ascontiguousarray(samples.T)).mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            speech = torchaudio.functional.resample(
                speech,
                sample_rate,
                target_sr)
        return speech

    # Torchaudio 2.11 routes file decoding through TorchCodec, whose CUDA
    # wheel requires optional NPP libraries.  SoundFile is already an
    # upstream CosyVoice dependency and deterministically handles the local
    # PCM reference WAV without pulling another multimedia runtime into the
    # worker.
    from cosyvoice.cli import frontend as cosy_frontend
    cosy_frontend.load_wav = load_wav
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(
        model_dir=str(installed_path(
            installation, "model", regular_file=False)),
        fp16=True,
        load_trt=False,
        load_vllm=False)
    if getattr(getattr(model, "model", None), "device", None) is None:
        raise RequestError("CosyVoice did not report its synthesis device.")
    if model.model.device.type != "cuda":
        raise RequestError(
            "CosyVoice did not load its synthesis model on CUDA.")
    speech_tokenizer_providers = (
        model.frontend.speech_tokenizer_session.get_providers())
    if (
            not speech_tokenizer_providers
            or speech_tokenizer_providers[0] != "CUDAExecutionProvider"):
        raise RequestError(
            "CosyVoice's speech tokenizer did not load on CUDA.")
    return model


def _synthesize_item(
        model: Any,
        installation: Mapping[str, Any],
        data: Mapping[str, Any]) -> dict:
    import torch
    seed_synthesis(data["seed"])
    voice = installation["voice"]
    prompt_audio = installed_path(
        installation, "prompt_audio", regular_file=True)
    chunks = []
    # The pinned model card explicitly uses this Mandarin reference
    # text/audio pair with inference_zero_shot for both English and Chinese.
    # The same multilingual zero-shot path is used for French.  CosyVoice's
    # cross-lingual method is documented here for fine-grained controls and
    # languages outside the model's normal text frontend, not as a required
    # replacement for English zero-shot synthesis.
    outputs = model.inference_zero_shot(
        data["text"],
        voice["prompt_text"],
        str(prompt_audio),
        stream=False,
        speed=data["speed"])
    for output in outputs:
        speech = output.get("tts_speech")
        if not isinstance(speech, torch.Tensor) or speech.numel() == 0:
            raise RequestError("CosyVoice returned an invalid audio chunk.")
        chunks.append(speech.detach().cpu().reshape(-1))
    if not chunks:
        raise RequestError("CosyVoice returned no audio.")
    torch.cuda.synchronize()
    audio = torch.cat(chunks)
    write_tensor_wav(
        audio,
        int(model.sample_rate),
        data["sample_rate_hz"],
        data["output_path"])
    return success_artifact(data)


def _synthesize(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voices()
    data = validate_synthesis_request(
        request,
        installation,
        supported_languages={"english", "french", "chinese"},
        voice_revisions=voices)
    _validate_cosy_item(data)
    with gpu_synthesis_lease(COSY_BACKEND):
        model = _load_model(installation)
        return _synthesize_item(model, installation, data)


def _synthesize_many(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voices()
    items = validate_synthesis_items(
        request,
        installation,
        supported_languages={"english", "french", "chinese"},
        voice_revisions=voices)
    for item in items:
        _validate_cosy_item(item)
    with clean_batch_outputs_on_error(items):
        with gpu_synthesis_lease(COSY_BACKEND):
            model = _load_model(installation)
            artifacts = [
                _synthesize_item(model, installation, item)["artifact"]
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
        backend=COSY_BACKEND,
        model_id=COSY_MODEL_ID,
        handler=handle))
