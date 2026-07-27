#!/usr/bin/env python3
"""CUDA-only Style-Bert-VITS2 JP-Extra JSON-lines worker."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping

from _autoanki_tts_worker_common import (
    RequestError,
    STYLE_BACKEND,
    STYLE_MODEL_ID,
    clean_batch_outputs_on_error,
    cuda_inventory,
    gpu_synthesis_lease,
    installed_path,
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


VOICE_ID = "neutral-japanese"


def _installation_and_voice() -> tuple[dict, dict[str, str]]:
    installation = load_installation(STYLE_BACKEND, STYLE_MODEL_ID)
    assets = [
        ("config", installed_path(
            installation, "voice_config", regular_file=True)),
        ("model", installed_path(
            installation, "voice_model", regular_file=True)),
        ("style-vectors", installed_path(
            installation, "voice_styles", regular_file=True)),
    ]
    # BERT is part of the synthesis identity even though it is not a voice.
    installed_path(installation, "bert_model", regular_file=False)
    revision = revision_for_assets(
        assets,
        text_assets=(("style", "Neutral"),))
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
    gpu_available, device_name, _problem = cuda_inventory()
    return status_response(
        backend=STYLE_BACKEND,
        model_id=STYLE_MODEL_ID,
        model_revision=model_revision,
        runtime_available=installation_ok,
        gpu_available=gpu_available,
        device_name=device_name,
        voices=voices if installation_ok else {})


def _load_model(installation: Mapping[str, Any]):
    gpu_available, _device_name, problem = cuda_inventory()
    if not gpu_available:
        raise RequestError(problem or "CUDA is unavailable.")

    from style_bert_vits2.constants import Languages
    from style_bert_vits2.models import infer as style_infer
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    bert_path = installed_path(
        installation, "bert_model", regular_file=False)
    bert_models.load_model(Languages.JP, str(bert_path))
    bert_models.load_tokenizer(Languages.JP, str(bert_path))

    model = TTSModel(
        model_path=installed_path(
            installation, "voice_model", regular_file=True),
        config_path=installed_path(
            installation, "voice_config", regular_file=True),
        style_vec_path=installed_path(
            installation, "voice_styles", regular_file=True),
        device="cuda")
    model.load()
    if (
            model.net_g is None
            or next(model.net_g.parameters()).device.type != "cuda"):
        raise RequestError(
            "Style-Bert-VITS2 did not load its synthesis model on CUDA.")

    # The official Japanese DeBERTa snapshot emits fp16 features, while this
    # JP-Extra voice checkpoint's BERT projection is fp32.  Bridge that
    # boundary on the small feature tensor instead of doubling the BERT
    # model's VRAM by converting all of its weights to fp32.
    projection_dtype = model.net_g.enc_p.bert_proj.weight.dtype
    extract_bert_feature = style_infer.extract_bert_feature

    def extract_compatible_bert_feature(*args, **kwargs):
        feature = extract_bert_feature(*args, **kwargs)
        return feature.to(dtype=projection_dtype)

    style_infer.extract_bert_feature = extract_compatible_bert_feature
    return model, Languages


def _synthesize_item(model, languages, data: Mapping[str, Any]) -> dict:
    import torch
    seed_synthesis(data["seed"])
    sample_rate, audio = model.infer(
        text=data["text"],
        language=languages.JP,
        style="Neutral",
        style_weight=1.0,
        length=1.0 / data["speed"],
        pitch_scale=math.pow(2.0, data["pitch"] / 12.0),
        intonation_scale=1.0,
        use_assist_text=False)
    if not torch.cuda.is_available():
        raise RequestError("CUDA became unavailable during synthesis.")
    torch.cuda.synchronize()
    write_tensor_wav(
        audio,
        int(sample_rate),
        data["sample_rate_hz"],
        data["output_path"])
    return success_artifact(data)


def _synthesize(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voice()
    data = validate_synthesis_request(
        request,
        installation,
        supported_languages={"japanese"},
        voice_revisions=voices)
    with gpu_synthesis_lease(STYLE_BACKEND):
        model, languages = _load_model(installation)
        return _synthesize_item(model, languages, data)


def _synthesize_many(request: Mapping[str, Any]) -> dict:
    installation, voices = _installation_and_voice()
    items = validate_synthesis_items(
        request,
        installation,
        supported_languages={"japanese"},
        voice_revisions=voices)
    with clean_batch_outputs_on_error(items):
        with gpu_synthesis_lease(STYLE_BACKEND):
            model, languages = _load_model(installation)
            artifacts = [
                _synthesize_item(model, languages, item)["artifact"]
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
        backend=STYLE_BACKEND,
        model_id=STYLE_MODEL_ID,
        handler=handle))
