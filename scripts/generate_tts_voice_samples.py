#!/usr/bin/env python3
"""Regenerate the legacy Style-Bert/CosyVoice listening comparison.

The installed Style-Bert model contains one speaker with seven styles.  The
installed CosyVoice source bundle contains two official reference recordings;
CosyVoice clones those references rather than offering a finite speaker list.
These are retained comparison backends: production Japanese now uses MeloTTS
JP and production Chinese uses Kokoro zm_010.  This script deliberately uses
only already-installed, redistributable assets and does not change routing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


STYLE_BACKEND = "style_bert_vits2_jp_extra"
COSY_BACKEND = "fun_cosyvoice3_0_5b"
STYLE_STYLES = (
    "Neutral",
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
)
ENGLISH_SAMPLE = "The quick brown fox jumps over the lazy dog."
JAPANESE_SAMPLE = "素早い茶色の狐が、のんびりした犬を飛び越える。"


def _tts_root() -> Path:
    return Path(
        os.environ.get(
            "AUTOANKI_TTS_HOME",
            Path.home() / ".local" / "share" / "autoanki" / "tts"))


def _runtime_python(root: Path, backend: str) -> Path:
    runtime = root / "runtimes" / backend
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _installation(root: Path, backend: str) -> dict:
    return json.loads(
        (root / "runtimes" / backend / "installation.json").read_text(
            encoding="utf-8"))


def _installed_path(root: Path, installation: dict, name: str) -> Path:
    return root / installation["paths"][name]


def _style_worker(root: Path, output: Path) -> list[dict]:
    runtime = root / "runtimes" / STYLE_BACKEND
    sys.path.insert(0, str(runtime))
    from _autoanki_tts_worker_common import (
        gpu_synthesis_lease,
        write_tensor_wav,
    )
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.models import infer as style_infer
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    installation = _installation(root, STYLE_BACKEND)
    bert_path = _installed_path(root, installation, "bert_model")
    bert_models.load_model(Languages.JP, str(bert_path))
    bert_models.load_tokenizer(Languages.JP, str(bert_path))
    with gpu_synthesis_lease(STYLE_BACKEND):
        model = TTSModel(
            model_path=_installed_path(
                root, installation, "voice_model"),
            config_path=_installed_path(
                root, installation, "voice_config"),
            style_vec_path=_installed_path(
                root, installation, "voice_styles"),
            device="cuda")
        model.load()
        projection_dtype = model.net_g.enc_p.bert_proj.weight.dtype
        extract_feature = style_infer.extract_bert_feature

        def compatible_feature(*args, **kwargs):
            return extract_feature(
                *args,
                **kwargs).to(dtype=projection_dtype)

        style_infer.extract_bert_feature = compatible_feature
        entries = []
        for style in STYLE_STYLES:
            sample_rate, audio = model.infer(
                text=JAPANESE_SAMPLE,
                language=Languages.JP,
                style=style,
                style_weight=1.0,
                length=1.0,
                pitch_scale=1.0,
                intonation_scale=1.0,
                use_assist_text=False)
            path = output / f"style_bert_jvnv_f1_{style.casefold()}.wav"
            write_tensor_wav(
                audio,
                int(sample_rate),
                int(sample_rate),
                path)
            entries.append({
                "backend": STYLE_BACKEND,
                "speaker": "jvnv-F1-jp",
                "style": style,
                "text": JAPANESE_SAMPLE,
                "file": path.name,
            })
    return entries


def _load_cosy_model(root: Path, installation: dict):
    source = _installed_path(root, installation, "source")
    sys.path.insert(0, str(source / "third_party" / "Matcha-TTS"))
    sys.path.insert(0, str(source))
    import onnxruntime

    onnxruntime.preload_dlls(directory="")
    import numpy
    import soundfile
    import torch
    import torchaudio.functional

    def load_wav(path: str, target_sr: int, min_sr: int = 16000):
        samples, sample_rate = soundfile.read(
            path,
            dtype="float32",
            always_2d=True)
        if sample_rate < min_sr:
            raise RuntimeError(
                f"Reference audio sample rate {sample_rate} is below "
                f"{min_sr} Hz.")
        speech = torch.from_numpy(
            numpy.ascontiguousarray(samples.T)).mean(
                dim=0,
                keepdim=True)
        if sample_rate != target_sr:
            speech = torchaudio.functional.resample(
                speech,
                sample_rate,
                target_sr)
        return speech

    from cosyvoice.cli import frontend as cosy_frontend
    cosy_frontend.load_wav = load_wav
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(
        model_dir=str(_installed_path(root, installation, "model")),
        fp16=True,
        load_trt=False,
        load_vllm=False)


def _save_cosy_output(
        model,
        outputs,
        path: Path,
        write_tensor_wav) -> None:
    import torch

    chunks = [
        value["tts_speech"].detach().cpu().reshape(-1)
        for value in outputs
    ]
    if not chunks:
        raise RuntimeError("CosyVoice produced no sample audio.")
    write_tensor_wav(
        torch.cat(chunks),
        int(model.sample_rate),
        int(model.sample_rate),
        path,
    )


def _cosy_worker(root: Path, output: Path) -> list[dict]:
    runtime = root / "runtimes" / COSY_BACKEND
    sys.path.insert(0, str(runtime))
    from _autoanki_tts_worker_common import (
        gpu_synthesis_lease,
        write_tensor_wav,
    )

    installation = _installation(root, COSY_BACKEND)
    source = _installed_path(root, installation, "source")
    zero_reference = _installed_path(
        root, installation, "prompt_audio")
    cross_reference = source / "asset" / "cross_lingual_prompt.wav"
    prompt_text = installation["voice"]["prompt_text"]
    with gpu_synthesis_lease(COSY_BACKEND):
        model = _load_cosy_model(root, installation)
        zero_path = output / "cosyvoice_official_zero_shot_reference.wav"
        _save_cosy_output(
            model,
            model.inference_zero_shot(
                ENGLISH_SAMPLE,
                prompt_text,
                str(zero_reference),
                stream=False,
                speed=1.0),
            zero_path,
            write_tensor_wav)
        cross_path = (
            output / "cosyvoice_official_cross_lingual_reference.wav")
        _save_cosy_output(
            model,
            model.inference_cross_lingual(
                "You are a helpful assistant.<|endofprompt|>"
                + ENGLISH_SAMPLE,
                str(cross_reference),
                stream=False,
                speed=1.0),
            cross_path,
            write_tensor_wav)
    return [
        {
            "backend": COSY_BACKEND,
            "speaker": "official zero-shot reference",
            "style": "default synthesis",
            "text": ENGLISH_SAMPLE,
            "file": zero_path.name,
        },
        {
            "backend": COSY_BACKEND,
            "speaker": "official cross-lingual reference",
            "style": "default synthesis",
            "text": ENGLISH_SAMPLE,
            "file": cross_path.name,
        },
    ]


def _run_child(root: Path, output: Path, backend: str) -> list[dict]:
    python = _runtime_python(root, backend)
    environment = os.environ.copy()
    environment["AUTOANKI_TTS_HOME"] = str(root)
    completed = subprocess.run(
        (
            python,
            Path(__file__).resolve(),
            "--worker",
            backend,
            "--output",
            output,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        env=environment,
        check=False)
    if completed.returncode:
        raise RuntimeError(
            f"{backend} sample generation failed with "
            f"exit code {completed.returncode}.")
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    raise RuntimeError(
        f"{backend} sample generation returned no JSON manifest.")


def _write_readme(output: Path, entries: list[dict]) -> None:
    lines = [
        "# Legacy Style-Bert/CosyVoice comparison",
        "",
        "The Style-Bert files use one Japanese speaker with seven learned "
        "speaking styles. This is no longer AutoAnki's production Japanese "
        "route; production uses MeloTTS `JP`.",
        "",
        "Fun-CosyVoice3 is a zero-shot voice-cloning model, so it has no "
        "finite built-in voice catalogue. These two files synthesize the "
        "English sample using voices cloned from the two reference recordings "
        "bundled with the official installed source. English and French still "
        "use the zero-shot reference; Chinese now uses Kokoro `zm_010`.",
        "",
        "Files:",
        "",
    ]
    lines.extend(
        f"- `{entry['file']}` — {entry['speaker']}; {entry['style']}"
        for entry in entries)
    (output / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8")


def _existing_style_entries(output: Path) -> list[dict] | None:
    entries = [
        {
            "backend": STYLE_BACKEND,
            "speaker": "jvnv-F1-jp",
            "style": style,
            "text": JAPANESE_SAMPLE,
            "file": f"style_bert_jvnv_f1_{style.casefold()}.wav",
        }
        for style in STYLE_STYLES
    ]
    return (
        entries
        if all(
            (output / entry["file"]).is_file()
            for entry in entries)
        else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "tts_voice_samples")
    parser.add_argument(
        "--worker",
        choices=(STYLE_BACKEND, COSY_BACKEND))
    arguments = parser.parse_args()
    root = _tts_root()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.worker:
        entries = (
            _style_worker(root, output)
            if arguments.worker == STYLE_BACKEND
            else _cosy_worker(root, output))
        print(json.dumps(entries, ensure_ascii=False))
        return 0

    style_entries = (
        _existing_style_entries(output)
        or _run_child(root, output, STYLE_BACKEND))
    entries = [
        *style_entries,
        *_run_child(root, output, COSY_BACKEND),
    ]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True) + "\n",
        encoding="utf-8")
    _write_readme(output, entries)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
