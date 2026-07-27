#!/usr/bin/env python3
"""Produce an auditable local TTS listening and timing comparison.

This is deliberately an evaluation utility, not an AutoAnki backend.  Run one
engine at a time in its isolated environment, then use ``report`` to combine
the raw measurements into a Markdown index beside the WAV files.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import wave


PROMPTS = {
    "japanese": (
        ("quick_brown_fox", "素早い茶色の狐が、怠け者の犬を飛び越える。"),
        ("word_cat", "猫。"),
        ("word_dog", "犬。"),
        ("word_book", "本。"),
    ),
    "chinese": (
        ("quick_brown_fox", "敏捷的棕色狐狸跳过了懒狗。"),
        ("word_cat", "猫。"),
        ("word_dog", "狗。"),
        ("word_book", "书。"),
    ),
}

REFERENCE_PROMPTS = {
    "japanese": "これは落ち着いた自然な日本語の音声です。毎日少しずつ勉強を続けます。",
    "chinese": "这是平静自然的中文语音。每天坚持学习一点。",
}

STYLE_BERT_REPOSITORY = "litagin/style_bert_vits2_jvnv"
STYLE_BERT_REVISION = "205830ca1d49e666ddfbf2a755f0108e9cade4dd"
STYLE_BERT_VOICES = (
    ("jvnv-M1-jp", "Neutral", "M1 neutral"),
    ("jvnv-M2-jp", "Neutral", "M2 neutral"),
    ("jvnv-F2-jp", "Neutral", "F2 neutral"),
)

KOKORO_JAPANESE_VOICES = (
    ("jf_alpha", "Japanese female alpha"),
    ("jf_gongitsune", "Japanese female gongitsune"),
    ("jm_kumo", "Japanese male kumo"),
)
KOKORO_CHINESE_VOICES = (
    ("zf_001", "Chinese female 001"),
    ("zf_021", "Chinese female 021"),
    ("zm_010", "Chinese male 010"),
    ("zm_052", "Chinese male 052"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    value = value.casefold().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def _write_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        # The existing Style-Bert runtime intentionally has a small dependency
        # set.  Keep this evaluator runnable there without mutating that
        # production environment merely to add an optional WAV writer.
        import numpy as np

        values = np.asarray(audio)
        if values.dtype.kind == "f":
            values = np.rint(np.clip(values, -1.0, 1.0) * 32767).astype(
                np.int16)
        elif values.dtype != np.int16:
            values = values.astype(np.int16)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1 if values.ndim == 1 else values.shape[1])
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(values.tobytes())
    else:
        sf.write(str(path), audio, sample_rate)


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _cuda_peak_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass
    return None


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@contextmanager
def _timed_cuda():
    _reset_cuda_peak()
    _cuda_sync()
    started = time.perf_counter()
    yield
    _cuda_sync()
    elapsed = time.perf_counter() - started
    return elapsed


def _measure_cuda(function):
    _reset_cuda_peak()
    _cuda_sync()
    started = time.perf_counter()
    value = function()
    _cuda_sync()
    return value, time.perf_counter() - started, _cuda_peak_mib()


def _write_results(output: Path, engine: str, entries: list[dict]) -> None:
    raw = output / "raw_timings"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{engine}.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "generated_at": _now(),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _entry(
        *,
        engine: str,
        language: str,
        voice: str,
        voice_detail: str,
        files: list[dict],
        model_load_seconds: float | None,
        prompt_setup_seconds: float | None,
        synthesis_seconds: float,
        peak_vram_mib: float | None,
        notes: str = "",
        startup_seconds: float | None = None,
    ) -> dict:
    audio_seconds = sum(item["duration_seconds"] for item in files)
    return {
        "engine": engine,
        "language": language,
        "voice": voice,
        "voice_detail": voice_detail,
        "files": files,
        "model_load_seconds": (
            round(model_load_seconds, 3)
            if model_load_seconds is not None else None),
        "prompt_setup_seconds": (
            round(prompt_setup_seconds, 3)
            if prompt_setup_seconds is not None else None),
        "synthesis_seconds": round(synthesis_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "real_time_factor": (
            round(synthesis_seconds / audio_seconds, 3)
            if audio_seconds else None),
        "peak_vram_mib": peak_vram_mib,
        "startup_seconds": (
            round(startup_seconds, 3)
            if startup_seconds is not None else None),
        "notes": notes,
    }


def _write_batch(
        output: Path,
        *,
        engine: str,
        language: str,
        voice: str,
        audio_by_prompt: list,
        sample_rate: int,
    ) -> list[dict]:
    files = []
    for (key, text), audio in zip(
            PROMPTS[language], audio_by_prompt, strict=True):
        relative = Path("audio") / language / _slug(engine) / _slug(voice) / (
            f"{key}.wav")
        target = output / relative
        _write_wav(target, audio, sample_rate)
        files.append({
            "prompt_key": key,
            "text": text,
            "path": relative.as_posix(),
            "duration_seconds": round(_audio_duration(target), 3),
        })
    return files


def _run_style(args) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.models import infer as style_infer
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    output = args.output
    asset_root = args.asset_root / "style_bert"
    asset_root.mkdir(parents=True, exist_ok=True)
    frontend_started = time.perf_counter()
    bert_models.load_model(Languages.JP, str(args.style_bert_path))
    bert_models.load_tokenizer(Languages.JP, str(args.style_bert_path))
    frontend_seconds = time.perf_counter() - frontend_started
    entries = []
    for model_name, style, description in STYLE_BERT_VOICES:
        paths = snapshot_download(
            repo_id=STYLE_BERT_REPOSITORY,
            revision=STYLE_BERT_REVISION,
            allow_patterns=(f"{model_name}/*",),
            local_dir=str(asset_root),
        )
        voice_root = Path(paths) / model_name
        config = voice_root / "config.json"
        model_path = next(voice_root.glob("*.safetensors"))
        styles = voice_root / "style_vectors.npy"

        def load_model():
            model = TTSModel(
                model_path=model_path,
                config_path=config,
                style_vec_path=styles,
                device="cuda",
            )
            model.load()
            projection_dtype = model.net_g.enc_p.bert_proj.weight.dtype
            extract_feature = style_infer.extract_bert_feature

            def compatible_feature(*call_args, **kwargs):
                return extract_feature(*call_args, **kwargs).to(
                    dtype=projection_dtype)

            style_infer.extract_bert_feature = compatible_feature
            return model

        model, load_seconds, load_peak = _measure_cuda(load_model)

        def synthesize():
            return [
                model.infer(
                    text=text,
                    language=Languages.JP,
                    style=style,
                    style_weight=1.0,
                    length=1.0,
                    pitch_scale=1.0,
                    intonation_scale=1.0,
                    use_assist_text=False,
                )
                for _key, text in PROMPTS["japanese"]
            ]

        results, synth_seconds, synth_peak = _measure_cuda(synthesize)
        sample_rates = {int(rate) for rate, _audio in results}
        if len(sample_rates) != 1:
            raise RuntimeError(f"{model_name} returned mixed sample rates.")
        files = _write_batch(
            output,
            engine="Style-Bert-VITS2 JP-Extra",
            language="japanese",
            voice=f"{model_name} {style}",
            audio_by_prompt=[audio for _rate, audio in results],
            sample_rate=sample_rates.pop(),
        )
        entries.append(_entry(
            engine="Style-Bert-VITS2 JP-Extra",
            language="japanese",
            voice=f"{model_name} {style}",
            voice_detail=description,
            files=files,
            model_load_seconds=load_seconds,
            prompt_setup_seconds=None,
            synthesis_seconds=synth_seconds,
            peak_vram_mib=max(
                value for value in (load_peak, synth_peak) if value is not None),
            notes=(
                "Frontend load shared by the three candidates: "
                f"{frontend_seconds:.3f}s."),
        ))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _write_results(output, "style_bert", entries)


def _run_kokoro(args) -> None:
    import torch
    from kokoro import KPipeline

    output = args.output
    entries = []

    def run_language(language, lang_code, repo_id, voices):
        pipeline, load_seconds, load_peak = _measure_cuda(
            lambda: KPipeline(
                lang_code=lang_code,
                repo_id=repo_id,
                device="cuda",
            ))
        for voice, description in voices:
            def synthesize():
                audios = []
                for _key, text in PROMPTS[language]:
                    generated = list(pipeline(text, voice=voice, speed=1.0))
                    if len(generated) != 1:
                        raise RuntimeError(
                            f"Kokoro split {voice} {text!r} into "
                            f"{len(generated)} outputs.")
                    _graphemes, _phonemes, audio = generated[0]
                    audios.append(audio)
                return audios

            audios, synth_seconds, synth_peak = _measure_cuda(synthesize)
            files = _write_batch(
                output,
                engine="Kokoro 82M",
                language=language,
                voice=voice,
                audio_by_prompt=audios,
                sample_rate=24000,
            )
            entries.append(_entry(
                engine="Kokoro 82M",
                language=language,
                voice=voice,
                voice_detail=description,
                files=files,
                model_load_seconds=load_seconds,
                prompt_setup_seconds=None,
                synthesis_seconds=synth_seconds,
                peak_vram_mib=max(
                    value for value in (load_peak, synth_peak)
                    if value is not None),
                notes=(
                    f"Repository: {repo_id}. Model load is shared by all "
                    f"{language} Kokoro voices."),
            ))

        # Generate a neutral reference of this language for Qwen Base.  It is
        # deliberately retained beside the demos so the clone provenance is
        # inspectable rather than hidden.
        reference_voice = (
            "jm_kumo" if language == "japanese" else "zm_010")
        ref_text = REFERENCE_PROMPTS[language]
        ref_audio, ref_seconds, _peak = _measure_cuda(
            lambda: list(pipeline(ref_text, voice=reference_voice))[0][2])
        reference_path = output / "references" / (
            f"kokoro_{language}_{reference_voice}.wav")
        _write_wav(reference_path, ref_audio, 24000)
        (output / "references" / f"kokoro_{language}_{reference_voice}.txt").write_text(
            ref_text + "\n", encoding="utf-8")
        entries.append({
            "engine": "Kokoro 82M",
            "language": language,
            "voice": f"{reference_voice} reference",
            "voice_detail": "Qwen Base reference generation only",
            "files": [{
                "prompt_key": "reference",
                "text": ref_text,
                "path": reference_path.relative_to(output).as_posix(),
                "duration_seconds": round(_audio_duration(reference_path), 3),
            }],
            "model_load_seconds": None,
            "prompt_setup_seconds": None,
            "synthesis_seconds": round(ref_seconds, 3),
            "audio_seconds": round(_audio_duration(reference_path), 3),
            "real_time_factor": round(
                ref_seconds / _audio_duration(reference_path), 3),
            "peak_vram_mib": _peak,
            "startup_seconds": None,
            "notes": "Reference clip for the Qwen Base test; not ranked as a candidate.",
        })
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    run_language(
        "japanese", "j", "hexgrad/Kokoro-82M", KOKORO_JAPANESE_VOICES)
    run_language(
        "chinese", "z", "hexgrad/Kokoro-82M-v1.1-zh",
        KOKORO_CHINESE_VOICES)
    _write_results(output, "kokoro", entries)


def _require_reference(output: Path, language: str) -> tuple[Path, str]:
    preferred = output / "references" / f"aivis_{language}_aida_normal.wav"
    if preferred.is_file():
        transcript = preferred.with_suffix(".txt")
        return preferred, transcript.read_text(encoding="utf-8").strip()
    reference = output / "references" / (
        f"kokoro_{language}_{'jm_kumo' if language == 'japanese' else 'zm_010'}.wav")
    transcript = reference.with_suffix(".txt")
    if not reference.is_file() or not transcript.is_file():
        raise RuntimeError(
            "Qwen Base needs the retained Kokoro/Aivis reference; run "
            "the Kokoro evaluation first.")
    return reference, transcript.read_text(encoding="utf-8").strip()


def _run_qwen(args) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    output = args.output
    entries = []

    def load_custom():
        return Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            revision="85e237c12c027371202489a0ec509ded67b5e4b5",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    custom, custom_load_seconds, custom_load_peak = _measure_cuda(load_custom)
    candidates = (
        ("chinese", "Dylan", "", "Native Beijing Chinese male"),
        ("chinese", "Uncle_Fu", "", "Native seasoned Chinese male"),
        ("japanese", "Dylan", "", "Cross-language Chinese voice; included by request"),
        ("japanese", "Uncle_Fu", "", "Cross-language Chinese voice; included by request"),
        ("japanese", "Ono_Anna", "", "Native Japanese speaker; upstream describes it as playful"),
        (
            "japanese",
            "Ono_Anna calm instruction",
            "Speak with a calm, natural, neutral adult Japanese delivery.",
            "Native Japanese speaker with neutral-delivery instruction",
        ),
    )
    for language, display_voice, instruction, detail in candidates:
        speaker = display_voice.split(" calm instruction", 1)[0]

        def synthesize():
            wavs, sample_rate = custom.generate_custom_voice(
                text=[text for _key, text in PROMPTS[language]],
                language=[language.title()] * len(PROMPTS[language]),
                speaker=[speaker] * len(PROMPTS[language]),
                instruct=[instruction] * len(PROMPTS[language]),
            )
            return wavs, sample_rate

        (wavs, sample_rate), synth_seconds, synth_peak = _measure_cuda(
            synthesize)
        files = _write_batch(
            output,
            engine="Qwen3-TTS 0.6B CustomVoice",
            language=language,
            voice=display_voice,
            audio_by_prompt=wavs,
            sample_rate=int(sample_rate),
        )
        entries.append(_entry(
            engine="Qwen3-TTS 0.6B CustomVoice",
            language=language,
            voice=display_voice,
            voice_detail=detail,
            files=files,
            model_load_seconds=custom_load_seconds,
            prompt_setup_seconds=None,
            synthesis_seconds=synth_seconds,
            peak_vram_mib=max(
                value for value in (custom_load_peak, synth_peak)
                if value is not None),
            notes="CustomVoice checkpoint pinned to 85e237c12c027371202489a0ec509ded67b5e4b5.",
        ))
    del custom
    gc.collect()
    torch.cuda.empty_cache()

    def load_base():
        return Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            revision="5d83992436eae1d760afd27aff78a71d676296fc",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    base, base_load_seconds, base_load_peak = _measure_cuda(load_base)
    for language in ("japanese", "chinese"):
        reference_path, reference_text = _require_reference(output, language)
        prompt, prompt_seconds, prompt_peak = _measure_cuda(
            lambda: base.create_voice_clone_prompt(
                ref_audio=str(reference_path),
                ref_text=reference_text,
                x_vector_only_mode=False,
            ))

        def synthesize_base():
            return base.generate_voice_clone(
                text=[text for _key, text in PROMPTS[language]],
                language=[language.title()] * len(PROMPTS[language]),
                voice_clone_prompt=prompt,
            )

        (wavs, sample_rate), synth_seconds, synth_peak = _measure_cuda(
            synthesize_base)
        files = _write_batch(
            output,
            engine="Qwen3-TTS 0.6B Base",
            language=language,
            voice=f"Base clone from {reference_path.stem}",
            audio_by_prompt=wavs,
            sample_rate=int(sample_rate),
        )
        entries.append(_entry(
            engine="Qwen3-TTS 0.6B Base",
            language=language,
            voice=f"Base clone from {reference_path.stem}",
            voice_detail=(
                "Voice-clone prompt made from retained, locally generated "
                f"reference: {reference_path.relative_to(output).as_posix()}"),
            files=files,
            model_load_seconds=base_load_seconds,
            prompt_setup_seconds=prompt_seconds,
            synthesis_seconds=synth_seconds,
            peak_vram_mib=max(
                value for value in (
                    base_load_peak, prompt_peak, synth_peak)
                if value is not None),
            notes="Base checkpoint pinned to 5d83992436eae1d760afd27aff78a71d676296fc.",
        ))
    _write_results(output, "qwen", entries)


def _run_melo(args) -> None:
    import torch
    from melo.api import TTS

    output = args.output
    warmup_dir = args.asset_root / "melo_warmups"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for language, code in (("japanese", "JP"), ("chinese", "ZH")):
        def load_model():
            model = TTS(language=code, device="cuda:0")
            # Melo lazily loads its multilingual BERT on the first Chinese
            # utterance.  Include that one-off work in startup rather than
            # mislabeling it as steady-state synthesis for the first demo.
            model.tts_to_file(
                "これはウォームアップです。" if code == "JP" else "这是预热音频。",
                model.hps.data.spk2id[code],
                str(warmup_dir / f"{code}.wav"),
                speed=1.0,
            )
            return model

        model, load_seconds, load_peak = _measure_cuda(load_model)
        speaker_ids = model.hps.data.spk2id
        if code not in speaker_ids:
            raise RuntimeError(
                f"Melo {code} does not expose expected speaker ID: "
                f"{speaker_ids!r}")
        output_dir = output / "audio" / language / "melotts" / code
        output_dir.mkdir(parents=True, exist_ok=True)

        def synthesize():
            targets = []
            for key, text in PROMPTS[language]:
                target = output_dir / f"{key}.wav"
                model.tts_to_file(
                    text,
                    speaker_ids[code],
                    str(target),
                    speed=1.0,
                )
                targets.append(target)
            return targets

        targets, synth_seconds, synth_peak = _measure_cuda(synthesize)
        files = []
        for (key, text), target in zip(PROMPTS[language], targets, strict=True):
            files.append({
                "prompt_key": key,
                "text": text,
                "path": target.relative_to(output).as_posix(),
                "duration_seconds": round(_audio_duration(target), 3),
            })
        entries.append(_entry(
            engine="MeloTTS",
            language=language,
            voice=code,
            voice_detail=(
                "MeloTTS exposes one fixed speaker for this language."),
            files=files,
            model_load_seconds=load_seconds,
            prompt_setup_seconds=None,
            synthesis_seconds=synth_seconds,
            peak_vram_mib=max(
                value for value in (load_peak, synth_peak)
                if value is not None),
            notes=(
                "Language checkpoint and speaker ID are both fixed by MeloTTS. "
                "A language-specific warm-up is included in model-load time."
            ),
        ))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _write_results(output, "melo", entries)


def _http_json(url: str, *, data: dict | None = None, method: str | None = None):
    if data is None:
        request = Request(url, method=method)
    else:
        request = Request(
            url,
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method or "POST",
        )
    with urlopen(request, timeout=180) as response:
        return response.read()


def _aivis_style_id(speakers: list, *, speaker_hint: str, style_hint: str):
    matching_speaker = next((
        speaker for speaker in speakers
        if speaker_hint.casefold() in speaker.get("name", "").casefold()),
        None)
    if matching_speaker is None:
        raise RuntimeError(
            f"Aivis speaker {speaker_hint!r} was not loaded. Available: "
            f"{[speaker.get('name') for speaker in speakers]!r}")
    style = next((
        candidate for candidate in matching_speaker.get("styles", ())
        if style_hint.casefold() in candidate.get("name", "").casefold()),
        None)
    if style is None:
        style = matching_speaker.get("styles", [None])[0]
    if not isinstance(style, dict) or "id" not in style:
        raise RuntimeError(
            f"Aivis speaker {speaker_hint!r} has no usable style.")
    return matching_speaker["name"], style["name"], style["id"]


def _run_aivis(args) -> None:
    output = args.output
    base_url = args.aivis_url.rstrip("/")
    started = time.perf_counter()
    while True:
        try:
            speakers = json.loads(_http_json(f"{base_url}/speakers"))
            break
        except Exception:
            if time.perf_counter() - started > args.aivis_ready_timeout:
                raise RuntimeError(
                    "AivisSpeech Engine did not become ready before the "
                    "evaluation timeout.")
            time.sleep(1)
    startup_seconds = time.perf_counter() - started
    candidates = (
        ("morioki", "normal", "morioki normal"),
        ("fumifumi", "normal", "fumifumi normal"),
        ("阿井田", "normal", "Aida Shigeru normal"),
        ("阿井田", "calm", "Aida Shigeru calm"),
    )
    entries = []
    for speaker_hint, style_hint, label in candidates:
        speaker_name, style_name, style_id = _aivis_style_id(
            speakers, speaker_hint=speaker_hint, style_hint=style_hint)
        files = []
        output_dir = output / "audio" / "japanese" / "aivisspeech" / _slug(label)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for key, text in PROMPTS["japanese"]:
            query_url = f"{base_url}/audio_query?" + urlencode({
                "text": text,
                "speaker": style_id,
            })
            query = json.loads(_http_json(query_url, method="POST"))
            audio = _http_json(
                f"{base_url}/synthesis?" + urlencode({"speaker": style_id}),
                data=query)
            target = output_dir / f"{key}.wav"
            target.write_bytes(audio)
            files.append({
                "prompt_key": key,
                "text": text,
                "path": target.relative_to(output).as_posix(),
                "duration_seconds": round(_audio_duration(target), 3),
            })
        synthesis_seconds = time.perf_counter() - started
        entries.append(_entry(
            engine="AivisSpeech Engine",
            language="japanese",
            voice=f"{speaker_name} · {style_name}",
            voice_detail=label,
            files=files,
            model_load_seconds=None,
            prompt_setup_seconds=None,
            synthesis_seconds=synthesis_seconds,
            peak_vram_mib=None,
            startup_seconds=startup_seconds,
            notes=(
                "HTTP API timing includes query and synthesis for all four "
                "clips. Engine startup is shared by all Aivis candidates."),
        ))

        if label == "Aida Shigeru normal":
            # Preserve a reference with exact transcript for Qwen Base.  This
            # is generated locally from a licensed Japanese synthesis model,
            # not taken from an unrelated person on the web.
            ref_text = REFERENCE_PROMPTS["japanese"]
            query = json.loads(_http_json(
                f"{base_url}/audio_query?" + urlencode({
                    "text": ref_text,
                    "speaker": style_id,
                }), method="POST"))
            reference = _http_json(
                f"{base_url}/synthesis?" + urlencode({"speaker": style_id}),
                data=query)
            reference_path = output / "references" / "aivis_japanese_aida_normal.wav"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_bytes(reference)
            reference_path.with_suffix(".txt").write_text(
                ref_text + "\n", encoding="utf-8")
    _write_results(output, "aivis", entries)


def _all_entries(output: Path) -> list[dict]:
    values = []
    for path in sorted((output / "raw_timings").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.extend(payload.get("entries", ()))
    return values


def _format_seconds(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _write_report(args) -> None:
    output = args.output
    entries = _all_entries(output)
    ranked = [
        entry for entry in entries
        if all(item.get("prompt_key") != "reference" for item in entry["files"])
    ]
    lines = [
        "# Local Japanese and Chinese TTS evaluation",
        "",
        f"Generated: {_now()}",
        "",
        "This folder contains every WAV used for the comparison. Each voice "
        "received one native-language quick-brown-fox equivalent plus three "
        "one-word utterances. Times are measured locally on the RTX 5050 "
        "Laptop GPU where the engine exposes CUDA; Aivis is reported as HTTP "
        "wall time because its external engine owns the device details.",
        "",
        "## Prompt set",
        "",
        "| Language | Full sentence | Three one-word clips |",
        "| --- | --- | --- |",
    ]
    for language in ("japanese", "chinese"):
        full = PROMPTS[language][0][1]
        words = " / ".join(text for _key, text in PROMPTS[language][1:])
        lines.append(f"| {language.title()} | {full} | {words} |")
    lines.extend([
        "",
        "## Measured batches",
        "",
        "Each row is one voice generating all four clips. `RTF` is synthesis "
        "wall time divided by the total emitted audio duration; lower is "
        "faster. Model-load and Base prompt time are kept separate so the "
        "steady-state cost is visible.",
        "",
        "| Language | Engine | Voice | Load s | Clone prompt s | Batch synth s | Audio s | RTF | Peak VRAM MiB | Demos |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for entry in ranked:
        target = entry["files"][0]["path"]
        lines.append(
            "| {language} | {engine} | {voice} | {load} | {prompt} | "
            "{synth} | {audio} | {rtf} | {vram} | [WAVs]({target}) |".format(
                language=entry["language"],
                engine=entry["engine"],
                voice=entry["voice"].replace("|", "\\|"),
                load=_format_seconds(entry.get("model_load_seconds")),
                prompt=_format_seconds(entry.get("prompt_setup_seconds")),
                synth=_format_seconds(entry["synthesis_seconds"]),
                audio=_format_seconds(entry["audio_seconds"]),
                rtf=_format_seconds(entry.get("real_time_factor")),
                vram=_format_seconds(entry.get("peak_vram_mib")),
                target=target,
            ))
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- Do not rank a voice solely by timing. Listen to both the sentence and "
        "the three isolated words; the latter are the actual Anki workload.",
        "- Qwen Dylan and Uncle_Fu are native Chinese voices. Their Japanese "
        "samples are a deliberate cross-language control, not a claim that "
        "they are ideal Japanese speakers.",
        "- Qwen Base has no supplied speaker. Its entries state the retained, "
        "locally generated reference WAV used to make the clone prompt. The "
        "reference and its exact transcript live in `references/`.",
        "- MeloTTS exposes only one built-in Japanese and one built-in Chinese "
        "speaker, so it is a speed/quality baseline rather than a voice roster.",
        "- Kokoro's own voice documentation warns that short utterances can be "
        "weak; this is why all three one-word clips are included.",
        "- AivisSpeech samples use the downloaded model files in the evaluation "
        "asset cache. Their audio and timing results are retained here; model "
        "weights themselves are intentionally not duplicated into this output "
        "folder.",
        "",
        "## Raw measurements",
        "",
        "Machine-readable per-engine data is in `raw_timings/`. Each entry "
        "contains the exact prompt text, file path, individual output duration, "
        "model load time, prompt construction time where applicable, batch "
        "generation time, and peak reported CUDA allocation.",
        "",
        "## Model and voice references",
        "",
        "- [Qwen3-TTS upstream repository](https://github.com/QwenLM/Qwen3-TTS)",
        "- [Kokoro voice roster](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)",
        "- [MeloTTS upstream repository](https://github.com/myshell-ai/MeloTTS)",
        "- [AivisSpeech Engine upstream repository](https://github.com/Aivis-Project/AivisSpeech-Engine)",
        "- [Style-Bert-VITS2 JVNV model card](https://huggingface.co/litagin/style_bert_vits2_jvnv)",
    ])
    (output / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps({"generated_at": _now(), "entries": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "engine",
        choices=("style", "kokoro", "qwen", "melo", "aivis", "report"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--style-bert-path", type=Path)
    parser.add_argument("--aivis-url", default="http://127.0.0.1:10101")
    parser.add_argument("--aivis-ready-timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _arguments(argv)
    args.output = args.output.resolve()
    args.asset_root = args.asset_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "style":
        if args.style_bert_path is None:
            raise SystemExit("--style-bert-path is required for the Style-Bert run.")
        _run_style(args)
    elif args.engine == "kokoro":
        _run_kokoro(args)
    elif args.engine == "qwen":
        _run_qwen(args)
    elif args.engine == "melo":
        _run_melo(args)
    elif args.engine == "aivis":
        _run_aivis(args)
    else:
        _write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
