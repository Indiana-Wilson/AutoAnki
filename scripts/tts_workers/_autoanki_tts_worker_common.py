"""Shared, dependency-light support for AutoAnki's isolated TTS workers.

This file is copied beside ``autoanki_worker.py`` in each isolated runtime.
Backend libraries are deliberately imported only by the backend worker so a
missing or broken model can still return a useful protocol status response.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping
import wave


PROTOCOL = "autoanki-local-tts"
VERSION = 1
INSTALLATION_SCHEMA_VERSION = 1

STYLE_BACKEND = "style_bert_vits2_jp_extra"
COSY_BACKEND = "fun_cosyvoice3_0_5b"
STYLE_MODEL_ID = "Style-Bert-VITS2 JP-Extra"
COSY_MODEL_ID = "Fun-CosyVoice3-0.5B"


class RequestError(RuntimeError):
    """A handled request or installation error."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RequestError(f"{label} must be a JSON object.")
    return value


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{label} must be a non-empty string.")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{label} must be a number.")
    result = float(value)
    if not (-1.0e100 < result < 1.0e100):
        raise RequestError(f"{label} must be finite.")
    return result


def require_sha256(value: Any, label: str) -> str:
    value = require_text(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef"
                               for character in value):
        raise RequestError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def runtime_root() -> Path:
    return Path(__file__).resolve().parent


def shared_root() -> Path:
    override = os.environ.get("AUTOANKI_TTS_HOME")
    if override:
        return Path(override).expanduser().resolve()
    # <root>/runtimes/<backend>/autoanki_worker.py
    return runtime_root().parent.parent


def _safe_relative_path(value: Any, label: str) -> Path:
    text = require_text(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise RequestError(f"{label} must stay inside the shared TTS root.")
    resolved = (shared_root() / path).resolve()
    try:
        resolved.relative_to(shared_root())
    except ValueError as error:
        raise RequestError(
            f"{label} must stay inside the shared TTS root.") from error
    return resolved


def load_installation(expected_backend: str, expected_model_id: str) -> dict:
    path = runtime_root() / "installation.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RequestError(
            f"The {expected_backend} runtime is not installed completely; "
            "installation.json is missing.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RequestError(
            f"The {expected_backend} installation manifest is unreadable.") \
            from error
    value = dict(require_mapping(value, "installation manifest"))
    if value.get("schema_version") != INSTALLATION_SCHEMA_VERSION:
        raise RequestError("The installation manifest schema is incompatible.")
    if value.get("backend") != expected_backend:
        raise RequestError("The installation manifest names another backend.")
    if (
            value.get("state") != "ready"
            and os.environ.get("AUTOANKI_TTS_INSTALL_SMOKE") != "1"):
        raise RequestError(
            f"The {expected_backend} runtime has not passed its installation "
            "smoke test.")
    model = require_mapping(value.get("model"), "installed model")
    if model.get("id") != expected_model_id:
        raise RequestError("The installation manifest names another model.")
    require_text(model.get("revision"), "installed model revision")
    runtime = require_mapping(value.get("runtime"), "installed runtime")
    expected_worker_revision = require_sha256(
        runtime.get("worker_sha256"), "installed worker revision")
    digest = hashlib.sha256()
    for filename in (
            "_autoanki_tts_worker_common.py",
            "autoanki_worker.py"):
        worker_path = runtime_root() / filename
        if not worker_path.is_file():
            raise RequestError(
                f"Installed worker file is missing: {worker_path}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(worker_path.read_bytes())
        digest.update(b"\0")
    if digest.hexdigest() != expected_worker_revision:
        raise RequestError(
            "The installed worker files differ from their revisioned "
            "installation manifest.")
    return value


def installed_path(
        installation: Mapping[str, Any],
        key: str,
        *,
        regular_file: bool | None = None) -> Path:
    paths = require_mapping(
        installation.get("paths"), "installation paths")
    path = _safe_relative_path(paths.get(key), f"installation path {key!r}")
    if regular_file is True and not path.is_file():
        raise RequestError(f"Installed file is missing: {path}")
    if regular_file is False and not path.is_dir():
        raise RequestError(f"Installed directory is missing: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def revision_for_assets(
        assets: Iterable[tuple[str, Path]],
        *,
        text_assets: Iterable[tuple[str, str]] = ()) -> str:
    """Hash asset identities, bytes, and synthesis-affecting text."""
    digest = hashlib.sha256()
    for name, path in sorted(assets, key=lambda item: item[0]):
        if not path.is_file():
            raise RequestError(f"Installed voice asset is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    for name, value in sorted(text_assets, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cuda_inventory() -> tuple[bool, str | None, str | None]:
    """Return CUDA availability, device name, and an import problem."""
    try:
        import torch
    except Exception as error:
        return False, None, f"PyTorch could not be imported: {error}"
    try:
        if not torch.cuda.is_available():
            return False, None, (
                "CUDA is unavailable in this isolated PyTorch runtime.")
        name = torch.cuda.get_device_name(0)
        # Force CUDA context creation so a driver/runtime mismatch is caught.
        torch.empty(1, device="cuda")
        return True, str(name), None
    except Exception as error:
        return False, None, f"CUDA initialization failed: {error}"


def validate_request_identity(
        request: Mapping[str, Any],
        *,
        backend: str,
        model_id: str) -> str:
    if request.get("protocol") != PROTOCOL or request.get("version") != VERSION:
        raise RequestError("The request uses an incompatible TTS protocol.")
    if request.get("backend") != backend:
        raise RequestError("The request was sent to the wrong TTS backend.")
    operation = request.get("operation")
    if operation not in {"status", "synthesize", "synthesize_many"}:
        raise RequestError("Unsupported TTS worker operation.")
    if operation != "synthesize_many":
        model = require_mapping(request.get("model"), "model")
        if model.get("id") != model_id:
            raise RequestError("The request names the wrong TTS model.")
        execution = require_mapping(request.get("execution"), "execution")
        if (
                execution.get("required_device") != "cuda"
                or execution.get("allow_cpu_fallback") is not False):
            raise RequestError(
                "This worker requires CUDA and never permits CPU fallback.")
    return operation


def validate_synthesis_items(
        request: Mapping[str, Any],
        installation: Mapping[str, Any],
        *,
        supported_languages: set[str],
        voice_revisions: Mapping[str, str]) -> list[dict]:
    """Prevalidate a batch before loading a model or creating any output."""
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise RequestError("synthesize_many requires a non-empty items array.")
    if len(items) > 4096:
        raise RequestError("A synthesis batch cannot exceed 4096 items.")
    prepared = []
    output_paths: set[Path] = set()
    cache_keys: set[str] = set()
    for index, item in enumerate(items):
        item = require_mapping(item, f"items[{index}]")
        expanded = dict(item)
        model = require_mapping(
            expanded.get("model"), f"items[{index}].model")
        if model.get("id") != installation["model"]["id"]:
            raise RequestError(
                f"items[{index}] names the wrong TTS model.")
        execution = require_mapping(
            expanded.get("execution"), f"items[{index}].execution")
        if (
                execution.get("required_device") != "cuda"
                or execution.get("allow_cpu_fallback") is not False):
            raise RequestError(
                f"items[{index}] must require CUDA without CPU fallback.")
        cache_key = require_sha256(
            expanded.get("cache_key"), f"items[{index}].cache_key")
        if cache_key in cache_keys:
            raise RequestError(
                "Every item in a synthesis batch needs a distinct cache key.")
        cache_keys.add(cache_key)
        data = validate_synthesis_request(
            expanded,
            installation,
            supported_languages=supported_languages,
            voice_revisions=voice_revisions)
        data["cache_key"] = cache_key
        if data["output_path"] in output_paths:
            raise RequestError(
                "Every item in a synthesis batch needs a distinct output "
                "path.")
        output_paths.add(data["output_path"])
        prepared.append(data)
    return prepared


def validate_synthesis_request(
        request: Mapping[str, Any],
        installation: Mapping[str, Any],
        *,
        supported_languages: set[str],
        voice_revisions: Mapping[str, str]) -> dict:
    model = require_mapping(request.get("model"), "model")
    installed_model = require_mapping(
        installation.get("model"), "installed model")
    if model.get("revision") != installed_model.get("revision"):
        raise RequestError(
            "The requested model revision is not the installed revision.")

    input_value = require_mapping(request.get("input"), "input")
    text = require_text(input_value.get("text"), "input text")
    language = require_text(input_value.get("language"), "input language")
    if language not in supported_languages:
        raise RequestError(
            f"This backend cannot synthesize language {language!r}.")
    role = input_value.get("role")
    if role not in {"word", "sentence"}:
        raise RequestError("Input role must be 'word' or 'sentence'.")

    voice = require_mapping(request.get("voice"), "voice")
    voice_id = require_text(voice.get("id"), "voice id")
    expected_revision = voice_revisions.get(voice_id)
    if expected_revision is None:
        raise RequestError(f"Voice {voice_id!r} is not installed.")
    if require_sha256(
            voice.get("revision_sha256"),
            "voice revision") != expected_revision:
        raise RequestError("The requested voice revision is not installed.")
    if (
            voice.get("reference_path") is not None
            or voice.get("reference_sha256") is not None):
        raise RequestError(
            "Custom reference audio is not supported by the fixed "
            "AutoAnki voices.")

    settings = require_mapping(request.get("settings"), "settings")
    emotion = settings.get("emotion", "neutral")
    if not isinstance(emotion, str) or emotion.casefold() != "neutral":
        raise RequestError("Only emotionally neutral synthesis is supported.")
    speed = require_number(settings.get("speed", 1.0), "speed")
    pitch = require_number(settings.get("pitch", 0.0), "pitch")
    energy = require_number(settings.get("energy", 1.0), "energy")
    seed = settings.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RequestError("seed must be an integer.")
    if not 0.5 <= speed <= 2.0:
        raise RequestError("speed must be between 0.5 and 2.0.")
    if not -12.0 <= pitch <= 12.0:
        raise RequestError("pitch must be between -12 and 12 semitones.")
    if energy != 1.0:
        raise RequestError(
            "This runtime does not implement energy adjustment; use 1.0.")

    output = require_mapping(request.get("output"), "output")
    if output.get("codec") != "wav" or output.get("encoder") != "pcm_s16le":
        raise RequestError(
            "The isolated workers currently produce pcm_s16le WAV only.")
    sample_rate = output.get("sample_rate_hz")
    channels = output.get("channels")
    if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate < 8000
            or sample_rate > 192000):
        raise RequestError("The requested WAV sample rate is invalid.")
    if channels != 1:
        raise RequestError("The isolated workers currently produce mono WAV.")
    path = Path(require_text(output.get("path"), "output path")).expanduser()
    if not path.is_absolute():
        raise RequestError("The output path must be absolute.")
    resolved_path = path.resolve()
    staging = (shared_root() / "staging").resolve()
    try:
        resolved_path.relative_to(staging)
    except ValueError as error:
        raise RequestError(
            "The output path must be inside the shared TTS staging "
            "directory.") from error
    if resolved_path.suffix.casefold() != ".wav":
        raise RequestError("The output path must have a .wav suffix.")
    if resolved_path.exists():
        raise RequestError("The output path must not already exist.")
    if not resolved_path.parent.is_dir():
        raise RequestError("The output directory does not exist.")

    prepared = {
        "text": text,
        "language": language,
        "role": role,
        "voice_id": voice_id,
        "speed": speed,
        "pitch": pitch,
        "seed": seed,
        "sample_rate_hz": sample_rate,
        "output_path": resolved_path,
    }
    if "cache_key" in request:
        prepared["cache_key"] = require_sha256(
            request.get("cache_key"), "cache key")
    return prepared


def seed_synthesis(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def gpu_synthesis_lease(backend: str):
    """Serialize model load and synthesis across processes and repositories."""
    lock_directory = shared_root() / "locks"
    lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Distinct from the parent's ``orchestration.lock``: AutoAnki holds that
    # across worker execution and cache publication, while this lease also
    # protects VRAM when a worker is invoked directly.
    lock_path = lock_directory / "worker-gpu.lock"
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        log(f"{backend}: waiting for shared GPU synthesis lease")
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        elif os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            def unlock():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            raise RequestError(
                "This platform has no configured cross-process GPU lease; "
                "unsafe concurrent synthesis was refused.")
        try:
            metadata = json.dumps({
                "backend": backend,
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }, separators=(",", ":")).encode("utf-8")
            stream.seek(0)
            stream.truncate()
            stream.write(metadata)
            stream.flush()
            os.fsync(stream.fileno())
            log(f"{backend}: acquired shared GPU synthesis lease")
            yield
        finally:
            unlock()


def write_tensor_wav(
        tensor: Any,
        source_sample_rate: int,
        target_sample_rate: int,
        output_path: Path) -> None:
    """Resample a mono tensor/array and atomically write PCM16 WAV."""
    import numpy
    import torch
    import torchaudio.functional

    audio = torch.as_tensor(tensor).detach().to(
        device="cpu", dtype=torch.float32)
    audio = audio.squeeze()
    if audio.ndim != 1 or audio.numel() == 0:
        raise RequestError("The TTS backend returned empty or non-mono audio.")
    # Style-Bert returns int16; CosyVoice returns normalized floats.
    peak = float(audio.abs().max())
    if peak > 2.0:
        audio = audio / 32768.0
    if source_sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0),
            source_sample_rate,
            target_sample_rate).squeeze(0)
    audio = audio.nan_to_num().clamp(-1.0, 1.0)
    pcm = (
        audio.mul(32767.0)
        .round()
        .to(dtype=torch.int16)
        .numpy()
        .astype(numpy.dtype("<i2"), copy=False)
        .tobytes())

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with wave.open(str(temporary), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(target_sample_rate)
            stream.writeframes(pcm)
        if temporary.stat().st_size <= 44:
            raise RequestError("The TTS backend produced no audible samples.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def status_response(
        *,
        backend: str,
        model_id: str,
        model_revision: str | None,
        runtime_available: bool,
        gpu_available: bool,
        device_name: str | None,
        voices: Mapping[str, str]) -> dict:
    return {
        "runtime": {"available": bool(runtime_available)},
        "gpu": {
            "available": bool(gpu_available),
            "device": "cuda" if gpu_available else None,
            "name": device_name,
        },
        "model": {
            "id": model_id,
            "revision": model_revision,
        },
        "voices": {
            voice_id: {"revision_sha256": revision}
            for voice_id, revision in sorted(voices.items())
        },
    }


def success_artifact(request_data: Mapping[str, Any]) -> dict:
    artifact = {
        "path": str(request_data["output_path"]),
        "device": "cuda",
        "codec": "wav",
        "encoder": "pcm_s16le",
    }
    if "cache_key" in request_data:
        artifact["cache_key"] = request_data["cache_key"]
    return {"artifact": artifact}


@contextmanager
def clean_batch_outputs_on_error(items: Iterable[Mapping[str, Any]]):
    """Prevent a failed all-or-nothing batch from leaving staged audio."""
    items = tuple(items)
    try:
        yield
    except Exception:
        for item in items:
            path = item.get("output_path")
            if isinstance(path, Path):
                path.unlink(missing_ok=True)
        raise


def _response_base(
        backend: str,
        operation: Any,
        *,
        ok: bool) -> dict:
    return {
        "protocol": PROTOCOL,
        "version": VERSION,
        "backend": backend,
        "operation": operation if isinstance(operation, str) else "invalid",
        "ok": ok,
    }


def run_worker(
        *,
        backend: str,
        model_id: str,
        handler: Callable[[Mapping[str, Any], str], Mapping[str, Any]]) -> int:
    """Read exactly one request and emit exactly one response."""
    operation: Any = "invalid"
    try:
        line = sys.stdin.readline()
        if not line:
            raise RequestError("The worker received no JSON request.")
        if sys.stdin.read().strip():
            raise RequestError("The worker accepts exactly one JSON request.")
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise RequestError("The worker received invalid JSON.") from error
        request = require_mapping(request, "request")
        operation = request.get("operation")
        operation = validate_request_identity(
            request, backend=backend, model_id=model_id)
        # Third-party model libraries frequently print progress and banners to
        # stdout.  Keep the JSON-lines transport unambiguous by treating every
        # backend emission as diagnostics; the one protocol response is
        # printed after leaving this redirect.
        with redirect_stdout(sys.stderr):
            payload = dict(handler(request, operation))
        response = _response_base(backend, operation, ok=True)
        response.update(payload)
    except Exception as error:
        # Handled failures stay machine-readable and use exit status zero per
        # protocol. Keep traceback-like detail on stderr, never stdout.
        message = str(error).strip() or type(error).__name__
        log(f"{backend}: {type(error).__name__}: {message}")
        response = _response_base(backend, operation, ok=False)
        response["error"] = {"message": message}
    print(
        json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":")),
        flush=True)
    return 0
