"""Local, GPU-only text-to-speech orchestration for enhanced Anki cards.

This module deliberately does not import either supported TTS runtime.  Each
runtime lives in an isolated environment and is addressed through a small
one-shot JSON subprocess protocol.  That keeps their large and occasionally
conflicting dependency trees out of AutoAnki's Python environment.

The public entry point is :class:`LocalTTSService`.  It normalises text,
selects the language-specific backend, verifies that the worker has a CUDA
device, content-addresses the request, and publishes immutable cached audio.
No model is downloaded automatically and CPU synthesis is never used as a
fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import errno
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


TTS_PROTOCOL_NAME = "autoanki-local-tts"
TTS_PROTOCOL_VERSION = 1
TTS_CACHE_SCHEMA_VERSION = 1
MEDIA_MANIFEST_SCHEMA_VERSION = 1
MAX_SYNTHESIS_BATCH_ITEMS = 50
SYNTHESIS_TIMEOUT_SECONDS_PER_ITEM = 30.0

STYLE_BERT_BACKEND = "style_bert_vits2_jp_extra"
COSYVOICE_BACKEND = "fun_cosyvoice3_0_5b"

STYLE_BERT_MODEL_ID = "Style-Bert-VITS2 JP-Extra"
COSYVOICE_MODEL_ID = "Fun-CosyVoice3-0.5B"

TTS_HOME_ENVIRONMENT_VARIABLE = "AUTOANKI_TTS_HOME"
STYLE_BERT_WORKER_ENVIRONMENT_VARIABLE = (
    "AUTOANKI_STYLE_BERT_VITS2_WORKER")
COSYVOICE_WORKER_ENVIRONMENT_VARIABLE = "AUTOANKI_COSYVOICE3_WORKER"

_SUPPORTED_ROLES = frozenset({"word", "sentence"})
_BLOCK_TAGS = frozenset({
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
})
_SILENT_TAGS = frozenset({"script", "style", "template"})


class LocalTTSError(RuntimeError):
    """Base class for local TTS failures."""


class UnsupportedTTSLanguageError(LocalTTSError, ValueError):
    """The requested language has no deliberately configured TTS route."""


class TTSRuntimeUnavailableError(LocalTTSError):
    """The selected worker, model, voice, or GPU is unavailable."""


class TTSWorkerProtocolError(LocalTTSError):
    """A worker violated the JSON protocol or returned unsafe output."""


class TTSWorkerExecutionError(LocalTTSError):
    """A correctly formed worker response reported a synthesis failure."""


class _TextExtractor(HTMLParser):
    """Extract readable text without inserting spaces around inline tags."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.silent_depth = 0

    def _separator(self):
        if self.parts and self.parts[-1] != " ":
            self.parts.append(" ")

    def handle_starttag(self, tag, attrs):
        del attrs
        tag = tag.lower()
        if tag in _SILENT_TAGS:
            self.silent_depth += 1
            return
        if not self.silent_depth and tag in _BLOCK_TAGS:
            self._separator()

    def handle_startendtag(self, tag, attrs):
        del attrs
        if not self.silent_depth and tag.lower() in _BLOCK_TAGS:
            self._separator()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _SILENT_TAGS:
            self.silent_depth = max(0, self.silent_depth - 1)
            return
        if not self.silent_depth and tag in _BLOCK_TAGS:
            self._separator()

    def handle_data(self, data):
        if not self.silent_depth:
            self.parts.append(data)


def normalize_tts_text(value: str) -> str:
    """Strip HTML, decode entities, collapse whitespace, and return NFC text."""
    if not isinstance(value, str):
        raise TypeError("TTS text must be a string.")
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception as error:
        raise ValueError("TTS text contains malformed HTML.") from error
    text = "".join(parser.parts)
    # HTMLParser decodes normal entities.  A bounded second pass also handles
    # data that was escaped once for storage and once again for an Anki field.
    for _index in range(2):
        decoded = unescape(text)
        if decoded == text:
            break
        text = decoded
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    if not text:
        raise ValueError("TTS text is empty after HTML removal.")
    return text


def _language_token(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .strip()
        .casefold()
        .replace("_", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


@dataclass(frozen=True)
class TTSRoute:
    language: str
    backend: str
    model_id: str
    default_voice: str


_ROUTES = {
    "english": TTSRoute(
        language="english",
        backend=COSYVOICE_BACKEND,
        model_id=COSYVOICE_MODEL_ID,
        default_voice="neutral-english"),
    "french": TTSRoute(
        language="french",
        backend=COSYVOICE_BACKEND,
        model_id=COSYVOICE_MODEL_ID,
        default_voice="neutral-french"),
    "chinese": TTSRoute(
        language="chinese",
        backend=COSYVOICE_BACKEND,
        model_id=COSYVOICE_MODEL_ID,
        default_voice="neutral-mandarin"),
    "japanese": TTSRoute(
        language="japanese",
        backend=STYLE_BERT_BACKEND,
        model_id=STYLE_BERT_MODEL_ID,
        default_voice="neutral-japanese"),
}

_EXACT_LANGUAGE_ALIASES = {
    "english": "english",
    "modern english": "english",
    "en": "english",
    "en-au": "english",
    "en-ca": "english",
    "en-gb": "english",
    "en-us": "english",
    "french": "french",
    "fr": "french",
    "fr-ca": "french",
    "fr-fr": "french",
    "japanese": "japanese",
    "ja": "japanese",
    "ja-jp": "japanese",
    "jp": "japanese",
    "chinese": "chinese",
    "mandarin": "chinese",
    "mandarin chinese": "chinese",
    "simplified chinese": "chinese",
    "traditional chinese": "chinese",
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh-hans": "chinese",
    "zh-hant": "chinese",
    "zh-hk": "chinese",
    "zh-tw": "chinese",
    "classical chinese": "chinese",
    "classical-chinese": "chinese",
    "classical-chinese-ming": "chinese",
    "classical-chinese-warring-states": "chinese",
    "classical-chinese-han": "chinese",
    "classical-chinese-wang-bi": "chinese",
    "classical chinese (ming)": "chinese",
    "classical chinese (warring states)": "chinese",
    "classical chinese (early han)": "chinese",
    "classical chinese (wang bi recension)": "chinese",
}

_ARCHAIC_LANGUAGE_MARKERS = (
    "middle english",
    "old english",
    "ancient ",
    "archaic ",
    "latin",
)


def resolve_tts_route(language: str) -> TTSRoute:
    """Resolve a modern language without silently treating archaic text as it."""
    if not isinstance(language, str) or not language.strip():
        raise UnsupportedTTSLanguageError(
            "An explicit modern language is required for enhanced audio.")
    token = _language_token(language)
    if any(marker in token for marker in _ARCHAIC_LANGUAGE_MARKERS):
        raise UnsupportedTTSLanguageError(
            f"Enhanced audio does not support the archaic language "
            f"{language!r}; no historically appropriate local voice is "
            "configured.")
    canonical = _EXACT_LANGUAGE_ALIASES.get(token)
    if canonical is None:
        raise UnsupportedTTSLanguageError(
            f"Enhanced audio has no local TTS route for {language!r}. "
            "Supported modern languages are English, French, Chinese, and "
            "Japanese.")
    return _ROUTES[canonical]


def get_tts_root() -> Path:
    """Return the shared, repository-independent TTS data directory."""
    override = os.environ.get(TTS_HOME_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser() / "autoanki" / "tts"
    return Path.home() / ".local" / "share" / "autoanki" / "tts"


def get_huggingface_cache_directory() -> Path:
    """Return Hugging Face's standard shared cache location."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


@dataclass(frozen=True)
class TTSPaths:
    root: Path
    artifacts: Path
    manifests: Path
    staging: Path
    runtimes: Path
    voices: Path
    huggingface_cache: Path

    @classmethod
    def shared(cls, root: Path | None = None) -> "TTSPaths":
        root = Path(root) if root is not None else get_tts_root()
        return cls(
            root=root,
            artifacts=root / "artifacts",
            manifests=root / "manifests",
            staging=root / "staging",
            runtimes=root / "runtimes",
            voices=root / "voices",
            huggingface_cache=get_huggingface_cache_directory(),
        )

    def ensure_writable_directories(self):
        for directory in (
                self.root,
                self.artifacts,
                self.manifests,
                self.staging,
                self.runtimes,
                self.voices):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)


@contextmanager
def _shared_orchestration_lease(paths: TTSPaths):
    """Serialize shared GPU/cache work across repositories and processes."""
    lock_path = paths.root / "locks" / "orchestration.lock"
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    except OSError as error:
        if error.errno in {
                errno.EACCES,
                errno.EAGAIN,
                getattr(errno, "EDEADLK", errno.EAGAIN)}:
            raise TTSRuntimeUnavailableError(
                "Could not acquire the shared local-TTS GPU/cache lease.") \
                from error
        raise
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@dataclass(frozen=True)
class OutputFormat:
    codec: str = "wav"
    encoder: str = "pcm_s16le"
    sample_rate_hz: int = 24000
    channels: int = 1

    def __post_init__(self):
        if (
                not isinstance(self.codec, str)
                or self.codec not in {"wav", "flac", "ogg", "mp3"}):
            raise ValueError(f"Unsupported audio codec: {self.codec!r}")
        if not isinstance(self.encoder, str) or not self.encoder.strip():
            raise ValueError("An audio encoder identity is required.")
        if (
                isinstance(self.sample_rate_hz, bool)
                or not isinstance(self.sample_rate_hz, int)
                or self.sample_rate_hz < 8000):
            raise ValueError("The audio sample rate is implausibly low.")
        if (
                isinstance(self.channels, bool)
                or not isinstance(self.channels, int)
                or self.channels not in {1, 2}):
            raise ValueError("Audio must use one or two channels.")


def _default_settings():
    return {
        "emotion": "neutral",
        "speed": 1.0,
        "pitch": 0.0,
        "energy": 1.0,
        "seed": 0,
    }


def _json_safe_mapping(value: Mapping[str, Any], *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    copied = dict(value)
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain canonical JSON values.") from error
    return json.loads(encoded)


@dataclass(frozen=True)
class AudioRequest:
    text: str
    language: str
    role: str
    voice_id: str | None = None
    reference_audio_path: Path | None = None
    settings: Mapping[str, Any] = field(default_factory=_default_settings)
    output_format: OutputFormat = field(default_factory=OutputFormat)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("Audio text must be a string.")
        if not isinstance(self.language, str):
            raise TypeError("Audio language must be a string.")
        if (
                not isinstance(self.role, str)
                or self.role not in _SUPPORTED_ROLES):
            raise ValueError(
                "Audio role must be either 'word' or 'sentence'.")
        if self.voice_id is not None:
            if (
                    not isinstance(self.voice_id, str)
                    or not self.voice_id.strip()):
                raise ValueError("A supplied voice identity cannot be empty.")
        if not isinstance(self.output_format, OutputFormat):
            raise TypeError("Audio output_format must be an OutputFormat.")
        object.__setattr__(
            self,
            "settings",
            _json_safe_mapping(self.settings, label="Synthesis settings"))
        if self.reference_audio_path is not None:
            object.__setattr__(
                self,
                "reference_audio_path",
                Path(self.reference_audio_path))


@dataclass(frozen=True)
class PreparedAudioRequest:
    text: str
    language: str
    role: str
    backend: str
    model_id: str
    model_revision: str
    voice_id: str
    voice_revision_sha256: str
    reference_audio_path: Path | None
    reference_audio_sha256: str | None
    settings: Mapping[str, Any]
    output_format: OutputFormat
    cache_key: str

    def identity(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "role": self.role,
            "backend": self.backend,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
            },
            "voice": {
                "id": self.voice_id,
                "revision_sha256": self.voice_revision_sha256,
                "reference_sha256": self.reference_audio_sha256,
            },
            "settings": dict(self.settings),
            "output": asdict(self.output_format),
        }


@dataclass(frozen=True)
class TTSBackendStatus:
    backend: str
    model_id: str
    model_revision: str | None
    runtime_available: bool
    gpu_available: bool
    device_name: str | None
    voices: Mapping[str, str]
    message: str

    @property
    def ready(self):
        return bool(
            self.runtime_available
            and self.gpu_available
            and self.model_revision
            and self.voices)


@dataclass(frozen=True)
class AudioArtifact:
    cache_key: str
    path: Path
    manifest_path: Path
    media_filename: str
    sha256: str
    byte_count: int
    codec: str
    encoder: str
    language: str
    role: str
    backend: str
    model_id: str
    model_revision: str
    voice_id: str

    def as_manifest_entry(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "media_filename": self.media_filename,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "codec": self.codec,
            "encoder": self.encoder,
            "language": self.language,
            "role": self.role,
            "backend": self.backend,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
            },
            "voice_id": self.voice_id,
        }


@dataclass(frozen=True)
class AudioSynthesisPlan:
    """Prepared requests plus the cache work needed to satisfy them."""

    requests: tuple[PreparedAudioRequest, ...]
    statuses: Mapping[str, TTSBackendStatus]
    cached_artifacts: Mapping[str, AudioArtifact]

    @property
    def requested_count(self) -> int:
        return len(self.requests)

    @property
    def unique_count(self) -> int:
        return len({
            request.cache_key
            for request in self.requests
        })

    @property
    def cache_hit_count(self) -> int:
        return len(self.cached_artifacts)

    @property
    def synthesis_count(self) -> int:
        return self.unique_count - self.cache_hit_count


@dataclass(frozen=True)
class WorkerCommand:
    argv: tuple[str, ...]
    required_paths: tuple[Path, ...] = ()

    def availability_problem(self) -> str | None:
        if not self.argv:
            return "No worker command is configured."
        executable = self.argv[0]
        if (
                os.sep in executable
                or (os.altsep is not None and os.altsep in executable)):
            if not Path(executable).expanduser().is_file():
                return f"Worker executable is missing: {executable}"
        elif shutil.which(executable) is None:
            return f"Worker executable is not on PATH: {executable}"
        missing = [
            str(path)
            for path in self.required_paths
            if not Path(path).expanduser().exists()
        ]
        if missing:
            return "Worker runtime is incomplete; missing " + ", ".join(missing)
        return None


def _command_from_environment(variable_name: str) -> WorkerCommand | None:
    value = os.environ.get(variable_name)
    if value is None:
        return None
    argv = tuple(shlex.split(value))
    if not argv:
        return WorkerCommand(())
    return WorkerCommand(argv)


def default_worker_commands(
        paths: TTSPaths | None = None) -> dict[str, WorkerCommand]:
    """Return commands for isolated shared runtimes, without installing them."""
    paths = paths or TTSPaths.shared()
    style_runtime = paths.runtimes / STYLE_BERT_BACKEND
    cosy_runtime = paths.runtimes / COSYVOICE_BACKEND
    if os.name == "nt":
        style_python = style_runtime / "Scripts" / "python.exe"
        cosy_python = cosy_runtime / "Scripts" / "python.exe"
    else:
        style_python = style_runtime / "bin" / "python"
        cosy_python = cosy_runtime / "bin" / "python"
    style_entry = style_runtime / "autoanki_worker.py"
    cosy_entry = cosy_runtime / "autoanki_worker.py"
    return {
        STYLE_BERT_BACKEND: (
            _command_from_environment(
                STYLE_BERT_WORKER_ENVIRONMENT_VARIABLE)
            or WorkerCommand(
                (str(style_python), str(style_entry)),
                required_paths=(style_python, style_entry))),
        COSYVOICE_BACKEND: (
            _command_from_environment(
                COSYVOICE_WORKER_ENVIRONMENT_VARIABLE)
            or WorkerCommand(
                (str(cosy_python), str(cosy_entry)),
                required_paths=(cosy_python, cosy_entry))),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_presentation_key(*identity_parts: Any) -> str:
    """Return a stable identity used for 50/50 audio/text presentation."""
    payload = json.dumps(
        identity_parts,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_audio_first(*identity_parts: Any) -> bool:
    """Choose a stable, unbiased presentation bit for one Anki note."""
    key = deterministic_presentation_key(*identity_parts)
    return int(key[:2], 16) < 128


def deterministic_choice_index(count: int, *identity_parts: Any) -> int:
    """Choose one aligned example reproducibly without runtime JS races."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("A deterministic choice needs a positive count.")
    key = deterministic_presentation_key(*identity_parts)
    return int(key[2:18], 16) % count


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))


class JSONSubprocessWorker:
    """Execute one protocol operation in a fresh subprocess."""

    def __init__(
            self,
            commands: Mapping[str, WorkerCommand] | None = None,
            *,
            paths: TTSPaths | None = None,
            timeout_seconds: float = 300.0):
        self.paths = paths or TTSPaths.shared()
        self.commands = dict(commands or default_worker_commands(self.paths))
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0 or not math.isfinite(
                self.timeout_seconds):
            raise ValueError("Worker timeout must be a positive finite value.")

    def _invoke(
            self,
            backend: str,
            payload: Mapping[str, Any],
            *,
            timeout_seconds: float | None = None) -> dict:
        command = self.commands.get(backend)
        if command is None:
            raise TTSRuntimeUnavailableError(
                f"No isolated worker is configured for {backend}.")
        problem = command.availability_problem()
        if problem:
            raise TTSRuntimeUnavailableError(problem)
        request = {
            "protocol": TTS_PROTOCOL_NAME,
            "version": TTS_PROTOCOL_VERSION,
            "backend": backend,
            **dict(payload),
        }
        environment = os.environ.copy()
        # Keep model files shared between AutoAnki and other repositories.
        environment.setdefault(
            "HF_HOME",
            str(self.paths.huggingface_cache))
        effective_timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else max(self.timeout_seconds, float(timeout_seconds)))
        try:
            completed = subprocess.run(
                command.argv,
                input=json.dumps(
                    request,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":")) + "\n",
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=effective_timeout,
                check=False,
                env=environment)
        except subprocess.TimeoutExpired as error:
            raise TTSWorkerExecutionError(
                f"The {backend} TTS worker timed out after "
                f"{effective_timeout:g} seconds.") from error
        except OSError as error:
            raise TTSRuntimeUnavailableError(
                f"The {backend} TTS worker could not start: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            raise TTSWorkerExecutionError(
                f"The {backend} TTS worker exited with code "
                f"{completed.returncode}"
                + (f": {detail}" if detail else "."))
        lines = [
            line
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        if len(lines) != 1:
            raise TTSWorkerProtocolError(
                f"The {backend} worker must return exactly one JSON object.")
        try:
            response = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise TTSWorkerProtocolError(
                f"The {backend} worker returned invalid JSON.") from error
        if not isinstance(response, dict):
            raise TTSWorkerProtocolError(
                f"The {backend} worker response is not an object.")
        if (
                response.get("protocol") != TTS_PROTOCOL_NAME
                or response.get("version") != TTS_PROTOCOL_VERSION
                or response.get("operation") != request.get("operation")
                or response.get("backend") != backend):
            raise TTSWorkerProtocolError(
                f"The {backend} worker response has incompatible identity.")
        if response.get("ok") is not True:
            error = response.get("error")
            if not isinstance(error, dict):
                raise TTSWorkerProtocolError(
                    f"The {backend} worker returned an invalid error.")
            message = error.get("message")
            if not isinstance(message, str) or not message.strip():
                raise TTSWorkerProtocolError(
                    f"The {backend} worker error has no message.")
            raise TTSWorkerExecutionError(message.strip())
        return response

    def status(self, route: TTSRoute) -> TTSBackendStatus:
        command = self.commands.get(route.backend)
        problem = (
            "No isolated worker is configured."
            if command is None
            else command.availability_problem()
        )
        if problem:
            return TTSBackendStatus(
                backend=route.backend,
                model_id=route.model_id,
                model_revision=None,
                runtime_available=False,
                gpu_available=False,
                device_name=None,
                voices={},
                message=problem)
        try:
            response = self._invoke(route.backend, {
                "operation": "status",
                "model": {"id": route.model_id},
                "execution": {
                    "required_device": "cuda",
                    "allow_cpu_fallback": False,
                },
            })
        except (TTSRuntimeUnavailableError, TTSWorkerExecutionError) as error:
            return TTSBackendStatus(
                backend=route.backend,
                model_id=route.model_id,
                model_revision=None,
                runtime_available=False,
                gpu_available=False,
                device_name=None,
                voices={},
                message=str(error))
        runtime = response.get("runtime")
        gpu = response.get("gpu")
        model = response.get("model")
        voices = response.get("voices")
        if not all(isinstance(value, dict) for value in (
                runtime, gpu, model, voices)):
            raise TTSWorkerProtocolError(
                f"The {route.backend} status response is incomplete.")
        if model.get("id") != route.model_id:
            raise TTSWorkerProtocolError(
                f"The {route.backend} worker loaded the wrong model.")
        revision = model.get("revision")
        if revision is not None and (
                not isinstance(revision, str) or not revision.strip()):
            raise TTSWorkerProtocolError(
                f"The {route.backend} model revision is invalid.")
        parsed_voices = {}
        for voice_id, metadata in voices.items():
            if (
                    not isinstance(voice_id, str)
                    or not voice_id
                    or not isinstance(metadata, dict)
                    or not _valid_sha256(
                        metadata.get("revision_sha256"))):
                raise TTSWorkerProtocolError(
                    f"The {route.backend} voice inventory is invalid.")
            parsed_voices[voice_id] = metadata["revision_sha256"]
        gpu_available = (
            gpu.get("available") is True
            and gpu.get("device") == "cuda")
        runtime_available = runtime.get("available") is True
        if not runtime_available:
            message = "The local TTS runtime or model is not installed."
        elif not gpu_available:
            message = (
                "A CUDA GPU is required; CPU fallback is deliberately "
                "disabled.")
        elif not revision:
            message = "The installed model did not report a revision."
        elif not parsed_voices:
            message = "No installed voice reported a content revision."
        else:
            message = "Ready for GPU synthesis."
        return TTSBackendStatus(
            backend=route.backend,
            model_id=route.model_id,
            model_revision=revision,
            runtime_available=runtime_available,
            gpu_available=gpu_available,
            device_name=(
                gpu.get("name")
                if isinstance(gpu.get("name"), str)
                else None),
            voices=parsed_voices,
            message=message)

    @staticmethod
    def _synthesis_item(
            request: PreparedAudioRequest,
            output_path: Path) -> dict:
        return {
            "cache_key": request.cache_key,
            "model": {
                "id": request.model_id,
                "revision": request.model_revision,
            },
            "input": {
                "text": request.text,
                "language": request.language,
                "role": request.role,
            },
            "voice": {
                "id": request.voice_id,
                "revision_sha256": request.voice_revision_sha256,
                "reference_path": (
                    str(request.reference_audio_path)
                    if request.reference_audio_path is not None
                    else None),
                "reference_sha256": request.reference_audio_sha256,
            },
            "settings": dict(request.settings),
            "output": {
                **asdict(request.output_format),
                "path": str(output_path),
            },
            "execution": {
                "required_device": "cuda",
                "allow_cpu_fallback": False,
            },
        }

    @staticmethod
    def _validate_worker_artifact(
            request: PreparedAudioRequest,
            output_path: Path,
            artifact: Any):
        if not isinstance(artifact, dict):
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker returned no audio artifact.")
        if artifact.get("cache_key") not in {None, request.cache_key}:
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker returned audio for the wrong "
                "cache identity.")
        if artifact.get("path") != str(output_path):
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker wrote an unexpected path.")
        if artifact.get("device") != "cuda":
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker did not use CUDA; the result "
                "was rejected rather than accepting a CPU fallback.")
        if (
                artifact.get("codec") != request.output_format.codec
                or artifact.get("encoder") != request.output_format.encoder):
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker changed the requested output "
                "format.")

    def synthesize(
            self,
            request: PreparedAudioRequest,
            output_path: Path):
        response = self._invoke(request.backend, {
            "operation": "synthesize",
            **self._synthesis_item(request, output_path),
        })
        self._validate_worker_artifact(
            request,
            output_path,
            response.get("artifact"))

    def synthesize_many(
            self,
            requests: Sequence[PreparedAudioRequest],
            output_paths: Sequence[Path]):
        """Load one backend once and synthesize a whole cache-miss batch."""
        requests = tuple(requests)
        output_paths = tuple(Path(path) for path in output_paths)
        if len(requests) != len(output_paths):
            raise ValueError(
                "Batch TTS requests and output paths must be aligned.")
        if not requests:
            return
        backend = requests[0].backend
        if any(request.backend != backend for request in requests):
            raise ValueError(
                "One TTS subprocess batch cannot mix backends.")
        response = self._invoke(
            backend,
            {
                "operation": "synthesize_many",
                "items": [
                    self._synthesis_item(request, output_path)
                    for request, output_path in zip(
                        requests,
                        output_paths,
                        strict=True)
                ],
            },
            timeout_seconds=(
                len(requests)
                * SYNTHESIS_TIMEOUT_SECONDS_PER_ITEM))
        artifacts = response.get("artifacts")
        if (
                not isinstance(artifacts, list)
                or len(artifacts) != len(requests)):
            raise TTSWorkerProtocolError(
                f"The {backend} worker returned the wrong number of batch "
                "artifacts.")
        by_cache_key = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise TTSWorkerProtocolError(
                    f"The {backend} worker returned an invalid batch "
                    "artifact.")
            cache_key = artifact.get("cache_key")
            if not _valid_sha256(cache_key) or cache_key in by_cache_key:
                raise TTSWorkerProtocolError(
                    f"The {backend} worker returned ambiguous batch "
                    "identities.")
            by_cache_key[cache_key] = artifact
        if set(by_cache_key) != {
                request.cache_key for request in requests}:
            raise TTSWorkerProtocolError(
                f"The {backend} worker returned audio for the wrong batch.")
        for request, output_path in zip(
                requests,
                output_paths,
                strict=True):
            self._validate_worker_artifact(
                request,
                output_path,
                by_cache_key[request.cache_key])


class LocalTTSService:
    """GPU-only synthesiser with immutable content-addressed caching."""

    def __init__(
            self,
            *,
            paths: TTSPaths | None = None,
            worker: JSONSubprocessWorker | None = None):
        self.paths = paths or TTSPaths.shared()
        self.worker = worker or JSONSubprocessWorker(paths=self.paths)
        # GPU synthesis is intentionally serial.  Besides avoiding VRAM
        # contention, this makes identical concurrent calls share one cache
        # publication rather than invoking the model twice.
        self._synthesis_lock = threading.RLock()

    def backend_status(self, language: str) -> TTSBackendStatus:
        return self.worker.status(resolve_tts_route(language))

    @staticmethod
    def _require_ready(
            status: TTSBackendStatus,
            route: TTSRoute):
        if not status.runtime_available:
            raise TTSRuntimeUnavailableError(
                f"{route.model_id} is unavailable: {status.message}")
        if not status.gpu_available:
            raise TTSRuntimeUnavailableError(
                f"{route.model_id} cannot run: {status.message}")
        if not status.model_revision:
            raise TTSRuntimeUnavailableError(
                f"{route.model_id} has no installed revision identity.")
        if not status.voices:
            raise TTSRuntimeUnavailableError(
                f"{route.model_id} has no revisioned installed voice.")

    def _prepare(
            self,
            request: AudioRequest,
            *,
            status: TTSBackendStatus | None = None) -> PreparedAudioRequest:
        route = resolve_tts_route(request.language)
        status = status or self.worker.status(route)
        self._require_ready(status, route)
        voice_id = request.voice_id or route.default_voice
        voice_revision = status.voices.get(voice_id)
        if voice_revision is None:
            raise TTSRuntimeUnavailableError(
                f"Voice {voice_id!r} is not installed for {route.model_id}.")
        reference_path = request.reference_audio_path
        reference_sha = None
        if reference_path is not None:
            if not reference_path.is_file():
                raise TTSRuntimeUnavailableError(
                    f"TTS reference audio is missing: {reference_path}")
            reference_sha = _sha256_file(reference_path)
        identity = {
            "text": normalize_tts_text(request.text),
            "language": route.language,
            "role": request.role,
            "backend": route.backend,
            "model": {
                "id": route.model_id,
                "revision": status.model_revision,
            },
            "voice": {
                "id": voice_id,
                "revision_sha256": voice_revision,
                "reference_sha256": reference_sha,
            },
            "settings": dict(request.settings),
            "output": asdict(request.output_format),
        }
        return PreparedAudioRequest(
            text=identity["text"],
            language=route.language,
            role=request.role,
            backend=route.backend,
            model_id=route.model_id,
            model_revision=status.model_revision,
            voice_id=voice_id,
            voice_revision_sha256=voice_revision,
            reference_audio_path=reference_path,
            reference_audio_sha256=reference_sha,
            settings=dict(request.settings),
            output_format=request.output_format,
            cache_key=_canonical_digest(identity),
        )

    def _paths_for(
            self,
            request: PreparedAudioRequest) -> tuple[Path, Path]:
        shard = request.cache_key[:2]
        artifact_path = (
            self.paths.artifacts
            / shard
            / f"{request.cache_key}.{request.output_format.codec}")
        manifest_path = (
            self.paths.manifests
            / shard
            / f"{request.cache_key}.json")
        return artifact_path, manifest_path

    def _cached(
            self,
            request: PreparedAudioRequest) -> AudioArtifact | None:
        artifact_path, manifest_path = self._paths_for(request)
        if not artifact_path.is_file() or not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected_identity = request.identity()
        if (
                manifest.get("schema_version") != TTS_CACHE_SCHEMA_VERSION
                or manifest.get("cache_key") != request.cache_key
                or manifest.get("request") != expected_identity):
            return None
        artifact = manifest.get("artifact")
        if not isinstance(artifact, dict):
            return None
        sha256 = artifact.get("sha256")
        byte_count = artifact.get("byte_count")
        media_filename = artifact.get("media_filename")
        if (
                not _valid_sha256(sha256)
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count <= 0
                or not isinstance(media_filename, str)
                or Path(media_filename).name != media_filename
                or not media_filename.startswith("_autoanki_tts_")
                or artifact_path.stat().st_size != byte_count
                or _sha256_file(artifact_path) != sha256):
            return None
        return AudioArtifact(
            cache_key=request.cache_key,
            path=artifact_path,
            manifest_path=manifest_path,
            media_filename=media_filename,
            sha256=sha256,
            byte_count=byte_count,
            codec=request.output_format.codec,
            encoder=request.output_format.encoder,
            language=request.language,
            role=request.role,
            backend=request.backend,
            model_id=request.model_id,
            model_revision=request.model_revision,
            voice_id=request.voice_id,
        )

    def _allocate_staging_path(
            self,
            request: PreparedAudioRequest) -> Path:
        artifact_path, manifest_path = self._paths_for(request)
        artifact_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        manifest_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.paths.staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{request.cache_key}.",
            suffix=f".{request.output_format.codec}",
            dir=self.paths.staging)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        # A worker must create the audio itself.  Leaving an empty file in
        # place can mask runtimes that merely claim success.
        temporary_path.unlink()
        return temporary_path

    def _publish_staged(
            self,
            request: PreparedAudioRequest,
            temporary_path: Path) -> AudioArtifact:
        artifact_path, manifest_path = self._paths_for(request)
        if not temporary_path.is_file():
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker reported success without "
                "creating audio.")
        byte_count = temporary_path.stat().st_size
        if byte_count <= 0:
            raise TTSWorkerProtocolError(
                f"The {request.backend} worker created empty audio.")
        sha256 = _sha256_file(temporary_path)
        temporary_path.replace(artifact_path)
        media_filename = (
            f"_autoanki_tts_{request.cache_key[:24]}."
            f"{request.output_format.codec}")
        manifest = {
            "schema_version": TTS_CACHE_SCHEMA_VERSION,
            "cache_key": request.cache_key,
            "request": request.identity(),
            "artifact": {
                "relative_path": str(
                    artifact_path.relative_to(self.paths.root)),
                "media_filename": media_filename,
                "sha256": sha256,
                "byte_count": byte_count,
            },
        }
        self._write_json_atomic(manifest_path, manifest)
        artifact = self._cached(request)
        if artifact is None:
            raise TTSWorkerProtocolError(
                "The newly published TTS artifact failed cache validation.")
        return artifact

    def _publish(
            self,
            request: PreparedAudioRequest) -> AudioArtifact:
        temporary_path = self._allocate_staging_path(request)
        try:
            self.worker.synthesize(request, temporary_path)
            return self._publish_staged(request, temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _publish_many(
            self,
            requests: Sequence[PreparedAudioRequest],
    ) -> tuple[AudioArtifact, ...]:
        """Publish one backend batch after a single model-loading process."""
        requests = tuple(requests)
        if not requests:
            return ()
        if len({request.cache_key for request in requests}) != len(requests):
            raise ValueError("A publish batch must contain unique requests.")
        if len({request.backend for request in requests}) != 1:
            raise ValueError("A publish batch cannot mix TTS backends.")
        temporary_paths = tuple(
            self._allocate_staging_path(request)
            for request in requests)
        try:
            batch_synthesizer = getattr(
                self.worker,
                "synthesize_many",
                None)
            if callable(batch_synthesizer):
                batch_synthesizer(requests, temporary_paths)
            else:
                # Small test doubles and explicitly supplied legacy workers
                # remain compatible.  The built-in worker always batches.
                for request, temporary_path in zip(
                        requests,
                        temporary_paths,
                        strict=True):
                    self.worker.synthesize(request, temporary_path)
            # Validate every output before publishing any cache entry.
            for request, temporary_path in zip(
                    requests,
                    temporary_paths,
                    strict=True):
                if (
                        not temporary_path.is_file()
                        or temporary_path.stat().st_size <= 0):
                    raise TTSWorkerProtocolError(
                        f"The {request.backend} batch did not create every "
                        "requested audio file.")
            return tuple(
                self._publish_staged(request, temporary_path)
                for request, temporary_path in zip(
                    requests,
                    temporary_paths,
                    strict=True))
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _write_json_atomic(path: Path, value: Mapping[str, Any]):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)

    def synthesize_many(
            self,
            requests: Iterable[AudioRequest]) -> tuple[AudioArtifact, ...]:
        """Synthesize requests in order, reusing each distinct request once."""
        requests = tuple(requests)
        if not requests:
            return ()
        with self._synthesis_lock:
            with _shared_orchestration_lease(self.paths):
                plan = self._plan(requests)
                return self._execute_plan(plan)

    def synthesize(self, request: AudioRequest) -> AudioArtifact:
        return self.synthesize_many((request,))[0]

    def _plan(
            self,
            requests: Sequence[AudioRequest]) -> AudioSynthesisPlan:
        statuses: dict[str, TTSBackendStatus] = {}
        prepared: list[PreparedAudioRequest] = []
        for request in requests:
            if not isinstance(request, AudioRequest):
                raise TypeError("Audio plans require AudioRequest values.")
            route = resolve_tts_route(request.language)
            if route.backend not in statuses:
                statuses[route.backend] = self.worker.status(route)
            prepared.append(self._prepare(
                request,
                status=statuses[route.backend]))
        cached = {}
        for request in prepared:
            if request.cache_key in cached:
                continue
            artifact = self._cached(request)
            if artifact is not None:
                cached[request.cache_key] = artifact
        return AudioSynthesisPlan(
            requests=tuple(prepared),
            statuses=dict(statuses),
            cached_artifacts=cached)

    def plan(
            self,
            requests: Iterable[AudioRequest]) -> AudioSynthesisPlan:
        """Prepare and de-duplicate work without running either TTS model."""
        requests = tuple(requests)
        with self._synthesis_lock:
            return self._plan(requests)

    def _execute_plan(
            self,
            plan: AudioSynthesisPlan) -> tuple[AudioArtifact, ...]:
        by_key: dict[str, AudioArtifact] = dict(plan.cached_artifacts)
        missing_by_backend: dict[str, list[PreparedAudioRequest]] = {}
        for request in plan.requests:
            if request.cache_key != _canonical_digest(request.identity()):
                raise ValueError(
                    "An audio plan contains a mismatched cache identity.")
            if request.cache_key in by_key:
                continue
            artifact = self._cached(request)
            if artifact is not None:
                by_key[request.cache_key] = artifact
                continue
            backend_requests = missing_by_backend.setdefault(
                request.backend,
                [])
            if all(
                    existing.cache_key != request.cache_key
                    for existing in backend_requests):
                backend_requests.append(request)
        for backend_requests in missing_by_backend.values():
            for start in range(
                    0,
                    len(backend_requests),
                    MAX_SYNTHESIS_BATCH_ITEMS):
                batch = backend_requests[
                    start:start + MAX_SYNTHESIS_BATCH_ITEMS]
                for request, artifact in zip(
                        batch,
                        self._publish_many(batch),
                        strict=True):
                    by_key[request.cache_key] = artifact
        return tuple(
            by_key[request.cache_key]
            for request in plan.requests)

    def execute_plan(
            self,
            plan: AudioSynthesisPlan) -> tuple[AudioArtifact, ...]:
        """Execute a previously prepared plan, preserving request order."""
        if not isinstance(plan, AudioSynthesisPlan):
            raise TypeError("Expected an AudioSynthesisPlan.")
        with self._synthesis_lock:
            with _shared_orchestration_lease(self.paths):
                return self._execute_plan(plan)


def require_tts_ready(
        language: str,
        *,
        service: LocalTTSService | None = None) -> TTSBackendStatus:
    """Fail before generation costs money when Enhanced audio cannot run."""
    service = service or LocalTTSService()
    status = service.backend_status(language)
    route = resolve_tts_route(language)
    LocalTTSService._require_ready(status, route)
    return status


def build_media_manifest(
        artifacts: Iterable[AudioArtifact]) -> dict:
    """Return a deterministic, de-duplicated manifest for deck packaging."""
    unique = {}
    for artifact in artifacts:
        if not isinstance(artifact, AudioArtifact):
            raise TypeError("Media manifests require AudioArtifact values.")
        previous = unique.get(artifact.cache_key)
        if previous is not None and previous.sha256 != artifact.sha256:
            raise ValueError(
                f"Conflicting audio exists for cache key "
                f"{artifact.cache_key}.")
        unique[artifact.cache_key] = artifact
    entries = [
        artifact.as_manifest_entry()
        for _key, artifact in sorted(unique.items())
    ]
    return {
        "schema_version": MEDIA_MANIFEST_SCHEMA_VERSION,
        "kind": "autoanki_enhanced_audio",
        "artifacts": entries,
    }


def materialize_media(
        artifacts: Iterable[AudioArtifact],
        destination: Path) -> dict:
    """Copy cached audio to a package directory and return its manifest.

    Existing identical files are retained.  A filename collision with
    different bytes is rejected rather than overwritten.
    """
    artifacts = tuple(artifacts)
    manifest = build_media_manifest(artifacts)
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    by_key = {artifact.cache_key: artifact for artifact in artifacts}
    for entry in manifest["artifacts"]:
        artifact = by_key[entry["cache_key"]]
        target = destination / artifact.media_filename
        if target.exists():
            if not target.is_file() or _sha256_file(target) != artifact.sha256:
                raise FileExistsError(
                    f"An unrelated media file already uses "
                    f"{artifact.media_filename}.")
            continue
        shutil.copy2(artifact.path, target)
        if _sha256_file(target) != artifact.sha256:
            target.unlink(missing_ok=True)
            raise OSError(
                f"Copied audio failed verification: {artifact.media_filename}")
    return manifest


__all__ = [
    "AudioArtifact",
    "AudioRequest",
    "AudioSynthesisPlan",
    "COSYVOICE_BACKEND",
    "COSYVOICE_MODEL_ID",
    "JSONSubprocessWorker",
    "LocalTTSError",
    "LocalTTSService",
    "MAX_SYNTHESIS_BATCH_ITEMS",
    "MEDIA_MANIFEST_SCHEMA_VERSION",
    "OutputFormat",
    "PreparedAudioRequest",
    "STYLE_BERT_BACKEND",
    "STYLE_BERT_MODEL_ID",
    "TTSBackendStatus",
    "TTSPaths",
    "TTSRoute",
    "TTSRuntimeUnavailableError",
    "TTSWorkerExecutionError",
    "TTSWorkerProtocolError",
    "UnsupportedTTSLanguageError",
    "WorkerCommand",
    "build_media_manifest",
    "default_worker_commands",
    "deterministic_audio_first",
    "deterministic_choice_index",
    "deterministic_presentation_key",
    "get_huggingface_cache_directory",
    "get_tts_root",
    "materialize_media",
    "normalize_tts_text",
    "require_tts_ready",
    "resolve_tts_route",
]
