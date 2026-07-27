#!/usr/bin/env python3
"""Install AutoAnki's local TTS backends into isolated shared runtimes.

Nothing is installed in AutoAnki's virtual environment. Runtime environments,
model snapshots, reference voices, and source checkouts live beneath
``~/.local/share/autoanki/tts`` by default, so other repositories can reuse
them. Existing runtimes are moved to timestamped backups when ``--force`` is
used; they are never recursively deleted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen
import venv


STYLE_BACKEND = "style_bert_vits2_jp_extra"
COSY_BACKEND = "fun_cosyvoice3_0_5b"
STYLE_MODEL_ID = "Style-Bert-VITS2 JP-Extra"
COSY_MODEL_ID = "Fun-CosyVoice3-0.5B"

# Pinned upstream identities make installations and audio cache identities
# reproducible. Update only after a GPU synthesis regression test.
STYLE_SOURCE_REVISION = "66de777e06392c0f313600be03c43ef96658b244"
COSY_SOURCE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
STYLE_VOICE_REPOSITORY = "litagin/style_bert_vits2_jvnv"
STYLE_VOICE_REVISION = "205830ca1d49e666ddfbf2a755f0108e9cade4dd"
COSY_MODEL_REPOSITORY = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSY_MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
STYLE_BERT_REPOSITORY = "ku-nlp/deberta-v2-large-japanese-char-wwm"
# The CosyVoice revision pins Whisper 20231117, whose ``triton<3`` metadata
# forces pip to replace modern CUDA PyTorch.  CosyVoice uses only Whisper's
# stable tokenizer and log-mel APIs.  This newer immutable release supports
# ``triton>=2`` and therefore preserves the CUDA 13 PyTorch stack required by
# Ada/Blackwell-era GPUs.
COSY_WHISPER_VERSION = "20250625"
# Upstream's ONNX Runtime 1.18 Linux wheel is linked against CUDA 11/cuDNN 8.
# This release supplies a Python 3.10 wheel with CUDA 12/cuDNN 9 side-by-side
# runtime dependencies and remains compatible with the model's ONNX graphs.
COSY_ONNXRUNTIME_VERSION = "1.23.2"

DEFAULT_TORCH_VERSION = "2.11.0"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"

STYLE_VOICE_FILES = (
    "jvnv-F1-jp/config.json",
    "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
    "jvnv-F1-jp/style_vectors.npy",
)
COSY_MODEL_FILES = (
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/*",
)
COSY_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "希望你以后能够做的比我还好呦。"
)

_HERE = Path(__file__).resolve().parent
_WORKERS = _HERE / "tts_workers"
_COMMON_WORKER = _WORKERS / "_autoanki_tts_worker_common.py"
_STYLE_WORKER = _WORKERS / "style_bert_vits2_worker.py"
_COSY_WORKER = _WORKERS / "fun_cosyvoice3_worker.py"


class InstallationError(RuntimeError):
    """An actionable installation failure."""


def _default_root() -> Path:
    override = os.environ.get("AUTOANKI_TTS_HOME")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser() / "autoanki" / "tts"
    return Path.home() / ".local" / "share" / "autoanki" / "tts"


def _run(
        arguments: Iterable[str | Path],
        *,
        environment: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = False) -> subprocess.CompletedProcess:
    argv = [str(value) for value in arguments]
    print("+", " ".join(argv), flush=True)
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None)
    if completed.returncode:
        detail = ""
        if capture:
            detail = (completed.stderr or completed.stdout or "").strip()
            if len(detail) > 3000:
                detail = detail[-3000:]
        raise InstallationError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(argv)}"
            + (f"\n{detail}" if detail else ""))
    return completed


def _python_version(executable: Path) -> tuple[int, int, int]:
    completed = _run(
        (executable, "-c", (
            "import json,sys;"
            "print(json.dumps(list(sys.version_info[:3])))")),
        capture=True)
    try:
        values = json.loads(completed.stdout)
        return tuple(int(value) for value in values)
    except Exception as error:
        raise InstallationError(
            f"Could not inspect Python interpreter {executable}.") from error


def _resolve_python(value: str | None, backend: str) -> Path:
    candidates = []
    if value:
        candidates.append(value)
    if backend == COSY_BACKEND:
        candidates.extend(("python3.10",))
    else:
        candidates.extend(("python3.11", "python3.10", sys.executable))
    for candidate in candidates:
        resolved = shutil.which(candidate) if not Path(candidate).is_file() \
            else candidate
        if not resolved:
            continue
        executable = Path(resolved).resolve()
        version = _python_version(executable)
        if backend == COSY_BACKEND and version[:2] != (3, 10):
            continue
        if backend == STYLE_BACKEND and not (
                (3, 10) <= version[:2] <= (3, 12)):
            continue
        return executable
    if backend == COSY_BACKEND:
        raise InstallationError(
            "CosyVoice's supported installer path requires Python 3.10. "
            "Install Python 3.10, then pass --cosy-python /path/to/python3.10.")
    raise InstallationError(
        "Style-Bert-VITS2 requires Python 3.10 through 3.12. "
        "Pass --style-python /path/to/a/compatible/python.")


def _runtime_python(runtime: Path) -> Path:
    if os.name == "nt":
        return runtime / "Scripts" / "python.exe"
    return runtime / "bin" / "python"


def _environment(root: Path) -> dict[str, str]:
    value = os.environ.copy()
    value["AUTOANKI_TTS_HOME"] = str(root)
    value.setdefault(
        "HF_HOME",
        str(Path(
            os.environ.get(
                "XDG_CACHE_HOME",
                Path.home() / ".cache")) / "huggingface"))
    value.setdefault("TOKENIZERS_PARALLELISM", "false")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise InstallationError(
            f"Installation path escaped the shared root: {path}") from error


def _write_json_atomic(path: Path, value: Mapping) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _create_runtime(
        root: Path,
        backend: str,
        python: Path,
        *,
        force: bool) -> tuple[Path, Path | None]:
    runtime = root / "runtimes" / backend
    backup = None
    if runtime.exists():
        if not force:
            raise InstallationError(
                f"Runtime already exists: {runtime}. Use --force to install "
                "a replacement while retaining a timestamped backup.")
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ")
        backup = runtime.with_name(f"{runtime.name}.backup-{timestamp}")
        if backup.exists():
            raise InstallationError(f"Backup target already exists: {backup}")
        runtime.replace(backup)
    try:
        runtime.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        builder = venv.EnvBuilder(
            with_pip=True,
            clear=False,
            symlinks=True,
            upgrade_deps=False)
        # EnvBuilder normally uses the running interpreter. Use the selected
        # interpreter to execute stdlib venv when it differs.
        if python.resolve() == Path(sys.executable).resolve():
            builder.create(runtime)
        else:
            _run((python, "-m", "venv", runtime))
    except BaseException:
        if runtime.exists():
            failed = runtime.with_name(
                f"{runtime.name}.failed-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            runtime.replace(failed)
        if backup is not None:
            backup.replace(runtime)
        raise
    return runtime, backup


def _restore_runtime_after_failure(
        runtime: Path,
        backup: Path | None) -> None:
    if runtime.exists():
        failed = runtime.with_name(
            f"{runtime.name}.failed-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
        runtime.replace(failed)
        print(f"Incomplete replacement retained at {failed}", file=sys.stderr)
    if backup is not None and backup.exists():
        backup.replace(runtime)
        print(f"Restored prior runtime from {backup}", file=sys.stderr)


def _pip(
        runtime_python: Path,
        *arguments: str,
        environment: Mapping[str, str]) -> None:
    _run(
        (runtime_python, "-m", "pip", "--disable-pip-version-check",
         *arguments),
        environment=environment)


def _install_cuda_torch(
        runtime_python: Path,
        *,
        version: str,
        index_url: str,
        environment: Mapping[str, str]) -> None:
    _pip(
        runtime_python,
        "install",
        "--index-url",
        index_url,
        f"torch=={version}",
        f"torchaudio=={version}",
        environment=environment)
    _assert_exact_cuda_torch(
        runtime_python,
        expected_version=version,
        environment=environment)


def _assert_exact_cuda_torch(
        runtime_python: Path,
        *,
        expected_version: str,
        environment: Mapping[str, str]) -> None:
    code = (
        "import json,torch;"
        "import torchaudio;"
        f"expected={expected_version!r};"
        "assert torch.__version__.split('+',1)[0]==expected,"
        "f'unexpected torch {torch.__version__}, expected {expected}';"
        "assert torchaudio.__version__.split('+',1)[0]==expected,"
        "f'unexpected torchaudio {torchaudio.__version__}, expected "
        "{expected}';"
        "assert torch.cuda.is_available(),"
        "'installed PyTorch cannot initialize CUDA';"
        "torch.empty(1,device='cuda');"
        "print(json.dumps({"
        "'torch':torch.__version__,"
        "'cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0),"
        "'capability':list(torch.cuda.get_device_capability(0))}))")
    _run(
        (runtime_python, "-c", code),
        environment=environment)


def _download_snapshot(
        runtime_python: Path,
        *,
        repository: str,
        revision: str,
        destination: Path,
        environment: Mapping[str, str],
        allow_patterns: Iterable[str] | None = None) -> None:
    payload = json.dumps({
        "repo_id": repository,
        "revision": revision,
        "local_dir": str(destination),
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
    })
    code = (
        "import json,sys;"
        "from huggingface_hub import snapshot_download;"
        "v=json.loads(sys.argv[1]);"
        "snapshot_download("
        "repo_id=v['repo_id'],revision=v['revision'],"
        "local_dir=v['local_dir'],"
        "allow_patterns=v['allow_patterns'])")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _run(
        (runtime_python, "-c", code, payload),
        environment=environment)


def _resolve_huggingface_revision(repository: str) -> str:
    url = (
        "https://huggingface.co/api/models/"
        f"{quote(repository, safe='/')}/revision/main")
    request = Request(url, headers={"User-Agent": "AutoAnki-TTS-Installer/1"})
    try:
        with urlopen(request, timeout=30) as response:
            value = json.load(response)
    except Exception as error:
        raise InstallationError(
            f"Could not resolve the current revision of {repository}: "
            f"{error}") from error
    revision = value.get("sha")
    if (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40,64}", revision)):
        raise InstallationError(
            f"Hugging Face returned no immutable revision for {repository}.")
    return revision


def _ensure_cosy_source(root: Path, *, force: bool) -> Path:
    destination = root / "sources" / "CosyVoice" / COSY_SOURCE_REVISION
    if destination.is_dir():
        completed = _run(
            ("git", "-C", destination, "rev-parse", "HEAD"),
            capture=True)
        if completed.stdout.strip() == COSY_SOURCE_REVISION:
            return destination
        if not force:
            raise InstallationError(
                f"Existing CosyVoice source has the wrong revision: "
                f"{destination}")
        backup = destination.with_name(
            f"{destination.name}.backup-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        destination.replace(backup)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.cloning")
    if temporary.exists():
        raise InstallationError(
            f"An earlier clone is incomplete: {temporary}")
    try:
        _run((
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/FunAudioLLM/CosyVoice.git", temporary))
        _run(("git", "-C", temporary, "checkout", "--detach",
              COSY_SOURCE_REVISION))
        _run(("git", "-C", temporary, "submodule", "update", "--init",
              "--recursive", "--depth", "1"))
        temporary.replace(destination)
    except BaseException:
        # Retain an incomplete clone for diagnosis; it is not treated as the
        # immutable source directory.
        raise
    return destination


def _requirement_name(line: str) -> str | None:
    value = line.split("#", 1)[0].strip()
    if not value or value.startswith("-"):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)", value)
    return match.group(1).replace("_", "-").casefold() if match else None


def _filtered_cosy_requirements(source: Path, runtime: Path) -> Path:
    original = source / "requirements.txt"
    if not original.is_file():
        raise InstallationError("CosyVoice requirements.txt is missing.")
    # These are either supplied separately (torch), server/training-only, or
    # TensorRT packages whose engines are deliberately disabled by the worker.
    excluded = {
        "deepspeed",
        "fastapi",
        "fastapi-cli",
        "gradio",
        "grpcio",
        "grpcio-tools",
        "onnxruntime-gpu",
        "openai-whisper",
        "tensorboard",
        "tensorrt-cu12",
        "tensorrt-cu12-bindings",
        "tensorrt-cu12-libs",
        "torch",
        "torchaudio",
        "uvicorn",
    }
    retained = []
    for line in original.read_text(encoding="utf-8").splitlines():
        name = _requirement_name(line)
        if name is None or name in excluded:
            continue
        retained.append(line)
    path = runtime / "autoanki_cosyvoice_requirements.txt"
    path.write_text(
        "\n".join(retained) + "\n",
        encoding="utf-8")
    return path


def _copy_worker(runtime: Path, source: Path) -> None:
    for origin, destination_name in (
            (_COMMON_WORKER, "_autoanki_tts_worker_common.py"),
            (source, "autoanki_worker.py")):
        if not origin.is_file():
            raise InstallationError(f"Worker source is missing: {origin}")
        destination = runtime / destination_name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_name}.",
            dir=runtime)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(origin, temporary)
            temporary.chmod(0o700)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _model_revision(
        *,
        model_revision: str,
        source_revision: str,
        environment_revision: str,
        worker_revision: str,
        extra_revision: str | None = None) -> str:
    components = [
        f"model:{model_revision}",
        f"runtime:{source_revision}",
        f"environment:{environment_revision}",
        f"worker:{worker_revision}",
    ]
    if extra_revision:
        components.append(f"frontend:{extra_revision}")
    return ";".join(components)


def _environment_fingerprint(
        runtime_python: Path,
        environment: Mapping[str, str]) -> str:
    completed = _run(
        (runtime_python, "-m", "pip", "freeze", "--all"),
        environment=environment,
        capture=True)
    normalized = "\n".join(sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip())) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _worker_fingerprint(runtime: Path) -> str:
    digest = hashlib.sha256()
    for filename in (
            "_autoanki_tts_worker_common.py",
            "autoanki_worker.py"):
        path = runtime / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _base_manifest(
        *,
        backend: str,
        model_id: str,
        model_revision: str,
        torch_version: str,
        torch_index_url: str) -> dict:
    return {
        "schema_version": 1,
        "state": "testing",
        "backend": backend,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": model_id,
            "revision": model_revision,
        },
        "runtime": {
            "torch_requested": torch_version,
            "torch_index_url": torch_index_url,
            "required_device": "cuda",
            "allow_cpu_fallback": False,
        },
    }


def _install_style(
        root: Path,
        python: Path,
        *,
        force: bool,
        defer_smoke: bool,
        torch_version: str,
        torch_index_url: str) -> Path:
    runtime, backup = _create_runtime(
        root, STYLE_BACKEND, python, force=force)
    environment = _environment(root)
    runtime_python = _runtime_python(runtime)
    try:
        _pip(
            runtime_python, "install", "--upgrade", "pip", "setuptools",
            "wheel", environment=environment)
        _install_cuda_torch(
            runtime_python,
            version=torch_version,
            index_url=torch_index_url,
            environment=environment)
        _pip(
        runtime_python,
            "install",
            f"git+https://github.com/litagin02/Style-Bert-VITS2.git@"
            f"{STYLE_SOURCE_REVISION}",
            "huggingface_hub",
            environment=environment)
        _assert_exact_cuda_torch(
            runtime_python,
            expected_version=torch_version,
            environment=environment)

        voice_root = (
            root / "models" / STYLE_BACKEND / STYLE_VOICE_REVISION)
        _download_snapshot(
            runtime_python,
            repository=STYLE_VOICE_REPOSITORY,
            revision=STYLE_VOICE_REVISION,
            destination=voice_root,
            allow_patterns=STYLE_VOICE_FILES,
            environment=environment)
        bert_revision = _resolve_huggingface_revision(
            STYLE_BERT_REPOSITORY)
        bert_root = (
            root / "models" / "huggingface"
            / "ku-nlp--deberta-v2-large-japanese-char-wwm"
            / bert_revision)
        _download_snapshot(
            runtime_python,
            repository=STYLE_BERT_REPOSITORY,
            revision=bert_revision,
            destination=bert_root,
            environment=environment)
        _copy_worker(runtime, _STYLE_WORKER)
        environment_revision = _environment_fingerprint(
            runtime_python, environment)
        worker_revision = _worker_fingerprint(runtime)
        manifest = _base_manifest(
            backend=STYLE_BACKEND,
            model_id=STYLE_MODEL_ID,
            model_revision=_model_revision(
                model_revision=STYLE_VOICE_REVISION,
                source_revision=STYLE_SOURCE_REVISION,
                environment_revision=environment_revision,
                worker_revision=worker_revision,
                extra_revision=bert_revision),
            torch_version=torch_version,
            torch_index_url=torch_index_url)
        manifest.update({
            "paths": {
                "voice_config": _relative(
                    root, voice_root / STYLE_VOICE_FILES[0]),
                "voice_model": _relative(
                    root, voice_root / STYLE_VOICE_FILES[1]),
                "voice_styles": _relative(
                    root, voice_root / STYLE_VOICE_FILES[2]),
                "bert_model": _relative(root, bert_root),
            },
            "voice": {
                "id": "neutral-japanese",
                "style": "Neutral",
                "repository": STYLE_VOICE_REPOSITORY,
                "repository_revision": STYLE_VOICE_REVISION,
                "license": "CC-BY-SA-4.0",
            },
            "source": {
                "repository": (
                    "https://github.com/litagin02/Style-Bert-VITS2.git"),
                "revision": STYLE_SOURCE_REVISION,
                "license": "AGPL-3.0",
            },
        })
        manifest["runtime"].update({
            "environment_sha256": environment_revision,
            "worker_sha256": worker_revision,
        })
        _write_json_atomic(runtime / "installation.json", manifest)
        if defer_smoke:
            print(
                "Style-Bert-VITS2 prepared in the testing state; run "
                "--verify-existing when sufficient VRAM is available.")
            return runtime
        smoke_directory = _smoke_test(
            root, runtime, STYLE_BACKEND, STYLE_MODEL_ID)
        manifest["state"] = "ready"
        manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
        manifest["smoke_audio_directory"] = _relative(
            root, smoke_directory)
        _write_json_atomic(runtime / "installation.json", manifest)
        return runtime
    except BaseException:
        _restore_runtime_after_failure(runtime, backup)
        raise


def _install_cosy(
        root: Path,
        python: Path,
        *,
        force: bool,
        defer_smoke: bool,
        torch_version: str,
        torch_index_url: str) -> Path:
    source = _ensure_cosy_source(root, force=force)
    runtime, backup = _create_runtime(
        root, COSY_BACKEND, python, force=force)
    environment = _environment(root)
    runtime_python = _runtime_python(runtime)
    try:
        _pip(
            runtime_python, "install", "--upgrade", "pip", "setuptools",
            "wheel", environment=environment)
        _install_cuda_torch(
            runtime_python,
            version=torch_version,
            index_url=torch_index_url,
            environment=environment)
        requirements = _filtered_cosy_requirements(source, runtime)
        _pip(
            runtime_python,
            "install",
            "-r",
            str(requirements),
            "huggingface_hub",
            environment=environment)
        # Upstream's old Whisper pin is incompatible with PyTorch's current
        # Triton runtime.  Build without isolation because Whisper's sdist
        # imports pkg_resources while computing its package metadata.
        _pip(
            runtime_python,
            "install",
            "--no-build-isolation",
            f"openai-whisper=={COSY_WHISPER_VERSION}",
            environment=environment)
        _pip(
            runtime_python,
            "install",
            f"onnxruntime-gpu[cuda,cudnn]=={COSY_ONNXRUNTIME_VERSION}",
            environment=environment)
        _assert_exact_cuda_torch(
            runtime_python,
            expected_version=torch_version,
            environment=environment)
        model_root = (
            root / "models" / COSY_BACKEND / COSY_MODEL_REVISION)
        _download_snapshot(
            runtime_python,
            repository=COSY_MODEL_REPOSITORY,
            revision=COSY_MODEL_REVISION,
            destination=model_root,
            allow_patterns=COSY_MODEL_FILES,
            environment=environment)
        prompt_audio = source / "asset" / "zero_shot_prompt.wav"
        if not prompt_audio.is_file():
            raise InstallationError(
                "The pinned CosyVoice neutral reference asset is missing.")
        _copy_worker(runtime, _COSY_WORKER)
        environment_revision = _environment_fingerprint(
            runtime_python, environment)
        worker_revision = _worker_fingerprint(runtime)
        manifest = _base_manifest(
            backend=COSY_BACKEND,
            model_id=COSY_MODEL_ID,
            model_revision=_model_revision(
                model_revision=COSY_MODEL_REVISION,
                source_revision=COSY_SOURCE_REVISION,
                environment_revision=environment_revision,
                worker_revision=worker_revision),
            torch_version=torch_version,
            torch_index_url=torch_index_url)
        manifest.update({
            "paths": {
                "model": _relative(root, model_root),
                "source": _relative(root, source),
                "prompt_audio": _relative(root, prompt_audio),
            },
            "voice": {
                "ids": [
                    "neutral-english",
                    "neutral-french",
                    "neutral-mandarin",
                ],
                "prompt_text": COSY_PROMPT_TEXT,
                "source": "official CosyVoice zero-shot example asset",
            },
            "source": {
                "repository": (
                    "https://github.com/FunAudioLLM/CosyVoice.git"),
                "revision": COSY_SOURCE_REVISION,
                "license": "Apache-2.0",
            },
            "model_source": {
                "repository": COSY_MODEL_REPOSITORY,
                "revision": COSY_MODEL_REVISION,
                "license": "Apache-2.0",
            },
        })
        manifest["runtime"].update({
            "environment_sha256": environment_revision,
            "worker_sha256": worker_revision,
            "compatibility_overrides": {
                "openai_whisper": COSY_WHISPER_VERSION,
                "onnxruntime_gpu": COSY_ONNXRUNTIME_VERSION,
            },
        })
        _write_json_atomic(runtime / "installation.json", manifest)
        if defer_smoke:
            print(
                "CosyVoice prepared in the testing state; run "
                "--verify-existing when sufficient VRAM is available.")
            return runtime
        smoke_directory = _smoke_test(
            root, runtime, COSY_BACKEND, COSY_MODEL_ID)
        manifest["state"] = "ready"
        manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
        manifest["smoke_audio_directory"] = _relative(
            root, smoke_directory)
        _write_json_atomic(runtime / "installation.json", manifest)
        return runtime
    except BaseException:
        _restore_runtime_after_failure(runtime, backup)
        raise


def _worker_request(
        root: Path,
        runtime: Path,
        request: Mapping) -> dict:
    environment = _environment(root)
    environment["AUTOANKI_TTS_INSTALL_SMOKE"] = "1"
    completed = subprocess.run(
        (_runtime_python(runtime), runtime / "autoanki_worker.py"),
        input=json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":")) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
        timeout=1800)
    if completed.returncode:
        raise InstallationError(
            f"TTS worker smoke test exited with {completed.returncode}: "
            f"{completed.stderr[-2000:]}")
    lines = [
        line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise InstallationError(
            "TTS worker smoke test did not return one JSON response.")
    try:
        response = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise InstallationError(
            "TTS worker smoke test returned invalid JSON.") from error
    if response.get("ok") is not True:
        raise InstallationError(
            "TTS worker smoke test failed: "
            f"{response.get('error', {}).get('message', 'unknown error')}")
    return response


def _smoke_test(
        root: Path,
        runtime: Path,
        backend: str,
        model_id: str) -> Path:
    identity = {
        "protocol": "autoanki-local-tts",
        "version": 1,
        "backend": backend,
    }
    status = _worker_request(
        root,
        runtime,
        {
            **identity,
            "operation": "status",
            "model": {"id": model_id},
            "execution": {
                "required_device": "cuda",
                "allow_cpu_fallback": False,
            },
        })
    if (
            status.get("runtime", {}).get("available") is not True
            or status.get("gpu", {}).get("available") is not True):
        raise InstallationError(
            "The installed worker did not report a ready CUDA runtime.")
    revision = status["model"]["revision"]
    voices = status["voices"]
    if backend == STYLE_BACKEND:
        samples = (("japanese", "こんにちは", "neutral-japanese"),)
    else:
        samples = (
            ("english", "Hello.", "neutral-english"),
            ("french", "Bonjour.", "neutral-french"),
            ("chinese", "你好。", "neutral-mandarin"),
        )
    staging = root / "staging"
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_paths = [
        staging / f".installer-{backend}-{index}.wav"
        for index in range(len(samples))
    ]
    for path in output_paths:
        if path.exists():
            raise InstallationError(
                f"Smoke-test staging path already exists: {path}")
    items = []
    for (language, text, voice_id), output_path in zip(
            samples, output_paths):
        items.append({
            "cache_key": (
                f"{len(items) + 1:064x}"
            ),
            "model": {
                "id": model_id,
                "revision": revision,
            },
            "input": {
                "text": text,
                "language": language,
                "role": "word",
            },
            "voice": {
                "id": voice_id,
                "revision_sha256": voices[voice_id]["revision_sha256"],
                "reference_path": None,
                "reference_sha256": None,
            },
            "settings": {
                "emotion": "neutral",
                "speed": 1.0,
                "pitch": 0.0,
                "energy": 1.0,
                "seed": 0,
            },
            "output": {
                "codec": "wav",
                "encoder": "pcm_s16le",
                "sample_rate_hz": 24000,
                "channels": 1,
                "path": str(output_path.resolve()),
            },
            "execution": {
                "required_device": "cuda",
                "allow_cpu_fallback": False,
            },
        })
    try:
        response = _worker_request(
            root,
            runtime,
            {
                **identity,
                "operation": "synthesize_many",
                "items": items,
            })
        artifacts = response.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(items):
            raise InstallationError(
                "TTS batch smoke test returned the wrong artifact count.")
        for output_path in output_paths:
            if not output_path.is_file() or output_path.stat().st_size <= 44:
                raise InstallationError(
                    f"TTS smoke output is missing or empty: {output_path}")
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ")
        diagnostics = (
            root / "diagnostics" / "installation-smoke"
            / f"{backend}-{timestamp}")
        diagnostics.mkdir(mode=0o700, parents=True, exist_ok=False)
        for (language, _text, _voice_id), output_path in zip(
                samples, output_paths):
            shutil.copy2(output_path, diagnostics / f"{language}.wav")
        print(
            f"Retained smoke-test audio for manual listening: {diagnostics}")
        return diagnostics
    finally:
        for output_path in output_paths:
            output_path.unlink(missing_ok=True)


def _write_notices(root: Path) -> None:
    path = root / "THIRD_PARTY_NOTICES.md"
    content = """# AutoAnki local TTS third-party notices

- Style-Bert-VITS2 source: AGPL-3.0,
  https://github.com/litagin02/Style-Bert-VITS2
- `litagin/style_bert_vits2_jvnv` voice assets: CC-BY-SA-4.0,
  https://huggingface.co/litagin/style_bert_vits2_jvnv
- FunAudioLLM CosyVoice source and Fun-CosyVoice3 model: Apache-2.0,
  https://github.com/FunAudioLLM/CosyVoice and
  https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

Review the upstream terms before distributing a runtime or generated audio.
"""
    path.write_text(content, encoding="utf-8")


def _verify_existing(root: Path, backend: str) -> Path:
    runtime = root / "runtimes" / backend
    manifest_path = runtime / "installation.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InstallationError(
            f"Cannot verify incomplete runtime {runtime}: {error}") from error
    if (
            manifest.get("schema_version") != 1
            or manifest.get("backend") != backend
            or manifest.get("state") not in {"testing", "ready"}):
        raise InstallationError(
            f"Runtime has no compatible testing/ready manifest: {runtime}")
    expected_model_id = (
        STYLE_MODEL_ID if backend == STYLE_BACKEND else COSY_MODEL_ID)
    if manifest.get("model", {}).get("id") != expected_model_id:
        raise InstallationError(
            f"Runtime manifest names the wrong model: {runtime}")
    smoke_directory = _smoke_test(
        root, runtime, backend, expected_model_id)
    manifest["state"] = "ready"
    manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
    manifest["smoke_audio_directory"] = _relative(
        root, smoke_directory)
    _write_json_atomic(manifest_path, manifest)
    return runtime


def _refresh_existing_worker(root: Path, backend: str) -> Path:
    """Atomically refresh worker code without reinstalling model assets."""
    runtime = root / "runtimes" / backend
    manifest_path = runtime / "installation.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InstallationError(
            f"Cannot refresh incomplete runtime {runtime}: {error}") from error
    if (
            manifest.get("schema_version") != 1
            or manifest.get("backend") != backend
            or manifest.get("state") not in {"testing", "ready"}):
        raise InstallationError(
            f"Runtime has no compatible testing/ready manifest: {runtime}")
    model = manifest.get("model")
    runtime_value = manifest.get("runtime")
    if not isinstance(model, dict) or not isinstance(runtime_value, dict):
        raise InstallationError(
            f"Runtime manifest is missing model/runtime metadata: {runtime}")

    torch_version = runtime_value.get("torch_requested")
    if not isinstance(torch_version, str) or not torch_version:
        raise InstallationError(
            f"Runtime manifest has no requested torch version: {runtime}")
    environment = _environment(root)
    _assert_exact_cuda_torch(
        _runtime_python(runtime),
        expected_version=torch_version,
        environment=environment)
    environment_revision = _environment_fingerprint(
        _runtime_python(runtime), environment)
    worker_source = (
        _STYLE_WORKER if backend == STYLE_BACKEND else _COSY_WORKER)
    _copy_worker(runtime, worker_source)
    worker_revision = _worker_fingerprint(runtime)
    revision = model.get("revision")
    if not isinstance(revision, str):
        raise InstallationError(
            f"Runtime manifest has no model revision: {runtime}")
    components = revision.split(";")
    for name, value in (
            ("environment", environment_revision),
            ("worker", worker_revision)):
        matches = [
            index
            for index, component in enumerate(components)
            if component.startswith(f"{name}:")
        ]
        if len(matches) != 1:
            raise InstallationError(
                f"Runtime model revision has no unique {name} component: "
                f"{runtime}")
        components[matches[0]] = f"{name}:{value}"
    model["revision"] = ";".join(components)
    runtime_value["environment_sha256"] = environment_revision
    runtime_value["worker_sha256"] = worker_revision
    manifest["state"] = "testing"
    manifest["worker_refreshed_at"] = (
        datetime.now(timezone.utc).isoformat())
    manifest.pop("verified_at", None)
    manifest.pop("smoke_audio_directory", None)
    _write_json_atomic(manifest_path, manifest)
    return runtime


def _plan(args, root: Path, backends: list[str]) -> dict:
    values = {
        "shared_root": str(root),
        "backends": backends,
        "main_virtualenv_unchanged": True,
        "cuda_required": True,
        "cpu_fallback": False,
        "torch": {
            "version": args.torch_version,
            "index_url": args.torch_index_url,
        },
        "pinned_revisions": {
            "style_source": STYLE_SOURCE_REVISION,
            "style_voice": STYLE_VOICE_REVISION,
            "cosy_source": COSY_SOURCE_REVISION,
            "cosy_model": COSY_MODEL_REVISION,
            "cosy_openai_whisper": COSY_WHISPER_VERSION,
            "cosy_onnxruntime_gpu": COSY_ONNXRUNTIME_VERSION,
        },
    }
    interpreters = {}
    for backend in backends:
        supplied = (
            args.style_python if backend == STYLE_BACKEND
            else args.cosy_python)
        try:
            interpreters[backend] = str(_resolve_python(
                supplied or args.python, backend))
        except InstallationError as error:
            interpreters[backend] = {"unavailable": str(error)}
    values["interpreters"] = interpreters
    return values


def _parse_arguments(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Install isolated, shared, CUDA-only local TTS runtimes."))
    parser.add_argument(
        "--backend",
        action="append",
        choices=("style", "cosy", "all"),
        required=True,
        help="Backend to install; repeat the option or use all.")
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Shared TTS root (default: %(default)s).")
    parser.add_argument(
        "--python",
        help="Fallback Python interpreter for either backend.")
    parser.add_argument(
        "--style-python",
        help="Python 3.10-3.12 interpreter for Style-Bert-VITS2.")
    parser.add_argument(
        "--cosy-python",
        help="Python 3.10 interpreter for CosyVoice.")
    parser.add_argument(
        "--torch-version",
        default=DEFAULT_TORCH_VERSION,
        help="Pinned CUDA torch and torchaudio version.")
    parser.add_argument(
        "--torch-index-url",
        default=DEFAULT_TORCH_INDEX_URL,
        help="Official CUDA wheel index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Install a replacement and retain the old runtime as a backup.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a plan without creating directories or downloading files.")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--defer-smoke",
        action="store_true",
        help=(
            "Prepare immutable runtimes in the testing state without loading "
            "the TTS models; later use --verify-existing."))
    phase.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Run real CUDA smoke tests for existing testing/ready runtimes "
            "without reinstalling them."))
    phase.add_argument(
        "--refresh-workers",
        action="store_true",
        help=(
            "Refresh only installed worker code/fingerprints, return the "
            "runtimes to testing, audit the environment fingerprint, and "
            "leave models/environments unchanged."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_arguments(argv)
    selected = set(args.backend)
    backends = []
    if "all" in selected or "style" in selected:
        backends.append(STYLE_BACKEND)
    if "all" in selected or "cosy" in selected:
        backends.append(COSY_BACKEND)
    root = args.root.expanduser().resolve()
    if root == Path("/") or root == Path.home().resolve():
        raise InstallationError(
            "Refusing to use a filesystem root or home directory itself as "
            "the TTS root.")
    if args.verify_existing:
        if args.dry_run:
            raise InstallationError(
                "--verify-existing and --dry-run cannot be combined.")
        verified = [
            str(_verify_existing(root, backend))
            for backend in backends
        ]
        print(json.dumps({
            "verified": verified,
            "shared_root": str(root),
            "cuda_required": True,
            "cpu_fallback": False,
        }, indent=2))
        return 0
    if args.refresh_workers:
        if args.dry_run:
            raise InstallationError(
                "--refresh-workers and --dry-run cannot be combined.")
        refreshed = [
            str(_refresh_existing_worker(root, backend))
            for backend in backends
        ]
        print(json.dumps({
            "refreshed": refreshed,
            "shared_root": str(root),
            "state": "testing",
            "models_unchanged": True,
            "environment_fingerprint_reconciled": True,
        }, indent=2))
        return 0
    plan = _plan(args, root, backends)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    for backend, interpreter in plan["interpreters"].items():
        if isinstance(interpreter, dict):
            raise InstallationError(interpreter["unavailable"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_notices(root)
    installed = []
    for backend in backends:
        python = Path(plan["interpreters"][backend])
        if backend == STYLE_BACKEND:
            runtime = _install_style(
                root,
                python,
                force=args.force,
                defer_smoke=args.defer_smoke,
                torch_version=args.torch_version,
                torch_index_url=args.torch_index_url)
        else:
            runtime = _install_cosy(
                root,
                python,
                force=args.force,
                defer_smoke=args.defer_smoke,
                torch_version=args.torch_version,
                torch_index_url=args.torch_index_url)
        installed.append(str(runtime))
    print(json.dumps({
        "installed": installed,
        "shared_root": str(root),
        "main_virtualenv_unchanged": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallationError as error:
        print(f"Installation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
