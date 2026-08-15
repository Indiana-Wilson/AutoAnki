"""Cooperative host-wide lease for large local CUDA model workloads.

The default deliberately matches AutoAnki's established isolated-worker
lock.  Other local applications can join the same queue without requiring an
AutoAnki runtime migration.  ``LOCAL_LLM_GPU_LOCK_PATH`` may move the shared
lock, but every participating process must receive the same absolute path.
The lease only serializes cooperating GPU workloads; it does not impose a
swap requirement or a fixed host-memory admission threshold.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat


LOCAL_LLM_GPU_LOCK_PATH_ENVIRONMENT_VARIABLE = (
    "LOCAL_LLM_GPU_LOCK_PATH")
AUTOANKI_TTS_HOME_ENVIRONMENT_VARIABLE = "AUTOANKI_TTS_HOME"


class LocalGPULeaseError(RuntimeError):
    """The cooperative local-GPU lease could not be used safely."""


@dataclass(frozen=True)
class LocalGPULeaseHandle:
    """An acquired lease and the descriptor that keeps it alive."""

    path: Path
    file_descriptor: int


def _default_tts_root() -> Path:
    override = os.environ.get(AUTOANKI_TTS_HOME_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home).expanduser() / "autoanki" / "tts"
    return Path.home() / ".local" / "share" / "autoanki" / "tts"


def local_llm_gpu_lock_path(*, tts_root: Path | None = None) -> Path:
    """Return the one lock path shared by cooperating local model jobs."""
    override = os.environ.get(
        LOCAL_LLM_GPU_LOCK_PATH_ENVIRONMENT_VARIABLE)
    if override is not None:
        if not override.strip():
            raise LocalGPULeaseError(
                f"{LOCAL_LLM_GPU_LOCK_PATH_ENVIRONMENT_VARIABLE} cannot "
                "be empty.")
        path = Path(override)
        if not path.is_absolute():
            raise LocalGPULeaseError(
                f"{LOCAL_LLM_GPU_LOCK_PATH_ENVIRONMENT_VARIABLE} must be "
                "an absolute path so every process opens the same lock.")
        return path
    root = Path(tts_root) if tts_root is not None else _default_tts_root()
    path = root.expanduser() / "locks" / "worker-gpu.lock"
    if not path.is_absolute():
        raise LocalGPULeaseError(
            "The local GPU lock path must be absolute.")
    return path


def _reject_symlink_ancestors(path: Path):
    for candidate in (path, *path.parents):
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise LocalGPULeaseError(
                "The local GPU lock path contains a symlink.")


def _lock_stream(path: Path):
    _reject_symlink_ancestors(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_ancestors(path.parent)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1):
            raise LocalGPULeaseError(
                "The local GPU lock is not owner-controlled.")
        return os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def local_llm_gpu_lease_handle(
        purpose: str,
        *,
        lock_path: Path | None = None):
    """Block for exclusive CUDA ownership and yield its live lock handle.

    Advisory OS locks are released automatically when the final descriptor
    for the locked open-file description closes. The small JSON payload is
    diagnostic only and never determines lock ownership.
    Callers must keep the context open through model unload and CUDA cache
    cleanup, not merely through model inference. On POSIX, a caller may pass
    ``file_descriptor`` to a child with ``pass_fds``; the inherited open-file
    description then keeps the lease held if the parent dies first.
    """
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("A local GPU lease requires a non-empty purpose.")
    if lock_path is None:
        path = local_llm_gpu_lock_path()
    else:
        path = Path(lock_path)
        if not path.is_absolute():
            raise LocalGPULeaseError(
                "The local GPU lock path must be absolute.")
    _reject_symlink_ancestors(path)

    with _lock_stream(path) as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
        acquired = False
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            elif os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    stream.fileno(),
                    msvcrt.LK_LOCK,  # type: ignore[attr-defined]
                    1)
            else:
                raise LocalGPULeaseError(
                    "This platform has no configured cross-process GPU "
                    "lease; unsafe concurrent CUDA work was refused.")
            acquired = True
            _reject_symlink_ancestors(path)
            descriptor_metadata = os.fstat(stream.fileno())
            pathname_metadata = os.lstat(path)
            if (
                    (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
                    != (pathname_metadata.st_dev, pathname_metadata.st_ino)
                    or not stat.S_ISREG(descriptor_metadata.st_mode)
                    or stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
                    or descriptor_metadata.st_uid != os.getuid()
                    or descriptor_metadata.st_nlink != 1):
                raise LocalGPULeaseError(
                    "The local GPU lock changed during acquisition.")
            metadata = json.dumps({
                "owner": "autoanki",
                "purpose": purpose.strip(),
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }, separators=(",", ":")).encode("utf-8")
            stream.seek(0)
            stream.truncate()
            stream.write(metadata)
            os.fsync(stream.fileno())
            yield LocalGPULeaseHandle(
                path=path,
                file_descriptor=stream.fileno())
        finally:
            if acquired and os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    stream.fileno(),
                    msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                    1)
            # On POSIX, close this process's descriptor instead of issuing
            # LOCK_UN against the shared open-file description. An inherited
            # child duplicate must continue holding the lease until it exits.


@contextmanager
def local_llm_gpu_lease(
        purpose: str,
        *,
        lock_path: Path | None = None):
    """Block until exclusive CUDA ownership is available, then yield its path."""
    with local_llm_gpu_lease_handle(
            purpose,
            lock_path=lock_path) as lease:
        yield lease.path


__all__ = [
    "AUTOANKI_TTS_HOME_ENVIRONMENT_VARIABLE",
    "LOCAL_LLM_GPU_LOCK_PATH_ENVIRONMENT_VARIABLE",
    "LocalGPULeaseError",
    "LocalGPULeaseHandle",
    "local_llm_gpu_lease",
    "local_llm_gpu_lease_handle",
    "local_llm_gpu_lock_path",
]
