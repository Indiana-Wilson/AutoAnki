"""Atomic per-chunk persistence and bounded concurrent request orchestration."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import re
import random
import tempfile
import threading
import time
import uuid

import runtime_paths
from source_generation.models import (
    GenerationChunk,
    GenerationPlan,
    SOURCE_GENERATION_SCHEMA_VERSION,
)
from source_generation.requests import normalise_source_request_contract


CHUNK_PENDING = "pending"
CHUNK_RUNNING = "running"
CHUNK_SUCCEEDED = "succeeded"
CHUNK_INVALID = "invalid_response"
CHUNK_CONNECTION_FAILED = "connection_failed"
CHUNK_FAILED = "failed"
CHUNK_CANCELLED = "cancelled"

RETRYABLE_MANUALLY = frozenset({
    CHUNK_INVALID,
    CHUNK_CONNECTION_FAILED,
    CHUNK_FAILED,
    CHUNK_CANCELLED,
})
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REQUEST_CONTRACT_FILE_NAME = "request_contract.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value, *, indent=None):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
    ) + "\n"


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent)
    try:
        with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path, value):
    _atomic_write_text(path, _canonical_json(value, indent=2))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid source-generation file: {path}") from error


@contextmanager
def _advisory_file_lock(path, *, blocking):
    """Hold one small cross-process lock file for the context's lifetime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            mode = (
                msvcrt.LK_LOCK
                if blocking
                else msvcrt.LK_NBLCK)
            try:
                msvcrt.locking(handle.fileno(), mode, 1)
                acquired = True
            except OSError as error:
                if (
                        blocking
                        or error.errno not in {
                            errno.EACCES,
                            errno.EAGAIN,
                            getattr(
                                errno,
                                "EDEADLK",
                                errno.EAGAIN),
                        }):
                    raise
        else:
            import fcntl

            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), flags)
                acquired = True
            except BlockingIOError:
                if blocking:
                    raise
        yield acquired
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


def _response_status_code(error):
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_request_error(error):
    """Classify only connection, timeout, 429, and server-side failures."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    status = _response_status_code(error)
    if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
        return True
    # Avoid importing an OpenAI SDK exception hierarchy here. These are its
    # connection-level names, and test doubles can use the same protocol.
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    }


def _is_invalid_response_error(error):
    """Recognize response/refusal exceptions without importing process_text."""
    return type(error).__name__ in {
        "GeneratedCardValidationError",
        "OpenAIResponseError",
        "OpenAIRefusalError",
        "OpenAIIncompleteResponseError",
    }


def _retry_after_seconds(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if headers is None:
        return None
    try:
        retry_after = headers.get("retry-after")
        if retry_after is None:
            retry_after = headers.get("Retry-After")
        retry_after_ms = headers.get("retry-after-ms")
        reset_requests = headers.get("x-ratelimit-reset-requests")
        reset_tokens = headers.get("x-ratelimit-reset-tokens")
    except AttributeError:
        return None
    candidates = []
    try:
        if retry_after is not None:
            candidates.append(float(retry_after))
    except (TypeError, ValueError):
        pass
    try:
        if retry_after_ms is not None:
            candidates.append(float(retry_after_ms) / 1000)
    except (TypeError, ValueError):
        pass
    for value in (reset_requests, reset_tokens):
        parsed = _parse_reset_duration(value)
        if parsed is not None:
            candidates.append(parsed)
    return max(0.0, max(candidates)) if candidates else None


_RESET_DURATION_PART = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")


def _parse_reset_duration(value):
    """Parse OpenAI-style durations such as ``1m2.5s`` or ``250ms``."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    position = 0
    seconds = 0.0
    multipliers = {
        "ms": 0.001,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
    }
    for match in _RESET_DURATION_PART.finditer(text):
        if match.start() != position:
            return None
        seconds += (
            float(match.group("value"))
            * multipliers[match.group("unit")])
        position = match.end()
    return seconds if position == len(text) else None


@dataclass(frozen=True)
class RetryPolicy:
    """Automatic retries apply only to transient connection/API failures."""

    max_retries: int = 3
    initial_backoff_seconds: float = 0.5
    maximum_backoff_seconds: float = 8.0
    jitter_fraction: float = 0.10

    def __post_init__(self):
        if (
                isinstance(self.max_retries, bool)
                or not isinstance(self.max_retries, int)
                or self.max_retries < 0):
            raise ValueError("Maximum transient retries cannot be negative.")
        if self.initial_backoff_seconds < 0:
            raise ValueError("Initial retry backoff cannot be negative.")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "Maximum retry backoff cannot be below initial backoff.")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("Retry jitter fraction must be between zero and one.")

    def delay(self, retry_number, error, *, random_unit=0.0):
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            # A server-provided Retry-After is authoritative. Capping it below
            # the requested wait would immediately violate the rate limit.
            base = retry_after
        else:
            base = min(
                self.initial_backoff_seconds * (2 ** (retry_number - 1)),
                self.maximum_backoff_seconds,
            )
        # Positive-only jitter never violates Retry-After and reduces workers
        # reconnecting in lockstep after a shared outage.
        return base * (
            1 + self.jitter_fraction * max(0.0, min(1.0, random_unit)))


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    chunk_id: str | None
    status: str
    message: str
    attempt: int | None
    timestamp: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    path: Path
    overall_status: str
    source_key: str
    source_title: str
    plan_id: str
    created_at: str
    updated_at: str
    chunks: tuple[dict, ...]

    @property
    def counts(self):
        result = {}
        for chunk in self.chunks:
            status = chunk["status"]
            result[status] = result.get(status, 0) + 1
        return result

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "path": str(self.path),
            "overall_status": self.overall_status,
            "source_key": self.source_key,
            "source_title": self.source_title,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "counts": self.counts,
            "chunks": list(self.chunks),
        }


class GenerationJobStore:
    """Own inspectable job directories; all mutable writes are atomic."""

    def __init__(self, root=None):
        self.root = Path(
            root
            or (
                runtime_paths.get_output_directory()
                / "source_generation_jobs"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _job_path(self, job_id):
        if (
                not isinstance(job_id, str)
                or _JOB_ID_PATTERN.fullmatch(job_id) is None):
            raise ValueError("Invalid source-generation job ID.")
        root = self.root.resolve()
        candidate = (root / job_id).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "Source-generation job path escapes its root.") from error
        return candidate

    def _chunk_path(self, job_id, chunk_id):
        if (
                not isinstance(chunk_id, str)
                or _JOB_ID_PATTERN.fullmatch(chunk_id) is None):
            raise ValueError("Invalid source-generation chunk ID.")
        return self._job_path(job_id) / "chunks" / chunk_id

    def create(
            self,
            plan,
            *,
            request_metadata=None,
            request_contract=None,
            job_id=None):
        if not isinstance(plan, GenerationPlan):
            raise TypeError("A source generation plan is required.")
        if request_metadata is None:
            request_metadata = {}
        if not isinstance(request_metadata, dict):
            raise TypeError("Source request metadata must be an object.")
        request_metadata = dict(request_metadata)
        embedded_contract = request_metadata.get("request_contract")
        if request_contract is None:
            request_contract = embedded_contract
        elif (
                embedded_contract is not None
                and normalise_source_request_contract(embedded_contract)
                != normalise_source_request_contract(request_contract)):
            raise ValueError(
                "Source request contract arguments disagree.")
        if request_contract is not None:
            request_contract = normalise_source_request_contract(
                request_contract)
            request_metadata["request_contract"] = request_contract
        job_id = job_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:12])
        target = self._job_path(job_id)
        if target.exists():
            raise FileExistsError(
                f"Source-generation job already exists: {job_id}")
        target.mkdir(parents=True)
        created_at = _utc_now()
        for chunk in plan.chunks:
            chunk_path = self._chunk_path(job_id, chunk.chunk_id)
            chunk_path.mkdir(parents=True)
            _atomic_write_json(
                chunk_path / "request.json",
                chunk.to_dict())
            _atomic_write_json(
                chunk_path / "status.json",
                {
                    "chunk_id": chunk.chunk_id,
                    "index": chunk.index,
                    "status": CHUNK_PENDING,
                    "attempts": 0,
                    "worker": "queued",
                    "last_error": None,
                    "latest_attempt_path": None,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "started_at": None,
                    "completed_at": None,
                })
        if request_contract is not None:
            _atomic_write_json(
                target / REQUEST_CONTRACT_FILE_NAME,
                request_contract)
        manifest = {
            "schema_version": SOURCE_GENERATION_SCHEMA_VERSION,
            "kind": "source_generation_job",
            "job_id": job_id,
            "overall_status": (
                "ready"
                if plan.chunks
                else "nothing_to_generate"),
            "created_at": created_at,
            "updated_at": created_at,
            "plan": plan.to_dict(include_chunks=False),
            "request_metadata": request_metadata,
        }
        _atomic_write_json(target / "manifest.json", manifest)
        self._write_combined(job_id)
        return self.snapshot(job_id)

    def _manifest(self, job_id):
        manifest = _read_json(self._job_path(job_id) / "manifest.json")
        if (
                manifest.get("schema_version")
                != SOURCE_GENERATION_SCHEMA_VERSION
                or manifest.get("kind") != "source_generation_job"
                or manifest.get("job_id") != job_id):
            raise ValueError("Unsupported or inconsistent generation job.")
        return manifest

    def load_request_contract(self, job_id):
        """Return a job's immutable paid-request settings, if retained."""
        path = self._job_path(job_id) / REQUEST_CONTRACT_FILE_NAME
        if not path.is_file():
            return None
        return normalise_source_request_contract(_read_json(path))

    def ensure_request_contract(self, job_id, request_contract):
        """Persist a legacy job's contract once; never replace it afterward."""
        request_contract = normalise_source_request_contract(
            request_contract)
        path = self._job_path(job_id) / REQUEST_CONTRACT_FILE_NAME
        with _advisory_file_lock(
                self._job_path(job_id) / ".request_contract.lock",
                blocking=True) as acquired:
            if not acquired:
                raise RuntimeError(
                    "Could not lock the source request contract.")
            existing = self.load_request_contract(job_id)
            if existing is not None:
                return existing
            _atomic_write_json(path, request_contract)
            return request_contract

    def chunk_lease(self, job_id, chunk_id, *, blocking=False):
        """Return a process-wide lease for exactly one paid request worker."""
        return _advisory_file_lock(
            self._chunk_path(job_id, chunk_id) / ".worker.lock",
            blocking=blocking)

    def finalization_lease(self, job_id, *, blocking=False):
        """Return the single process-wide package/import lease for a job."""
        return _advisory_file_lock(
            self._job_path(job_id) / ".finalization.lock",
            blocking=blocking)

    def chunk_ids(self, job_id):
        chunks_path = self._job_path(job_id) / "chunks"
        if not chunks_path.is_dir():
            return ()
        statuses = []
        for path in chunks_path.iterdir():
            if not path.is_dir():
                continue
            status = _read_json(path / "status.json")
            statuses.append((status["index"], status["chunk_id"]))
        return tuple(
            chunk_id
            for _index, chunk_id in sorted(statuses))

    def load_chunk(self, job_id, chunk_id):
        return GenerationChunk.from_dict(_read_json(
            self._chunk_path(job_id, chunk_id) / "request.json"))

    def chunk_status(self, job_id, chunk_id):
        return _read_json(
            self._chunk_path(job_id, chunk_id) / "status.json")

    def latest_attempt_path(self, job_id, chunk_id):
        status = self.chunk_status(job_id, chunk_id)
        relative = status.get("latest_attempt_path")
        if not isinstance(relative, str) or not relative:
            return None
        chunk_path = self._chunk_path(job_id, chunk_id)
        candidate = (chunk_path / relative).resolve()
        try:
            candidate.relative_to(chunk_path.resolve())
        except ValueError as error:
            raise ValueError(
                "Saved source attempt path escapes its chunk.") from error
        return candidate

    def _set_chunk_status(self, job_id, chunk_id, **updates):
        with self._lock:
            path = self._chunk_path(job_id, chunk_id) / "status.json"
            status = _read_json(path)
            status.update(updates)
            status["updated_at"] = _utc_now()
            _atomic_write_json(path, status)
            return status

    def begin_attempt(self, job_id, chunk_id):
        with self._lock:
            status = self.chunk_status(job_id, chunk_id)
            attempt = int(status["attempts"]) + 1
            relative = f"attempts/{attempt:04d}"
            attempt_path = self._chunk_path(
                job_id,
                chunk_id) / relative
            attempt_path.mkdir(parents=True, exist_ok=False)
            now = _utc_now()
            self._set_chunk_status(
                job_id,
                chunk_id,
                status=CHUNK_RUNNING,
                attempts=attempt,
                latest_attempt_path=relative,
                last_error=None,
                started_at=status.get("started_at") or now,
                completed_at=None,
            )
            return attempt, attempt_path

    def write_raw(self, attempt_path, raw_text):
        _atomic_write_text(Path(attempt_path) / "raw.txt", raw_text)

    def write_validated(self, attempt_path, value):
        _atomic_write_json(Path(attempt_path) / "validated.json", value)

    def write_error(self, attempt_path, error, *, transient):
        status_code = _response_status_code(error)
        value = {
            "type": type(error).__name__,
            "message": str(error),
            "transient": bool(transient),
            "status_code": status_code,
            "timestamp": _utc_now(),
        }
        _atomic_write_json(Path(attempt_path) / "error.json", value)
        return value

    def write_recovery_error(self, attempt_path, error):
        """Retain local recovery failure without replacing prior evidence."""
        value = {
            "type": type(error).__name__,
            "message": str(error),
            "transient": False,
            "status_code": _response_status_code(error),
            "timestamp": _utc_now(),
        }
        _atomic_write_json(
            Path(attempt_path) / "recovery_error.json",
            value)
        return value

    def _append_event(self, event):
        with self._lock:
            path = self._job_path(event.job_id) / "events.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as output:
                output.write(_canonical_json(event.to_dict()))
                output.flush()

    def event(
            self,
            job_id,
            *,
            chunk_id,
            status,
            message,
            attempt=None):
        event = JobEvent(
            job_id=job_id,
            chunk_id=chunk_id,
            status=status,
            message=message,
            attempt=attempt,
            timestamp=_utc_now(),
        )
        self._append_event(event)
        return event

    def _overall_status(self, statuses):
        values = {status["status"] for status in statuses}
        if not values:
            return "nothing_to_generate"
        if values == {CHUNK_SUCCEEDED}:
            return "completed"
        if CHUNK_RUNNING in values:
            return "running"
        if CHUNK_PENDING in values:
            return "ready"
        if CHUNK_CANCELLED in values and values <= {
                CHUNK_CANCELLED,
                CHUNK_SUCCEEDED}:
            return "cancelled"
        return "completed_with_failures"

    def _refresh_manifest(self, job_id):
        with self._lock:
            manifest = self._manifest(job_id)
            statuses = [
                self.chunk_status(job_id, chunk_id)
                for chunk_id in self.chunk_ids(job_id)
            ]
            manifest["overall_status"] = self._overall_status(statuses)
            manifest["updated_at"] = _utc_now()
            _atomic_write_json(
                self._job_path(job_id) / "manifest.json",
                manifest)
            return manifest

    def _write_combined(self, job_id):
        chunks = []
        cards = []
        for chunk_id in self.chunk_ids(job_id):
            status = self.chunk_status(job_id, chunk_id)
            if status["status"] != CHUNK_SUCCEEDED:
                continue
            attempt_relative = status.get("latest_attempt_path")
            validated_path = (
                self._chunk_path(job_id, chunk_id)
                / attempt_relative
                / "validated.json")
            validated = _read_json(validated_path)
            chunks.append({
                "chunk_id": chunk_id,
                "output": validated,
            })
            if (
                    isinstance(validated, dict)
                    and isinstance(validated.get("cards"), list)):
                cards.extend(validated["cards"])
        manifest = self._manifest(job_id)
        value = {
            "schema_version": SOURCE_GENERATION_SCHEMA_VERSION,
            "job_id": job_id,
            "source_key": manifest["plan"]["source_key"],
            "plan_id": manifest["plan"]["plan_id"],
            "complete": (
                len(chunks) == manifest["plan"]["chunk_count"]),
            "completed_chunk_count": len(chunks),
            "total_chunk_count": manifest["plan"]["chunk_count"],
            "chunks": chunks,
            "cards": cards,
        }
        _atomic_write_json(
            self._job_path(job_id) / "combined.json",
            value)
        return value

    def refresh(self, job_id):
        with self._lock:
            self._write_combined(job_id)
            self._refresh_manifest(job_id)
            return self.snapshot(job_id)

    def snapshot(self, job_id):
        manifest = self._manifest(job_id)
        statuses = tuple(
            self.chunk_status(job_id, chunk_id)
            for chunk_id in self.chunk_ids(job_id))
        return JobSnapshot(
            job_id=job_id,
            path=self._job_path(job_id),
            overall_status=manifest["overall_status"],
            source_key=manifest["plan"]["source_key"],
            source_title=manifest["plan"]["source_title"],
            plan_id=manifest["plan"]["plan_id"],
            created_at=manifest["created_at"],
            updated_at=manifest["updated_at"],
            chunks=statuses,
        )

    def list(self):
        snapshots = []
        for path in sorted(
                self.root.iterdir(),
                key=lambda item: item.name,
                reverse=True):
            if not path.is_dir():
                continue
            try:
                snapshots.append(self.snapshot(path.name))
            except (OSError, TypeError, ValueError):
                continue
        return tuple(snapshots)

    def inspect_chunk(self, job_id, chunk_id):
        chunk_path = self._chunk_path(job_id, chunk_id)
        attempts = []
        attempts_path = chunk_path / "attempts"
        if attempts_path.is_dir():
            for attempt_path in sorted(attempts_path.iterdir()):
                if not attempt_path.is_dir():
                    continue
                record = {"attempt": int(attempt_path.name)}
                for name, filename in (
                        ("raw_text", "raw.txt"),
                        ("validated", "validated.json"),
                        ("error", "error.json"),
                        ("recovery_error", "recovery_error.json")):
                    path = attempt_path / filename
                    if not path.is_file():
                        continue
                    record[name] = (
                        path.read_text(encoding="utf-8")
                        if name == "raw_text"
                        else _read_json(path))
                attempts.append(record)
        return {
            "request": _read_json(chunk_path / "request.json"),
            "request_contract": self.load_request_contract(job_id),
            "status": self.chunk_status(job_id, chunk_id),
            "attempts": attempts,
        }

    def reset_for_manual_retry(self, job_id, chunk_ids):
        selected = set(chunk_ids)
        available = set(self.chunk_ids(job_id))
        unknown = selected - available
        if unknown:
            raise KeyError(
                "Unknown source-generation chunks: "
                + ", ".join(sorted(unknown)))
        reset = []
        for chunk_id in self.chunk_ids(job_id):
            if chunk_id not in selected:
                continue
            status = self.chunk_status(job_id, chunk_id)
            if status["status"] not in RETRYABLE_MANUALLY:
                continue
            self._set_chunk_status(
                job_id,
                chunk_id,
                status=CHUNK_PENDING,
                worker="queued",
                last_error=None,
                completed_at=None,
            )
            reset.append(chunk_id)
        self.refresh(job_id)
        return tuple(reset)

    def recover_interrupted(self, job_id):
        recovered = []
        for chunk_id in self.chunk_ids(job_id):
            status = self.chunk_status(job_id, chunk_id)
            if status["status"] != CHUNK_RUNNING:
                continue
            with self.chunk_lease(
                    job_id,
                    chunk_id,
                    blocking=False) as acquired:
                if not acquired:
                    continue
                # The worker may have completed between the first status read
                # and lease acquisition. Never roll a completed result back.
                status = self.chunk_status(job_id, chunk_id)
                if status["status"] != CHUNK_RUNNING:
                    continue
                self._set_chunk_status(
                    job_id,
                    chunk_id,
                    status=CHUNK_PENDING,
                    last_error={
                        "type": "InterruptedRun",
                        "message": (
                            "The previous process stopped while this chunk was "
                            "running; its last attempt was retained."),
                        "transient": True,
                        "status_code": None,
                        "timestamp": _utc_now(),
                    },
                )
                recovered.append(chunk_id)
        if recovered:
            self.refresh(job_id)
        return tuple(recovered)


class _RequestStartGate:
    def __init__(self, interval_seconds, *, clock, sleeper):
        self.interval = interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = self.clock()
                delay = max(0.0, self.next_start - now)
                if delay <= 0:
                    self.next_start = (
                        max(now, self.next_start) + self.interval)
                    return
            # Sleeping while holding the scheduling lock would prevent a
            # concurrent 429/Retry-After response from extending the global
            # deadline. Re-check after every wake-up so newly deferred work
            # cannot leave on the obsolete schedule.
            self.sleeper(delay)

    def defer(self, delay_seconds):
        """Apply a server-requested cooldown to every not-yet-started call."""
        if delay_seconds <= 0:
            return
        with self.lock:
            self.next_start = max(
                self.next_start,
                self.clock() + delay_seconds)


class GenerationJobRunner:
    """Run or resume only incomplete chunks using bounded worker threads."""

    def __init__(
            self,
            store,
            job_id,
            request_callable,
            response_validator,
            *,
            concurrency=None,
            request_stagger_ms=None,
            retry_policy=None,
            progress_callback=None,
            sleeper=time.sleep,
            clock=time.monotonic,
            random_source=random.random):
        if not isinstance(store, GenerationJobStore):
            raise TypeError("A GenerationJobStore is required.")
        if not callable(request_callable):
            raise TypeError("A paid request callable must be injected.")
        if not callable(response_validator):
            raise TypeError("A response validator must be injected.")
        manifest = store._manifest(job_id)
        config = manifest["plan"]["config"]
        self.store = store
        self.job_id = job_id
        self.request_callable = request_callable
        self.response_validator = response_validator
        self.concurrency = int(
            concurrency
            if concurrency is not None
            else config["concurrency"])
        if not 1 <= self.concurrency <= 64:
            raise ValueError("Generation concurrency must be between 1 and 64.")
        stagger_ms = (
            request_stagger_ms
            if request_stagger_ms is not None
            else config["request_stagger_ms"])
        if stagger_ms < 0:
            raise ValueError("Request staggering cannot be negative.")
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=config["max_transient_retries"])
        self.progress_callback = progress_callback
        self.sleeper = sleeper
        self.random_source = random_source
        self.cancel_event = threading.Event()
        self.gate = _RequestStartGate(
            stagger_ms / 1000,
            clock=clock,
            sleeper=sleeper)

    def _emit(self, *, chunk_id, status, message, attempt=None):
        event = self.store.event(
            self.job_id,
            chunk_id=chunk_id,
            status=status,
            message=message,
            attempt=attempt)
        if self.progress_callback is not None:
            try:
                self.progress_callback(event)
            except Exception:
                # UI observers must never strand or duplicate paid work.
                pass

    def cancel(self):
        self.cancel_event.set()
        self._emit(
            chunk_id=None,
            status="cancelling",
            message=(
                "Cancellation requested; completed paid responses will be "
                "retained and no queued request will start."))

    def _raw_text(self, response):
        if isinstance(response, str):
            return response
        if isinstance(response, bytes):
            return response.decode("utf-8")
        return _canonical_json(response)

    def _mark_cancelled(self, chunk_id):
        self.store._set_chunk_status(
            self.job_id,
            chunk_id,
            status=CHUNK_CANCELLED,
            completed_at=_utc_now(),
        )
        self._emit(
            chunk_id=chunk_id,
            status=CHUNK_CANCELLED,
            message="Chunk was cancelled before its request started.")

    def _run_chunk(self, chunk_id):
        # A second AutoAnki process may inspect or even try to resume this
        # same saved job. The OS releases this lease automatically on process
        # death, allowing genuine interrupted work to be recovered.
        with self.store.chunk_lease(
                self.job_id,
                chunk_id,
                blocking=False) as acquired:
            if not acquired:
                return
            if self.store.chunk_status(
                    self.job_id,
                    chunk_id)["status"] != CHUNK_PENDING:
                return
            self._run_owned_chunk(chunk_id)

    def _reuse_retained_response(self, chunk_id, chunk):
        """Validate crash-retained output locally before another paid call."""
        status = self.store.chunk_status(
            self.job_id,
            chunk_id)
        attempt_path = self.store.latest_attempt_path(
            self.job_id,
            chunk_id)
        if attempt_path is None:
            return False
        # A normal explicit retry retains its old invalid raw response for
        # inspection; it must still make the newly authorized request. Only an
        # InterruptedRun marker identifies raw output whose validation/status
        # transition itself may have been cut short.
        if (status.get("last_error") or {}).get("type") != "InterruptedRun":
            return False
        raw_path = attempt_path / "raw.txt"
        if not raw_path.is_file():
            validated_path = attempt_path / "validated.json"
            if not validated_path.is_file():
                return False
            error = RuntimeError(
                "A retained validated response has no corresponding raw "
                "response, so its semantic validity cannot be verified.")
            error_record = self.store.write_recovery_error(
                attempt_path,
                error)
            self.store._set_chunk_status(
                self.job_id,
                chunk_id,
                status=CHUNK_FAILED,
                worker="recovery",
                last_error=error_record,
                completed_at=_utc_now(),
            )
            self._emit(
                chunk_id=chunk_id,
                status=CHUNK_FAILED,
                message=(
                    "Retained output could not be safely revalidated; no new "
                    "paid request was made."),
                attempt=status.get("attempts"))
            return True
        attempt = self.store.chunk_status(
            self.job_id,
            chunk_id).get("attempts")
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
            validated = self.response_validator(raw_text, chunk)
            self.store.write_validated(
                attempt_path,
                validated)
        except Exception as error:
            error_record = self.store.write_recovery_error(
                attempt_path,
                error)
            final_status = (
                CHUNK_FAILED
                if isinstance(error, OSError)
                else CHUNK_INVALID)
            self.store._set_chunk_status(
                self.job_id,
                chunk_id,
                status=final_status,
                worker="recovery",
                last_error=error_record,
                completed_at=_utc_now(),
            )
            self._emit(
                chunk_id=chunk_id,
                status=final_status,
                message=(
                    "The retained raw response failed local validation; no "
                    "new paid request was made."),
                attempt=attempt)
            return True

        self.store._set_chunk_status(
            self.job_id,
            chunk_id,
            status=CHUNK_SUCCEEDED,
            worker="recovered",
            last_error=None,
            completed_at=_utc_now(),
        )
        self._emit(
            chunk_id=chunk_id,
            status=CHUNK_SUCCEEDED,
            message=(
                "Locally validated and reused the raw response retained "
                "before the previous process stopped."),
            attempt=attempt)
        return True

    def _run_owned_chunk(self, chunk_id):
        if self.cancel_event.is_set():
            self._mark_cancelled(chunk_id)
            return
        chunk = self.store.load_chunk(self.job_id, chunk_id)
        if self._reuse_retained_response(chunk_id, chunk):
            return
        automatic_retry = 0
        while True:
            if self.cancel_event.is_set():
                self._mark_cancelled(chunk_id)
                return
            attempt, attempt_path = self.store.begin_attempt(
                self.job_id,
                chunk_id)
            self.store._set_chunk_status(
                self.job_id,
                chunk_id,
                worker=threading.current_thread().name)
            self._emit(
                chunk_id=chunk_id,
                status=CHUNK_RUNNING,
                message=(
                    f"Requesting source chunk {chunk.index}/{chunk.total}."),
                attempt=attempt)
            self.gate.wait()
            if self.cancel_event.is_set():
                # The attempt directory explains why the paid call did not
                # happen; it is still useful recovery evidence.
                error = RuntimeError(
                    "Cancelled after rate-limit wait and before request.")
                error_record = self.store.write_error(
                    attempt_path,
                    error,
                    transient=False)
                self.store._set_chunk_status(
                    self.job_id,
                    chunk_id,
                    status=CHUNK_CANCELLED,
                    last_error=error_record,
                    completed_at=_utc_now(),
                )
                self._emit(
                    chunk_id=chunk_id,
                    status=CHUNK_CANCELLED,
                    message=(
                        "Chunk was cancelled after its scheduling wait and "
                        "before its request started."),
                    attempt=attempt)
                return
            try:
                response = self.request_callable(chunk)
            except Exception as error:
                transient = is_transient_request_error(error)
                retained_raw = getattr(
                    error,
                    "raw_response_text",
                    None)
                if isinstance(retained_raw, str) and retained_raw:
                    self.store.write_raw(
                        attempt_path,
                        retained_raw)
                error_record = self.store.write_error(
                    attempt_path,
                    error,
                    transient=transient)
                if (
                        transient
                        and automatic_retry < self.retry_policy.max_retries):
                    automatic_retry += 1
                    delay = self.retry_policy.delay(
                        automatic_retry,
                        error,
                        random_unit=self.random_source())
                    self.store._set_chunk_status(
                        self.job_id,
                        chunk_id,
                        status=CHUNK_RUNNING,
                        last_error=error_record,
                    )
                    self._emit(
                        chunk_id=chunk_id,
                        status="retrying_connection",
                        message=(
                            "Transient connection/API failure; retrying "
                            f"automatically in {delay:.2f}s."),
                        attempt=attempt)
                    # The next loop reaches the shared start gate. Deferring
                    # that gate makes a 429/reset header slow every queued
                    # worker, rather than only the worker that observed it.
                    self.gate.defer(delay)
                    continue
                if transient:
                    final_status = CHUNK_CONNECTION_FAILED
                elif _is_invalid_response_error(error):
                    final_status = CHUNK_INVALID
                else:
                    final_status = CHUNK_FAILED
                self.store._set_chunk_status(
                    self.job_id,
                    chunk_id,
                    status=final_status,
                    last_error=error_record,
                    completed_at=_utc_now(),
                )
                self._emit(
                    chunk_id=chunk_id,
                    status=final_status,
                    message=(
                        "Connection retries were exhausted."
                        if transient
                        else (
                            "The API response was invalid and requires manual "
                            "review."
                            if final_status == CHUNK_INVALID
                            else "Request failed and requires manual review."
                        )),
                    attempt=attempt)
                return

            raw_text = self._raw_text(response)
            self.store.write_raw(attempt_path, raw_text)
            try:
                validated = self.response_validator(raw_text, chunk)
                self.store.write_validated(attempt_path, validated)
            except Exception as error:
                error_record = self.store.write_error(
                    attempt_path,
                    error,
                    transient=False)
                self.store._set_chunk_status(
                    self.job_id,
                    chunk_id,
                    status=CHUNK_INVALID,
                    last_error=error_record,
                    completed_at=_utc_now(),
                )
                self._emit(
                    chunk_id=chunk_id,
                    status=CHUNK_INVALID,
                    message=(
                        "The response was retained but failed validation; "
                        "manual retry or inspection is required."),
                    attempt=attempt)
                return

            self.store._set_chunk_status(
                self.job_id,
                chunk_id,
                status=CHUNK_SUCCEEDED,
                last_error=None,
                completed_at=_utc_now(),
            )
            self._emit(
                chunk_id=chunk_id,
                status=CHUNK_SUCCEEDED,
                message=(
                    f"Validated source chunk {chunk.index}/{chunk.total}."),
                attempt=attempt)
            return

    def run(self, chunk_ids=None):
        """Resume pending chunks; invalid output is never retried implicitly."""
        self.store.recover_interrupted(self.job_id)
        available = self.store.chunk_ids(self.job_id)
        if chunk_ids is None:
            selected = available
        else:
            selected_set = set(chunk_ids)
            unknown = selected_set - set(available)
            if unknown:
                raise KeyError(
                    "Unknown source-generation chunks: "
                    + ", ".join(sorted(unknown)))
            selected = tuple(
                chunk_id
                for chunk_id in available
                if chunk_id in selected_set)
        pending = tuple(
            chunk_id
            for chunk_id in selected
            if self.store.chunk_status(
                self.job_id,
                chunk_id)["status"] == CHUNK_PENDING)
        if not pending:
            return self.store.refresh(self.job_id)
        self._emit(
            chunk_id=None,
            status="running",
            message=(
                f"Starting {len(pending)} source chunks with at most "
                f"{self.concurrency} concurrent requests."))
        with ThreadPoolExecutor(
                max_workers=self.concurrency,
                thread_name_prefix="autoanki-source") as executor:
            futures = {
                executor.submit(self._run_chunk, chunk_id): chunk_id
                for chunk_id in pending
            }
            for future in as_completed(futures):
                # Worker exceptions should be exceptional internal failures,
                # since request and validation failures are persisted above.
                future.result()
        snapshot = self.store.refresh(self.job_id)
        self._emit(
            chunk_id=None,
            status=snapshot.overall_status,
            message=(
                "Source generation pass finished; all completed output and "
                "failures were retained."))
        return snapshot

    def retry_one(self, chunk_id):
        """Explicitly authorize one failed chunk for another paid request."""
        reset = self.store.reset_for_manual_retry(
            self.job_id,
            (chunk_id,))
        if not reset:
            raise ValueError(
                "Only failed, invalid, or cancelled chunks can be retried.")
        self.cancel_event.clear()
        return self.run(reset)

    def retry_all(self):
        """Explicitly authorize every failed chunk for another paid request."""
        failed = tuple(
            chunk_id
            for chunk_id in self.store.chunk_ids(self.job_id)
            if self.store.chunk_status(
                self.job_id,
                chunk_id)["status"] in RETRYABLE_MANUALLY)
        reset = self.store.reset_for_manual_retry(
            self.job_id,
            failed)
        self.cancel_event.clear()
        return self.run(reset)
