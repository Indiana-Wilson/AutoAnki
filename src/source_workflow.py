"""Application-level orchestration for the GUI's From Source workflow.

Planning, estimation, and job inspection remain free operations.  OpenAI,
Codex, local model loading, and Anki mutations occur only from their explicit
GUI callbacks.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unicodedata

import anki_integration
import codex_source_retrieval
from corpus_pipeline.tokenizers import (
    CkipHanTokenizer,
    recommended_hardware_batch_size,
)
import pipeline_store
import process_text
import source_deck
from source_generation import (
    AnkiExclusionSpec,
    GenerationJobRunner,
    MODEL_MAX_OUTPUT_TOKENS,
    SOURCE_REQUEST_MODEL,
    SourceGenerationBackend,
    build_source_request_contract,
    make_pipeline_response_validator,
    render_chunk_input,
)
import source_preparation


SOURCE_MODEL = SOURCE_REQUEST_MODEL
SOURCE_WORKFLOW_SCHEMA_VERSION = 1
SOURCE_WORKFLOW_FILE_NAME = "workflow.json"
_FINALIZE_ROW_ID = "finalize"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path, value):
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
            json.dump(
                value,
                output,
                ensure_ascii=False,
                sort_keys=True,
                indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid source workflow file: {path}") from error


class SourceWorkflowController:
    """Default callbacks wired into :class:`gui.AutoAnkiApp`."""

    def __init__(
            self,
            *,
            backend=None,
            openai_client_factory=None,
            anki_client_factory=anki_integration.AnkiConnectClient,
            package_creator=source_deck.create_source_package,
            package_importer=source_deck.import_source_package,
            source_preparer=source_preparation.prepare_source_file,
            retrieval_job_creator=(
                codex_source_retrieval.create_retrieval_job),
            retrieval_job_runner=(
                codex_source_retrieval.run_retrieval_job),
            clock=time.monotonic):
        self.anki_client_factory = anki_client_factory
        self.openai_client_factory = (
            openai_client_factory
            or (
                lambda api_key: process_text.OpenAI(
                    api_key=api_key,
                    max_retries=0)))
        self.package_creator = package_creator
        self.package_importer = package_importer
        self.source_preparer = source_preparer
        self.retrieval_job_creator = retrieval_job_creator
        self.retrieval_job_runner = retrieval_job_runner
        self.clock = clock
        self._anki_vocabulary_cache = {}
        self._anki_cache_seconds = 30
        self._anki_cache_lock = threading.RLock()
        self._active_jobs = set()
        self._active_lock = threading.RLock()
        self.backend = backend or SourceGenerationBackend(
            exclusion_resolver=self._resolve_anki_exclusion)
        if (
                backend is not None
                and backend.exclusion_resolver is None):
            backend.exclusion_resolver = self._resolve_anki_exclusion

    def gui_hooks(self):
        return {
            "source_catalog_loader": self.catalogue,
            "source_estimator": self.estimate,
            "source_generate_callback": self.generate,
            "source_prepare_callback": self.prepare,
            "source_codex_callback": self.retrieve_with_codex,
            "source_jobs_loader": self.job_rows,
            "source_retry_callback": self.retry,
            "source_inspect_callback": self.inspect,
            "source_preview_loader": self.preview,
            "source_anki_options_loader": self.anki_options,
            "manual_input_filter_callback": self.filter_manual_input,
        }

    def catalogue(self):
        return self.backend.catalogue()

    def estimate(self, request):
        return self.backend.estimate(request)

    def preview(self, request):
        """Load retained source metadata without contacting OpenAI or Anki."""
        return self.backend.preview(request)

    def _ready_anki_client(self):
        client = self.anki_client_factory()
        anki_integration.ensure_anki_running(client)
        anki_integration.wait_for_collection_ready(client)
        return client

    def _resolve_anki_exclusion(
            self,
            specification,
            _loaded_source,
            *,
            force_refresh=False):
        identity = (
            specification.deck_name,
            specification.note_type,
            specification.field_name,
            specification.card_template_name,
        )
        # Keep the collection read inside this lock. Concurrent estimate
        # refreshes for the same settings then share one AnkiConnect read
        # instead of launching a thundering herd of identical requests.
        with self._anki_cache_lock:
            cached = self._anki_vocabulary_cache.get(identity)
            now = self.clock()
            if (
                    not force_refresh
                    and cached is not None
                    and now - cached[0] <= self._anki_cache_seconds):
                return cached[1]
            client = self._ready_anki_client()
            vocabulary = anki_integration.read_existing_vocabulary(
                client,
                anki_integration.AnkiVocabularySource(
                    deck_name=specification.deck_name,
                    note_type_name=specification.note_type,
                    field_name=specification.field_name,
                    card_template_name=(
                        specification.card_template_name)))
            self._anki_vocabulary_cache[identity] = (
                self.clock(),
                vocabulary)
            return vocabulary

    def _refresh_paid_source_exclusion(self, request):
        """Read Anki again immediately before freezing a paid source plan."""
        if not request.get("exclude_anki"):
            return None
        specifications = self._anki_exclusions_from_request(request)
        if not specifications:
            raise ValueError(
                "Anki exclusion requires a deck, note type, and field.")
        return frozenset(
            word
            for specification in specifications
            for word in self._resolve_anki_exclusion(
                specification,
                None,
                force_refresh=True))

    @staticmethod
    def _anki_exclusions_from_request(request):
        values = request.get("anki_exclusions")
        if values is None:
            singular = request.get("anki_exclusion")
            values = () if singular is None else (singular,)
        if not isinstance(values, (tuple, list)):
            raise TypeError("Anki exclusions must be a list.")
        specifications = tuple(dict.fromkeys(
            AnkiExclusionSpec.from_mapping(value)
            for value in values))
        if any(
                specification is None
                or not specification.note_type
                or not specification.field_name
                for specification in specifications):
            raise ValueError(
                "Every Anki exclusion requires a deck, note type, and field.")
        return specifications

    def anki_options(self, request):
        """Resolve note types and fields without changing the collection."""
        if not isinstance(request, dict):
            raise TypeError("Anki option request must be an object.")
        action = request.get("action")
        client = self._ready_anki_client()
        if action == "note_types":
            deck = str(request.get("deck", "")).strip()
            if not deck:
                raise ValueError("Choose an Anki deck first.")
            return {
                "note_types": anki_integration.list_note_types_in_deck(
                    client,
                    deck),
            }
        if action == "note_type_details":
            note_type = str(request.get("note_type", "")).strip()
            if not note_type:
                raise ValueError("Choose an Anki note type first.")
            return {
                "fields": anki_integration.get_note_type_fields(
                    client,
                    note_type),
            }
        raise ValueError("Unknown Anki option request.")

    def filter_manual_input(self, request):
        """Remove exact learned line-items before any manual paid request."""
        if not isinstance(request, dict):
            raise TypeError("Manual vocabulary filter request must be an object.")
        text = request.get(
            "text",
            request.get("input_text", ""))
        if not isinstance(text, str):
            raise TypeError("Manual input must be text.")
        specifications = self._anki_exclusions_from_request(request)
        if not specifications:
            raise ValueError(
                "Choose an Anki deck, note type, and field for filtering.")
        learned = {
            unicodedata.normalize("NFC", str(term))
            for specification in specifications
            for term in self._resolve_anki_exclusion(
                specification,
                None,
                force_refresh=True)
        }
        retained_lines = []
        excluded_count = 0
        remaining_count = 0
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                retained_lines.append(line)
                continue
            if unicodedata.normalize("NFC", candidate) in learned:
                excluded_count += 1
                continue
            retained_lines.append(line)
            remaining_count += 1
        filtered_text = "\n".join(retained_lines).strip()
        if not filtered_text:
            remaining_count = 0
        return {
            "filtered_text": filtered_text,
            "excluded_count": excluded_count,
            "remaining_count": remaining_count,
            "original_count": (
                excluded_count + remaining_count),
        }

    def _job_path(self, job_id):
        return self.backend.jobs.snapshot(job_id).path

    def _manifest(self, job_id):
        manifest = _read_json(
            self._job_path(job_id) / "manifest.json")
        if manifest.get("job_id") != job_id:
            raise ValueError("Source job manifest and path disagree.")
        return manifest

    def _workflow_path(self, job_id):
        return self._job_path(job_id) / SOURCE_WORKFLOW_FILE_NAME

    def _workflow(self, job_id):
        path = self._workflow_path(job_id)
        if not path.is_file():
            manifest = self._manifest(job_id)
            return {
                "schema_version": SOURCE_WORKFLOW_SCHEMA_VERSION,
                "job_id": job_id,
                "state": "waiting_for_chunks",
                "source_title": manifest["plan"]["source_title"],
                "deck_name": source_deck.source_deck_name(
                    manifest["plan"]["source_title"]),
                "finalize_attempts": 0,
                "package_path": None,
                "notes_created": None,
                "import_result": None,
                "last_error": None,
                "updated_at": manifest["updated_at"],
            }
        value = _read_json(path)
        if (
                value.get("schema_version")
                != SOURCE_WORKFLOW_SCHEMA_VERSION
                or value.get("job_id") != job_id):
            raise ValueError("Unsupported or inconsistent source workflow.")
        return value

    def _update_workflow(self, job_id, **updates):
        value = self._workflow(job_id)
        value.update(updates)
        value["updated_at"] = _utc_now()
        _atomic_write_json(self._workflow_path(job_id), value)
        return value

    def _pipeline_for_job(self, job_id):
        metadata = self._manifest(job_id).get("request_metadata", {})
        return pipeline_store.pipeline_from_mapping(
            metadata.get("pipeline"))

    def _request_contract_for_job(self, job_id, pipeline=None):
        contract = self.backend.jobs.load_request_contract(job_id)
        metadata = self._manifest(job_id).get("request_metadata", {})
        if contract is not None:
            origin = (
                "created_with_job"
                if metadata.get("request_contract") is not None
                else "legacy_reconstructed")
            return contract, origin
        pipeline = pipeline or self._pipeline_for_job(job_id)
        chunks = tuple(
            self.backend.jobs.load_chunk(job_id, chunk_id)
            for chunk_id in self.backend.jobs.chunk_ids(job_id))
        contract = self.backend.jobs.ensure_request_contract(
            job_id,
            build_source_request_contract(
                pipeline,
                chunks=chunks))
        return contract, "legacy_reconstructed"

    def _paid_request_callable(self, pipeline, client, request_contract):
        def request(chunk):
            try:
                max_output_tokens = request_contract[
                    "max_output_tokens_by_chunk"][chunk.chunk_id]
            except KeyError as error:
                raise ValueError(
                    "The saved source request contract has no output limit "
                    f"for chunk {chunk.chunk_id}.") from error
            request_options = {
                "model": request_contract["model"],
                "input": (
                    request_contract["composed_prompt"]
                    + render_chunk_input(chunk)),
                "reasoning": request_contract["reasoning"],
                "text": {
                    "format": request_contract["response_format"],
                },
                "max_output_tokens": max_output_tokens,
            }
            tools = request_contract.get("tools", ())
            if tools:
                request_options["tools"] = tools
                request_options["tool_choice"] = "auto"
                request_options["max_tool_calls"] = request_contract[
                    "max_tool_calls"]
            response = client.responses.create(
                **request_options)
            return process_text.extract_response_text(response)

        return request

    def _run_job(self, job_id, pipeline, client, chunk_ids=None):
        request_contract, _origin = self._request_contract_for_job(
            job_id,
            pipeline)
        runner = GenerationJobRunner(
            self.backend.jobs,
            job_id,
            self._paid_request_callable(
                pipeline,
                client,
                request_contract),
            make_pipeline_response_validator(pipeline))
        return runner.run(chunk_ids)

    def _mark_active(self, job_id, active):
        with self._active_lock:
            if active:
                self._active_jobs.add(job_id)
            else:
                self._active_jobs.discard(job_id)

    def _is_active(self, job_id):
        with self._active_lock:
            return job_id in self._active_jobs

    def generate(self, request):
        if request.get("paid_confirmed") is not True:
            raise PermissionError(
                "Paid source generation requires explicit confirmation.")
        self._refresh_paid_source_exclusion(request)
        snapshot = self.backend.create_job(request)
        self._update_workflow(
            snapshot.job_id,
            state=(
                "waiting_for_chunks"
                if snapshot.chunks
                else "nothing_to_generate"),
            last_error=None)
        if not snapshot.chunks:
            return {
                "job_id": snapshot.job_id,
                "no_new_vocabulary": True,
                "message": (
                    "No new vocabulary remains. No OpenAI request or Anki "
                    "deck was created."),
            }

        api_key = process_text.get_api_key()
        if not api_key:
            raise process_text.MissingAPIKeyError(
                "No OpenAI API key is configured. Add one in Advanced.")
        pipeline = pipeline_store.pipeline_from_mapping(
            request.get("pipeline"))
        client = self.openai_client_factory(api_key)
        self._mark_active(snapshot.job_id, True)
        try:
            completed = self._run_job(
                snapshot.job_id,
                pipeline,
                client)
            finalized = self._finalize_if_complete(
                snapshot.job_id,
                pipeline,
                completed)
        finally:
            self._mark_active(snapshot.job_id, False)
        return {
            "job_id": snapshot.job_id,
            "status": completed.overall_status,
            "counts": completed.counts,
            "requires_attention": (
                completed.overall_status != "completed"),
            "message": (
                finalized.get("message")
                if finalized is not None
                else (
                    "The generation pass finished with retained failures. "
                    "Inspect Jobs & Failures; invalid responses were not "
                    "retried.")),
        }

    def _combined(self, job_id):
        value = _read_json(self._job_path(job_id) / "combined.json")
        if value.get("job_id") != job_id:
            raise ValueError("Combined output belongs to another source job.")
        return value

    def _finalize_if_complete(self, job_id, pipeline, snapshot=None):
        # A snapshot supplied by another runner may already be stale. Re-read
        # before deciding, then take the same lease used by finalization so a
        # second process cannot overwrite an imported workflow with
        # "waiting_for_chunks".
        snapshot = self.backend.jobs.refresh(job_id)
        if snapshot.overall_status == "completed":
            return self._finalize(job_id, pipeline)
        with self.backend.jobs.finalization_lease(
                job_id,
                blocking=False) as acquired:
            if not acquired:
                workflow = self._workflow(job_id)
                if workflow["state"] == "imported":
                    return workflow
                return {
                    **workflow,
                    "finalization_in_progress": True,
                    "message": (
                        "Deck packaging/import is already running in another "
                        "AutoAnki process. No duplicate import was started."),
                }
            snapshot = self.backend.jobs.refresh(job_id)
            if snapshot.overall_status == "completed":
                return self._finalize_with_lease(job_id, pipeline)
            workflow = self._workflow(job_id)
            if workflow["state"] in {
                    "imported",
                    "packaging",
                    "packaged",
                    "importing",
                    "failed"}:
                return workflow
            self._update_workflow(
                job_id,
                state="waiting_for_chunks",
                last_error=None)
            return None

    def _finalize(self, job_id, pipeline):
        with self.backend.jobs.finalization_lease(
                job_id,
                blocking=False) as acquired:
            if not acquired:
                # Another AutoAnki process owns packaging/import. Do not wait
                # indefinitely and, above all, do not duplicate its import.
                workflow = self._workflow(job_id)
                if workflow["state"] == "imported":
                    return workflow
                return {
                    **workflow,
                    "finalization_in_progress": True,
                    "message": (
                        "Deck packaging/import is already running in another "
                        "AutoAnki process. No duplicate import was started."),
                }
            # Re-read only after ownership. The other process may have
            # completed between the caller's snapshot and lease acquisition.
            return self._finalize_with_lease(job_id, pipeline)

    def _finalize_with_lease(self, job_id, pipeline):
        workflow = self._workflow(job_id)
        if workflow["state"] == "imported":
            return workflow
        manifest = self._manifest(job_id)
        combined = self._combined(job_id)
        if not combined.get("complete"):
            raise RuntimeError(
                "Not every source chunk has validated output.")

        package_path = self._job_path(job_id) / "source_deck.apkg"
        attempts = int(workflow.get("finalize_attempts", 0)) + 1
        try:
            if (
                    not package_path.is_file()
                    or workflow.get("notes_created") is None):
                self._update_workflow(
                    job_id,
                    state="packaging",
                    finalize_attempts=attempts,
                    last_error=None)
                _path, notes_created = self.package_creator(
                    {"cards": combined["cards"]},
                    source_title=manifest["plan"]["source_title"],
                    source_key=manifest["plan"]["source_key"],
                    pipeline=pipeline,
                    output_path=package_path)
                workflow = self._update_workflow(
                    job_id,
                    state="packaged",
                    package_path=str(package_path),
                    notes_created=notes_created,
                    last_error=None)
            self._update_workflow(
                job_id,
                state="importing",
                finalize_attempts=attempts,
                package_path=str(package_path),
                last_error=None)
            imported = self.package_importer(package_path)
        except Exception as error:
            self._update_workflow(
                job_id,
                state="failed",
                finalize_attempts=attempts,
                package_path=(
                    str(package_path)
                    if package_path.is_file()
                    else None),
                last_error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": _utc_now(),
                })
            raise

        deck_name = source_deck.source_deck_name(
            manifest["plan"]["source_title"])
        return self._update_workflow(
            job_id,
            state="imported",
            package_path=str(package_path),
            deck_name=deck_name,
            import_result=imported,
            imported_at=_utc_now(),
            last_error=None,
            message=(
                f'Created and imported “{deck_name}” in place. '
                "Its cards were not moved and the deck was not deleted."))

    def _finalization_is_active(self, job_id):
        with self.backend.jobs.finalization_lease(
                job_id,
                blocking=False) as acquired:
            return not acquired

    def _finalize_row(self, snapshot):
        workflow = self._workflow(snapshot.job_id)
        state = workflow["state"]
        finalization_active = (
            state not in {"imported", "nothing_to_generate"}
            and self._finalization_is_active(snapshot.job_id))
        if state == "imported":
            status = "completed"
            detail = workflow.get("message", "Deck imported.")
        elif state == "nothing_to_generate":
            status = "completed"
            detail = "No vocabulary remained; no requests or deck."
        elif finalization_active:
            status = "running"
            detail = "Deck packaging/import is active in another process."
        elif state == "failed":
            status = "failed"
            detail = (workflow.get("last_error") or {}).get(
                "message",
                "Deck packaging or import failed.")
        elif (
                state == "waiting_for_chunks"
                and snapshot.overall_status == "completed"):
            status = "failed"
            detail = (
                "Every OpenAI chunk is complete, but deck finalization did "
                "not run. Retry this row to package and import without "
                "repeating OpenAI.")
        elif (
                state in {"packaging", "packaged", "importing"}
                and not self._is_active(snapshot.job_id)):
            # A process may have stopped after writing this state. Stable note
            # GUIDs make an explicitly authorized re-import recoverable.
            status = "failed"
            detail = (
                "The previous run stopped during deck finalization. Inspect "
                "it, then retry this row to recover without repeating OpenAI.")
        elif state in {"packaging", "packaged", "importing"}:
            status = "running"
            detail = f"Deck {state} is in progress."
        else:
            status = "pending"
            detail = "Waiting for every OpenAI chunk to validate."
        return {
            "job_id": (
                f"{snapshot.job_id}::{_FINALIZE_ROW_ID}"),
            "parent_job_id": snapshot.job_id,
            "source_name": snapshot.source_title,
            "chunk_label": "Deck",
            "worker": "package/import",
            "status": status,
            "attempts": workflow.get("finalize_attempts", 0),
            "detail": detail,
            "updated_at": workflow.get("updated_at", ""),
        }

    def job_rows(self):
        for snapshot in self.backend.jobs.list():
            if not self._is_active(snapshot.job_id):
                self.backend.jobs.recover_interrupted(
                    snapshot.job_id)
        payload = self.backend.job_rows()
        rows = list(payload.get("jobs", ()))
        for snapshot in self.backend.jobs.list():
            rows.append(self._finalize_row(snapshot))
        return {"jobs": rows}

    def inspect(self, request):
        row_id = (
            request.get("job_id")
            if isinstance(request, dict)
            else request)
        job_id, child_id = self.backend.split_row_id(row_id)
        request_contract, contract_origin = (
            self._request_contract_for_job(job_id))
        if child_id != _FINALIZE_ROW_ID:
            inspection = self.backend.inspect(request)
            inspection["request_contract"] = request_contract
            inspection["request_contract_origin"] = contract_origin
            return inspection
        combined = self._combined(job_id)
        return {
            "workflow": self._workflow(job_id),
            "request_contract": request_contract,
            "request_contract_origin": contract_origin,
            "combined": {
                key: value
                for key, value in combined.items()
                if key not in {"cards", "chunks"}
            },
            "job_path": str(self._job_path(job_id)),
        }

    def retry(self, request):
        if request.get("paid_confirmed") is not True:
            raise PermissionError(
                "Source retries require explicit confirmation.")
        regular_rows = []
        finalize_jobs = set()
        for row_id in request.get("job_ids", ()):
            job_id, child_id = self.backend.split_row_id(row_id)
            if child_id == _FINALIZE_ROW_ID:
                finalize_jobs.add(job_id)
            else:
                regular_rows.append(row_id)

        client = None
        grouped = {}
        if regular_rows:
            api_key = process_text.get_api_key()
            if not api_key:
                raise process_text.MissingAPIKeyError(
                    "No OpenAI API key is configured.")
            grouped = self.backend.manual_retry_targets({
                "job_ids": tuple(regular_rows),
                "paid_confirmed": True,
            })
            client = self.openai_client_factory(api_key)

        completed_jobs = set()
        for job_id, chunk_ids in grouped.items():
            if not chunk_ids:
                continue
            pipeline = self._pipeline_for_job(job_id)
            self._mark_active(job_id, True)
            try:
                snapshot = self._run_job(
                    job_id,
                    pipeline,
                    client,
                    chunk_ids)
                if snapshot.overall_status == "completed":
                    self._finalize(job_id, pipeline)
                completed_jobs.add(job_id)
            finally:
                self._mark_active(job_id, False)

        for job_id in finalize_jobs - completed_jobs:
            pipeline = self._pipeline_for_job(job_id)
            self._mark_active(job_id, True)
            try:
                self._finalize_if_complete(job_id, pipeline)
            finally:
                self._mark_active(job_id, False)
        selected_parent_job_ids = set(grouped) | finalize_jobs
        remaining_failures = sum(
            1
            for snapshot in self.backend.jobs.list()
            if snapshot.job_id in selected_parent_job_ids
            for chunk in snapshot.chunks
            if chunk["status"] in {
                "cancelled",
                "connection_failed",
                "failed",
                "invalid_response",
                "pending",
            })
        return {
            "requires_attention": remaining_failures > 0,
            "message": (
                "The explicitly selected retries finished. Completed OpenAI "
                "chunks were not repeated."
                + (
                    f" {remaining_failures:,} failed request(s) still need "
                    "inspection."
                    if remaining_failures
                    else "")),
        }

    @staticmethod
    def _document_tokenizer(language_key, prefer_cuda):
        device = "auto" if prefer_cuda else "cpu"
        batch_size = (
            recommended_hardware_batch_size()
            if prefer_cuda
            else 128)
        options = {
            "device": device,
            "batch_size": batch_size,
        }
        if language_key == "classical_chinese_warring_states":
            return CkipHanTokenizer.for_shanggu(**options)
        if language_key == "classical_chinese_ming":
            return CkipHanTokenizer.for_jindai(**options)
        raise ValueError(
            "Source preparation currently supports Classical Chinese "
            "(Warring States) and Classical Chinese (Ming).")

    def prepare(self, request):
        if request.get("generate_cards"):
            raise PermissionError(
                "File preparation cannot start paid card generation.")
        language_key = request["language_key"]
        prepared = self.source_preparer(
            request["path"],
            title=request["title"],
            language_key=language_key,
            tokenizer=self._document_tokenizer(
                language_key,
                bool(request.get("prefer_cuda", True))))
        return {
            **prepared.to_dict(),
            "message": (
                f'Prepared “{prepared.title}”: '
                f"{prepared.unique_word_count:,} unique candidates in "
                f"{prepared.section_count:,} sections. Inspect the retained "
                "source before authorizing generation."),
        }

    @staticmethod
    def _merge_retrieved_files(result):
        if len(result.files) == 1:
            return result.files[0]
        sections = []
        for path in result.files:
            for title, text in source_preparation.extract_source_sections(path):
                sections.append(
                    f"{path.name} — {title}\n{text.strip()}")
        if not sections:
            raise ValueError("Codex retrieved no extractable source text.")
        merged_path = result.job_path / "retrieved" / (
            "autoanki_combined_source.txt")
        merged_path.write_text(
            "\n\f\n".join(sections) + "\n",
            encoding="utf-8",
            newline="\n")
        return merged_path

    def retrieve_with_codex(self, request):
        if request.get("retrieval_authorized") is not True:
            raise PermissionError(
                "Codex retrieval requires explicit confirmation.")
        if request.get("generate_cards"):
            raise PermissionError(
                "Codex retrieval cannot start paid card generation.")
        retrieval_request = (
            codex_source_retrieval.CodexRetrievalRequest(
                description=request.get(
                    "description",
                    request.get("request", "")),
                source_title=request.get(
                    "source_title",
                    request.get("title", "")),
                language_key=request["language_key"]))
        job_path = self.retrieval_job_creator(retrieval_request)
        result = self.retrieval_job_runner(job_path)
        source_path = self._merge_retrieved_files(result)
        prepared = self.source_preparer(
            source_path,
            title=request.get(
                "source_title",
                request.get("title", result.title)),
            language_key=request["language_key"],
            tokenizer=self._document_tokenizer(
                request["language_key"],
                True))
        warnings = (
            " ".join(result.warnings)
            if result.warnings
            else "No retrieval warnings were reported.")
        return {
            **prepared.to_dict(),
            "codex_job_id": result.job_id,
            "source_urls": result.source_urls,
            "warnings": result.warnings,
            "message": (
                f'Codex retrieved and prepared “{prepared.title}” with '
                f"{prepared.unique_word_count:,} unique candidates. "
                f"{warnings} Inspect the saved files before generation."),
        }
