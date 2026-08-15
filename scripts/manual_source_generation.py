#!/usr/bin/env python3
"""Resumable, offline ingestion of manually authored source responses.

This command deliberately has no provider-request operation.  It uses the
normal source planner, frozen request contract, response validator, job store,
and finalizer, but the response for each chunk must come from a local JSON
file written by a human.
"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui_preferences
import anki_integration
import pipeline_store
import source_deck
from source_generation import SourceGenerationBackend, render_chunk_input
from source_generation.jobs import CHUNK_SUCCEEDED
from source_generation.model_catalog import source_model_profile
from source_workflow import SourceWorkflowController


SOURCE_KEY = "journey_to_the_west"
SOURCE_TITLE = "Journey to the West"
SOURCE_LANGUAGE_KEY = "classical_chinese_ming"
EXPECTED_UNIQUE_TOKEN_COUNT = 24_224
MANUAL_RESPONSES_DIRECTORY = "manual_responses"
MANUAL_PACKETS_DIRECTORY = "manual_packets"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _positive_integer(value, name, *, maximum=None):
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if parsed < 1 or (maximum is not None and parsed > maximum):
        suffix = (
            f" no greater than {maximum:,}"
            if maximum is not None
            else "")
        raise ValueError(f"{name} must be a positive integer{suffix}.")
    return parsed


def _select_pipeline(pipelines, pipeline_id=None):
    pipelines = tuple(pipelines)
    if not pipelines:
        pipelines = (pipeline_store.default_pipeline(),)
    if pipeline_id is None:
        return pipelines[0]
    for pipeline in pipelines:
        if pipeline.pipeline_id == pipeline_id:
            return pipeline
    raise ValueError(f"Unknown pipeline ID: {pipeline_id}")


def build_journey_request(pipelines, preferences, *, pipeline_id=None):
    """Reproduce the current From Source settings without learned exclusions."""
    pipeline = _select_pipeline(pipelines, pipeline_id)
    settings = pipeline_store.get_language_settings(
        pipeline,
        SOURCE_LANGUAGE_KEY)

    direction_keys = tuple(dict.fromkeys(
        preferences.get("source_card_directions", ())))
    known_directions = {
        direction.key
        for direction in pipeline_store.list_directions()
    }
    if not direction_keys or not set(direction_keys) <= known_directions:
        raise ValueError(
            "Current source card directions are empty or invalid.")

    source_cards = []
    for card in settings.cards:
        enabled = card.direction_key in direction_keys
        effective_fields = pipeline_store.get_effective_fields(
            settings,
            card)
        if (
                enabled
                and card.direction_key == "context"
                and not effective_fields):
            effective_fields = (
                pipeline_store.FieldSetting(
                    "dictionary_meaning",
                    "english"),)
        source_cards.append(replace(
            card,
            enabled=enabled,
            fields=tuple(effective_fields)))
    settings = replace(
        settings,
        cards=tuple(source_cards),
        target_deck=f"Vocabulary from {SOURCE_TITLE}",
        separate_target_decks=False,
        share_field_settings=False)
    settings_by_key = {
        item.language_key: item
        for item in pipeline.language_settings
    }
    settings_by_key[settings.language_key] = settings
    pipeline = pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        tuple(settings_by_key.values()),
        active_language_key=SOURCE_LANGUAGE_KEY)
    pipeline_store.validate_pipelines((pipeline,))

    chunk_size = _positive_integer(
        preferences.get("source_chunk_size", "30"),
        "Source chunk size",
        maximum=10_000)
    model = preferences.get("source_model_key", "gpt-5.4-mini")
    model_profile = source_model_profile(model)
    execution_mode = preferences.get(
        "source_execution_key",
        "standard")
    if execution_mode not in {"standard", "economy"}:
        raise ValueError("Current source execution mode is invalid.")
    # This manual run uses the current compact contract, rather than creating
    # a rollback-format job if an old GUI preference happens to be retained.
    request_protocol = "v10"
    reasoning_effort = preferences.get("source_reasoning_key", "low")
    if request_protocol == "v8":
        reasoning_effort = "low"
    if model_profile.local:
        execution_mode = "standard"
        request_protocol = "v10"
        reasoning_effort = "none"
        concurrency = model_profile.recommended_concurrency
        request_stagger_ms = 0
    elif execution_mode == "economy":
        concurrency = 1
        request_stagger_ms = 0
    else:
        concurrency = _positive_integer(
            preferences.get("source_concurrency", "8"),
            "Source concurrency",
            maximum=64)
        request_stagger_ms = int(str(preferences.get(
            "source_request_stagger_ms",
            "100")).strip())
        if not 0 <= request_stagger_ms <= 60_000:
            raise ValueError(
                "Source request stagger must be between 0 and 60000 ms.")

    context_mode = preferences.get("source_context_key", "sentence")
    use_source_examples = bool(
        preferences.get("source_use_source_examples", False)
        and context_mode != "none"
        and "context" in direction_keys)
    if use_source_examples:
        context_mode = "sentence"
    automatic_repair = bool(
        preferences.get("source_automatic_repair", False)
        and execution_mode == "standard")

    return {
        "source_key": SOURCE_KEY,
        "source_name": SOURCE_TITLE,
        "source_language_key": SOURCE_LANGUAGE_KEY,
        "chunk_size": chunk_size,
        # This workflow intentionally covers the complete saved vocabulary.
        "source_prefix_token_limit": None,
        "context_mode": context_mode,
        "concurrency": concurrency,
        "request_stagger_ms": request_stagger_ms,
        "max_transient_retries": model_profile.default_transient_retries,
        "model": model,
        "request_protocol": request_protocol,
        "reasoning_effort": reasoning_effort,
        "execution_mode": execution_mode,
        "automatic_repair": automatic_repair,
        "max_automatic_repairs": model_profile.max_automatic_repairs,
        # A manual response cannot invoke a provider's built-in search tool.
        "allow_web_search": False,
        "use_source_for_example_sentences": use_source_examples,
        "include_source_context_nuance": bool(
            use_source_examples
            and "context" in direction_keys
            and preferences.get("source_include_context_nuance", False)),
        "source_card_directions": list(direction_keys),
        "separate_source_decks": bool(
            preferences.get("source_separate_decks", False)),
        "pipeline": pipeline,
        "pipelines": (pipeline,),
        # The user requested every unique token, including learned words.
        "anki_exclusions": (),
        "exclude_anki": False,
        "output_deck_name": f"Vocabulary from {SOURCE_TITLE}",
        "keep_imported_deck": True,
        "move_cards_after_import": False,
        "delete_imported_deck": False,
    }


def create_journey_job(
        backend,
        *,
        pipeline_path=None,
        preferences_path=None,
        pipeline_id=None):
    request = build_journey_request(
        pipeline_store.load_pipelines(pipeline_path),
        gui_preferences.load_preferences(preferences_path),
        pipeline_id=pipeline_id)
    estimate = backend.estimate(request)
    candidate_count = estimate.get("candidate_count")
    if candidate_count != EXPECTED_UNIQUE_TOKEN_COUNT:
        raise RuntimeError(
            "Refusing to create a partial Journey to the West job: the "
            f"latest audited build planned {candidate_count!r} candidates, "
            f"not {EXPECTED_UNIQUE_TOKEN_COUNT:,}.")

    # The explicit offline marker authorizes persistence without pretending
    # that paid dispatch was approved.  This script never constructs a runner
    # or supplies a provider request callable.
    snapshot = backend.create_job({
        **request,
        "estimate": estimate,
        "paid_confirmed": False,
        "manual_offline_responses": True,
    })
    (snapshot.path / MANUAL_RESPONSES_DIRECTORY).mkdir(exist_ok=True)
    (snapshot.path / MANUAL_PACKETS_DIRECTORY).mkdir(exist_ok=True)
    return snapshot, estimate


def _manual_response_path(snapshot, chunk_id):
    directory = snapshot.path / MANUAL_RESPONSES_DIRECTORY
    directory.mkdir(exist_ok=True)
    return directory / f"{chunk_id}.json"


def missing_chunks(backend, job_id):
    _require_manual_job(backend, job_id)
    backend.jobs.recover_interrupted(job_id)
    snapshot = backend.jobs.refresh(job_id)
    missing = []
    for status in snapshot.chunks:
        if status["status"] == CHUNK_SUCCEEDED:
            continue
        chunk_id = status["chunk_id"]
        chunk = backend.jobs.load_chunk(job_id, chunk_id)
        missing.append({
            "chunk_id": chunk_id,
            "index": chunk.index,
            "total": chunk.total,
            "status": status["status"],
            "attempts": status["attempts"],
            "start_rank": chunk.start_rank,
            "end_rank": chunk.end_rank,
            "word_count": len(chunk.words),
            "terms": [word.surface for word in chunk.words],
            "response_path": str(
                _manual_response_path(snapshot, chunk_id)),
        })
    return snapshot, tuple(missing)


def _manifest(backend, job_id):
    path = backend.jobs.snapshot(job_id).path / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("job_id") != job_id:
        raise ValueError("Source job manifest and path disagree.")
    return value


def _require_manual_job(backend, job_id):
    manifest = _manifest(backend, job_id)
    if not manifest.get("request_metadata", {}).get(
            "manual_offline_responses", False):
        raise ValueError(
            "This is not a manually authored offline source job.")
    return manifest


def _job_validation_context(backend, job_id):
    manifest = _require_manual_job(backend, job_id)
    metadata = manifest.get("request_metadata", {})
    pipeline = pipeline_store.pipeline_from_mapping(
        metadata.get("pipeline"))
    contract = backend.jobs.load_request_contract(job_id)
    if contract is None:
        raise ValueError(
            "The manual workflow requires a job with a frozen request "
            "contract.")
    translation_memory = (
        metadata.get("translation_memory_by_chunk", {})
        if metadata.get("automatic_repair", False)
        else {})
    if not isinstance(translation_memory, dict):
        raise ValueError("Saved source translation memory is malformed.")
    validator = SourceWorkflowController._response_validator(
        pipeline,
        contract,
        translation_memory)
    return pipeline, contract, translation_memory, validator


def write_authoring_packet(backend, job_id, chunk_id, output_path=None):
    snapshot = backend.jobs.snapshot(job_id)
    chunk = backend.jobs.load_chunk(job_id, chunk_id)
    _pipeline, contract, translation_memory, _validator = (
        _job_validation_context(backend, job_id))
    chunk_memory = translation_memory.get(chunk_id, {})
    packet = {
        "job_id": job_id,
        "chunk_id": chunk_id,
        "status": backend.jobs.chunk_status(job_id, chunk_id)["status"],
        "raw_response_path": str(
            _manual_response_path(snapshot, chunk_id)),
        "instructions": contract["composed_prompt"],
        "chunk_input": json.loads(render_chunk_input(
            chunk,
            protocol_version=contract["schema_version"],
            source_context_translation_memory=chunk_memory)),
        "response_format": (
            SourceWorkflowController._response_format_for_chunk(
                contract,
                chunk)),
    }
    output_path = Path(output_path or (
        snapshot.path
        / MANUAL_PACKETS_DIRECTORY
        / f"{chunk_id}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def _read_raw_response(path):
    if str(path) == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def ingest_manual_response(
        backend,
        job_id,
        chunk_id,
        raw_response_path=None,
        *,
        replace_succeeded=False):
    """Strictly validate, then checkpoint one human-authored response."""
    snapshot = backend.jobs.snapshot(job_id)
    raw_response_path = (
        raw_response_path
        if raw_response_path is not None
        else _manual_response_path(snapshot, chunk_id))
    raw_text = _read_raw_response(raw_response_path)
    chunk = backend.jobs.load_chunk(job_id, chunk_id)
    _pipeline, _contract, _memory, validator = (
        _job_validation_context(backend, job_id))

    # Validation precedes begin_attempt, so malformed or semantically invalid
    # drafts never alter durable job state or consume an attempt number.
    validated = validator(raw_text, chunk)
    backend.jobs.recover_interrupted(job_id)

    with backend.jobs.chunk_lease(
            job_id,
            chunk_id,
            blocking=False) as acquired:
        if not acquired:
            raise RuntimeError(
                "Another process currently owns this source chunk.")
        status = backend.jobs.chunk_status(job_id, chunk_id)
        if status["status"] == CHUNK_SUCCEEDED:
            attempt_path = backend.jobs.latest_attempt_path(
                job_id,
                chunk_id)
            raw_path = (
                attempt_path / "raw.txt"
                if attempt_path is not None
                else None)
            if (
                    raw_path is not None
                    and raw_path.is_file()
                    and raw_path.read_text(encoding="utf-8") == raw_text):
                return backend.jobs.refresh(job_id), False
            if not replace_succeeded:
                raise ValueError(
                    "This chunk already succeeded with a different retained "
                    "response. Pass --replace-succeeded to retain the old "
                    "attempt and checkpoint this validated revision as a new "
                    "attempt.")

        attempt, attempt_path = backend.jobs.begin_attempt(
            job_id,
            chunk_id)
        backend.jobs._set_chunk_status(
            job_id,
            chunk_id,
            worker="manual-offline")
        backend.jobs.write_raw(attempt_path, raw_text)
        backend.jobs.write_validated(attempt_path, validated)
        completed_at = _utc_now()
        backend.jobs._set_chunk_status(
            job_id,
            chunk_id,
            status=CHUNK_SUCCEEDED,
            worker="manual-offline",
            last_error=None,
            completed_at=completed_at)
        backend.jobs.event(
            job_id,
            chunk_id=chunk_id,
            status=CHUNK_SUCCEEDED,
            message=(
                "Strictly validated and ingested a manually authored "
                + (
                    "offline revision."
                    if status["status"] == CHUNK_SUCCEEDED
                    else "offline response.")),
            attempt=attempt)
    return backend.jobs.refresh(job_id), True


def refresh_combined(backend, job_id):
    _require_manual_job(backend, job_id)
    snapshot = backend.jobs.refresh(job_id)
    return snapshot, snapshot.path / "combined.json"


def _forbid_openai_client(*_args, **_kwargs):
    raise AssertionError(
        "The offline manual workflow must never construct an OpenAI client.")


def finalize_job(
        backend,
        job_id,
        *,
        confirm_import=False,
        media_staging_root=None,
        anki_timeout=300):
    """Run the existing package/audio/import finalizer, but only if complete."""
    _require_manual_job(backend, job_id)
    snapshot = backend.jobs.refresh(job_id)
    if snapshot.overall_status != "completed":
        missing_count = sum(
            count
            for status, count in snapshot.counts.items()
            if status != CHUNK_SUCCEEDED)
        raise RuntimeError(
            "Refusing to package or import an incomplete job: "
            f"{missing_count:,} chunk(s) have not succeeded.")
    if not confirm_import:
        raise PermissionError(
            "Finalization creates audio, packages the deck, and imports it "
            "through AnkiConnect. Re-run with --confirm-import.")

    anki_timeout = _positive_integer(
        anki_timeout,
        "AnkiConnect timeout")
    with ExitStack() as stack:
        package_creator = source_deck.create_source_package
        if media_staging_root is not None:
            media_staging_root = Path(media_staging_root).resolve()
            if not media_staging_root.is_dir():
                raise ValueError(
                    "The media staging root must be an existing directory: "
                    f"{media_staging_root}")
            media_directory = Path(stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=".autoanki-source-audio-",
                    dir=media_staging_root)))
            package_creator = partial(
                source_deck.create_source_package,
                audio_media_directory=media_directory)

        anki_client = anki_integration.AnkiConnectClient(
            timeout=anki_timeout)
        package_importer = partial(
            source_deck.import_source_package,
            client=anki_client)
        controller = SourceWorkflowController(
            backend=backend,
            openai_client_factory=_forbid_openai_client,
            package_creator=package_creator,
            package_importer=package_importer)
        pipeline = controller._pipeline_for_job(job_id)
        controller._require_enhanced_audio_ready(pipeline)
        return controller._finalize(job_id, pipeline)


def _backend(args):
    return SourceGenerationBackend(
        corpus_root=args.corpus_root,
        jobs_root=args.jobs_root)


def _print_json(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _command_create(args):
    snapshot, estimate = create_journey_job(
        _backend(args),
        pipeline_path=args.pipeline_config,
        preferences_path=args.preferences,
        pipeline_id=args.pipeline_id)
    _print_json({
        "job_id": snapshot.job_id,
        "job_path": str(snapshot.path),
        "candidate_count": estimate["candidate_count"],
        "chunk_count": len(snapshot.chunks),
        "manual_responses_path": str(
            snapshot.path / MANUAL_RESPONSES_DIRECTORY),
        "status": snapshot.overall_status,
    })


def _command_missing(args):
    snapshot, missing = missing_chunks(_backend(args), args.job_id)
    selected = missing[:args.limit] if args.limit is not None else missing
    if args.json:
        _print_json({
            "job_id": args.job_id,
            "overall_status": snapshot.overall_status,
            "missing_count": len(missing),
            "chunks": selected,
        })
        return
    print(
        f"{len(missing):,} of {len(snapshot.chunks):,} chunks missing "
        f"({snapshot.overall_status})")
    for item in selected:
        print(
            f"{item['chunk_id']}  {item['status']}  "
            f"r{item['start_rank']}-r{item['end_rank']}  "
            + " ".join(item["terms"]))
        print(f"  {item['response_path']}")


def _command_packet(args):
    path = write_authoring_packet(
        _backend(args),
        args.job_id,
        args.chunk_id,
        args.output)
    print(path)


def _command_ingest(args):
    snapshot, created = ingest_manual_response(
        _backend(args),
        args.job_id,
        args.chunk_id,
        args.raw_json,
        replace_succeeded=args.replace_succeeded)
    _print_json({
        "job_id": args.job_id,
        "chunk_id": args.chunk_id,
        "new_attempt": created,
        "overall_status": snapshot.overall_status,
        "counts": snapshot.counts,
        "combined_path": str(snapshot.path / "combined.json"),
    })


def _command_refresh(args):
    snapshot, path = refresh_combined(_backend(args), args.job_id)
    _print_json({
        "job_id": args.job_id,
        "overall_status": snapshot.overall_status,
        "counts": snapshot.counts,
        "combined_path": str(path),
    })


def _command_finalize(args):
    result = finalize_job(
        _backend(args),
        args.job_id,
        confirm_import=args.confirm_import,
        media_staging_root=args.media_staging_root,
        anki_timeout=args.anki_timeout)
    _print_json(result)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="Override the processed-corpus root (primarily for tests).")
    parser.add_argument(
        "--jobs-root",
        type=Path,
        help="Override the source-generation job root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create",
        help="Create the complete 24,224-token Journey v10 job.")
    create.add_argument("--pipeline-config", type=Path)
    create.add_argument("--preferences", type=Path)
    create.add_argument("--pipeline-id")
    create.set_defaults(handler=_command_create)

    missing = subparsers.add_parser(
        "missing",
        help="List every chunk which has not strictly succeeded.")
    missing.add_argument("job_id")
    missing.add_argument("--limit", type=int)
    missing.add_argument("--json", action="store_true")
    missing.set_defaults(handler=_command_missing)

    packet = subparsers.add_parser(
        "packet",
        help="Write one chunk's exact prompt, input, and JSON schema.")
    packet.add_argument("job_id")
    packet.add_argument("chunk_id")
    packet.add_argument("--output", type=Path)
    packet.set_defaults(handler=_command_packet)

    ingest = subparsers.add_parser(
        "ingest",
        help="Strictly validate and checkpoint one manual JSON response.")
    ingest.add_argument("job_id")
    ingest.add_argument("chunk_id")
    ingest.add_argument(
        "raw_json",
        nargs="?",
        type=Path,
        help=(
            "Response JSON path; defaults to the chunk path shown by "
            "missing. Use - for stdin."))
    ingest.add_argument(
        "--replace-succeeded",
        action="store_true",
        help=(
            "After strict validation, preserve the earlier successful "
            "attempt and checkpoint this different response as a new "
            "successful revision."))
    ingest.set_defaults(handler=_command_ingest)

    refresh = subparsers.add_parser(
        "refresh",
        help="Atomically rebuild combined.json from succeeded chunks.")
    refresh.add_argument("job_id")
    refresh.set_defaults(handler=_command_refresh)

    finalize = subparsers.add_parser(
        "finalize",
        help="Package, synthesize audio, and import only a complete job.")
    finalize.add_argument("job_id")
    finalize.add_argument("--confirm-import", action="store_true")
    finalize.add_argument(
        "--media-staging-root",
        type=Path,
        help=(
            "Create the temporary package-media directory under this "
            "existing path (for example, /dev/shm when disk is tight)."))
    finalize.add_argument(
        "--anki-timeout",
        type=int,
        default=300,
        help="AnkiConnect request timeout in seconds (default: 300).")
    finalize.set_defaults(handler=_command_finalize)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        report = getattr(error, "validation_report", None)
        if isinstance(report, dict):
            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2),
                file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
