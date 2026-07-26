#!/usr/bin/env python3
"""Archived local Ollama compact-v10 evaluator.

This driver deliberately stops after source-job validation.  It never calls
the packaging/finalization path and injects guards that fail if an Anki or
package hook is reached accidentally.

The requested counts are counts of *unique first-seen vocabulary entries*.
For each stage, the driver derives the running-token prefix containing exactly
that many entries, then verifies the saved plan before inference begins.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
from source_generation import (
    SourceGenerationBackend,
    load_processed_source,
)
from source_generation.model_catalog import (
    LOCAL_SOURCE_MODELS,
    QWEN3_14B_SOURCE_MODEL,
    source_model_profile,
)
from source_generation.translation_memory import (
    SourceContextTranslationMemory,
)
from source_workflow import SourceWorkflowController


REPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "evaluations"
    / "local_source_v10")
_TOKEN_OCCURRENCE_PREFIX = "token:"
_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_PROBLEM_FIELDS = (
    "problem_id",
    "code",
    "title",
    "message",
    "path",
    "location",
    "scope",
    "term",
    "field_name",
    "expected",
    "actual",
    "suggestion",
    "overrideable",
    "source_rank",
    "context_id",
    "full_retry_required",
    "exception_type",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_integer(value, *, name):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a positive integer.") from error
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"{name} must be a positive integer.")
    return number


def _nonnegative_integer(value, *, name):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a non-negative integer.") from error
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"{name} must be a non-negative integer.")
    return number


def _counts(value):
    try:
        result = tuple(
            _positive_integer(item.strip(), name="Each stage count")
            for item in str(value).split(",")
            if item.strip())
    except argparse.ArgumentTypeError:
        raise
    if not result:
        raise argparse.ArgumentTypeError(
            "At least one progressive stage count is required.")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise argparse.ArgumentTypeError(
            "Stage counts must be strictly increasing.")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run fail-fast compact-v10 evaluations against a shared local "
            "Ollama model without packaging or importing an Anki deck."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Verify the exact plans and installed model identity; no inference.
  python scripts/evaluate_local_source_v10.py --plan-only \\
      --model ollama/qwen3:14b --counts 1,3,10,30,100

  # Fail-fast production-path evaluation, using Qwen's five-word request cap.
  python scripts/evaluate_local_source_v10.py --execute-local \\
      --model ollama/qwen3:14b --counts 1,3,10,30,100

  # Exercise larger single-request batches deliberately.
  python scripts/evaluate_local_source_v10.py --execute-local \\
      --model ollama/gemma4:12b --counts 1,3,10 --chunk-size 10
""")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--execute-local",
        action="store_true",
        help=(
            "Acknowledge that local inference will consume substantial CPU, "
            "GPU, RAM, and time."))
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Audit exact unique-word prefixes and estimates without running "
            "inference or creating source jobs."))
    parser.add_argument(
        "--source",
        default="daodejing_wang_bi",
        help="Processed corpus key (default: daodejing_wang_bi).")
    parser.add_argument(
        "--model",
        choices=tuple(sorted(LOCAL_SOURCE_MODELS)),
        default=QWEN3_14B_SOURCE_MODEL)
    parser.add_argument(
        "--counts",
        type=_counts,
        default=_counts("1,3"),
        help=(
            "Strictly increasing unique-word stage sizes, for example "
            "1,3,10,30,100 (default: 1,3)."))
    parser.add_argument(
        "--chunk-size",
        type=lambda value: _positive_integer(
            value,
            name="Chunk size"),
        help=(
            "Maximum words per request. By default, use the selected "
            "model's conservative recommended size. A stage smaller than "
            "this value remains one request."))
    parser.add_argument(
        "--pipeline-id",
        help="Saved Card Setup pipeline ID; default: the first saved pipeline.")
    parser.add_argument(
        "--context-mode",
        choices=(
            "sentence",
            "sentence_neighbors",
            "chunk_span",
            "none",
        ),
        default="sentence")
    parser.add_argument(
        "--source-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use retained source passages for contextual examples "
            "(default: enabled)."))
    parser.add_argument(
        "--automatic-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the production automatic-repair loop. Local profiles allow "
            "up to five follow-ups (default: enabled)."))
    parser.add_argument(
        "--max-automatic-repairs",
        type=lambda value: _nonnegative_integer(
            value,
            name="Maximum automatic repairs"),
        help="Override the local model profile's automatic-repair cap.")
    parser.add_argument(
        "--transient-retries",
        type=lambda value: _nonnegative_integer(
            value,
            name="Transient retries"),
        help="Override the local model profile's transient connection retries.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to larger stages after a failed validation stage.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--session-name",
        help=(
            "Stable output-directory name. It must not already exist; "
            "otherwise a timestamped name is generated."))
    return parser.parse_args()


def _session_directory(args):
    if args.session_name is not None:
        if _SAFE_SESSION_NAME.fullmatch(args.session_name) is None:
            raise ValueError(
                "Session names may contain only letters, digits, dots, "
                "underscores, and hyphens.")
        name = args.session_name
    else:
        model_name = re.sub(
            r"[^a-z0-9]+",
            "-",
            args.model.casefold()).strip("-")
        timestamp = datetime.now(
            timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{timestamp}-{model_name}-{uuid.uuid4().hex[:8]}"
    path = (args.output_dir / name).resolve()
    if path.exists():
        raise FileExistsError(
            f"Evaluation session already exists: {path}")
    path.mkdir(parents=True)
    return path


def _pipeline_for_source(source_language_key, pipeline_id):
    pipelines = pipeline_store.load_pipelines()
    if not pipelines:
        pipelines = (pipeline_store.default_pipeline(),)
    if pipeline_id is None:
        pipeline = pipelines[0]
    else:
        matches = tuple(
            pipeline
            for pipeline in pipelines
            if pipeline.pipeline_id == pipeline_id)
        if len(matches) != 1:
            raise ValueError(
                f"No unique saved pipeline has ID {pipeline_id!r}.")
        pipeline = matches[0]
    settings = pipeline_store.get_language_settings(
        pipeline,
        source_language_key)
    settings_by_key = {
        item.language_key: item
        for item in pipeline.language_settings
    }
    settings_by_key[settings.language_key] = settings
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        tuple(settings_by_key.values()),
        active_language_key=source_language_key)


def _first_occurrence_token_index(word):
    occurrence_id = word.first_occurrence_id
    if (
            not isinstance(occurrence_id, str)
            or not occurrence_id.startswith(_TOKEN_OCCURRENCE_PREFIX)
            or not occurrence_id[
                len(_TOKEN_OCCURRENCE_PREFIX):].isdecimal()):
        raise ValueError(
            f"Source rank {word.rank} has an invalid first occurrence ID.")
    index = int(occurrence_id[len(_TOKEN_OCCURRENCE_PREFIX):])
    if index < 1:
        raise ValueError(
            f"Source rank {word.rank} has an invalid token index.")
    return index


def _selection_for_unique_count(source, count):
    words = source.build.unique_words
    if count > len(words):
        raise ValueError(
            f"The source contains only {len(words):,} unique words; "
            f"stage {count:,} is impossible.")
    selected = tuple(words[:count])
    expected_ranks = tuple(range(1, count + 1))
    actual_ranks = tuple(word.rank for word in selected)
    if actual_ranks != expected_ranks:
        raise ValueError(
            "The processed source vocabulary does not have contiguous "
            "first-occurrence ranks.")
    prefix_token_limit = _first_occurrence_token_index(selected[-1])
    if (
            count < len(words)
            and _first_occurrence_token_index(words[count])
            <= prefix_token_limit):
        raise ValueError(
            "The next unique word shares or precedes the selected running-"
            "token boundary, so this source cannot express the requested "
            "unique count as an exact prefix.")
    return {
        "unique_count": count,
        "prefix_token_limit": prefix_token_limit,
        "expected_ranks": expected_ranks,
        "expected_terms": tuple(word.surface for word in selected),
    }


def _request_for_stage(
        args,
        source,
        pipeline,
        selection,
        *,
        chunk_size):
    if args.source_examples and args.context_mode == "none":
        raise ValueError(
            "Source examples require a retained source context.")
    request = {
        "source_key": source.key,
        "source_name": source.title,
        "source_language_key": (
            source.build.snapshot.source_language_key),
        "run_path": str(source.run_path),
        "chunk_size": min(selection["unique_count"], chunk_size),
        "source_prefix_token_limit": (
            selection["prefix_token_limit"]),
        "context_mode": args.context_mode,
        "concurrency": 1,
        "request_stagger_ms": 0,
        "model": args.model,
        "request_protocol": "v10",
        "reasoning_effort": "none",
        "execution_mode": "standard",
        "automatic_repair": args.automatic_repair,
        "excluded_words": (),
        "anki_exclusions": (),
        "exclude_anki": False,
        "allow_web_search": False,
        "use_source_for_example_sentences": args.source_examples,
        "pipeline": pipeline,
        "output_deck_name": f"Vocabulary from {source.title}",
        "keep_imported_deck": True,
        "move_cards_after_import": False,
        "delete_imported_deck": False,
    }
    if args.max_automatic_repairs is not None:
        request["max_automatic_repairs"] = (
            args.max_automatic_repairs)
    if args.transient_retries is not None:
        request["max_transient_retries"] = args.transient_retries
    return request


def _verify_plan(plan, selection):
    ranks = tuple(
        word.rank
        for chunk in plan.chunks
        for word in chunk.words)
    terms = tuple(
        word.surface
        for chunk in plan.chunks
        for word in chunk.words)
    problems = []
    if ranks != selection["expected_ranks"]:
        problems.append({
            "code": "evaluation_rank_selection_mismatch",
            "expected": list(selection["expected_ranks"]),
            "actual": list(ranks),
        })
    if terms != selection["expected_terms"]:
        problems.append({
            "code": "evaluation_term_selection_mismatch",
            "expected": list(selection["expected_terms"]),
            "actual": list(terms),
        })
    if plan.prefix_unique_word_count != selection["unique_count"]:
        problems.append({
            "code": "evaluation_prefix_unique_count_mismatch",
            "expected": selection["unique_count"],
            "actual": plan.prefix_unique_word_count,
        })
    if plan.word_count != selection["unique_count"]:
        problems.append({
            "code": "evaluation_selected_word_count_mismatch",
            "expected": selection["unique_count"],
            "actual": plan.word_count,
        })
    if plan.excluded_word_count != 0:
        problems.append({
            "code": "evaluation_unexpected_exclusions",
            "expected": 0,
            "actual": plan.excluded_word_count,
        })
    if problems:
        raise ValueError(
            "Exact unique-word plan verification failed: "
            + json.dumps(
                problems,
                ensure_ascii=False,
                separators=(",", ":")))
    return {
        "plan_id": plan.plan_id,
        "source_prefix_token_count": plan.source_prefix_token_count,
        "prefix_unique_word_count": plan.prefix_unique_word_count,
        "selected_word_count": plan.word_count,
        "excluded_word_count": plan.excluded_word_count,
        "chunk_ids": [chunk.chunk_id for chunk in plan.chunks],
        "chunk_word_counts": [
            len(chunk.words)
            for chunk in plan.chunks
        ],
        "ranks": list(ranks),
        "terms": list(terms),
    }


def _problem_summary(problem):
    if not isinstance(problem, dict):
        return {
            "code": "malformed_validation_problem",
            "actual": repr(problem),
        }
    return {
        key: problem[key]
        for key in _PROBLEM_FIELDS
        if key in problem
    }


def _error_summary(error):
    if not isinstance(error, dict):
        return None
    return {
        key: error[key]
        for key in (
            "type",
            "message",
            "transient",
            "status_code",
            "timestamp",
        )
        if key in error
    }


def _response_summary(response):
    if not isinstance(response, dict):
        return None
    return {
        key: response[key]
        for key in (
            "response_id",
            "model",
            "status",
            "service_tier",
            "received_at",
            "usage",
        )
        if key in response
    }


def _attempt_summary(job_path, chunk_id, attempt):
    error = (
        attempt.get("error")
        if isinstance(attempt.get("error"), dict)
        else attempt.get("recovery_error"))
    validation = (
        error.get("validation")
        if isinstance(error, dict)
        and isinstance(error.get("validation"), dict)
        else {})
    problems = [
        _problem_summary(problem)
        for problem in validation.get("problems", ())
    ]
    number = int(attempt["attempt"])
    attempt_path = (
        Path(job_path)
        / "chunks"
        / chunk_id
        / "attempts"
        / f"{number:04d}")
    raw_path = attempt_path / "raw.txt"
    repaired_path = attempt_path / "repaired_raw.txt"
    return {
        "attempt": number,
        "artifact_path": str(attempt_path),
        "raw_sha256": (
            hashlib.sha256(raw_path.read_bytes()).hexdigest()
            if raw_path.is_file()
            else None),
        "repaired_raw_sha256": (
            hashlib.sha256(repaired_path.read_bytes()).hexdigest()
            if repaired_path.is_file()
            else None),
        "response": _response_summary(attempt.get("response")),
        "local_dispatch": attempt.get("local_dispatch"),
        "automatic_repair_dispatch": attempt.get(
            "automatic_repair_dispatch"),
        "repair_scope": attempt.get("repair_scope"),
        "local_repair": attempt.get("local_repair"),
        "error": _error_summary(error),
        "validation_problem_count": len(problems),
        "validation_problems": problems,
        "validated_card_count": (
            len(attempt["validated"].get("cards", ()))
            if isinstance(attempt.get("validated"), dict)
            else None),
    }


def _collect_stage_report(
        backend,
        snapshot,
        pipeline,
        source,
        selection,
        plan_summary,
        estimate):
    job_path = snapshot.path
    combined = _read_json(job_path / "combined.json")
    manifest = _read_json(job_path / "manifest.json")
    chunks = []
    historical_problem_codes = Counter()
    final_problem_codes = Counter()
    all_problems = []
    automatic_dispatches = 0
    for chunk_id in backend.jobs.chunk_ids(snapshot.job_id):
        inspection = backend.jobs.inspect_chunk(
            snapshot.job_id,
            chunk_id)
        attempts = [
            _attempt_summary(job_path, chunk_id, attempt)
            for attempt in inspection["attempts"]
        ]
        for attempt in attempts:
            for problem in attempt["validation_problems"]:
                code = problem.get(
                    "code",
                    "uncoded_validation_problem")
                historical_problem_codes[code] += 1
                all_problems.append({
                    "chunk_id": chunk_id,
                    "attempt": attempt["attempt"],
                    **problem,
                })
            if attempt["automatic_repair_dispatch"] is not None:
                automatic_dispatches += 1
        latest_problems = (
            attempts[-1]["validation_problems"]
            if attempts
            else [])
        for problem in latest_problems:
            final_problem_codes[
                problem.get(
                    "code",
                    "uncoded_validation_problem")
            ] += 1
        chunks.append({
            "chunk_id": chunk_id,
            "status": inspection["status"].get("status"),
            "attempt_count": inspection["status"].get("attempts", 0),
            "last_error": inspection["status"].get("last_error"),
            "attempts": attempts,
        })

    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field
    cards = (
        combined.get("cards", ())
        if isinstance(combined.get("cards"), list)
        else ())
    terms = [
        card.get(term_field)
        for card in cards
        if isinstance(card, dict)
        and isinstance(card.get(term_field), str)
    ]
    term_counts = Counter(terms)
    expected_terms = selection["expected_terms"]
    expected_term_set = set(expected_terms)
    actual_term_set = set(terms)
    missing_terms = [
        term
        for term in expected_terms
        if term not in actual_term_set
    ]
    unexpected_terms = sorted(
        actual_term_set - expected_term_set)
    additional_sense_counts = {
        term: max(0, term_counts.get(term, 0) - 1)
        for term in expected_terms
    }
    additional_sense_count = sum(
        additional_sense_counts.values())
    terms_with_additional_senses = sum(
        count > 0
        for count in additional_sense_counts.values())
    quality_warnings = []
    if (
            selection["unique_count"] >= 3
            and additional_sense_count == 0):
        quality_warnings.append({
            "code": "no_additional_senses_detected",
            "message": (
                "The structurally valid result supplied no additional "
                "dictionary senses. This is reported as a semantic quality "
                "warning rather than a language-independent validation "
                "failure."),
        })
    exact_coverage = (
        not missing_terms
        and not unexpected_terms
        and len(actual_term_set) == selection["unique_count"])
    final_validation_passed = (
        snapshot.overall_status == "completed"
        and combined.get("complete") is True
        and exact_coverage
        and len(cards) >= selection["unique_count"]
        and all(
            chunk["status"] == "succeeded"
            for chunk in chunks))
    package_path = job_path / "source_deck.apkg"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "local_source_v10_evaluation_stage",
        "created_at": _utc_now(),
        "status": (
            "passed"
            if final_validation_passed
            else "failed"),
        "final_validation_passed": final_validation_passed,
        "anki_mutation_performed": False,
        "package_or_import_attempted": False,
        "unexpected_package_exists": package_path.is_file(),
        "source": {
            "key": source.key,
            "title": source.title,
            "build_id": source.build_id,
            "run_path": str(source.run_path),
            "language_key": (
                source.build.snapshot.source_language_key),
        },
        "selection": {
            "requested_unique_count": selection["unique_count"],
            "source_prefix_token_limit": (
                selection["prefix_token_limit"]),
            "expected_ranks": list(selection["expected_ranks"]),
            "expected_terms": list(expected_terms),
        },
        "plan": plan_summary,
        "pipeline_id": pipeline.pipeline_id,
        "model": manifest.get(
            "request_metadata",
            {}).get("model"),
        "provider": manifest.get(
            "request_metadata",
            {}).get("provider"),
        "model_runtime_snapshot": manifest.get(
            "request_metadata",
            {}).get("model_runtime_snapshot"),
        "estimate": {
            key: estimate.get(key)
            for key in (
                "request_count",
                "word_count",
                "model",
                "provider",
                "local_model",
                "estimated_cost_usd",
                "estimated_cost_aud",
                "largest_request_input_tokens",
                "largest_request_output_tokens",
                "largest_request_high_total_tokens",
                "request_limit_warning",
                "translation_memory_hit_count",
            )
        },
        "job": {
            "job_id": snapshot.job_id,
            "path": str(job_path),
            "overall_status": snapshot.overall_status,
            "combined_complete": combined.get("complete"),
            "completed_chunk_count": combined.get(
                "completed_chunk_count"),
            "total_chunk_count": combined.get(
                "total_chunk_count"),
        },
        "output": {
            "canonical_card_count": len(cards),
            "unique_term_count": len(actual_term_set),
            "term_field": term_field,
            "term_counts": {
                term: term_counts[term]
                for term in expected_terms
                if term in term_counts
            },
            "additional_sense_count": additional_sense_count,
            "additional_sense_counts_by_term": (
                additional_sense_counts),
            "terms_with_additional_senses": (
                terms_with_additional_senses),
            "terms_with_additional_senses_ratio": (
                terms_with_additional_senses
                / selection["unique_count"]),
            "quality_warnings": quality_warnings,
            "missing_terms": missing_terms,
            "unexpected_terms": unexpected_terms,
            "exact_unique_term_coverage": exact_coverage,
        },
        "validation": {
            "automatic_repair_dispatch_count": automatic_dispatches,
            "historical_problem_code_counts": dict(
                sorted(historical_problem_codes.items())),
            "final_problem_code_counts": dict(
                sorted(final_problem_codes.items())),
            "problems": all_problems,
            "chunks": chunks,
        },
        "usage": backend.jobs.usage_summary(snapshot.job_id),
    }


def _planned_stage_report(
        args,
        source,
        pipeline,
        selection,
        plan_summary,
        estimate):
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "local_source_v10_evaluation_stage",
        "created_at": _utc_now(),
        "status": "planned",
        "final_validation_passed": None,
        "anki_mutation_performed": False,
        "package_or_import_attempted": False,
        "source": {
            "key": source.key,
            "title": source.title,
            "build_id": source.build_id,
            "run_path": str(source.run_path),
            "language_key": (
                source.build.snapshot.source_language_key),
        },
        "selection": {
            "requested_unique_count": selection["unique_count"],
            "source_prefix_token_limit": (
                selection["prefix_token_limit"]),
            "expected_ranks": list(selection["expected_ranks"]),
            "expected_terms": list(selection["expected_terms"]),
        },
        "plan": plan_summary,
        "pipeline_id": pipeline.pipeline_id,
        "model": args.model,
        "model_runtime_snapshot": estimate.get(
            "model_runtime_snapshot"),
        "estimate": estimate,
    }


def _forbid_anki_or_packaging(*_args, **_kwargs):
    raise RuntimeError(
        "The local evaluation driver forbids Anki and package operations.")


def main():
    args = _parse_args()
    profile = source_model_profile(args.model)
    chunk_size = args.chunk_size or profile.recommended_chunk_size
    if (
            args.max_automatic_repairs is not None
            and args.max_automatic_repairs
            > profile.max_automatic_repairs):
        raise SystemExit(
            f"{args.model} allows at most "
            f"{profile.max_automatic_repairs} automatic repairs.")

    session_path = _session_directory(args)
    source = load_processed_source(args.source)
    source_language_key = (
        source.build.snapshot.source_language_key)
    pipeline = _pipeline_for_source(
        source_language_key,
        args.pipeline_id)
    backend = SourceGenerationBackend(
        jobs_root=session_path / "jobs",
        translation_memory=SourceContextTranslationMemory(
            root=session_path / "translation_memory"))
    controller = SourceWorkflowController(
        backend=backend,
        anki_client_factory=_forbid_anki_or_packaging,
        package_creator=_forbid_anki_or_packaging,
        package_importer=_forbid_anki_or_packaging)
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "local_source_v10_progressive_evaluation",
        "created_at": _utc_now(),
        "session_path": str(session_path),
        "mode": (
            "execute_local"
            if args.execute_local
            else "plan_only"),
        "source": args.source,
        "source_build_id": source.build_id,
        "model": args.model,
        "pipeline_id": pipeline.pipeline_id,
        "counts": list(args.counts),
        "maximum_words_per_request": chunk_size,
        "automatic_repair": args.automatic_repair,
        "maximum_automatic_repairs": (
            args.max_automatic_repairs
            if args.max_automatic_repairs is not None
            else profile.max_automatic_repairs),
        "anki_mutation_performed": False,
        "stages": [],
        "all_passed": None,
        "stopped_after_count": None,
    }
    _atomic_write_json(session_path / "summary.json", summary)

    all_passed = True
    for stage_number, count in enumerate(args.counts, start=1):
        selection = _selection_for_unique_count(
            source,
            count)
        request = _request_for_stage(
            args,
            source,
            pipeline,
            selection,
            chunk_size=chunk_size)
        _loaded, plan = backend._plan(request)
        plan_summary = _verify_plan(plan, selection)
        print(
            f"[{stage_number}/{len(args.counts)}] "
            f"{count:,} unique words -> running-token prefix "
            f"{selection['prefix_token_limit']:,}; "
            f"{len(plan.chunks)} request(s).",
            flush=True)
        stage_path = (
            session_path
            / "stages"
            / f"{stage_number:04d}-{count:06d}-unique")
        stage_path.mkdir(parents=True)
        try:
            estimate = controller.estimate(request)
            if args.plan_only:
                report = _planned_stage_report(
                    args,
                    source,
                    pipeline,
                    selection,
                    plan_summary,
                    estimate)
                stage_passed = (
                    estimate.get("request_limit_warning") is None)
            else:
                authorized_request = {
                    **request,
                    "model_runtime_snapshot": estimate.get(
                        "model_runtime_snapshot"),
                    "estimate": estimate,
                    "paid_confirmed": True,
                }
                snapshot = backend.create_job(
                    authorized_request)
                client = controller._client_for_model(
                    args.model,
                    estimate.get("model_runtime_snapshot"))
                completed = controller._run_job(
                    snapshot.job_id,
                    pipeline,
                    client)
                report = _collect_stage_report(
                    backend,
                    completed,
                    pipeline,
                    source,
                    selection,
                    plan_summary,
                    estimate)
                stage_passed = report[
                    "final_validation_passed"]
        except Exception as error:
            report = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "kind": "local_source_v10_evaluation_stage",
                "created_at": _utc_now(),
                "status": "driver_error",
                "final_validation_passed": False,
                "anki_mutation_performed": False,
                "package_or_import_attempted": False,
                "source": {
                    "key": source.key,
                    "build_id": source.build_id,
                    "run_path": str(source.run_path),
                },
                "selection": {
                    "requested_unique_count": count,
                    "source_prefix_token_limit": (
                        selection["prefix_token_limit"]),
                },
                "plan": plan_summary,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            stage_passed = False

        report_path = stage_path / "report.json"
        _atomic_write_json(report_path, report)
        summary["stages"].append({
            "stage_number": stage_number,
            "requested_unique_count": count,
            "source_prefix_token_limit": (
                selection["prefix_token_limit"]),
            "report_path": str(report_path),
            "status": report["status"],
            "passed": stage_passed,
            "job_id": (
                report.get("job", {}).get("job_id")
                if isinstance(report.get("job"), dict)
                else None),
        })
        if not stage_passed:
            all_passed = False
            summary["stopped_after_count"] = count
        summary["all_passed"] = all_passed
        summary["updated_at"] = _utc_now()
        _atomic_write_json(session_path / "summary.json", summary)
        print(
            f"  {report['status']}: {report_path}",
            flush=True)
        if not stage_passed and not args.continue_on_failure:
            break

    print(f"Session report: {session_path / 'summary.json'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
