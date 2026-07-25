#!/usr/bin/env python3
"""Explicit paid smoke test for AutoAnki's compact source protocol."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation import (
    ContextMode,
    PaidResponse,
    SourceGenerationConfig,
    build_source_request_contract,
    estimate_text_tokens,
    load_processed_source,
    make_pipeline_response_validator,
    plan_source_generation,
    price_source_usage,
    render_chunk_input,
)
from source_workflow import _paid_response
from source_generation.repair import (
    CompactRepairScope,
    build_compact_repair_chunk,
)


def _pipeline_for_source(source_language_key):
    pipelines = pipeline_store.load_pipelines()
    pipeline = (
        pipelines[0]
        if pipelines
        else pipeline_store.default_pipeline())
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


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2)
        + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _request_options(contract, chunk):
    return {
        "model": contract["model"],
        "input": (
            contract["composed_prompt"]
            + render_chunk_input(
                chunk,
                protocol_version=9)),
        "reasoning": contract["reasoning"],
        "text": {"format": contract["response_format"]},
        "max_output_tokens": contract[
            "max_output_tokens_by_chunk"][chunk.chunk_id],
        "prompt_cache_key": contract["prompt_cache_key"],
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-paid",
        action="store_true",
        help="Required acknowledgement that this calls the paid API.")
    parser.add_argument(
        "--source",
        default="daodejing_wang_bi")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--count",
        type=int,
        choices=(1, 3, 10, 100))
    selection.add_argument(
        "--ranks",
        help=(
            "Comma-separated source ranks for a targeted test; each request "
            "must remain within the user's paid-test authorization."))
    parser.add_argument(
        "--reasoning",
        choices=("none", "low"),
        required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "output"
            / "evaluations"
            / "source_v9"))
    parser.add_argument(
        "--force",
        action="store_true")
    return parser.parse_args()


def _target_ranks(value):
    if value is None:
        return None
    try:
        ranks = tuple(sorted({
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }))
    except ValueError as error:
        raise SystemExit("--ranks must contain positive integers.") from error
    if not ranks or ranks[0] < 1:
        raise SystemExit("--ranks must contain positive integers.")
    return ranks


def main():
    args = _parse_args()
    if not args.execute_paid:
        raise SystemExit(
            "Refusing the API call without --execute-paid.")
    target_ranks = _target_ranks(args.ranks)
    if target_ranks is None:
        selection_name = f"{args.count:03d}"
        planned_chunk_size = args.count
    else:
        selection_name = "targeted_" + "-".join(
            f"{rank:03d}"
            for rank in target_ranks)
        planned_chunk_size = max(target_ranks)
    result_path = (
        args.output_dir
        / f"{args.source}_{selection_name}_{args.reasoning}.json")
    if result_path.is_file() and not args.force:
        print(f"Retained result already exists: {result_path}")
        return 0
    if result_path.is_file():
        sequence = 1
        while True:
            archive = result_path.with_name(
                f"{result_path.stem}.previous_{sequence:02d}.json")
            if not archive.exists():
                result_path.replace(archive)
                break
            sequence += 1

    source = load_processed_source(args.source)
    source_language_key = (
        source.build.snapshot.source_language_key)
    pipeline = _pipeline_for_source(source_language_key)
    config = SourceGenerationConfig(
        source_key=source.key,
        chunk_size=planned_chunk_size,
        context_mode=ContextMode.SENTENCE,
        concurrency=1,
        request_stagger_ms=0,
        max_transient_retries=0,
        request_protocol="v9",
        reasoning_effort=args.reasoning,
        execution_mode="standard")
    plan = plan_source_generation(source, config)
    chunk = plan.chunks[0]
    if target_ranks is None:
        if len(chunk.words) != args.count:
            raise RuntimeError(
                "The first planned chunk does not contain the requested "
                "count.")
    else:
        words_by_rank = {
            word.rank: word
            for word in chunk.words
        }
        missing_ranks = set(target_ranks) - set(words_by_rank)
        if missing_ranks:
            raise RuntimeError(
                "Target ranks are absent from the planned source: "
                + ", ".join(str(rank) for rank in sorted(missing_ranks)))
        context_ids = {
            words_by_rank[rank].context_id
            for rank in target_ranks
            if words_by_rank[rank].context_id is not None
        }
        ordered_context_ids = tuple(
            context.context_id
            for context in chunk.contexts
            if context.context_id in context_ids)
        chunk = build_compact_repair_chunk(
            chunk,
            CompactRepairScope(
                replace_ranks=target_ranks,
                replace_context_ids=(),
                request_ranks=target_ranks,
                request_context_ids=ordered_context_ids))
    contract = build_source_request_contract(
        pipeline,
        chunks=(chunk,),
        use_source_for_example_sentences=True,
        protocol_version=9,
        reasoning_effort=args.reasoning)
    options = _request_options(contract, chunk)
    api_key = process_text.get_api_key()
    if not api_key:
        raise process_text.MissingAPIKeyError(
            "No OpenAI API key is configured.")
    client = process_text.OpenAI(
        api_key=api_key,
        max_retries=0,
        timeout=900.0)
    response = client.responses.create(**options)
    response_error = None
    try:
        paid = _paid_response(response)
    except Exception as error:
        paid = getattr(error, "paid_response", None)
        if not isinstance(paid, PaidResponse):
            raise
        response_error = {
            "type": type(error).__name__,
            "message": str(error),
        }
    validator = make_pipeline_response_validator(
        pipeline,
        use_compact_source_results=True)
    valid = True
    validation_error = None
    if response_error is not None:
        valid = False
        validated = None
        validation_error = response_error
    else:
        try:
            validated = validator(
                paid.raw_text,
                chunk)
        except Exception as error:
            valid = False
            validated = None
            validation_error = {
                "type": type(error).__name__,
                "message": str(error),
                "validation_report": getattr(
                    error,
                    "validation_report",
                    None),
            }

    try:
        parsed = json.loads(paid.raw_text)
    except (TypeError, json.JSONDecodeError):
        parsed = paid.raw_text
    term_results = (
        parsed.get(
            process_text.SOURCE_TERM_RESULTS_KEY,
            [])
        if isinstance(parsed, dict)
        else [])
    additional_senses = sum(
        len(item.get(
            process_text.SOURCE_ADDITIONAL_SENSES_KEY,
            ()))
        for item in term_results
        if isinstance(item, dict))
    usage = paid.usage or {
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "web_search_calls": 0,
    }
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "source_build_id": source.build_id,
        "word_count": len(chunk.words),
        "target_ranks": (
            list(target_ranks)
            if target_ranks is not None
            else None),
        "terms": [
            {
                "rank": word.rank,
                "term": word.surface,
            }
            for word in chunk.words
        ],
        "context_count": len(chunk.contexts),
        "reasoning_effort": args.reasoning,
        "protocol": "v9",
        "model": contract["model"],
        "prompt_estimated_tokens": estimate_text_tokens(
            contract["composed_prompt"]),
        "payload_estimated_tokens": estimate_text_tokens(
            render_chunk_input(
                chunk,
                protocol_version=9)),
        "schema_estimated_tokens": estimate_text_tokens(
            json.dumps(
                contract["response_format"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"))),
        "max_output_tokens": options["max_output_tokens"],
        "response_id": paid.response_id,
        "response_status": paid.status,
        "usage": usage,
        "cost": price_source_usage(
            usage,
            execution_mode="standard"),
        "valid": valid,
        "validation_error": validation_error,
        "card_count": (
            len(validated.get("cards", ()))
            if isinstance(validated, dict)
            else None),
        "term_result_count": len(term_results),
        "additional_sense_count": additional_senses,
        "raw_response": parsed,
    }
    _write_json(result_path, record)
    print(json.dumps(
        {
            key: record[key]
            for key in (
                "word_count",
                "reasoning_effort",
                "prompt_estimated_tokens",
                "payload_estimated_tokens",
                "schema_estimated_tokens",
                "usage",
                "cost",
                "valid",
                "card_count",
                "additional_sense_count",
            )
        },
        ensure_ascii=False,
        indent=2))
    print(f"Saved {result_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
