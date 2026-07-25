#!/usr/bin/env python3
"""Run one explicitly paid selective repair against a saved live evaluation."""

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
    SourceGenerationConfig,
    build_source_request_contract,
    load_processed_source,
    make_pipeline_response_validator,
    plan_source_generation,
    price_source_usage,
    render_chunk_input,
)
from source_generation.repair import (
    CompactRepairScope,
    build_compact_repair_chunk,
    derive_compact_repair_scope,
    merge_compact_repair,
)
from source_workflow import _paid_response


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--reasoning",
        choices=("none", "low"),
        required=True)
    args = parser.parse_args()
    if not args.execute_paid:
        raise SystemExit("Refusing the API call without --execute-paid.")
    base_record = json.loads(
        args.evaluation.read_text(encoding="utf-8"))
    if base_record.get("valid"):
        raise SystemExit("The saved evaluation is already valid.")
    source = load_processed_source(base_record["source"])
    pipeline = _pipeline_for_source(
        source.build.snapshot.source_language_key)
    target_ranks = base_record.get("target_ranks")
    if target_ranks is not None:
        if (
                not isinstance(target_ranks, list)
                or not target_ranks
                or any(
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank < 1
                    for rank in target_ranks)):
            raise SystemExit("The saved target ranks are malformed.")
        target_ranks = tuple(sorted(set(target_ranks)))
        planned_chunk_size = max(target_ranks)
    else:
        planned_chunk_size = base_record["word_count"]
    plan = plan_source_generation(
        source,
        SourceGenerationConfig(
            source_key=source.key,
            chunk_size=planned_chunk_size,
            context_mode=ContextMode.SENTENCE,
            concurrency=1,
            request_stagger_ms=0,
            request_protocol="v9",
            reasoning_effort=args.reasoning))
    chunk = plan.chunks[0]
    if target_ranks is not None:
        words_by_rank = {
            word.rank: word
            for word in chunk.words
        }
        context_ids = {
            words_by_rank[rank].context_id
            for rank in target_ranks
            if rank in words_by_rank
            and words_by_rank[rank].context_id is not None
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
    scope = derive_compact_repair_scope(
        base_record["validation_error"]["validation_report"],
        chunk)
    if scope is None:
        raise SystemExit(
            "The saved failure cannot be repaired selectively.")
    repair_chunk = build_compact_repair_chunk(
        chunk,
        scope)
    contract = build_source_request_contract(
        pipeline,
        chunks=(chunk,),
        use_source_for_example_sentences=True,
        protocol_version=9,
        reasoning_effort=args.reasoning)
    options = {
        "model": contract["model"],
        "input": (
            contract["composed_prompt"]
            + render_chunk_input(
                repair_chunk,
                protocol_version=9)),
        "reasoning": contract["reasoning"],
        "text": {"format": contract["response_format"]},
        "max_output_tokens": contract[
            "max_output_tokens_by_chunk"][chunk.chunk_id],
        "prompt_cache_key": contract["prompt_cache_key"],
    }
    api_key = process_text.get_api_key()
    if not api_key:
        raise process_text.MissingAPIKeyError(
            "No OpenAI API key is configured.")
    client = process_text.OpenAI(
        api_key=api_key,
        max_retries=0,
        timeout=900.0)
    paid = _paid_response(
        client.responses.create(**options))
    validator = make_pipeline_response_validator(
        pipeline,
        use_compact_source_results=True)
    validator(paid.raw_text, repair_chunk)
    merged_text = merge_compact_repair(
        json.dumps(
            base_record["raw_response"],
            ensure_ascii=False),
        paid.raw_text,
        chunk,
        scope)
    validated = validator(merged_text, chunk)
    merged_response = json.loads(merged_text)
    merged_term_results = merged_response.get(
        process_text.SOURCE_TERM_RESULTS_KEY,
        ())
    merged_additional_sense_count = sum(
        len(item.get(
            process_text.SOURCE_ADDITIONAL_SENSES_KEY,
            ()))
        for item in merged_term_results
        if isinstance(item, dict))
    aggregate_usage = {
        key: (
            int(base_record["usage"].get(key, 0))
            + int((paid.usage or {}).get(key, 0)))
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "visible_output_tokens",
            "total_tokens",
            "web_search_calls",
        )
    }
    output_path = args.evaluation.with_name(
        f"{args.evaluation.stem}.repaired.json")
    result = {
        **base_record,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "validation_error": None,
        "raw_response": merged_response,
        "card_count": len(validated.get("cards", ())),
        "term_result_count": len(merged_term_results),
        "additional_sense_count": merged_additional_sense_count,
        "selective_repair": {
            "scope": scope.to_dict(),
            "response_id": paid.response_id,
            "reasoning_effort": args.reasoning,
            "usage": paid.usage,
            "raw_response": json.loads(paid.raw_text),
        },
        "aggregate_usage": aggregate_usage,
        "aggregate_cost": price_source_usage(
            aggregate_usage,
            execution_mode="standard"),
    }
    _write_json(output_path, result)
    print(json.dumps(
        {
            "valid": True,
            "scope": scope.to_dict(),
            "repair_usage": paid.usage,
            "aggregate_cost": result["aggregate_cost"],
            "card_count": result["card_count"],
            "additional_sense_count": result[
                "additional_sense_count"],
        },
        ensure_ascii=False,
        indent=2))
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
