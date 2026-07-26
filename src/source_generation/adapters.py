"""Plain-mapping adapters for GUI/controller integration."""

from dataclasses import asdict, replace
import hashlib
import hmac
import json
import threading

import pipeline_store
from source_generation.jobs import GenerationJobStore
from source_generation.model_catalog import source_model_profile
from source_generation.models import ContextMode, SourceGenerationConfig
from source_generation.planning import (
    estimate_plan_cost,
    list_processed_source_summaries,
    load_processed_source,
    plan_source_generation,
    resolve_latest_processed_source_path,
)
from source_generation.requests import (
    build_source_request_contract,
    source_request_contract_digest,
)
from source_generation.translation_memory import (
    SourceContextTranslationMemory,
)


SOURCE_ESTIMATE_AUTHORIZATION_SCHEMA_VERSION = 3


def _source_example_setting(request, plan, pipeline):
    enabled = bool(request.get("use_source_for_example_sentences", False))
    if not enabled:
        return False
    if plan.config.context_mode is ContextMode.NONE:
        raise ValueError(
            "Using source text for example sentences requires source "
            "context. Choose a context option other than None.")
    # The contract builder also checks this, but doing it here keeps the
    # source-specific error next to the other source-plan validation.
    if not pipeline_store.requires_sentences(pipeline):
        raise ValueError(
            "Using source text for example sentences requires the Context "
            "card direction.")
    return True


def _request_protocol_version(config):
    return int(config.request_protocol.removeprefix("v"))


def _estimate_response_format_arguments(config, request_contract):
    response_formats = request_contract.get(
        "response_formats_by_chunk")
    if config.request_protocol in {"v9", "v10"}:
        return {
            "response_format": request_contract["response_format"],
            "response_formats_by_chunk": None,
        }
    if request_contract.get("use_source_for_example_sentences"):
        return {
            "response_format": None,
            "response_formats_by_chunk": response_formats,
        }
    return {
        "response_format": request_contract["response_format"],
        "response_formats_by_chunk": None,
    }


def _normalise_model_runtime_snapshot(value, model):
    source_model_profile(model)
    if value not in (None, {}):
        raise ValueError(
            "OpenAI models cannot use a model-runtime snapshot.")
    return None


def _estimate_authorization_fingerprint(
        plan,
        request_contract,
        estimate,
        model_runtime_snapshot=None):
    payload = {
        "schema_version": SOURCE_ESTIMATE_AUTHORIZATION_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "execution_policy": {
            "model": plan.config.model,
            "concurrency": plan.config.concurrency,
            "request_stagger_ms": plan.config.request_stagger_ms,
            "max_transient_retries": (
                plan.config.max_transient_retries),
            "request_protocol": plan.config.request_protocol,
            "reasoning_effort": plan.config.reasoning_effort,
            "execution_mode": plan.config.execution_mode,
            "automatic_repair": plan.config.automatic_repair,
            "max_automatic_repairs": (
                plan.config.max_automatic_repairs),
        },
        "request_contract_sha256": source_request_contract_digest(
            request_contract),
        "model_runtime_snapshot": model_runtime_snapshot,
        # Binding the displayed cost inputs as well as request contents means
        # a local pricing/estimator update also requires fresh authorization.
        "estimate": estimate.to_dict(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_translation_memory_snapshot(value, plan):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            "The authorized translation-memory snapshot must be an object.")
    chunks = {
        chunk.chunk_id: chunk
        for chunk in plan.chunks
    }
    if set(value) - set(chunks):
        raise ValueError(
            "The translation-memory snapshot contains an unknown chunk.")
    result = {}
    for chunk_id, raw_hits in value.items():
        if not isinstance(raw_hits, dict):
            raise TypeError(
                "Translation-memory chunk hits must be an object.")
        expected_context_ids = {
            context.context_id
            for context in chunks[chunk_id].contexts
        }
        hits = {}
        for context_id, raw_hit in raw_hits.items():
            if (
                    context_id not in expected_context_ids
                    or not isinstance(raw_hit, dict)
                    or not isinstance(raw_hit.get("cache_key"), str)
                    or not raw_hit["cache_key"].startswith("sctm1-")
                    or not isinstance(raw_hit.get("translation"), str)
                    or not raw_hit["translation"].strip()
                    or not isinstance(
                        raw_hit.get("translation_sha256"),
                        str)
                    or raw_hit["translation_sha256"]
                    != hashlib.sha256(
                        raw_hit["translation"].encode(
                            "utf-8")).hexdigest()
                    or not isinstance(raw_hit.get("entry_sha256"), str)):
                raise ValueError(
                    "The translation-memory snapshot does not match its "
                    "source chunk.")
            hits[context_id] = dict(raw_hit)
        result[chunk_id] = hits
    return result


class SourceGenerationBackend:
    """Free planning/status callbacks suitable for ``AutoAnkiApp`` hooks.

    No method here makes an OpenAI request. ``create_job`` requires the GUI's
    explicit paid-confirmation flag but only persists pending request units.
    A controller may then construct ``GenerationJobRunner`` with its own paid
    request callable.
    """

    def __init__(
            self,
            *,
            corpus_root=None,
            jobs_root=None,
            exclusion_resolver=None,
            translation_memory=None):
        self.corpus_root = corpus_root
        self.jobs = GenerationJobStore(jobs_root)
        self.exclusion_resolver = exclusion_resolver
        self.translation_memory = (
            translation_memory
            or SourceContextTranslationMemory())
        self._loaded_sources = {}
        self._source_lock = threading.RLock()

    def catalogue(self):
        preset_names = {
            "daodejing_wang_bi": "Daodejing [Wang Bi]",
            "daodejing_mawangdui": "Daodejing [Mawangdui]",
            "journey_to_the_west": "Journey to the West",
        }
        sources = []
        for source in list_processed_source_summaries(self.corpus_root):
            if source["key"] == "daodejing_huijiao":
                # Superseded unsourced eclectic collation. Retain its files
                # for recovery, but do not present it as a third Daodejing.
                continue
            value = dict(source)
            preset_name = preset_names.get(source["key"])
            value["preset"] = preset_name is not None
            if preset_name is not None:
                value["name"] = preset_name
                value["title"] = preset_name
            sources.append(value)
        return {"sources": sources}

    def _load_source(self, source_key, run_path=None):
        selected_path = (
            resolve_latest_processed_source_path(
                source_key,
                corpus_root=self.corpus_root)
            if run_path is None
            else run_path)
        cache_key = (source_key, str(selected_path))
        with self._source_lock:
            loaded = self._loaded_sources.get(cache_key)
            if loaded is None:
                loaded = load_processed_source(
                    source_key,
                    run_path=selected_path,
                    corpus_root=self.corpus_root)
                self._loaded_sources[cache_key] = loaded
        return loaded

    def _plan(self, request):
        config = SourceGenerationConfig.from_mapping(request)
        loaded = self._load_source(
            config.source_key,
            config.run_path)
        exclusion_requested = bool(request.get("exclude_anki"))
        exclusions = config.anki_exclusions
        if exclusion_requested and not exclusions:
            raise ValueError(
                "Anki exclusion requires a deck, note type, and field.")
        if exclusion_requested:
            if any(
                    specification.note_type is None
                    or specification.field_name is None
                    for specification in exclusions):
                raise ValueError(
                    "Anki exclusion requires a deck, note type, and field.")
            if self.exclusion_resolver is None:
                raise RuntimeError(
                    "Anki vocabulary exclusion is not connected.")
            excluded_words = tuple(sorted({
                word
                for specification in exclusions
                for word in self.exclusion_resolver(
                    specification,
                    loaded)
            }))
            config = replace(
                config,
                excluded_words=excluded_words)
        plan = plan_source_generation(loaded, config)
        display_name = str(request.get("source_name", "")).strip()
        if display_name:
            plan = replace(
                plan,
                source_title=display_name)
        return loaded, plan

    def preview(self, request):
        """Return one read-only page of first-occurrence source metadata."""
        if not isinstance(request, dict):
            raise TypeError("Source preview request must be an object.")
        source_key = str(request.get("source_key", "")).strip()
        if not source_key:
            raise ValueError("Choose a processed source to inspect.")
        offset = request.get("offset", 0)
        limit = request.get("limit", 500)
        for name, value in (("offset", offset), ("limit", limit)):
            if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < (0 if name == "offset" else 1)):
                requirement = (
                    "a non-negative integer"
                    if name == "offset"
                    else "a positive integer")
                raise ValueError(
                    f"Source preview {name} must be {requirement}.")
        if limit > 10000:
            raise ValueError(
                "Source preview limit cannot exceed 10,000 words.")

        loaded = self._load_source(
            source_key,
            request.get("run_path"))
        build = loaded.build
        total = len(build.unique_words)
        if offset > total:
            raise ValueError(
                "Source preview offset is beyond the vocabulary list.")
        contexts = {
            context.context_id: context.text
            for context in build.contexts
        }
        sections = {
            section.section_id: section.title
            for section in build.snapshot.sections
        }
        items = []
        for word in build.unique_words[offset:offset + limit]:
            items.append({
                "rank": word.rank,
                "term": word.surface,
                "section_title": sections.get(word.section_id, ""),
                "previous_sentence": (
                    contexts.get(word.previous_sentence_id, "")
                    if word.previous_sentence_id is not None
                    else ""),
                "current_sentence": contexts.get(word.sentence_id, ""),
                "next_sentence": (
                    contexts.get(word.next_sentence_id, "")
                    if word.next_sentence_id is not None
                    else ""),
            })
        return {
            "source_key": loaded.key,
            "total": total,
            "offset": offset,
            "items": items,
        }

    def estimate(self, request):
        """GUI ``source_estimator`` callback returning its expected aliases."""
        _loaded, plan = self._plan(request)
        model_runtime_snapshot = _normalise_model_runtime_snapshot(
            request.get("model_runtime_snapshot"),
            plan.config.model)
        pipeline = request.get("pipeline")
        if pipeline is None:
            raise ValueError(
                "Card Setup must provide a generation pipeline.")
        use_source_examples = _source_example_setting(
            request,
            plan,
            pipeline)
        memory_enabled = bool(
            plan.config.automatic_repair
            and use_source_examples
            and plan.config.request_protocol == "v10")
        translation_memory_by_chunk = (
            self.translation_memory.lookup_chunks(
                pipeline.language_key,
                plan.chunks)
            if memory_enabled
            else {})
        request_contract = build_source_request_contract(
            pipeline,
            chunks=plan.chunks,
            allow_web_search=bool(request.get("allow_web_search")),
            use_source_for_example_sentences=use_source_examples,
            protocol_version=_request_protocol_version(plan.config),
            reasoning_effort=plan.config.reasoning_effort,
            model=plan.config.model,
            translation_memory_enabled=memory_enabled)
        estimate = estimate_plan_cost(
            plan,
            pipeline,
            prompt_text=request_contract["composed_prompt"],
            web_search_enabled=bool(request.get("allow_web_search")),
            use_source_for_example_sentences=use_source_examples,
            request_protocol=plan.config.request_protocol,
            reasoning_effort=plan.config.reasoning_effort,
            execution_mode=plan.config.execution_mode,
            translation_memory_by_chunk=(
                translation_memory_by_chunk),
            **_estimate_response_format_arguments(
                plan.config,
                request_contract))
        contract_digest = source_request_contract_digest(
            request_contract)
        authorization_fingerprint = (
            _estimate_authorization_fingerprint(
                plan,
                request_contract,
                estimate,
                model_runtime_snapshot))
        result = estimate.to_dict()
        request_assumptions = estimate.assumptions
        largest_input = request_assumptions[
            "largest_request_estimated_input_tokens"]
        largest_output = request_assumptions[
            "largest_request_estimated_output_tokens"]
        high_output = request_assumptions[
            "largest_request_high_output_tokens"]
        high_total = request_assumptions[
            "largest_request_high_total_tokens"]
        warning = None
        model_profile = source_model_profile(plan.config.model)
        if largest_input > model_profile.max_input_tokens:
            warning = (
                "The selected prompt, schema, and source payload may exceed "
                "the model's maximum input. Reduce words per request or "
                "source context.")
        elif high_output > model_profile.max_output_tokens:
            warning = (
                "The selected chunk size may exceed the model's maximum "
                "output. Reduce words per request before generation.")
        elif high_total > model_profile.max_context_tokens:
            warning = (
                "The selected context and card detail may exceed the model's "
                "context window. Reduce words per request or source context.")
        result.update({
            "plan_id": plan.plan_id,
            "request_contract_sha256": contract_digest,
            "authorization_fingerprint": authorization_fingerprint,
            "source_key": plan.source_key,
            "source_name": plan.source_title,
            "original_candidate_count": plan.original_word_count,
            "source_prefix_token_limit": (
                plan.config.source_prefix_token_limit),
            "source_prefix_token_count": (
                plan.source_prefix_token_count),
            "prefix_unique_candidate_count": (
                plan.prefix_unique_word_count),
            "prefix_omitted_candidate_count": max(
                0,
                plan.original_word_count
                - (plan.prefix_unique_word_count or 0)),
            "excluded_candidate_count": plan.excluded_word_count,
            "chunk_size": plan.config.chunk_size,
            "context_mode": plan.config.context_mode.value,
            "request_protocol": plan.config.request_protocol,
            "reasoning_effort": plan.config.reasoning_effort,
            "execution_mode": plan.config.execution_mode,
            "model": plan.config.model,
            "provider": model_profile.provider,
            "local_model": model_profile.local,
            "model_runtime_snapshot": model_runtime_snapshot,
            "automatic_repair": plan.config.automatic_repair,
            "max_automatic_repairs": (
                plan.config.max_automatic_repairs),
            "translation_memory_by_chunk": (
                translation_memory_by_chunk),
            "translation_memory_hit_count": sum(
                len(hits)
                for hits in translation_memory_by_chunk.values()),
            "largest_request_input_tokens": largest_input,
            "largest_request_output_tokens": largest_output,
            "largest_request_high_output_tokens": high_output,
            "largest_request_high_total_tokens": high_total,
            "request_limit_warning": warning,
        })
        return result

    def create_job(self, request):
        """Persist pending chunks after, but without consuming, authorization."""
        if request.get("paid_confirmed") is not True:
            raise PermissionError(
                "Paid source generation requires explicit confirmation.")
        _loaded, plan = self._plan(request)
        model_runtime_snapshot = _normalise_model_runtime_snapshot(
            request.get("model_runtime_snapshot"),
            plan.config.model)
        pipeline = request.get("pipeline")
        if pipeline is None:
            raise ValueError(
                "Card Setup must provide a generation pipeline.")
        use_source_examples = _source_example_setting(
            request,
            plan,
            pipeline)
        memory_enabled = bool(
            plan.config.automatic_repair
            and use_source_examples
            and plan.config.request_protocol == "v10")
        authorized_estimate = request.get("estimate")
        translation_memory_by_chunk = (
            _normalise_translation_memory_snapshot(
                (
                    authorized_estimate.get(
                        "translation_memory_by_chunk")
                    if isinstance(authorized_estimate, dict)
                    else None),
                plan)
            if memory_enabled
            else {})
        request_contract = build_source_request_contract(
            pipeline,
            chunks=plan.chunks,
            allow_web_search=bool(request.get("allow_web_search")),
            use_source_for_example_sentences=use_source_examples,
            protocol_version=_request_protocol_version(plan.config),
            reasoning_effort=plan.config.reasoning_effort,
            model=plan.config.model,
            translation_memory_enabled=memory_enabled)
        estimate = estimate_plan_cost(
            plan,
            pipeline,
            prompt_text=request_contract["composed_prompt"],
            web_search_enabled=bool(request.get("allow_web_search")),
            use_source_for_example_sentences=use_source_examples,
            request_protocol=plan.config.request_protocol,
            reasoning_effort=plan.config.reasoning_effort,
            execution_mode=plan.config.execution_mode,
            translation_memory_by_chunk=(
                translation_memory_by_chunk),
            **_estimate_response_format_arguments(
                plan.config,
                request_contract))
        expected_authorization = _estimate_authorization_fingerprint(
            plan,
            request_contract,
            estimate,
            model_runtime_snapshot)
        if (
                plan.chunks
                and (
                not isinstance(authorized_estimate, dict)
                or authorized_estimate.get("plan_id") != plan.plan_id
                or not isinstance(
                    authorized_estimate.get(
                        "authorization_fingerprint"),
                    str)
                or not hmac.compare_digest(
                    authorized_estimate["authorization_fingerprint"],
                    expected_authorization))):
            raise PermissionError(
                "The source, learned-word list, prompt, response schema, "
                "model, protocol, reasoning, execution mode, or cost estimate "
                "changed after authorization. "
                "Recalculate the estimate and authorize the paid requests "
                "again. No provider request was made.")
        high_output = estimate.assumptions[
            "largest_request_high_output_tokens"]
        largest_input = estimate.assumptions[
            "largest_request_estimated_input_tokens"]
        high_total = estimate.assumptions[
            "largest_request_high_total_tokens"]
        model_profile = source_model_profile(plan.config.model)
        if largest_input > model_profile.max_input_tokens:
            raise ValueError(
                "The estimated prompt, schema, and source payload for one "
                f"chunk can exceed {model_profile.max_input_tokens:,} input "
                "tokens. Reduce words per request or source context.")
        if high_output > model_profile.max_output_tokens:
            raise ValueError(
                "The estimated response for one chunk can exceed "
                f"{model_profile.max_output_tokens:,} output tokens. Reduce "
                "words per request.")
        if high_total > model_profile.max_context_tokens:
            raise ValueError(
                "The estimated input and response for one chunk can exceed "
                f"{model_profile.max_context_tokens:,} total tokens. Reduce "
                "the chunk size or context.")
        metadata = {
            "paid_confirmed_at_creation": True,
            "authorization_bypassed_for_empty_plan": (
                not plan.chunks),
            "pipeline": asdict(pipeline),
            "request_protocol": plan.config.request_protocol,
            "reasoning_effort": plan.config.reasoning_effort,
            "execution_mode": plan.config.execution_mode,
            "model": plan.config.model,
            "provider": model_profile.provider,
            "model_runtime_snapshot": model_runtime_snapshot,
            "automatic_repair": plan.config.automatic_repair,
            "max_automatic_repairs": (
                plan.config.max_automatic_repairs),
            "translation_memory_by_chunk": (
                translation_memory_by_chunk),
            "request_contract": request_contract,
            "estimate": {
                **estimate.to_dict(),
                "plan_id": plan.plan_id,
                "request_contract_sha256": (
                    source_request_contract_digest(
                        request_contract)),
                "authorization_fingerprint": expected_authorization,
            },
            "output_deck_name": request.get(
                "output_deck_name",
                f"Vocabulary from {plan.source_title}"),
            "keep_imported_deck": bool(
                request.get("keep_imported_deck", True)),
            "move_cards_after_import": bool(
                request.get("move_cards_after_import", False)),
            "delete_imported_deck": bool(
                request.get("delete_imported_deck", False)),
        }
        return self.jobs.create(
            plan,
            request_metadata=metadata,
            request_contract=request_contract)

    @staticmethod
    def _row_id(job_id, chunk_id):
        return f"{job_id}::{chunk_id}"

    @staticmethod
    def split_row_id(row_id):
        if not isinstance(row_id, str) or "::" not in row_id:
            raise ValueError("Select a valid saved source request.")
        job_id, chunk_id = row_id.split("::", 1)
        if not job_id or not chunk_id:
            raise ValueError("Select a valid saved source request.")
        return job_id, chunk_id

    def job_rows(self):
        """GUI ``source_jobs_loader`` callback, one row per request chunk."""
        rows = []
        for snapshot in self.jobs.list():
            for chunk in snapshot.chunks:
                error = chunk.get("last_error") or {}
                validation = error.get("validation")
                problem_count = (
                    validation.get("problem_count", 0)
                    if isinstance(validation, dict)
                    else 0)
                review = chunk.get("validation_review") or {}
                detail = error.get("message", "")
                if chunk["status"] == "invalid_response":
                    if problem_count:
                        accepted_count = int(review.get(
                            "accepted_problem_count",
                            0))
                        remaining_count = int(review.get(
                            "remaining_problem_count",
                            problem_count))
                        detail = (
                            f"Validation failed — {problem_count:,} "
                            f"problem(s), {accepted_count:,} manually "
                            f"accepted, {remaining_count:,} remaining: "
                            f"{detail}")
                    elif detail:
                        detail = "Validation failed: " + detail
                rows.append({
                    "job_id": self._row_id(
                        snapshot.job_id,
                        chunk["chunk_id"]),
                    "parent_job_id": snapshot.job_id,
                    "chunk_id": chunk["chunk_id"],
                    "source_name": snapshot.source_title,
                    "chunk_label": (
                        f'{chunk["index"]}/{len(snapshot.chunks)}'),
                    "worker": chunk.get("worker", "queued"),
                    "status": chunk["status"],
                    "attempts": chunk["attempts"],
                    "detail": detail,
                    "detail_kind": (
                        "validation_error"
                        if chunk["status"] == "invalid_response"
                        else (
                            "connection_error"
                            if chunk["status"] == "connection_failed"
                            else chunk["status"])),
                    "has_validation_error": (
                        chunk["status"] == "invalid_response"),
                    "validation_problem_count": problem_count,
                    "accepted_problem_count": review.get(
                        "accepted_problem_count",
                        (
                            chunk.get("validation_override") or {}).get(
                                "accepted_problem_count",
                                0)),
                    "remaining_problem_count": review.get(
                        "remaining_problem_count",
                        problem_count),
                    "created_at": snapshot.created_at,
                    "updated_at": chunk["updated_at"],
                })
        return {"jobs": rows}

    def inspect(self, request):
        """GUI ``source_inspect_callback`` callback."""
        row_id = (
            request.get("job_id")
            if isinstance(request, dict)
            else request)
        job_id, chunk_id = self.split_row_id(row_id)
        return self.jobs.inspect_chunk(job_id, chunk_id)

    def manual_retry_targets(self, request):
        """Resolve GUI row IDs after its paid-retry confirmation.

        This resets only the requested failed chunks. The controller still
        injects and runs the paid callable, so merely resolving a target cannot
        contact OpenAI.
        """
        if request.get("paid_confirmed") is not True:
            raise PermissionError(
                "Paid source retries require explicit confirmation.")
        grouped = {}
        for row_id in request.get("job_ids", ()):
            job_id, chunk_id = self.split_row_id(row_id)
            grouped.setdefault(job_id, []).append(chunk_id)
        result = {}
        for job_id, chunk_ids in grouped.items():
            reset = self.jobs.reset_for_manual_retry(
                job_id,
                chunk_ids)
            pending = tuple(
                chunk_id
                for chunk_id in self.jobs.chunk_ids(job_id)
                if (
                    chunk_id in chunk_ids
                    and self.jobs.chunk_status(
                        job_id,
                        chunk_id)["status"] == "pending"))
            result[job_id] = tuple(dict.fromkeys(
                (*reset, *pending)))
        return result
