"""Load audited corpora, repartition them, and estimate paid generation."""

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import runtime_paths
from corpus_pipeline.catalogue import get_corpus_spec
from corpus_pipeline.models import CORPUS_SCHEMA_VERSION
from corpus_pipeline.storage import (
    build_id,
    read_vocabulary_view,
    resolve_corpus_base,
)
from source_generation.models import (
    ContextMode,
    ContextUnit,
    CostEstimate,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    LoadedSource,
    OutputDetail,
    Pricing,
    SourceGenerationConfig,
)


DEFAULT_PRICING = Pricing(
    model="gpt-5.4-mini",
    input_usd_per_million=0.75,
    output_usd_per_million=4.50,
    label="standard pricing published 2026-07-24",
)
# OpenAI bills in USD.  This dated conversion keeps saved estimates
# reproducible instead of silently changing between authorization and job
# creation. RBA AUD/USD at 4:00 pm on 2026-07-24 was 0.6975.
AUD_USD_RATE = 0.6975
USD_TO_AUD_RATE = 1.0 / AUD_USD_RATE
AUD_EXCHANGE_RATE_LABEL = "RBA AUD/USD 0.6975 at 2026-07-24 16:00 AEST"

# Standard web search is USD $10 / 1,000 calls. Search-content tokens are
# additionally billed at model input rates. Their volume is not known before
# the model searches, so the upper estimate budgets 8,000 retrieved tokens.
WEB_SEARCH_USD_PER_CALL = 0.01
WEB_SEARCH_CONTENT_TOKENS_PER_CALL_HIGH = 8_000
MODEL_CONTEXT_WINDOW_TOKENS = 400_000
MODEL_MAX_OUTPUT_TOKENS = 128_000

# Deliberately visible assumptions: this is a planning estimate, not a promise
# about model verbosity.  Values are expected output tokens per generated sense.
_FIELD_OUTPUT_TOKENS = {
    "translation": 12,
    "dictionary_meaning": 42,
    "pronunciation": 10,
    "part_of_speech": 7,
    "register": 10,
    "nuance": 20,
}
_TERM_OUTPUT_TOKENS = 5
_JSON_OVERHEAD_TOKENS_PER_CARD = 13
_EXAMPLE_SENTENCE_OUTPUT_TOKENS = 72
_RESPONSE_ENVELOPE_TOKENS = 12


@dataclass(frozen=True)
class _DesiredInterval:
    start: int
    end: int
    word_rank: int


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_latest_build_path(corpus_root, source_key):
    base = resolve_corpus_base(corpus_root, source_key)
    pointer_path = base / "latest_build.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No processed build exists for source {source_key!r}.")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("The latest processed-source pointer is invalid.") \
            from error
    if pointer.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(
            "The latest processed-source pointer uses another schema.")
    expected_id = pointer.get("build_id")
    relative_path = pointer.get("relative_path")
    if not isinstance(expected_id, str) or not expected_id:
        raise ValueError("The latest processed-source pointer has no build ID.")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("The latest processed-source pointer has no path.")
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(
            "The latest processed-source pointer escapes corpus storage.") \
            from error
    if candidate.name != expected_id:
        raise ValueError(
            "The latest processed-source path and build ID disagree.")
    if not candidate.is_dir():
        raise FileNotFoundError("The latest processed-source build is missing.")
    return candidate


def _source_title(source_key, build):
    try:
        return get_corpus_spec(source_key).title
    except KeyError:
        edition = build.snapshot.edition or source_key
        local_prefix = "Local document:"
        if edition.startswith(local_prefix):
            return edition[len(local_prefix):].strip() or source_key
        return edition


def load_processed_source(
        source_key,
        *,
        run_path=None,
        corpus_root=None):
    """Load and fully audit one saved tokenization run.

    Passing a new generation chunk size never reaches this function's
    tokenizer: all later partitioning uses ``build.unique_words`` directly.
    """
    corpus_root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory()).resolve()
    selected_path = (
        Path(run_path).resolve()
        if run_path is not None
        else _safe_latest_build_path(corpus_root, source_key)
    )
    build = read_vocabulary_view(selected_path)
    if build.snapshot.spec_key != source_key:
        raise ValueError(
            "The selected processed run belongs to another source.")
    identifier = build_id(build)
    if selected_path.name != identifier:
        raise ValueError(
            "The processed run directory and audited build ID disagree.")
    return LoadedSource(
        key=source_key,
        title=_source_title(source_key, build),
        build_id=identifier,
        run_path=selected_path,
        build=build,
    )


def list_processed_sources(corpus_root=None):
    """Return audited latest builds discoverable under the corpus root."""
    root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory())
    if not root.is_dir():
        return ()
    sources = []
    for pointer in sorted(root.glob("*/latest_build.json")):
        source_key = pointer.parent.name
        try:
            sources.append(load_processed_source(
                source_key,
                corpus_root=root))
        except (OSError, TypeError, ValueError):
            # A corrupt build is not presented as selectable. The caller can
            # still explicitly load it and receive the precise audit failure.
            continue
    return tuple(sources)


def list_processed_source_summaries(corpus_root=None):
    """List latest builds from their small manifests without loading tokens.

    The selected build is still fully loaded and audited by
    :func:`load_processed_source` before an estimate or paid job is planned.
    This lightweight path keeps opening the Tk window instantaneous even for
    a novel with hundreds of megabytes of retained occurrence metadata.
    """
    root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory())
    if not root.is_dir():
        return ()
    summaries = []
    for pointer_path in sorted(root.glob("*/latest_build.json")):
        source_key = pointer_path.parent.name
        try:
            build_path = _safe_latest_build_path(root, source_key)
            manifest = json.loads(
                (build_path / "manifest.json").read_text(
                    encoding="utf-8"))
            if (
                    manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
                    or manifest.get("kind") != "corpus_build"
                    or manifest.get("spec_key") != source_key
                    or manifest.get("build_id") != build_path.name):
                raise ValueError("Inconsistent processed-source manifest.")
            counts = manifest["counts"]
            word_count = counts["unique_words"]
            section_count = counts["sections"]
            if (
                    isinstance(word_count, bool)
                    or not isinstance(word_count, int)
                    or word_count < 0
                    or isinstance(section_count, bool)
                    or not isinstance(section_count, int)
                    or section_count < 0):
                raise ValueError("Invalid processed-source counts.")
            try:
                title = get_corpus_spec(source_key).title
            except KeyError:
                title = str(manifest.get("edition") or source_key)
                prefix = "Local document:"
                if title.startswith(prefix):
                    title = title[len(prefix):].strip() or source_key
            summaries.append({
                "key": source_key,
                "source_key": source_key,
                "name": title,
                "title": title,
                "build_id": build_path.name,
                "run_path": str(build_path),
                "word_count": word_count,
                "section_count": section_count,
                "source_language_key": str(
                    manifest.get("source_language_key", "")),
            })
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return tuple(summaries)


def _partition_words(words, chunk_size):
    return tuple(
        tuple(words[start:start + chunk_size])
        for start in range(0, len(words), chunk_size)
        if words[start:start + chunk_size]
    )


def _desired_intervals(build, words, mode):
    if mode == ContextMode.NONE:
        return ()
    contexts = {
        context.context_id: context
        for context in build.contexts
    }
    if mode == ContextMode.CHUNK_SPAN:
        sentences = [
            contexts[word.sentence_id]
            for word in words
        ]
        start = min(sentence.start_offset for sentence in sentences)
        end = max(sentence.end_offset for sentence in sentences)
        return tuple(
            _DesiredInterval(
                start=start,
                end=end,
                word_rank=word.rank)
            for word in words
        )

    desired = []
    for word in words:
        sentence_ids = [word.sentence_id]
        if mode == ContextMode.SENTENCE_NEIGHBORS:
            if word.previous_sentence_id is not None:
                sentence_ids.insert(0, word.previous_sentence_id)
            if word.next_sentence_id is not None:
                sentence_ids.append(word.next_sentence_id)
        sentences = [contexts[context_id] for context_id in sentence_ids]
        desired.append(_DesiredInterval(
            start=min(item.start_offset for item in sentences),
            end=max(item.end_offset for item in sentences),
            word_rank=word.rank))
    return tuple(desired)


def _merge_intervals(intervals):
    if not intervals:
        return ()
    ordered = sorted(
        intervals,
        key=lambda item: (item.start, item.end, item.word_rank))
    merged = []
    start = ordered[0].start
    end = ordered[0].end
    ranks = [ordered[0].word_rank]
    for interval in ordered[1:]:
        # Merge repeated or genuinely overlapping windows. Merely adjacent
        # sentences remain independently reusable context units.
        if interval.start < end:
            end = max(end, interval.end)
            ranks.append(interval.word_rank)
            continue
        merged.append((start, end, tuple(sorted(set(ranks)))))
        start = interval.start
        end = interval.end
        ranks = [interval.word_rank]
    merged.append((start, end, tuple(sorted(set(ranks)))))
    return tuple(merged)


def _make_context_units(
        build,
        words,
        mode,
        *,
        source_build_id=None):
    intervals = _merge_intervals(
        _desired_intervals(build, words, mode))
    if not intervals:
        return (), {}
    sentences = tuple(
        context
        for context in build.contexts
        if context.kind == "sentence")
    sentence_starts = tuple(
        sentence.start_offset
        for sentence in sentences)
    sentence_ends = tuple(
        sentence.end_offset
        for sentence in sentences)
    sections = build.snapshot.sections
    stable_build_id = source_build_id or build_id(build)
    units = []
    rank_to_context = {}
    for start, end, ranks in intervals:
        identity = _canonical_json({
            "snapshot": stable_build_id,
            "mode": mode.value,
            "start": start,
            "end": end,
        })
        context_id = "source-context-" + _sha256_text(identity)[:20]
        section_ids = tuple(
            section.section_id
            for section in sections
            if section.start_offset < end and start < section.end_offset)
        first_sentence = bisect_right(sentence_ends, start)
        after_last_sentence = bisect_left(sentence_starts, end)
        sentence_ids = tuple(
            sentence.context_id
            for sentence in sentences[
                first_sentence:after_last_sentence])
        units.append(ContextUnit(
            context_id=context_id,
            mode=mode.value,
            start_offset=start,
            end_offset=end,
            text=build.snapshot.canonical_text[start:end],
            section_ids=section_ids,
            sentence_ids=sentence_ids,
            word_ranks=ranks,
        ))
        for rank in ranks:
            if rank in rank_to_context:
                raise ValueError(
                    "A source word was assigned to multiple context units.")
            rank_to_context[rank] = context_id
    if set(rank_to_context) != {word.rank for word in words}:
        raise ValueError("Not every source word was assigned retained context.")
    return tuple(units), rank_to_context


def plan_source_generation(loaded_source, config):
    """Repartition saved vocabulary and deduplicate context within each query."""
    if not isinstance(loaded_source, LoadedSource):
        raise TypeError("A loaded processed source is required.")
    config = SourceGenerationConfig.from_mapping(config)
    if loaded_source.key != config.source_key:
        raise ValueError(
            "Source generation settings select another processed source.")

    excluded = {
        unicodedata.normalize("NFC", word.strip())
        for word in config.excluded_words
        if word.strip()
    }
    selected_words = tuple(
        word
        for word in loaded_source.build.unique_words
        if word.normalized not in excluded
        and unicodedata.normalize("NFC", word.surface) not in excluded
    )
    groups = _partition_words(selected_words, config.chunk_size)
    total = len(groups)
    chunks = []
    for index, words in enumerate(groups, start=1):
        context_units, rank_to_context = _make_context_units(
            loaded_source.build,
            words,
            config.context_mode,
            source_build_id=loaded_source.build_id)
        chunk_id = (
            f"{index:06d}-r{words[0].rank}-r{words[-1].rank}")
        chunks.append(GenerationChunk(
            chunk_id=chunk_id,
            index=index,
            total=total,
            start_rank=words[0].rank,
            end_rank=words[-1].rank,
            words=tuple(
                GenerationWord(
                    rank=word.rank,
                    surface=word.surface,
                    normalized=word.normalized,
                    section_id=word.section_id,
                    sentence_id=word.sentence_id,
                    context_id=rank_to_context.get(word.rank))
                for word in words
            ),
            contexts=context_units,
        ))

    identity = {
        "source_build_id": loaded_source.build_id,
        "chunk_size": config.chunk_size,
        "context_mode": config.context_mode.value,
        # Concurrency and staggering affect execution, not request contents.
        "selected_ranks": [
            word.rank
            for word in selected_words
        ],
    }
    return GenerationPlan(
        plan_id=_sha256_text(_canonical_json(identity))[:24],
        source_key=loaded_source.key,
        source_title=loaded_source.title,
        source_build_id=loaded_source.build_id,
        source_run_path=str(loaded_source.run_path),
        config=config,
        original_word_count=len(loaded_source.build.unique_words),
        excluded_word_count=(
            len(loaded_source.build.unique_words) - len(selected_words)),
        chunks=tuple(chunks),
    )


def estimate_text_tokens(text):
    """Return a deterministic tokenizer-free approximation.

    Han/kana characters tend to consume close to one token each, while runs of
    Latin text tend toward roughly four characters per token. Punctuation and
    whitespace still carry a smaller cost. The final price range deliberately
    remains wider than this heuristic's likely error.
    """
    weighted = 0.0
    for character in str(text):
        codepoint = ord(character)
        if (
                0x3400 <= codepoint <= 0x9FFF
                or 0x3040 <= codepoint <= 0x30FF):
            weighted += 1.0
        elif character.isascii() and character.isalnum():
            weighted += 0.25
        elif character.isspace():
            weighted += 0.08
        elif character.isascii():
            weighted += 0.35
        else:
            weighted += 0.75
    return max(1, math.ceil(weighted)) if text else 0


def _schema_token_estimate(detail):
    field_names = [
        detail.term_field_name,
        *detail.field_keys,
    ]
    if detail.include_example_sentences:
        field_names.append("Sentences")
    # Strict JSON schema keywords and the repeated property/required names.
    return 80 + 2 * sum(
        estimate_text_tokens(field_name)
        for field_name in field_names)


def estimate_chunk_output_tokens(
        detail,
        word_count,
        *,
        senses_per_word=1.15,
        high_multiplier=1.0):
    """Estimate one response size for rate-limit and model-limit planning."""
    if isinstance(detail, dict):
        detail = OutputDetail.from_mapping(detail)
    elif not isinstance(detail, OutputDetail):
        detail = OutputDetail.from_pipeline(detail)
    if (
            isinstance(word_count, bool)
            or not isinstance(word_count, int)
            or word_count < 0):
        raise ValueError("Chunk word count cannot be negative.")
    if (
            isinstance(senses_per_word, bool)
            or not isinstance(senses_per_word, (int, float))
            or senses_per_word < 1):
        raise ValueError("Expected senses per word must be at least one.")
    if (
            isinstance(high_multiplier, bool)
            or not isinstance(high_multiplier, (int, float))
            or high_multiplier < 1):
        raise ValueError("Output safety multiplier must be at least one.")
    tokens_per_sense = (
        _TERM_OUTPUT_TOKENS
        + _JSON_OVERHEAD_TOKENS_PER_CARD
        + sum(
            _FIELD_OUTPUT_TOKENS.get(field_key, 24)
            for field_key in detail.field_keys)
        + (
            _EXAMPLE_SENTENCE_OUTPUT_TOKENS
            if detail.include_example_sentences
            else 0)
    )
    return math.ceil(
        (
            word_count
            * senses_per_word
            * tokens_per_sense
            + _RESPONSE_ENVELOPE_TOKENS)
        * high_multiplier)


def estimate_plan_cost(
        plan,
        detail,
        *,
        prompt_text="",
        pricing=DEFAULT_PRICING,
        senses_per_word=1.15,
        web_search_enabled=False):
    """Estimate standard API cost without contacting OpenAI.

    ``detail`` can be an :class:`OutputDetail`, its mapping form, or a saved
    ``PipelineConfig``.  Prompt and schema costs are multiplied by the number
    of requests, so changing the generation chunk size immediately changes the
    estimate.
    """
    if not isinstance(plan, GenerationPlan):
        raise TypeError("A source generation plan is required.")
    if isinstance(detail, dict):
        detail = OutputDetail.from_mapping(detail)
    elif not isinstance(detail, OutputDetail):
        detail = OutputDetail.from_pipeline(detail)
    if (
            isinstance(senses_per_word, bool)
            or not isinstance(senses_per_word, (int, float))
            or senses_per_word < 1):
        raise ValueError("Expected senses per word must be at least one.")
    if not isinstance(pricing, Pricing):
        raise TypeError("A pricing profile is required.")

    request_count = len(plan.chunks)
    prompt_tokens_per_request = estimate_text_tokens(prompt_text)
    prompt_tokens = prompt_tokens_per_request * request_count
    payload_tokens = sum(
        estimate_text_tokens(_canonical_json(chunk.request_payload()))
        for chunk in plan.chunks)
    schema_tokens_per_request = _schema_token_estimate(detail)
    schema_tokens = schema_tokens_per_request * request_count
    input_tokens = prompt_tokens + payload_tokens + schema_tokens

    request_output_tokens = tuple(
        estimate_chunk_output_tokens(
            detail,
            len(chunk.words),
            senses_per_word=senses_per_word)
        for chunk in plan.chunks)
    output_tokens = sum(request_output_tokens)
    request_input_tokens = tuple(
        prompt_tokens_per_request
        + schema_tokens_per_request
        + estimate_text_tokens(_canonical_json(chunk.request_payload()))
        for chunk in plan.chunks)
    largest_input = max(request_input_tokens, default=0)
    largest_output = max(request_output_tokens, default=0)
    largest_output_high = max(
        (
            estimate_chunk_output_tokens(
                detail,
                len(chunk.words),
                senses_per_word=senses_per_word,
                high_multiplier=1.50)
            for chunk in plan.chunks
        ),
        default=0)
    largest_total_high = largest_input + largest_output_high
    input_usd = (
        input_tokens
        * pricing.input_usd_per_million
        / 1_000_000)
    output_usd = (
        output_tokens
        * pricing.output_usd_per_million
        / 1_000_000)
    total = input_usd + output_usd
    web_search_high_usd = 0.0
    if web_search_enabled:
        web_search_high_usd = request_count * (
            WEB_SEARCH_USD_PER_CALL
            + (
                WEB_SEARCH_CONTENT_TOKENS_PER_CALL_HIGH
                * pricing.input_usd_per_million
                / 1_000_000))
    return CostEstimate(
        model=pricing.model,
        pricing_label=pricing.label,
        request_count=request_count,
        word_count=plan.word_count,
        prompt_tokens=prompt_tokens,
        payload_tokens=payload_tokens,
        schema_tokens=schema_tokens,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_input_usd=input_usd,
        estimated_output_usd=output_usd,
        estimated_total_usd=total,
        low_total_usd=total * 0.70,
        high_total_usd=total * 1.50 + web_search_high_usd,
        usd_to_aud_rate=USD_TO_AUD_RATE,
        assumptions={
            "token_estimator": (
                "deterministic character-class approximation; no API call"),
            "senses_per_word": senses_per_word,
            "prompt_tokens_per_request": prompt_tokens_per_request,
            "schema_tokens_per_request": schema_tokens_per_request,
            "field_output_tokens_per_sense": {
                field_key: _FIELD_OUTPUT_TOKENS.get(field_key, 24)
                for field_key in detail.field_keys
            },
            "example_sentence_tokens_per_sense": (
                _EXAMPLE_SENTENCE_OUTPUT_TOKENS
                if detail.include_example_sentences
                else 0),
            "context_deduplication_scope": "within each request",
            "web_search_enabled": bool(web_search_enabled),
            "web_search_max_calls_per_request": (
                1 if web_search_enabled else 0),
            "web_search_high_content_tokens_per_call": (
                WEB_SEARCH_CONTENT_TOKENS_PER_CALL_HIGH
                if web_search_enabled
                else 0),
            "web_search_high_cost_usd": web_search_high_usd,
            "web_search_tool_usd_per_call": (
                WEB_SEARCH_USD_PER_CALL
                if web_search_enabled
                else 0.0),
            "usd_to_aud_rate": USD_TO_AUD_RATE,
            "aud_exchange_rate": AUD_EXCHANGE_RATE_LABEL,
            "largest_request_estimated_input_tokens": largest_input,
            "largest_request_estimated_output_tokens": largest_output,
            "largest_request_high_output_tokens": largest_output_high,
            "largest_request_high_total_tokens": largest_total_high,
            "model_context_window_tokens": MODEL_CONTEXT_WINDOW_TOKENS,
            "model_max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
        },
    )
