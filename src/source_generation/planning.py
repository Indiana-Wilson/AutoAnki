"""Load audited corpora, repartition them, and estimate paid generation."""

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path

import runtime_paths
from corpus_pipeline.catalogue import get_corpus_spec
from corpus_pipeline.models import CORPUS_SCHEMA_VERSION, TokenOccurrence
from corpus_pipeline.processing import normalize_word
from corpus_pipeline.storage import (
    build_id,
    read_vocabulary_view,
    resolve_corpus_base,
    verify_artifact_hashes,
)
from source_generation.models import (
    ContextOccurrenceSpan,
    ContextMode,
    ContextUnit,
    CostEstimate,
    DEFAULT_SOURCE_MODEL,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    LoadedSource,
    OccurrenceLocator,
    OCCURRENCE_LOCATOR_WINDOW_CHARS,
    OCCURRENCE_TARGET_CLOSE,
    OCCURRENCE_TARGET_OPEN,
    OutputDetail,
    Pricing,
    SourceGenerationConfig,
)
from source_generation.model_catalog import source_model_profile


STANDARD_PRICING = Pricing(
    model=DEFAULT_SOURCE_MODEL,
    input_usd_per_million=0.75,
    output_usd_per_million=4.50,
    label="standard pricing published 2026-07-24",
    cached_input_usd_per_million=0.075,
)
BATCH_PRICING = Pricing(
    model=DEFAULT_SOURCE_MODEL,
    input_usd_per_million=0.375,
    output_usd_per_million=2.25,
    label="Batch/Flex pricing published 2026-07-24",
    cached_input_usd_per_million=0.0375,
)
DEFAULT_PRICING = STANDARD_PRICING
_PRICING_BY_MODEL_AND_MODE = {
    (STANDARD_PRICING.model, "standard"): STANDARD_PRICING,
    (BATCH_PRICING.model, "economy"): BATCH_PRICING,
}
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
MODEL_MAX_INPUT_TOKENS = 272_000
MODEL_MAX_OUTPUT_TOKENS = 128_000
# Reasoning tokens are billed as output tokens. Their exact count is not
# knowable before a request, so new low-effort source estimates reserve a
# conservative fraction on top of the visible structured response estimate.
LOW_REASONING_OUTPUT_RESERVE_MULTIPLIER = 1.50

# Prompt caching begins only for sufficiently long shared prefixes. The
# central estimate assumes that an identical prompt/schema prefix is cold once
# and cached on later requests. The high estimate remains fully cold.
PROMPT_CACHE_MINIMUM_TOKENS = 1_024

# These are explicit planning assumptions derived from the response shapes the
# application requests. They are not API guarantees. Generated-example mode
# retains the historical expectation of two senses per word. Source-example
# mode always has one lexical-only contextual sense, then a variable number of
# additional generated-example senses.
GENERATED_SENSES_PER_WORD_LOW_MULTIPLIER = 0.625
GENERATED_SENSES_PER_WORD_HIGH_MULTIPLIER = 1.50
SOURCE_ADDITIONAL_SENSES_PER_WORD_LOW = 0.50
SOURCE_ADDITIONAL_SENSES_PER_WORD_CENTRAL = 1.00
SOURCE_ADDITIONAL_SENSES_PER_WORD_HIGH = 2.00
SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_LOW = 0.75
SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_CENTRAL = 1.00
SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_HIGH = 1.50
LOW_REASONING_RESERVE_MULTIPLIERS = {
    "low": 1.05,
    "central": 1.20,
    "high": 1.50,
}

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
_EXAMPLE_SENTENCE_OUTPUT_TOKENS = 54
_EXAMPLE_TRANSLATION_OUTPUT_TOKENS = 54
_RESPONSE_ENVELOPE_TOKENS = 12
_AUTOMATIC_REPAIR_FIXED_INPUT_TOKENS = 400
_AUTOMATIC_REPAIR_INPUT_TOKENS_PER_PAIR = 220
_AUTOMATIC_REPAIR_OUTPUT_TOKENS_PER_PAIR = 80
_AUTOMATIC_REPAIR_MAX_PAIRS_PER_CALL = 64
_SOURCE_TERM_RESULT_OVERHEAD_TOKENS = 8
_SOURCE_CONTEXT_TRANSLATION_OVERHEAD_TOKENS = 8
_SOURCE_CONTEXT_NUANCE_OUTPUT_TOKENS = 28


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


def resolve_latest_processed_source_path(source_key, corpus_root=None):
    """Resolve the currently published immutable run for one source.

    Keeping this lightweight pointer lookup separate from full build loading
    lets long-lived GUI backends key their cache by the actual build path.
    A source re-prepared while the application is open must not keep resolving
    a stale object cached under the ambiguous value ``run_path=None``.
    """
    root = Path(
        corpus_root
        or runtime_paths.get_corpus_output_directory()).resolve()
    return _safe_latest_build_path(root, source_key)


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
        else resolve_latest_processed_source_path(
            source_key,
            corpus_root=corpus_root)
    )
    build = read_vocabulary_view(selected_path)
    manifest = json.loads(
        (selected_path / "manifest.json").read_text(encoding="utf-8"))
    token_occurrence_count = manifest.get(
        "counts",
        {}).get("token_occurrences")
    if (
            isinstance(token_occurrence_count, bool)
            or not isinstance(token_occurrence_count, int)
            or token_occurrence_count < 0):
        raise ValueError(
            "The processed source has an invalid token-occurrence count.")
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
        token_occurrence_count=token_occurrence_count,
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
            build_path = resolve_latest_processed_source_path(
                source_key,
                corpus_root=root)
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
            token_occurrence_count = counts["token_occurrences"]
            section_count = counts["sections"]
            if (
                    isinstance(word_count, bool)
                    or not isinstance(word_count, int)
                    or word_count < 0
                    or isinstance(token_occurrence_count, bool)
                    or not isinstance(token_occurrence_count, int)
                    or token_occurrence_count < 0
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
                "token_occurrence_count": token_occurrence_count,
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


def _first_occurrence_token_index(word):
    """Read the audited running-token index behind one vocabulary rank."""
    prefix = "token:"
    occurrence_id = word.first_occurrence_id
    if (
            not isinstance(occurrence_id, str)
            or not occurrence_id.startswith(prefix)
            or not occurrence_id[len(prefix):].isdecimal()):
        raise ValueError(
            "A source vocabulary item has an invalid first occurrence ID.")
    token_index = int(occurrence_id[len(prefix):])
    if token_index < 1:
        raise ValueError(
            "A source vocabulary item has an invalid first token index.")
    return token_index


def _source_token_occurrence_count(loaded_source):
    count = loaded_source.token_occurrence_count
    if count is not None:
        return count
    occurrences = getattr(loaded_source.build, "occurrences", None)
    if occurrences is not None:
        return len(occurrences)
    # Production vocabulary views carry the manifest count on LoadedSource.
    # This fallback keeps manually constructed integrations usable without
    # loading the large repeated-occurrence artifact.
    return max(
        (
            _first_occurrence_token_index(word)
            for word in loaded_source.build.unique_words
        ),
        default=0)


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


def _make_occurrence_locator(word, context):
    """Persist a bounded, visibly marked locator for one audited occurrence."""
    if context is None:
        return None
    relative_start = word.start_offset - context.start_offset
    relative_end = word.end_offset - context.start_offset
    if (
            relative_start < 0
            or relative_end <= relative_start
            or relative_end > len(context.text)
            or context.text[relative_start:relative_end] != word.surface):
        raise ValueError(
            "A source word's audited occurrence does not match its retained "
            "context.")
    ordinal = 0
    cursor = 0
    while True:
        match_start = context.text.find(word.surface, cursor)
        if match_start < 0 or match_start > relative_start:
            break
        ordinal += 1
        if match_start == relative_start:
            break
        cursor = match_start + len(word.surface)
    if match_start != relative_start:
        raise ValueError(
            "A source word's audited occurrence cannot be located inside its "
            "retained context.")
    before = context.text[
        max(
            0,
            relative_start - OCCURRENCE_LOCATOR_WINDOW_CHARS):
        relative_start]
    target = context.text[relative_start:relative_end]
    after = context.text[
        relative_end:
        relative_end + OCCURRENCE_LOCATOR_WINDOW_CHARS]
    return OccurrenceLocator(
        before=before,
        target=target,
        after=after,
        literal_match_ordinal=ordinal,
        marked_excerpt=(
            before
            + OCCURRENCE_TARGET_OPEN
            + target
            + OCCURRENCE_TARGET_CLOSE
            + after),
    )


def _load_selected_source_occurrences(loaded_source, normalized_terms):
    """Stream only requested token occurrences from a verified saved build."""
    occurrence_path = loaded_source.run_path / "occurrences.jsonl"
    if not occurrence_path.is_file():
        return {}
    verify_artifact_hashes(loaded_source.run_path)
    selected = {
        normalized: []
        for normalized in normalized_terms
    }
    with occurrence_path.open(
            "r",
            encoding="utf-8",
            newline="") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                occurrence = TokenOccurrence(**json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Invalid occurrences.jsonl record on line "
                    f"{line_number}.") from error
            if occurrence.normalized in selected:
                selected[occurrence.normalized].append(occurrence)
    return {
        normalized: tuple(occurrences)
        for normalized, occurrences in selected.items()
    }


def _chunk_context_occurrences(
        loaded_source,
        words,
        contexts_by_id,
        rank_to_context,
        occurrences_by_normalized):
    """Return complete tokenizer spans relative to each retained context."""
    canonical_text = loaded_source.build.snapshot.canonical_text
    result = {}
    for word in words:
        context = contexts_by_id.get(rank_to_context.get(word.rank))
        if context is None:
            continue
        spans = []
        for occurrence in occurrences_by_normalized.get(
                word.normalized,
                ()):
            if (
                    occurrence.start_offset < context.start_offset
                    or occurrence.end_offset > context.end_offset):
                continue
            if (
                    canonical_text[
                        occurrence.start_offset:occurrence.end_offset]
                    != occurrence.surface):
                raise ValueError(
                    "A saved token occurrence disagrees with canonical "
                    "source text.")
            relative_start = (
                occurrence.start_offset - context.start_offset)
            relative_end = occurrence.end_offset - context.start_offset
            if (
                    context.text[relative_start:relative_end]
                    != occurrence.surface):
                raise ValueError(
                    "A saved token occurrence disagrees with its retained "
                    "source context.")
            spans.append(ContextOccurrenceSpan(
                relative_start,
                relative_end,
                occurrence.surface))
        result[word.rank] = tuple(spans)
    return result


def plan_source_generation(loaded_source, config):
    """Repartition saved vocabulary and deduplicate context within each query."""
    if not isinstance(loaded_source, LoadedSource):
        raise TypeError("A loaded processed source is required.")
    config = SourceGenerationConfig.from_mapping(config)
    if loaded_source.key != config.source_key:
        raise ValueError(
            "Source generation settings select another processed source.")

    source_language_key = (
        loaded_source.build.snapshot.source_language_key)
    excluded = {
        normalize_word(
            word.strip(),
            source_language_key)
        for word in config.excluded_words
        if word.strip()
    }
    all_words = loaded_source.build.unique_words
    if config.source_prefix_token_limit is None:
        prefix_words = all_words
        prefix_token_count = None
    else:
        prefix_words = tuple(
            word
            for word in all_words
            if _first_occurrence_token_index(word)
            <= config.source_prefix_token_limit)
        prefix_token_count = min(
            config.source_prefix_token_limit,
            _source_token_occurrence_count(loaded_source))
    selected_words = tuple(
        word
        for word in prefix_words
        if word.normalized not in excluded
        and normalize_word(
            word.surface,
            source_language_key) not in excluded
    )
    groups = _partition_words(selected_words, config.chunk_size)
    occurrences_by_normalized = (
        _load_selected_source_occurrences(
            loaded_source,
            {
                word.normalized
                for word in selected_words
            })
        if config.request_protocol == "v10"
        else {})
    total = len(groups)
    chunks = []
    for index, words in enumerate(groups, start=1):
        context_units, rank_to_context = _make_context_units(
            loaded_source.build,
            words,
            config.context_mode,
            source_build_id=loaded_source.build_id)
        contexts_by_id = {
            context.context_id: context
            for context in context_units
        }
        context_occurrences_by_rank = (
            _chunk_context_occurrences(
                loaded_source,
                words,
                contexts_by_id,
                rank_to_context,
                occurrences_by_normalized)
            if config.request_protocol == "v10"
            else {})
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
                    context_id=rank_to_context.get(word.rank),
                    start_offset=word.start_offset,
                    end_offset=word.end_offset,
                    occurrence_locator=_make_occurrence_locator(
                        word,
                        contexts_by_id.get(
                            rank_to_context.get(word.rank))),
                    context_occurrences=(
                        context_occurrences_by_rank.get(
                            word.rank,
                            ())))
                for word in words
            ),
            contexts=context_units,
        ))

    identity = {
        "source_build_id": loaded_source.build_id,
        "chunk_size": config.chunk_size,
        "context_mode": config.context_mode.value,
        "source_prefix_token_limit": config.source_prefix_token_limit,
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
        original_word_count=len(all_words),
        excluded_word_count=len(prefix_words) - len(selected_words),
        chunks=tuple(chunks),
        source_prefix_token_count=prefix_token_count,
        prefix_unique_word_count=len(prefix_words),
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
        field_names.extend((
            "Sentences",
            "Sentence Translations (English)",
        ))
    # Strict JSON schema keywords and the repeated property/required names.
    return 80 + 2 * sum(
        estimate_text_tokens(field_name)
        for field_name in field_names)


def _tokens_per_sense(
        detail,
        *,
        include_term,
        include_generated_examples):
    return (
        (_TERM_OUTPUT_TOKENS if include_term else 0)
        + _JSON_OVERHEAD_TOKENS_PER_CARD
        + sum(
            _FIELD_OUTPUT_TOKENS.get(field_key, 24)
            for field_key in detail.field_keys)
        + (
            (
                _EXAMPLE_SENTENCE_OUTPUT_TOKENS
                + _EXAMPLE_TRANSLATION_OUTPUT_TOKENS)
            if (
                include_generated_examples
                and detail.include_example_sentences)
            else 0)
    )


def estimate_chunk_output_tokens(
        detail,
        word_count,
        *,
        senses_per_word=2.0,
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
    tokens_per_sense = _tokens_per_sense(
        detail,
        include_term=True,
        include_generated_examples=True)
    return math.ceil(
        (
            word_count
            * senses_per_word
            * tokens_per_sense
            + _RESPONSE_ENVELOPE_TOKENS)
        * high_multiplier)


def _source_output_detail_from_pipeline(pipeline):
    """Describe only fields requested for source lexical-direction cards."""
    import pipeline_store

    settings = pipeline_store.get_source_lexical_field_settings(pipeline)
    return OutputDetail(
        field_keys=tuple(
            setting.field_key
            for setting in settings),
        response_language_keys=tuple(
            setting.target_language_key
            for setting in settings),
        include_example_sentences=bool(settings),
        term_field_name=(
            pipeline_store.get_language(
                pipeline.language_key).term_field),
    )


def _source_chunk_output_tokens(
        detail,
        chunk,
        *,
        additional_senses_per_word,
        context_translation_token_ratio,
        remembered_context_ids=(),
        include_source_context_nuance=False):
    """Estimate the visible compact source-context response shape.

    Contextual entries are lexical-only, additional entries include generated
    examples when lexical cards are selected, and each deduplicated source
    sentence is translated exactly once. Terms and original source sentences
    are restored locally rather than echoed by the model.
    """
    contextual_tokens_per_word = _tokens_per_sense(
        detail,
        include_term=False,
        include_generated_examples=False)
    additional_tokens_per_sense = _tokens_per_sense(
        detail,
        include_term=False,
        include_generated_examples=detail.include_example_sentences)
    effective_additional_senses = (
        additional_senses_per_word
        if detail.field_keys
        else 0)
    context_translation_tokens = sum(
        (
            math.ceil(
                estimate_text_tokens(
                    getattr(context, "text", ""))
                * context_translation_token_ratio)
            + _SOURCE_CONTEXT_TRANSLATION_OVERHEAD_TOKENS)
        for context in chunk.contexts
        if context.context_id not in remembered_context_ids
    )
    context_nuance_tokens = (
        len(chunk.contexts) * _SOURCE_CONTEXT_NUANCE_OUTPUT_TOKENS
        if include_source_context_nuance
        else 0)
    return math.ceil(
        (
            len(chunk.words)
            * (
                contextual_tokens_per_word
                + _SOURCE_TERM_RESULT_OVERHEAD_TOKENS
                + (
                    effective_additional_senses
                    * additional_tokens_per_sense))
        )
        + context_translation_tokens
        + context_nuance_tokens
        + _RESPONSE_ENVELOPE_TOKENS)


def estimate_source_chunk_output_tokens(
        detail,
        chunk,
        *,
        include_source_context_nuance=False,
        high_multiplier=1.0):
    """Estimate a source-sentence response for output-limit reservation."""
    if isinstance(detail, dict):
        detail = OutputDetail.from_mapping(detail)
    elif not isinstance(detail, OutputDetail):
        detail = _source_output_detail_from_pipeline(detail)
    if not isinstance(include_source_context_nuance, bool):
        raise TypeError("Source-context nuance mode must be true or false.")
    if (
            isinstance(high_multiplier, bool)
            or not isinstance(high_multiplier, (int, float))
            or high_multiplier < 1):
        raise ValueError("Output safety multiplier must be at least one.")
    return math.ceil(
        _source_chunk_output_tokens(
            detail,
            chunk,
            additional_senses_per_word=(
                SOURCE_ADDITIONAL_SENSES_PER_WORD_HIGH),
            context_translation_token_ratio=(
                SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_HIGH),
            include_source_context_nuance=(
                include_source_context_nuance))
        * high_multiplier)


def _reasoning_multiplier(reasoning_effort, scenario):
    if reasoning_effort == "none":
        return 1.0
    if reasoning_effort == "low":
        return LOW_REASONING_RESERVE_MULTIPLIERS[scenario]
    raise ValueError(
        'Source reasoning effort must be either "none" or "low".')


def _pricing_for_execution_mode(
        execution_mode,
        model=DEFAULT_SOURCE_MODEL):
    try:
        return _PRICING_BY_MODEL_AND_MODE[(model, execution_mode)]
    except KeyError as error:
        if execution_mode not in {"standard", "economy"}:
            raise ValueError(
                'Source execution mode must be either "standard" or '
                '"economy".') from error
        raise ValueError(
            f"No source-generation pricing is configured for {model!r}.") \
            from error


def _usd_for_tokens(tokens, rate_per_million):
    return tokens * rate_per_million / 1_000_000


def price_source_usage(
        usage,
        *,
        execution_mode="standard",
        model=DEFAULT_SOURCE_MODEL,
        pricing=None):
    """Price an exact normalized source-generation usage summary.

    ``cached_input_tokens`` and ``uncached_input_tokens`` are disjoint.
    ``output_tokens`` includes both visible and reasoning output, matching the
    billable total returned by the Responses API.
    """
    if not isinstance(usage, dict):
        raise TypeError("Source usage must be an object.")
    token_counts = {}
    for key in (
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens"):
        value = usage.get(key, 0)
        if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0):
            raise ValueError(
                f"Source usage {key} must be a non-negative integer.")
        token_counts[key] = value
    web_search_calls = usage.get("web_search_calls", 0)
    if (
            isinstance(web_search_calls, bool)
            or not isinstance(web_search_calls, int)
            or web_search_calls < 0):
        raise ValueError(
            "Source usage web_search_calls must be a non-negative integer.")

    if pricing is None:
        pricing = _pricing_for_execution_mode(execution_mode, model)
    elif isinstance(pricing, dict):
        pricing = Pricing(**pricing)
    if not isinstance(pricing, Pricing):
        raise TypeError("A pricing profile is required.")
    cached_input_rate = (
        pricing.cached_input_usd_per_million
        if pricing.cached_input_usd_per_million is not None
        else pricing.input_usd_per_million)
    uncached_input_usd = _usd_for_tokens(
        token_counts["uncached_input_tokens"],
        pricing.input_usd_per_million)
    cached_input_usd = _usd_for_tokens(
        token_counts["cached_input_tokens"],
        cached_input_rate)
    cache_write_rate = (
        pricing.cache_write_input_usd_per_million
        if pricing.cache_write_input_usd_per_million is not None
        else pricing.input_usd_per_million)
    cache_write_input_usd = _usd_for_tokens(
        token_counts["cache_write_input_tokens"],
        cache_write_rate)
    output_usd = _usd_for_tokens(
        token_counts["output_tokens"],
        pricing.output_usd_per_million)
    web_search_usd = (
        web_search_calls * WEB_SEARCH_USD_PER_CALL)
    input_usd = (
        uncached_input_usd
        + cached_input_usd
        + cache_write_input_usd)
    total_usd = input_usd + output_usd + web_search_usd

    return {
        "execution_mode": execution_mode,
        "model": pricing.model,
        "pricing_label": pricing.label,
        "pricing": pricing.to_dict(),
        **token_counts,
        "web_search_calls": web_search_calls,
        "input_tokens": (
            token_counts["uncached_input_tokens"]
            + token_counts["cached_input_tokens"]
            + token_counts["cache_write_input_tokens"]),
        "total_tokens": (
            token_counts["uncached_input_tokens"]
            + token_counts["cached_input_tokens"]
            + token_counts["cache_write_input_tokens"]
            + token_counts["output_tokens"]),
        "uncached_input_usd": uncached_input_usd,
        "cached_input_usd": cached_input_usd,
        "cache_write_input_usd": cache_write_input_usd,
        "input_usd": input_usd,
        "output_usd": output_usd,
        "web_search_usd": web_search_usd,
        "total_usd": total_usd,
        "uncached_input_aud": uncached_input_usd * USD_TO_AUD_RATE,
        "cached_input_aud": cached_input_usd * USD_TO_AUD_RATE,
        "cache_write_input_aud": (
            cache_write_input_usd * USD_TO_AUD_RATE),
        "input_aud": input_usd * USD_TO_AUD_RATE,
        "output_aud": output_usd * USD_TO_AUD_RATE,
        "web_search_aud": web_search_usd * USD_TO_AUD_RATE,
        "total_aud": total_usd * USD_TO_AUD_RATE,
        "usd_to_aud_rate": USD_TO_AUD_RATE,
        "aud_exchange_rate": AUD_EXCHANGE_RATE_LABEL,
    }


def estimate_plan_cost(
        plan,
        detail,
        *,
        prompt_text="",
        pricing=None,
        senses_per_word=2.0,
        web_search_enabled=False,
        response_format=None,
        use_source_for_example_sentences=False,
        request_protocol=None,
        reasoning_effort=None,
        execution_mode=None,
        response_formats_by_chunk=None,
        translation_memory_by_chunk=None,
        include_source_context_nuance=False):
    """Estimate one source plan without contacting OpenAI.

    A fixed ``response_format`` represents a compact reusable schema.
    ``response_formats_by_chunk`` represents v8's exact dynamic schemas.
    """
    if not isinstance(plan, GenerationPlan):
        raise TypeError("A source generation plan is required.")
    if not isinstance(use_source_for_example_sentences, bool):
        raise TypeError(
            "Source-example response mode must be true or false.")
    if not isinstance(include_source_context_nuance, bool):
        raise TypeError("Source-context nuance mode must be true or false.")
    if (
            include_source_context_nuance
            and not use_source_for_example_sentences):
        raise ValueError(
            "Source-context nuance requires retained source sentence cards.")
    if isinstance(detail, dict):
        detail = OutputDetail.from_mapping(detail)
    elif not isinstance(detail, OutputDetail):
        detail = (
            _source_output_detail_from_pipeline(detail)
            if use_source_for_example_sentences
            else OutputDetail.from_pipeline(detail))
    if (
            isinstance(senses_per_word, bool)
            or not isinstance(senses_per_word, (int, float))
            or senses_per_word < 1):
        raise ValueError("Expected senses per word must be at least one.")
    reasoning_effort = (
        plan.config.reasoning_effort
        if reasoning_effort is None
        else reasoning_effort)
    request_protocol = (
        plan.config.request_protocol
        if request_protocol is None
        else request_protocol)
    execution_mode = (
        plan.config.execution_mode
        if execution_mode is None
        else execution_mode)
    if request_protocol not in {"v8", "v9", "v10"}:
        raise ValueError(
            'Source request protocol must be "v8", "v9", or "v10".')
    if request_protocol == "v8" and reasoning_effort != "low":
        raise ValueError(
            'Source request protocol "v8" requires reasoning effort "low".')
    _reasoning_multiplier(reasoning_effort, "central")
    selected_pricing = _pricing_for_execution_mode(
        execution_mode,
        plan.config.model)
    if pricing is None:
        pricing = selected_pricing
    if not isinstance(pricing, Pricing):
        raise TypeError("A pricing profile is required.")
    if (
            response_format is not None
            and response_formats_by_chunk is not None):
        raise ValueError(
            "Choose either one fixed response format or per-chunk response "
            "formats, not both.")
    if (
            request_protocol in {"v9", "v10"}
            and response_formats_by_chunk is not None):
        raise ValueError(
            "Compact source request protocols require one fixed response "
            "format.")
    if (
            request_protocol == "v8"
            and use_source_for_example_sentences
            and response_format is not None):
        raise ValueError(
            "Source-context request protocol v8 requires per-chunk response "
            "formats.")
    if (
            request_protocol == "v8"
            and use_source_for_example_sentences
            and response_formats_by_chunk is None):
        raise ValueError(
            "Source-context request protocol v8 requires per-chunk response "
            "formats.")
    if translation_memory_by_chunk is None:
        translation_memory_by_chunk = {}
    if not isinstance(translation_memory_by_chunk, dict):
        raise TypeError(
            "Source translation-memory snapshot must be an object.")
    expected_chunk_ids = {
        chunk.chunk_id
        for chunk in plan.chunks
    }
    unknown_memory_chunks = (
        set(translation_memory_by_chunk) - expected_chunk_ids)
    if unknown_memory_chunks:
        raise ValueError(
            "Source translation memory contains unknown request chunks.")
    for chunk_id, hits in translation_memory_by_chunk.items():
        if not isinstance(hits, dict):
            raise TypeError(
                f"Translation-memory hits for {chunk_id} must be an object.")

    # Imported here to keep planning's module import independent from the
    # request module, whose contract builder imports estimator constants.
    from source_generation.requests import render_chunk_input

    request_count = len(plan.chunks)
    prompt_tokens_per_request = estimate_text_tokens(prompt_text)
    prompt_tokens = prompt_tokens_per_request * request_count
    payload_tokens_by_chunk = {
        chunk.chunk_id: estimate_text_tokens(
            render_chunk_input(
                chunk,
                protocol_version=int(
                    request_protocol.removeprefix("v")),
                source_context_translation_memory=(
                    translation_memory_by_chunk.get(
                        chunk.chunk_id))))
        for chunk in plan.chunks
    }
    payload_tokens = sum(payload_tokens_by_chunk.values())

    schema_cache_shared = True
    if response_format is not None:
        if not isinstance(response_format, dict):
            raise TypeError("The fixed response format must be an object.")
        schema_tokens_per_request = estimate_text_tokens(
            _canonical_json(response_format))
        schema_tokens_by_chunk = {
            chunk.chunk_id: schema_tokens_per_request
            for chunk in plan.chunks
        }
        schema_estimation_mode = "fixed_exact"
    elif response_formats_by_chunk is None:
        schema_tokens_per_request = _schema_token_estimate(detail)
        schema_tokens_by_chunk = {
            chunk.chunk_id: schema_tokens_per_request
            for chunk in plan.chunks
        }
        schema_estimation_mode = "detail_approximation"
    else:
        if not isinstance(response_formats_by_chunk, dict):
            raise TypeError(
                "Per-chunk response formats must be an object.")
        expected_chunk_ids = {
            chunk.chunk_id
            for chunk in plan.chunks
        }
        if set(response_formats_by_chunk) != expected_chunk_ids:
            raise ValueError(
                "Per-chunk response formats do not match the generation "
                "plan.")
        canonical_formats = {
            chunk_id: _canonical_json(response_formats_by_chunk[chunk_id])
            for chunk_id in expected_chunk_ids
        }
        schema_tokens_by_chunk = {
            chunk_id: estimate_text_tokens(canonical)
            for chunk_id, canonical in canonical_formats.items()
        }
        schema_tokens_per_request = max(
            schema_tokens_by_chunk.values(),
            default=0)
        schema_cache_shared = len(set(canonical_formats.values())) <= 1
        schema_estimation_mode = "per_chunk_exact"
    schema_tokens = sum(schema_tokens_by_chunk.values())
    input_tokens = prompt_tokens + payload_tokens + schema_tokens

    if use_source_for_example_sentences:
        visible_output_tokens_by_scenario = {
            "low": tuple(
                _source_chunk_output_tokens(
                    detail,
                    chunk,
                    additional_senses_per_word=(
                        SOURCE_ADDITIONAL_SENSES_PER_WORD_LOW),
                    context_translation_token_ratio=(
                        SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_LOW),
                    remembered_context_ids=set(
                        translation_memory_by_chunk.get(
                            chunk.chunk_id,
                            ())),
                    include_source_context_nuance=(
                        include_source_context_nuance))
                for chunk in plan.chunks),
            "central": tuple(
                _source_chunk_output_tokens(
                    detail,
                    chunk,
                    additional_senses_per_word=(
                        SOURCE_ADDITIONAL_SENSES_PER_WORD_CENTRAL),
                    context_translation_token_ratio=(
                        SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_CENTRAL),
                    remembered_context_ids=set(
                        translation_memory_by_chunk.get(
                            chunk.chunk_id,
                            ())),
                    include_source_context_nuance=(
                        include_source_context_nuance))
                for chunk in plan.chunks),
            "high": tuple(
                _source_chunk_output_tokens(
                    detail,
                    chunk,
                    additional_senses_per_word=(
                        SOURCE_ADDITIONAL_SENSES_PER_WORD_HIGH),
                    context_translation_token_ratio=(
                        SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_HIGH),
                    remembered_context_ids=set(
                        translation_memory_by_chunk.get(
                            chunk.chunk_id,
                            ())),
                    include_source_context_nuance=(
                        include_source_context_nuance))
                for chunk in plan.chunks),
        }
        response_shape = "source_sentences_plus_lexical_senses"
        output_shape_assumptions = {
            "contextual_senses_per_word": 1.0,
            "contextual_senses_include_generated_examples": False,
            "additional_senses_per_word": {
                "low": SOURCE_ADDITIONAL_SENSES_PER_WORD_LOW,
                "central": SOURCE_ADDITIONAL_SENSES_PER_WORD_CENTRAL,
                "high": SOURCE_ADDITIONAL_SENSES_PER_WORD_HIGH,
            },
            "additional_senses_include_generated_examples": bool(
                detail.include_example_sentences),
            "source_context_translations": (
                "one output translation per deduplicated context"),
            "source_context_nuance": bool(
                include_source_context_nuance),
            "source_context_translation_token_ratio": {
                "low": SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_LOW,
                "central": (
                    SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_CENTRAL),
                "high": SOURCE_CONTEXT_TRANSLATION_TOKEN_RATIO_HIGH,
            },
        }
    else:
        low_senses_per_word = max(
            1.0,
            senses_per_word
            * GENERATED_SENSES_PER_WORD_LOW_MULTIPLIER)
        high_senses_per_word = (
            senses_per_word
            * GENERATED_SENSES_PER_WORD_HIGH_MULTIPLIER)
        visible_output_tokens_by_scenario = {
            "low": tuple(
                estimate_chunk_output_tokens(
                    detail,
                    len(chunk.words),
                    senses_per_word=low_senses_per_word)
                for chunk in plan.chunks),
            "central": tuple(
                estimate_chunk_output_tokens(
                    detail,
                    len(chunk.words),
                    senses_per_word=senses_per_word)
                for chunk in plan.chunks),
            "high": tuple(
                estimate_chunk_output_tokens(
                    detail,
                    len(chunk.words),
                    senses_per_word=high_senses_per_word)
                for chunk in plan.chunks),
        }
        response_shape = "generated_examples_for_each_sense"
        output_shape_assumptions = {
            "senses_per_word": {
                "low": low_senses_per_word,
                "central": senses_per_word,
                "high": high_senses_per_word,
            },
            "generated_examples_for_each_sense": bool(
                detail.include_example_sentences),
        }

    output_tokens_by_scenario = {
        scenario: tuple(
            math.ceil(
                visible_tokens
                * _reasoning_multiplier(
                    reasoning_effort,
                    scenario))
            for visible_tokens in visible_by_chunk)
        for scenario, visible_by_chunk
        in visible_output_tokens_by_scenario.items()
    }
    request_output_tokens = output_tokens_by_scenario["central"]
    output_tokens = sum(request_output_tokens)
    low_output_tokens = sum(output_tokens_by_scenario["low"])
    high_output_tokens = sum(output_tokens_by_scenario["high"])
    request_input_tokens = tuple(
        prompt_tokens_per_request
        + schema_tokens_by_chunk[chunk.chunk_id]
        + payload_tokens_by_chunk[chunk.chunk_id]
        for chunk in plan.chunks
    )
    largest_input = max(request_input_tokens, default=0)
    largest_output = max(request_output_tokens, default=0)
    largest_output_high = max(
        output_tokens_by_scenario["high"],
        default=0)
    largest_total_high = largest_input + largest_output_high

    shared_cache_prefix_tokens = prompt_tokens_per_request
    cache_mode = source_model_profile(
        plan.config.model).prompt_cache_mode
    if cache_mode == "implicit" and schema_cache_shared:
        shared_cache_prefix_tokens += schema_tokens_per_request
    cache_eligible = (
        cache_mode != "none"
        and
        request_count > 1
        and shared_cache_prefix_tokens >= PROMPT_CACHE_MINIMUM_TOKENS)
    cache_write_input_tokens = (
        shared_cache_prefix_tokens
        if (
            cache_eligible
            and cache_mode == "explicit_write")
        else 0)
    cached_input_tokens = (
        (request_count - 1) * shared_cache_prefix_tokens
        if cache_eligible
        else 0)
    cached_input_tokens = min(input_tokens, cached_input_tokens)
    cache_write_input_tokens = min(
        input_tokens - cached_input_tokens,
        cache_write_input_tokens)
    uncached_input_tokens = (
        input_tokens
        - cached_input_tokens
        - cache_write_input_tokens)
    cached_input_rate = (
        pricing.cached_input_usd_per_million
        if pricing.cached_input_usd_per_million is not None
        else pricing.input_usd_per_million)
    cached_input_usd = _usd_for_tokens(
        cached_input_tokens,
        cached_input_rate)
    cache_write_input_rate = (
        pricing.cache_write_input_usd_per_million
        if pricing.cache_write_input_usd_per_million is not None
        else pricing.input_usd_per_million)
    cache_write_input_usd = _usd_for_tokens(
        cache_write_input_tokens,
        cache_write_input_rate)
    uncached_input_usd = _usd_for_tokens(
        uncached_input_tokens,
        pricing.input_usd_per_million)
    input_usd = (
        cached_input_usd
        + cache_write_input_usd
        + uncached_input_usd)
    if (
            cache_eligible
            and cache_mode == "explicit_write"):
        high_cache_write_tokens = min(
            input_tokens,
            request_count * shared_cache_prefix_tokens)
        cold_input_usd = (
            _usd_for_tokens(
                high_cache_write_tokens,
                cache_write_input_rate)
            + _usd_for_tokens(
                input_tokens - high_cache_write_tokens,
                pricing.input_usd_per_million))
    else:
        cold_input_usd = _usd_for_tokens(
            input_tokens,
            pricing.input_usd_per_million)
    low_output_usd = _usd_for_tokens(
        low_output_tokens,
        pricing.output_usd_per_million)
    output_usd = _usd_for_tokens(
        output_tokens,
        pricing.output_usd_per_million)
    high_output_usd = _usd_for_tokens(
        high_output_tokens,
        pricing.output_usd_per_million)
    total = input_usd + output_usd

    web_search_high_usd = 0.0
    if web_search_enabled:
        web_search_high_usd = request_count * (
            WEB_SEARCH_USD_PER_CALL
            + _usd_for_tokens(
                WEB_SEARCH_CONTENT_TOKENS_PER_CALL_HIGH,
                pricing.input_usd_per_million))
    automatic_repair_max_requests = (
        request_count * plan.config.max_automatic_repairs
        if plan.config.automatic_repair
        else 0)
    automatic_repair_input_tokens = 0
    automatic_repair_output_tokens = 0
    if automatic_repair_max_requests:
        per_round_input = 0
        per_round_output = 0
        for chunk in plan.chunks:
            target_count = min(
                _AUTOMATIC_REPAIR_MAX_PAIRS_PER_CALL,
                len(chunk.words) * 4)
            per_round_input += (
                _AUTOMATIC_REPAIR_FIXED_INPUT_TOKENS
                + (
                    target_count
                    * _AUTOMATIC_REPAIR_INPUT_TOKENS_PER_PAIR))
            per_round_output += (
                _RESPONSE_ENVELOPE_TOKENS
                + (
                    target_count
                    * _AUTOMATIC_REPAIR_OUTPUT_TOKENS_PER_PAIR))
        automatic_repair_input_tokens = (
            per_round_input
            * plan.config.max_automatic_repairs)
        automatic_repair_output_tokens = (
            per_round_output
            * plan.config.max_automatic_repairs)
    automatic_repair_reserve_usd = (
        _usd_for_tokens(
            automatic_repair_input_tokens,
            pricing.input_usd_per_million)
        + _usd_for_tokens(
            automatic_repair_output_tokens,
            pricing.output_usd_per_million))
    translation_memory_hit_count = sum(
        len(hits)
        for hits in translation_memory_by_chunk.values())
    translation_memory_digest = (
        hashlib.sha256(
            _canonical_json(
                translation_memory_by_chunk).encode("utf-8")).hexdigest()
        if translation_memory_hit_count
        else None)

    return CostEstimate(
        model=pricing.model,
        pricing_label=pricing.label,
        request_count=request_count,
        word_count=plan.word_count,
        prompt_tokens=prompt_tokens,
        payload_tokens=payload_tokens,
        schema_tokens=schema_tokens,
        estimated_input_tokens=input_tokens,
        estimated_cached_input_tokens=cached_input_tokens,
        estimated_cache_write_input_tokens=cache_write_input_tokens,
        estimated_uncached_input_tokens=uncached_input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_input_usd=input_usd,
        estimated_cached_input_usd=cached_input_usd,
        estimated_cache_write_input_usd=cache_write_input_usd,
        estimated_uncached_input_usd=uncached_input_usd,
        estimated_output_usd=output_usd,
        estimated_total_usd=total,
        low_total_usd=input_usd + low_output_usd,
        high_total_usd=(
            cold_input_usd
            + high_output_usd
            + web_search_high_usd),
        usd_to_aud_rate=USD_TO_AUD_RATE,
        assumptions={
            "token_estimator": (
                "deterministic character-class approximation; no API call"),
            "estimate_scenarios": (
                "empirical low/central/high response-shape assumptions; "
                "not an API guarantee"),
            "response_shape": response_shape,
            "output_shape": output_shape_assumptions,
            "reasoning_effort": reasoning_effort,
            "reasoning_output_reserve_multiplier": (
                _reasoning_multiplier(
                    reasoning_effort,
                    "central")),
            "reasoning_output_reserve_multipliers": {
                scenario: _reasoning_multiplier(
                    reasoning_effort,
                    scenario)
                for scenario in ("low", "central", "high")
            },
            "visible_output_tokens": {
                scenario: sum(tokens)
                for scenario, tokens
                in visible_output_tokens_by_scenario.items()
            },
            "output_tokens": {
                scenario: sum(tokens)
                for scenario, tokens
                in output_tokens_by_scenario.items()
            },
            "execution_mode": execution_mode,
            "request_protocol": request_protocol,
            "pricing": pricing.to_dict(),
            "prompt_tokens_per_request": prompt_tokens_per_request,
            "schema_estimation_mode": schema_estimation_mode,
            "schema_tokens_per_request": schema_tokens_per_request,
            "schema_tokens_by_chunk": dict(schema_tokens_by_chunk),
            "central_cache_assumption": (
                "the first identical prompt/schema prefix is cold and later "
                "eligible prefixes are cached; payloads and dynamic schemas "
                "remain uncached"),
            "high_input_assumption": "all input tokens are cold",
            "prompt_cache_minimum_tokens": (
                PROMPT_CACHE_MINIMUM_TOKENS),
            "shared_cache_prefix_tokens_per_later_request": (
                shared_cache_prefix_tokens
                if cache_eligible
                else 0),
            "estimated_cached_input_tokens": cached_input_tokens,
            "estimated_cache_write_input_tokens": (
                cache_write_input_tokens),
            "estimated_uncached_input_tokens": uncached_input_tokens,
            "cold_input_usd": cold_input_usd,
            "field_output_tokens_per_sense": {
                field_key: _FIELD_OUTPUT_TOKENS.get(field_key, 24)
                for field_key in detail.field_keys
            },
            "example_sentence_tokens_per_sense": (
                _EXAMPLE_SENTENCE_OUTPUT_TOKENS
                if (
                    detail.include_example_sentences
                    and not use_source_for_example_sentences)
                else 0),
            "example_translation_tokens_per_sense": (
                _EXAMPLE_TRANSLATION_OUTPUT_TOKENS
                if (
                    detail.include_example_sentences
                    and not use_source_for_example_sentences)
                else 0),
            "context_deduplication_scope": "within each request",
            "translation_memory_hit_count": (
                translation_memory_hit_count),
            "translation_memory_sha256": (
                translation_memory_digest),
            "automatic_repair_enabled": (
                plan.config.automatic_repair),
            "automatic_repair_max_requests": (
                automatic_repair_max_requests),
            "automatic_repair_reserve_input_tokens": (
                automatic_repair_input_tokens),
            "automatic_repair_reserve_output_tokens": (
                automatic_repair_output_tokens),
            "automatic_repair_reserve_usd": (
                automatic_repair_reserve_usd),
            "automatic_repair_reserve_aud": (
                automatic_repair_reserve_usd * USD_TO_AUD_RATE),
            "automatic_repair_reserve_basis": (
                "conservative upper reserve if every base request uses all "
                "authorized micro-repair dispatches at up to 64 example "
                "pairs per call; excluded from headline and range"),
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
            "model_max_input_tokens": MODEL_MAX_INPUT_TOKENS,
            "model_max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
        },
    )
