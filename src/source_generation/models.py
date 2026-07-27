"""Immutable records for source-based generation planning and persistence."""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from corpus_pipeline.models import CorpusVocabularyView
from source_generation.model_catalog import (
    DEFAULT_SOURCE_MODEL,
    SUPPORTED_SOURCE_MODELS,
    source_model_profile,
)


SOURCE_GENERATION_SCHEMA_VERSION = 1
MAX_AUTOMATIC_REPAIR_ATTEMPTS = 3
OCCURRENCE_LOCATOR_WINDOW_CHARS = 24
OCCURRENCE_TARGET_OPEN = "⟪TARGET⟫"
OCCURRENCE_TARGET_CLOSE = "⟪/TARGET⟫"


class ContextMode(str, Enum):
    """How much retained source text accompanies each vocabulary item."""

    NONE = "none"
    SENTENCE = "sentence"
    SENTENCE_NEIGHBORS = "sentence_neighbors"
    CHUNK_SPAN = "chunk_span"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        aliases = {
            "current": cls.SENTENCE,
            "current_sentence": cls.SENTENCE,
            "current±1": cls.SENTENCE_NEIGHBORS,
            "current+/-1": cls.SENTENCE_NEIGHBORS,
            "neighbors": cls.SENTENCE_NEIGHBORS,
            "whole_chunk": cls.CHUNK_SPAN,
            "section": cls.CHUNK_SPAN,
        }
        text = str(value)
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown source context mode {value!r}; choose {choices}.") \
                from error


@dataclass(frozen=True)
class AnkiExclusionSpec:
    """A future-facing description of vocabulary that Anki should exclude.

    An Anki integration resolves this description to ``excluded_words`` before
    planning.  Keeping the description in the configuration makes a job fully
    inspectable without coupling this offline package to a running Anki client.
    """

    deck_name: str
    note_type: str | None = None
    field_name: str | None = None
    card_template_name: str | None = None

    def __post_init__(self):
        if not isinstance(self.deck_name, str) or not self.deck_name.strip():
            raise ValueError("An Anki exclusion deck name is required.")
        if self.note_type is not None and (
                not isinstance(self.note_type, str)
                or not self.note_type.strip()):
            raise ValueError("Anki exclusion note type cannot be empty.")
        if self.field_name is not None and (
                not isinstance(self.field_name, str)
                or not self.field_name.strip()):
            raise ValueError("Anki exclusion field name cannot be empty.")
        if self.card_template_name is not None and (
                not isinstance(self.card_template_name, str)
                or not self.card_template_name.strip()):
            raise ValueError(
                "Anki exclusion card template name cannot be empty.")

    @classmethod
    def from_mapping(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Anki exclusion settings must be an object.")
        return cls(
            deck_name=value.get("deck_name", value.get("deck", "")),
            note_type=value.get("note_type", value.get("model")),
            field_name=value.get("field_name", value.get("field")),
            card_template_name=value.get(
                "card_template_name",
                value.get("card_template")),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SourceGenerationConfig:
    """Free planning options and bounded paid-work concurrency settings."""

    source_key: str
    chunk_size: int = 30
    source_prefix_token_limit: int | None = None
    context_mode: ContextMode = ContextMode.SENTENCE
    run_path: str | None = None
    concurrency: int = 8
    request_stagger_ms: int = 100
    max_transient_retries: int = 3
    model: str = DEFAULT_SOURCE_MODEL
    request_protocol: str = "v10"
    reasoning_effort: str = "low"
    execution_mode: str = "standard"
    automatic_repair: bool = False
    max_automatic_repairs: int = MAX_AUTOMATIC_REPAIR_ATTEMPTS
    excluded_words: tuple[str, ...] = ()
    retain_excluded_source_contexts: bool = False
    exclude_anki: AnkiExclusionSpec | None = None
    anki_exclusions: tuple[AnkiExclusionSpec, ...] = ()

    def __post_init__(self):
        if not isinstance(self.source_key, str) or not self.source_key.strip():
            raise ValueError("A processed source key is required.")
        if (
                isinstance(self.chunk_size, bool)
                or not isinstance(self.chunk_size, int)
                or self.chunk_size < 1):
            raise ValueError("Generation chunk size must be a positive integer.")
        if (
                self.source_prefix_token_limit is not None
                and (
                    isinstance(self.source_prefix_token_limit, bool)
                    or not isinstance(self.source_prefix_token_limit, int)
                    or self.source_prefix_token_limit < 1)):
            raise ValueError(
                "Source prefix length must be a positive integer.")
        if (
                isinstance(self.concurrency, bool)
                or not isinstance(self.concurrency, int)
                or not 1 <= self.concurrency <= 64):
            raise ValueError("Generation concurrency must be between 1 and 64.")
        if (
                isinstance(self.request_stagger_ms, bool)
                or not isinstance(self.request_stagger_ms, int)
                or not 0 <= self.request_stagger_ms <= 60_000):
            raise ValueError(
                "Request staggering must be between 0 and 60000 milliseconds.")
        if (
                isinstance(self.max_transient_retries, bool)
                or not isinstance(self.max_transient_retries, int)
                or not 0 <= self.max_transient_retries <= 20):
            raise ValueError(
                "Transient retry count must be between 0 and 20.")
        if (
                not isinstance(self.model, str)
                or self.model not in SUPPORTED_SOURCE_MODELS):
            raise ValueError(
                "Source model must be one of: "
                + ", ".join(sorted(SUPPORTED_SOURCE_MODELS))
                + ".")
        if (
                not isinstance(self.request_protocol, str)
                or self.request_protocol not in {"v8", "v9", "v10"}):
            raise ValueError(
                'Source request protocol must be "v8", "v9", or "v10".')
        if (
                not isinstance(self.reasoning_effort, str)
                or self.reasoning_effort not in {"none", "low"}):
            raise ValueError(
                'Source reasoning effort must be either "none" or "low".')
        if (
                not isinstance(self.execution_mode, str)
                or self.execution_mode not in {"standard", "economy"}):
            raise ValueError(
                'Source execution mode must be either "standard" or '
                '"economy".')
        if not isinstance(self.automatic_repair, bool):
            raise ValueError(
                "Automatic repair must be enabled or disabled.")
        model_profile = source_model_profile(self.model)
        maximum_automatic_repairs = model_profile.max_automatic_repairs
        if (
                model_profile.local
                and self.automatic_repair
                and self.max_automatic_repairs
                == MAX_AUTOMATIC_REPAIR_ATTEMPTS):
            object.__setattr__(
                self,
                "max_automatic_repairs",
                maximum_automatic_repairs)
        if (
                isinstance(self.max_automatic_repairs, bool)
                or not isinstance(self.max_automatic_repairs, int)
                or not 0 <= self.max_automatic_repairs
                <= maximum_automatic_repairs):
            raise ValueError(
                "Automatic repairs must be between 0 and "
                f"{maximum_automatic_repairs} for the selected model.")
        if self.automatic_repair and self.execution_mode != "standard":
            raise ValueError(
                "Automatic paid repair currently requires Standard "
                "processing.")
        if (
                model_profile.local
                and self.request_protocol != "v10"):
            raise ValueError(
                "Local source models require request protocol v10.")
        if (
                model_profile.local
                and self.reasoning_effort != "none"):
            raise ValueError(
                "Local source models require reasoning effort none.")
        if (
                model_profile.local
                and self.execution_mode != "standard"):
            raise ValueError(
                "Local source models require Standard processing.")
        if (
                self.request_protocol == "v8"
                and self.reasoning_effort != "low"):
            raise ValueError(
                'Source request protocol "v8" requires reasoning effort '
                '"low" to preserve its frozen behavior.')
        if not isinstance(self.context_mode, ContextMode):
            object.__setattr__(
                self,
                "context_mode",
                ContextMode.parse(self.context_mode))
        if self.run_path is not None and not isinstance(self.run_path, str):
            raise ValueError("Processed source run path must be text.")
        if not isinstance(self.excluded_words, tuple):
            object.__setattr__(
                self,
                "excluded_words",
                tuple(self.excluded_words))
        if any(
                not isinstance(word, str)
                for word in self.excluded_words):
            raise ValueError("Excluded vocabulary entries must be text.")
        if not isinstance(self.retain_excluded_source_contexts, bool):
            raise ValueError(
                "Retaining source contexts for excluded words must be "
                "enabled or disabled.")
        if (
                self.retain_excluded_source_contexts
                and self.context_mode is ContextMode.NONE):
            raise ValueError(
                "Retaining source contexts for sentence cards requires a "
                "context mode other than None.")
        if (
                self.exclude_anki is not None
                and not isinstance(self.exclude_anki, AnkiExclusionSpec)):
            object.__setattr__(
                self,
                "exclude_anki",
                AnkiExclusionSpec.from_mapping(self.exclude_anki))
        if not isinstance(self.anki_exclusions, tuple):
            object.__setattr__(
                self,
                "anki_exclusions",
                tuple(self.anki_exclusions))
        exclusions = tuple(
            AnkiExclusionSpec.from_mapping(item)
            for item in self.anki_exclusions)
        if self.exclude_anki is not None and self.exclude_anki not in exclusions:
            exclusions = (self.exclude_anki,) + exclusions
        object.__setattr__(self, "anki_exclusions", exclusions)

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Source generation settings must be an object.")
        source_key = value.get("source_key", value.get("source"))
        exclusion_value = value.get("anki_exclusion")
        if exclusion_value is None:
            candidate = value.get("exclude_anki")
            exclusion_value = candidate if isinstance(candidate, dict) else None
        plural_exclusions = value.get("anki_exclusions", ())
        if plural_exclusions is None:
            plural_exclusions = ()
        if not isinstance(plural_exclusions, (tuple, list)):
            raise TypeError("Anki exclusions must be a list.")
        request_protocol = value.get("request_protocol")
        if request_protocol is None and value.get("protocol_version") is not None:
            request_protocol = f'v{value["protocol_version"]}'
        if request_protocol is None:
            request_protocol = "v10"
        return cls(
            source_key=source_key,
            run_path=(
                str(value["run_path"])
                if value.get("run_path") is not None
                else None),
            chunk_size=value.get("chunk_size", 30),
            source_prefix_token_limit=value.get(
                "source_prefix_token_limit"),
            context_mode=ContextMode.parse(
                value.get("context_mode", ContextMode.SENTENCE.value)),
            concurrency=value.get("concurrency", 8),
            request_stagger_ms=value.get("request_stagger_ms", 100),
            max_transient_retries=value.get(
                "max_transient_retries",
                3),
            model=value.get("model", DEFAULT_SOURCE_MODEL),
            request_protocol=request_protocol,
            reasoning_effort=value.get(
                "reasoning_effort",
                "low"),
            execution_mode=value.get(
                "execution_mode",
                value.get("processing_mode", "standard")),
            automatic_repair=value.get("automatic_repair", False),
            max_automatic_repairs=value.get(
                "max_automatic_repairs",
                MAX_AUTOMATIC_REPAIR_ATTEMPTS),
            excluded_words=tuple(value.get("excluded_words", ())),
            retain_excluded_source_contexts=value.get(
                "retain_excluded_source_contexts",
                bool(value.get(
                    "use_source_for_example_sentences",
                    False))),
            exclude_anki=AnkiExclusionSpec.from_mapping(
                exclusion_value),
            anki_exclusions=tuple(
                AnkiExclusionSpec.from_mapping(item)
                for item in plural_exclusions),
        )

    def to_dict(self):
        result = asdict(self)
        result["context_mode"] = self.context_mode.value
        return result


@dataclass(frozen=True)
class LoadedSource:
    """One semantically audited corpus build selected for generation."""

    key: str
    title: str
    build_id: str
    run_path: Path
    build: CorpusVocabularyView
    token_occurrence_count: int | None = None

    def to_dict(self):
        return {
            "key": self.key,
            "name": self.title,
            "source_key": self.key,
            "title": self.title,
            "build_id": self.build_id,
            "run_path": str(self.run_path),
            "word_count": len(self.build.unique_words),
            "token_occurrence_count": self.token_occurrence_count,
            "section_count": len(self.build.snapshot.sections),
            "source_language_key": (
                self.build.snapshot.source_language_key),
        }


@dataclass(frozen=True)
class ContextUnit:
    """One deduplicated source interval included once in a request."""

    context_id: str
    mode: str
    start_offset: int
    end_offset: int
    text: str
    section_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    word_ranks: tuple[int, ...]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OccurrenceLocator:
    """Persisted prompt guidance identifying one repeated source spelling."""

    before: str
    target: str
    after: str
    literal_match_ordinal: int
    marked_excerpt: str

    def __post_init__(self):
        if not all(
                isinstance(value, str)
                for value in (
                    self.before,
                    self.target,
                    self.after,
                    self.marked_excerpt)):
            raise TypeError("Source occurrence locator text must be strings.")
        if not self.target:
            raise ValueError("A source occurrence locator target is required.")
        if (
                isinstance(self.literal_match_ordinal, bool)
                or not isinstance(self.literal_match_ordinal, int)
                or self.literal_match_ordinal < 1):
            raise ValueError(
                "A source occurrence locator ordinal must be positive.")
        expected_excerpt = (
            self.before
            + OCCURRENCE_TARGET_OPEN
            + self.target
            + OCCURRENCE_TARGET_CLOSE
            + self.after)
        if self.marked_excerpt != expected_excerpt:
            raise ValueError(
                "A source occurrence marked excerpt is inconsistent.")

    @classmethod
    def from_mapping(cls, value):
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("A source occurrence locator must be an object.")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ContextOccurrenceSpan:
    """One complete tokenizer occurrence, relative to a retained context."""

    start_offset: int
    end_offset: int
    surface: str

    def __post_init__(self):
        if (
                isinstance(self.start_offset, bool)
                or not isinstance(self.start_offset, int)
                or self.start_offset < 0
                or isinstance(self.end_offset, bool)
                or not isinstance(self.end_offset, int)
                or self.end_offset <= self.start_offset):
            raise ValueError(
                "A context occurrence requires a non-empty non-negative "
                "offset span.")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError(
                "A context occurrence requires a non-empty surface.")
        if len(self.surface) != self.end_offset - self.start_offset:
            raise ValueError(
                "A context occurrence surface must match its offset width.")

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("A context occurrence must be an object.")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GenerationWord:
    """A ranked source word pointing to, but not repeating, its context."""

    rank: int
    surface: str
    normalized: str
    section_id: str
    sentence_id: str
    context_id: str | None
    # New plans retain the audited first occurrence. Defaults keep legacy
    # chunks loadable and, importantly, keep their retried request bytes
    # unchanged because request_payload omits absent spans.
    start_offset: int | None = None
    end_offset: int | None = None
    occurrence_locator: OccurrenceLocator | None = None
    context_occurrences: tuple[ContextOccurrenceSpan, ...] = ()

    def __post_init__(self):
        if (
                self.occurrence_locator is not None
                and not isinstance(
                    self.occurrence_locator,
                    OccurrenceLocator)):
            object.__setattr__(
                self,
                "occurrence_locator",
                OccurrenceLocator.from_mapping(
                    self.occurrence_locator))
        if not isinstance(self.context_occurrences, (tuple, list)):
            raise TypeError(
                "Source context occurrences must be a sequence.")
        object.__setattr__(
            self,
            "context_occurrences",
            tuple(
                ContextOccurrenceSpan.from_mapping(value)
                for value in self.context_occurrences))

    def to_dict(self):
        value = asdict(self)
        if not self.context_occurrences:
            value.pop("context_occurrences")
        return value


@dataclass(frozen=True)
class GenerationChunk:
    """One independently requestable and recoverable paid-work unit."""

    chunk_id: str
    index: int
    total: int
    start_rank: int
    end_rank: int
    words: tuple[GenerationWord, ...]
    contexts: tuple[ContextUnit, ...]
    # Excluded/previously learned words can still anchor a retained source
    # sentence card. They are persisted for ordering and attribution, but are
    # deliberately omitted from the provider payload's lexical ``words``
    # collection so no definition is requested or packaged for them.
    context_anchors: tuple[GenerationWord, ...] = ()
    # A retained context can be repeated as lexical input when one sentence
    # straddles a word-count chunk boundary, but its sentence translation is
    # owned by exactly one chunk. ``None`` preserves legacy saved chunks,
    # where every retained context was a translation target; an explicit
    # empty tuple means this chunk owns no source-sentence translation.
    source_context_translation_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        target_ids = self.source_context_translation_ids
        if target_ids is None:
            return
        if not isinstance(target_ids, (tuple, list)):
            raise TypeError(
                "Source-context translation IDs must be a sequence.")
        target_ids = tuple(target_ids)
        if (
                any(
                    not isinstance(context_id, str) or not context_id
                    for context_id in target_ids)
                or len(target_ids) != len(set(target_ids))):
            raise ValueError(
                "Source-context translation IDs must be unique non-empty "
                "strings.")
        retained_ids = {
            context.context_id
            for context in self.contexts
        }
        if not set(target_ids) <= retained_ids:
            raise ValueError(
                "Every source-context translation ID must identify a "
                "retained context in the same chunk.")
        object.__setattr__(
            self,
            "source_context_translation_ids",
            target_ids)

    @property
    def requested_source_context_ids(self):
        """Return translation targets while preserving legacy chunk meaning."""
        if self.source_context_translation_ids is None:
            return tuple(
                context.context_id
                for context in self.contexts)
        return self.source_context_translation_ids

    def to_dict(self):
        value = {
            "chunk_id": self.chunk_id,
            "index": self.index,
            "total": self.total,
            "start_rank": self.start_rank,
            "end_rank": self.end_rank,
            "words": [word.to_dict() for word in self.words],
            "contexts": [context.to_dict() for context in self.contexts],
            "context_anchors": [
                word.to_dict()
                for word in self.context_anchors
            ],
        }
        if self.source_context_translation_ids is not None:
            value["source_context_translation_ids"] = list(
                self.source_context_translation_ids)
        return value

    def request_payload(self):
        """Return the compact object appended to the generation prompt."""
        contexts_by_id = {
            context.context_id: context
            for context in self.contexts
        }

        def word_payload(word):
            value = {
                "rank": word.rank,
                "term": word.surface,
                "context_id": word.context_id,
            }
            context = contexts_by_id.get(word.context_id)
            if (
                    context is not None
                    and word.start_offset is not None
                    and word.end_offset is not None):
                value["occurrence_span"] = [
                    word.start_offset - context.start_offset,
                    word.end_offset - context.start_offset,
                ]
                if word.occurrence_locator is not None:
                    value["occurrence_locator"] = (
                        word.occurrence_locator.to_dict())
            return value

        value = {
            "chunk_id": self.chunk_id,
            "rank_range": [self.start_rank, self.end_rank],
            "words": [
                word_payload(word)
                for word in self.words
            ],
            "contexts": [
                {
                    "context_id": context.context_id,
                    "text": context.text,
                }
                for context in self.contexts
            ],
        }
        if self.source_context_translation_ids is not None:
            value["source_context_translation_ids"] = list(
                self.source_context_translation_ids)
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(
            chunk_id=value["chunk_id"],
            index=value["index"],
            total=value["total"],
            start_rank=value["start_rank"],
            end_rank=value["end_rank"],
            words=tuple(
                GenerationWord(**word)
                for word in value["words"]),
            contexts=tuple(
                ContextUnit(
                    **{
                        **context,
                        "section_ids": tuple(context["section_ids"]),
                        "sentence_ids": tuple(context["sentence_ids"]),
                        "word_ranks": tuple(context["word_ranks"]),
                    })
                for context in value["contexts"]),
            context_anchors=tuple(
                GenerationWord(**word)
                for word in value.get("context_anchors", ())),
            source_context_translation_ids=(
                tuple(value["source_context_translation_ids"])
                if "source_context_translation_ids" in value
                else None),
        )


@dataclass(frozen=True)
class GenerationPlan:
    """A deterministic free plan derived from an existing tokenized build."""

    plan_id: str
    source_key: str
    source_title: str
    source_build_id: str
    source_run_path: str
    config: SourceGenerationConfig
    original_word_count: int
    excluded_word_count: int
    chunks: tuple[GenerationChunk, ...]
    source_prefix_token_count: int | None = None
    prefix_unique_word_count: int | None = None

    @property
    def word_count(self):
        return sum(len(chunk.words) for chunk in self.chunks)

    def to_dict(self, *, include_chunks=True):
        result = {
            "schema_version": SOURCE_GENERATION_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "source_key": self.source_key,
            "source_title": self.source_title,
            "source_build_id": self.source_build_id,
            "source_run_path": self.source_run_path,
            "config": self.config.to_dict(),
            "original_word_count": self.original_word_count,
            "excluded_word_count": self.excluded_word_count,
            "word_count": self.word_count,
            "chunk_count": len(self.chunks),
            "source_prefix_token_count": self.source_prefix_token_count,
            "prefix_unique_word_count": self.prefix_unique_word_count,
        }
        if include_chunks:
            result["chunks"] = [
                chunk.to_dict()
                for chunk in self.chunks
            ]
        return result


@dataclass(frozen=True)
class OutputDetail:
    """Response-size inputs derived from Card Setup selections."""

    field_keys: tuple[str, ...]
    response_language_keys: tuple[str, ...] = ()
    include_example_sentences: bool = False
    term_field_name: str = "Term"

    def __post_init__(self):
        if not isinstance(self.field_keys, tuple):
            object.__setattr__(self, "field_keys", tuple(self.field_keys))
        if not isinstance(self.response_language_keys, tuple):
            object.__setattr__(
                self,
                "response_language_keys",
                tuple(self.response_language_keys))

    @classmethod
    def from_pipeline(cls, pipeline):
        # Imported lazily so corpus-only workflows do not need GUI state.
        import pipeline_store

        settings = pipeline_store.get_requested_field_settings(pipeline)
        return cls(
            field_keys=tuple(
                setting.field_key
                for setting in settings),
            response_language_keys=tuple(
                setting.target_language_key
                for setting in settings),
            include_example_sentences=(
                pipeline_store.requires_sentences(pipeline)),
            term_field_name=(
                pipeline_store.get_language(
                    pipeline.language_key).term_field),
        )

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("Output detail must be an object.")
        return cls(
            field_keys=tuple(value.get("field_keys", ())),
            response_language_keys=tuple(
                value.get("response_language_keys", ())),
            include_example_sentences=bool(
                value.get("include_example_sentences", False)),
            term_field_name=value.get("term_field_name", "Term"),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Pricing:
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    label: str = "standard"
    cached_input_usd_per_million: float | None = None
    cache_write_input_usd_per_million: float | None = None

    def to_dict(self):
        result = asdict(self)
        # Keep estimates compact when a provider has no distinct cache-write
        # price class.
        if self.cache_write_input_usd_per_million is None:
            result.pop("cache_write_input_usd_per_million")
        return result


@dataclass(frozen=True)
class CostEstimate:
    """An explicitly approximate token and USD breakdown."""

    model: str
    pricing_label: str
    request_count: int
    word_count: int
    prompt_tokens: int
    payload_tokens: int
    schema_tokens: int
    estimated_input_tokens: int
    estimated_cached_input_tokens: int
    estimated_cache_write_input_tokens: int
    estimated_uncached_input_tokens: int
    estimated_output_tokens: int
    estimated_input_usd: float
    estimated_cached_input_usd: float
    estimated_cache_write_input_usd: float
    estimated_uncached_input_usd: float
    estimated_output_usd: float
    estimated_total_usd: float
    low_total_usd: float
    high_total_usd: float
    usd_to_aud_rate: float
    assumptions: dict[str, Any]

    @property
    def candidate_count(self):
        return self.word_count

    @property
    def input_tokens(self):
        return self.estimated_input_tokens

    @property
    def output_tokens(self):
        return self.estimated_output_tokens

    @property
    def cached_input_tokens(self):
        return self.estimated_cached_input_tokens

    @property
    def cache_write_input_tokens(self):
        return self.estimated_cache_write_input_tokens

    @property
    def uncached_input_tokens(self):
        return self.estimated_uncached_input_tokens

    @property
    def estimated_cost_usd(self):
        return self.estimated_total_usd

    @property
    def estimated_cost_low_usd(self):
        return self.low_total_usd

    @property
    def estimated_cost_high_usd(self):
        return self.high_total_usd

    @property
    def estimated_cost_aud(self):
        return self.estimated_total_usd * self.usd_to_aud_rate

    @property
    def estimated_cost_low_aud(self):
        return self.low_total_usd * self.usd_to_aud_rate

    @property
    def estimated_cost_high_aud(self):
        return self.high_total_usd * self.usd_to_aud_rate

    def to_dict(self):
        result = asdict(self)
        result.update({
            "candidate_count": self.candidate_count,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_cost_low_usd": self.estimated_cost_low_usd,
            "estimated_cost_high_usd": self.estimated_cost_high_usd,
            "estimated_cost_aud": self.estimated_cost_aud,
            "estimated_cost_low_aud": self.estimated_cost_low_aud,
            "estimated_cost_high_aud": self.estimated_cost_high_aud,
        })
        return result
