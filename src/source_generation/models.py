"""Immutable records for source-based generation planning and persistence."""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from corpus_pipeline.models import CorpusVocabularyView


SOURCE_GENERATION_SCHEMA_VERSION = 1


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
    context_mode: ContextMode = ContextMode.SENTENCE
    run_path: str | None = None
    concurrency: int = 8
    request_stagger_ms: int = 100
    max_transient_retries: int = 3
    excluded_words: tuple[str, ...] = ()
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
        return cls(
            source_key=source_key,
            run_path=(
                str(value["run_path"])
                if value.get("run_path") is not None
                else None),
            chunk_size=value.get("chunk_size", 30),
            context_mode=ContextMode.parse(
                value.get("context_mode", ContextMode.SENTENCE.value)),
            concurrency=value.get("concurrency", 8),
            request_stagger_ms=value.get("request_stagger_ms", 100),
            max_transient_retries=value.get(
                "max_transient_retries",
                3),
            excluded_words=tuple(value.get("excluded_words", ())),
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

    def to_dict(self):
        return {
            "key": self.key,
            "name": self.title,
            "source_key": self.key,
            "title": self.title,
            "build_id": self.build_id,
            "run_path": str(self.run_path),
            "word_count": len(self.build.unique_words),
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
class GenerationWord:
    """A ranked source word pointing to, but not repeating, its context."""

    rank: int
    surface: str
    normalized: str
    section_id: str
    sentence_id: str
    context_id: str | None

    def to_dict(self):
        return asdict(self)


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

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "index": self.index,
            "total": self.total,
            "start_rank": self.start_rank,
            "end_rank": self.end_rank,
            "words": [word.to_dict() for word in self.words],
            "contexts": [context.to_dict() for context in self.contexts],
        }

    def request_payload(self):
        """Return the compact object appended to the generation prompt."""
        return {
            "chunk_id": self.chunk_id,
            "rank_range": [self.start_rank, self.end_rank],
            "words": [
                {
                    "rank": word.rank,
                    "term": word.surface,
                    "context_id": word.context_id,
                }
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

    def to_dict(self):
        return asdict(self)


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
    estimated_output_tokens: int
    estimated_input_usd: float
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
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_cost_low_usd": self.estimated_cost_low_usd,
            "estimated_cost_high_usd": self.estimated_cost_high_usd,
            "estimated_cost_aud": self.estimated_cost_aud,
            "estimated_cost_low_aud": self.estimated_cost_low_aud,
            "estimated_cost_high_aud": self.estimated_cost_high_aud,
        })
        return result
