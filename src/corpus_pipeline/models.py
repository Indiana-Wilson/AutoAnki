"""Immutable records shared by corpus sources, tokenizers, and storage."""

from dataclasses import asdict, dataclass
from typing import Any


CORPUS_SCHEMA_VERSION = 1
CORPUS_PROCESSING_VERSION = "ordered-contextual-vocabulary-v4"
CORPUS_NORMALIZATION_POLICY = "NFC-exact"
CORPUS_CONTEXT_POLICY = "paragraph-and-neighbor-sentences-v2"
OFFSET_CONVENTION = (
    "Zero-based Python Unicode code-point offsets into canonical_text; "
    "start is inclusive and end is exclusive."
)


class CorpusValidationError(ValueError):
    """A corpus artifact is incomplete, inconsistent, or contaminated."""


@dataclass(frozen=True)
class SourcePage:
    """One exact MediaWiki revision and its unmodified wikitext."""

    page_key: str
    order: int
    title: str
    url: str
    revision_id: int
    revision_timestamp: str
    revision_sha1: str
    retrieved_at: str
    raw_wikitext: str
    raw_sha256: str

    def to_dict(self, *, include_wikitext=False):
        result = asdict(self)
        if not include_wikitext:
            result.pop("raw_wikitext")
        return result


@dataclass(frozen=True)
class TextSection:
    """A logical chapter located inside the canonical cleaned text."""

    section_id: str
    order: int
    title: str
    source_page_key: str
    text: str
    start_offset: int = 0
    end_offset: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ContextSpan:
    """A paragraph or sentence retained for later contextual definition."""

    context_id: str
    kind: str
    section_id: str
    start_offset: int
    end_offset: int
    text: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TokenSpan:
    """A tokenizer result using offsets local to its input string."""

    surface: str
    start_offset: int
    end_offset: int
    confidence: float | None = None


@dataclass(frozen=True)
class TokenizerIdentity:
    """Enough information to reproduce or distinguish tokenizer output."""

    backend: str
    backend_version: str
    model: str
    model_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def to_dict(self):
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model": self.model,
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class TokenOccurrence:
    """One lexical token occurrence in canonical source order."""

    occurrence_id: str
    token_index: int
    surface: str
    normalized: str
    section_id: str
    start_offset: int
    end_offset: int
    paragraph_id: str
    sentence_id: str
    confidence: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class UniqueWord:
    """A word ranked by its first occurrence in the work."""

    rank: int
    surface: str
    normalized: str
    first_occurrence_id: str
    occurrence_count: int
    section_id: str
    start_offset: int
    end_offset: int
    paragraph_id: str
    sentence_id: str
    previous_sentence_id: str | None = None
    next_sentence_id: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BuildConfig:
    """Options that affect deterministic vocabulary output."""

    chunk_size: int = 500
    normalization: str = CORPUS_NORMALIZATION_POLICY
    context_policy: str = CORPUS_CONTEXT_POLICY
    include_section_titles: bool = False

    def __post_init__(self):
        if (
                isinstance(self.chunk_size, bool)
                or not isinstance(self.chunk_size, int)
                or self.chunk_size < 1):
            raise ValueError(
                "Corpus chunk size must be a positive integer.")
        if self.normalization != CORPUS_NORMALIZATION_POLICY:
            raise ValueError(
                "Only exact NFC corpus normalization is implemented.")
        if self.context_policy != CORPUS_CONTEXT_POLICY:
            raise ValueError(
                "Only paragraph-and-neighbor-sentences context is "
                "implemented.")
        if not isinstance(self.include_section_titles, bool):
            raise ValueError(
                "Section-title inclusion must be true or false.")

    def to_dict(self):
        result = asdict(self)
        # Preserve the identity shape of older title-excluding configurations.
        if not self.include_section_titles:
            result.pop("include_section_titles")
        return result


@dataclass(frozen=True)
class CorpusSnapshot:
    """Fetched, cleaned, and offset-aligned source text."""

    spec_key: str
    edition: str
    source_language_key: str
    pages: tuple[SourcePage, ...]
    sections: tuple[TextSection, ...]
    canonical_text: str
    cleaner_version: str
    include_section_titles: bool = False


@dataclass(frozen=True)
class CorpusBuild:
    """Complete local tokenization result before serialization."""

    snapshot: CorpusSnapshot
    config: BuildConfig
    tokenizer: TokenizerIdentity
    contexts: tuple[ContextSpan, ...]
    occurrences: tuple[TokenOccurrence, ...]
    unique_words: tuple[UniqueWord, ...]


@dataclass(frozen=True)
class CorpusVocabularyView:
    """Generation-facing build data without every repeated occurrence.

    Occurrence artifacts remain hash-verified on disk.  Omitting their
    deserialization avoids constructing hundreds of megabytes of short-lived
    Python objects whenever a large source estimate changes.
    """

    snapshot: CorpusSnapshot
    config: BuildConfig
    tokenizer: TokenizerIdentity
    contexts: tuple[ContextSpan, ...]
    unique_words: tuple[UniqueWord, ...]


def json_ready(value: Any):
    """Convert supported records into deterministic JSON-compatible values."""
    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
