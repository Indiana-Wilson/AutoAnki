"""Reusable source-to-vocabulary preparation for large Chinese works.

This package deliberately does not import AutoAnki's OpenAI or Anki modules.
Fetching, cleaning, tokenising, and auditing a corpus are free local
preparation steps that can later be exposed through the main GUI.
"""

from corpus_pipeline.models import (
    BuildConfig,
    CorpusBuild,
    CorpusSnapshot,
    ContextSpan,
    SourcePage,
    TextSection,
    TokenOccurrence,
    TokenSpan,
    TokenizerIdentity,
    UniqueWord,
)

__all__ = (
    "BuildConfig",
    "CorpusBuild",
    "CorpusSnapshot",
    "ContextSpan",
    "SourcePage",
    "TextSection",
    "TokenOccurrence",
    "TokenSpan",
    "TokenizerIdentity",
    "UniqueWord",
)
