"""Deterministic dictionary refinement for suspicious historical spans."""

from collections import Counter
from dataclasses import replace
from importlib import metadata
import logging
from pathlib import Path

from corpus_pipeline.models import (
    CorpusBuild,
    TokenOccurrence,
    UniqueWord,
)
from corpus_pipeline.contexts import sentence_neighbor_ids
from corpus_pipeline.processing import is_han_word
from corpus_pipeline.resources import (
    JIEBA_TRADITIONAL_REVISION,
    JIEBA_TRADITIONAL_SHA256,
    sha256_file,
)


JIEBA_VERSION = "0.42.1"
JIEBA_REFINER_VERSION = "jieba-traditional-long-span-v1"
JIEBA_MAX_OUTPUT_GRAPHEMES = 4


def _is_variation_selector(character):
    codepoint = ord(character)
    return (
        0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _han_graphemes(text):
    graphemes = []
    for character in text:
        if _is_variation_selector(character) and graphemes:
            graphemes[-1] += character
        else:
            graphemes.append(character)
    return tuple(graphemes)


class JiebaLongSpanRefiner:
    """Split only unresolved >4-Han CKIP spans with a pinned dictionary."""

    def __init__(
            self,
            dictionary_path,
            *,
            segmenter=None,
            package_version=None,
            dictionary_sha256=JIEBA_TRADITIONAL_SHA256,
            dictionary_revision=JIEBA_TRADITIONAL_REVISION):
        self.dictionary_path = Path(dictionary_path)
        if (
                not self.dictionary_path.is_file()
                or self.dictionary_path.is_symlink()
                or sha256_file(self.dictionary_path)
                != dictionary_sha256):
            raise ValueError(
                "The Traditional Jieba dictionary failed its pinned hash.")
        self.dictionary_sha256 = dictionary_sha256
        self.dictionary_revision = dictionary_revision

        if segmenter is None:
            try:
                import jieba
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "Dictionary refinement requires the optional packages "
                    "in requirements-corpus.txt.") from error
            installed_version = metadata.version("jieba")
            if installed_version != JIEBA_VERSION:
                raise RuntimeError(
                    f"Jieba {JIEBA_VERSION} is required; found "
                    f"{installed_version}.")
            jieba.default_logger.setLevel(logging.WARNING)
            segmenter = jieba.Tokenizer(str(self.dictionary_path))
            segmenter.initialize()
            package_version = installed_version
        self.segmenter = segmenter
        self.package_version = package_version or "injected-test-segmenter"

    @property
    def identity_options(self):
        return (
            ("dictionary_refiner_version", JIEBA_REFINER_VERSION),
            ("dictionary_refiner_package", f"jieba=={self.package_version}"),
            ("dictionary_refiner_revision", self.dictionary_revision),
            ("dictionary_refiner_sha256", self.dictionary_sha256),
            ("dictionary_refiner_hmm", "false"),
            (
                "dictionary_refiner_max_output_graphemes",
                str(JIEBA_MAX_OUTPUT_GRAPHEMES),
            ),
        )

    def split(self, surface):
        """Return a complete, offset-free partition of one all-Han span."""
        if not is_han_word(surface):
            raise ValueError("Dictionary refinement accepts only Han spans.")
        pieces = tuple(self.segmenter.cut(surface, HMM=False))
        if not pieces or "".join(pieces) != surface:
            raise ValueError(
                "Jieba did not preserve the complete source span.")

        bounded = []
        for piece in pieces:
            graphemes = _han_graphemes(piece)
            if len(graphemes) > JIEBA_MAX_OUTPUT_GRAPHEMES:
                bounded.extend(graphemes)
            else:
                bounded.append(piece)
        if "".join(bounded) != surface:
            raise ValueError(
                "Dictionary refinement did not preserve source text.")
        return tuple(bounded)


def refine_build_long_spans(build: CorpusBuild, refiner):
    """Apply the dictionary fallback and rebuild ordered vocabulary records."""
    existing_options = dict(build.tokenizer.options)
    option_names = {
        name
        for name, _value in refiner.identity_options
    }
    present = option_names.intersection(existing_options)
    if present:
        expected = dict(refiner.identity_options)
        if all(existing_options.get(name) == value
               for name, value in expected.items()):
            return build
        raise ValueError(
            "This build already records another dictionary refinement.")

    expanded = []
    for occurrence in build.occurrences:
        if (
                len(_han_graphemes(occurrence.surface))
                <= JIEBA_MAX_OUTPUT_GRAPHEMES
                or not is_han_word(occurrence.surface)):
            pieces = (occurrence.surface,)
        else:
            pieces = refiner.split(occurrence.surface)

        cursor = occurrence.start_offset
        was_split = len(pieces) > 1
        for piece in pieces:
            end = cursor + len(piece)
            expanded.append(replace(
                occurrence,
                occurrence_id="",
                token_index=0,
                surface=piece,
                normalized=piece,
                start_offset=cursor,
                end_offset=end,
                confidence=None if was_split else occurrence.confidence,
            ))
            cursor = end
        if cursor != occurrence.end_offset:
            raise ValueError(
                "Dictionary refinement changed a source occurrence span.")

    occurrences = tuple(
        replace(
            occurrence,
            occurrence_id=f"token:{index:09d}",
            token_index=index,
        )
        for index, occurrence in enumerate(expanded, start=1)
    )
    counts = Counter(
        occurrence.normalized
        for occurrence in occurrences
    )
    first_occurrences = {}
    for occurrence in occurrences:
        first_occurrences.setdefault(
            occurrence.normalized,
            occurrence,
        )
    sentence_neighbors = sentence_neighbor_ids(build.contexts)
    unique_words = tuple(
        UniqueWord(
            rank=rank,
            surface=occurrence.surface,
            normalized=normalized,
            first_occurrence_id=occurrence.occurrence_id,
            occurrence_count=counts[normalized],
            section_id=occurrence.section_id,
            start_offset=occurrence.start_offset,
            end_offset=occurrence.end_offset,
            paragraph_id=occurrence.paragraph_id,
            sentence_id=occurrence.sentence_id,
            previous_sentence_id=sentence_neighbors[
                occurrence.sentence_id][0],
            next_sentence_id=sentence_neighbors[
                occurrence.sentence_id][1],
        )
        for rank, (normalized, occurrence) in enumerate(
            first_occurrences.items(),
            start=1,
        )
    )
    tokenizer = replace(
        build.tokenizer,
        options=build.tokenizer.options + refiner.identity_options,
    )
    return replace(
        build,
        tokenizer=tokenizer,
        occurrences=occurrences,
        unique_words=unique_words,
    )
