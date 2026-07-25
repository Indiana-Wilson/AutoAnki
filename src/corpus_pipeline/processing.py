"""Offset-safe token processing and first-occurrence vocabulary ranking."""

from bisect import bisect_right
from dataclasses import replace
import re
import unicodedata

from corpus_pipeline.contexts import (
    build_contexts,
    sentence_neighbor_ids,
)
from corpus_pipeline.models import (
    CorpusBuild,
    CorpusValidationError,
    TokenSpan,
    TokenOccurrence,
    UniqueWord,
)


_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
    (0x30000, 0x323AF),
)
_HAN_MARKS = frozenset((0x3005, 0x3007, 0x303B))
_VARIATION_RANGES = (
    (0xFE00, 0xFE0F),
    (0xE0100, 0xE01EF),
)
HISTORICAL_ENGLISH_LANGUAGE_KEYS = frozenset({
    "middle_english",
    "old_english",
})
HISTORICAL_ENGLISH_NORMALIZATION_POLICY = "NFC-casefold-v1"
_WORD_APOSTROPHES = frozenset(("'", "\u2019", "\u02bc"))
_WORD_HYPHENS = frozenset(("-", "\u2010", "\u2011"))
_EDITORIAL_NUMBER_LINE = re.compile(
    r"^[\[(]?(?:[A-Za-z]\s*)?\d+[a-z]?[.)\]\-]?$")
_ROMAN_NUMERALS = frozenset("IVXLCDM")


def _in_ranges(codepoint, ranges):
    return any(start <= codepoint <= end for start, end in ranges)


def is_han_component(character):
    """Return whether one code point is Han text or a variation selector."""
    codepoint = ord(character)
    return (
        codepoint in _HAN_MARKS
        or _in_ranges(codepoint, _HAN_RANGES)
        or _in_ranges(codepoint, _VARIATION_RANGES)
    )


def is_han_word(text):
    """Return whether a token is wholly Han text (plus variation selectors)."""
    found_han = False
    for character in text:
        codepoint = ord(character)
        if (
                codepoint in _HAN_MARKS
                or _in_ranges(codepoint, _HAN_RANGES)):
            found_han = True
            continue
        if _in_ranges(codepoint, _VARIATION_RANGES):
            continue
        return False
    return found_han


def _is_word_letter(character):
    # Unicode classifies the ordinal indicators ª and º as letters. They
    # annotate numbers and citations rather than forming historical-English
    # words.
    return character.isalpha() and character not in "\u00aa\u00ba"


def _is_word_mark(character):
    return unicodedata.category(character).startswith("M")


def _line_bounds(text, start, end):
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    return line_start, line_end


def _is_editorial_historical_english_span(text, start, end):
    """Reject standalone numbering labels while retaining lexical text."""
    line_start, line_end = _line_bounds(text, start, end)
    line = text[line_start:line_end].strip()
    if _EDITORIAL_NUMBER_LINE.fullmatch(line):
        return True

    surface = text[start:end]
    if (
            surface.casefold() == "c"
            and start > 0
            and text[start - 1] == "&"
            and end < len(text)
            and text[end] == "."):
        return True
    if (
            surface
            and all(
                character.upper() in _ROMAN_NUMERALS
                for character in surface)
            and text[end:end + 2].replace("_", "").startswith("\u00ba")):
        return True
    roman = (
        bool(surface)
        and all(character in _ROMAN_NUMERALS for character in surface))
    if not roman:
        return False
    if (
            not text[line_start:start].strip()
            and end < len(text)
            and text[end] in ".)"):
        return True
    # Roman-numeral chapter/page labels are editorial when they occupy the
    # line by themselves (allowing conventional surrounding punctuation).
    stripped = line.strip("[]().: \t")
    return stripped == surface


def historical_english_word_ranges(
        text,
        *,
        reject_editorial_labels=True):
    """Return deterministic Unicode word ranges for historical English.

    A word begins with any Unicode letter, retains combining diacritics, and
    may contain an apostrophe or a true hyphen only when that connector joins
    two letter runs. Digits and standalone editorial numbering labels are not
    vocabulary. The returned ranges always reproduce the source text exactly.
    """
    ranges = []
    cursor = 0
    while cursor < len(text):
        if not _is_word_letter(text[cursor]):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text) and _is_word_mark(text[cursor]):
            cursor += 1
        while cursor < len(text):
            character = text[cursor]
            if _is_word_letter(character):
                cursor += 1
                while (
                        cursor < len(text)
                        and _is_word_mark(text[cursor])):
                    cursor += 1
                continue
            if (
                    character in _WORD_APOSTROPHES | _WORD_HYPHENS
                    and cursor + 1 < len(text)
                    and _is_word_letter(text[cursor + 1])):
                cursor += 2
                while (
                        cursor < len(text)
                        and _is_word_mark(text[cursor])):
                    cursor += 1
                continue
            break
        if (
                not reject_editorial_labels
                or not _is_editorial_historical_english_span(
                    text,
                    start,
                    cursor)):
            ranges.append((start, cursor))
    return tuple(ranges)


def is_historical_english_word(text):
    """Return whether text is exactly one historical-English word span."""
    return historical_english_word_ranges(
        text,
        reject_editorial_labels=False) == ((0, len(text)),)


def normalize_word(surface, source_language_key):
    """Return the language-aware vocabulary identity for one source form."""
    normalized = unicodedata.normalize("NFC", surface)
    if source_language_key in HISTORICAL_ENGLISH_LANGUAGE_KEYS:
        # Sentence-initial capitalization is not a distinct vocabulary item.
        # Keep the first surface/offset while counting all case variants.
        return normalized.casefold()
    return normalized


def _han_subspans(token):
    """Keep maximal Han runs if a tokenizer joins text to punctuation."""
    runs = []
    run_start = None
    for relative_offset, character in enumerate(token.surface):
        if is_han_component(character):
            if run_start is None:
                run_start = relative_offset
            continue
        if run_start is not None:
            surface = token.surface[run_start:relative_offset]
            if is_han_word(surface):
                runs.append(TokenSpan(
                    surface=surface,
                    start_offset=token.start_offset + run_start,
                    end_offset=token.start_offset + relative_offset,
                    confidence=token.confidence))
            run_start = None
    if run_start is not None:
        surface = token.surface[run_start:]
        if is_han_word(surface):
            runs.append(TokenSpan(
                surface=surface,
                start_offset=token.start_offset + run_start,
                end_offset=token.end_offset,
                confidence=token.confidence))
    return tuple(runs)


def _historical_english_subspans(token):
    """Keep exact lexical runs if a tokenizer joins words to punctuation."""
    return tuple(
        TokenSpan(
            surface=token.surface[start:end],
            start_offset=token.start_offset + start,
            end_offset=token.start_offset + end,
            confidence=token.confidence)
        for start, end in historical_english_word_ranges(
            token.surface,
            reject_editorial_labels=False)
    )


def _lexical_subspans(token, source_language_key):
    if source_language_key in HISTORICAL_ENGLISH_LANGUAGE_KEYS:
        return _historical_english_subspans(token)
    return _han_subspans(token)


class _ContextIndex:
    def __init__(self, contexts, kind):
        selected = [
            context
            for context in contexts
            if context.kind == kind
        ]
        self.contexts = selected
        self.starts = [context.start_offset for context in selected]
        self.kind = kind

    def containing(self, start, end, section_id):
        index = bisect_right(self.starts, start) - 1
        if index >= 0:
            context = self.contexts[index]
            if (
                    context.section_id == section_id
                    and context.start_offset <= start
                    and end <= context.end_offset):
                return context
        raise CorpusValidationError(
            f"Token at {start}:{end} has no containing {self.kind}.")


def build_vocabulary(
        snapshot,
        tokenizer,
        config,
        *,
        progress_callback=None):
    """Tokenize a cleaned snapshot without contacting OpenAI or Anki."""
    contexts = build_contexts(
        snapshot.canonical_text,
        snapshot.sections,
        source_language_key=snapshot.source_language_key)
    sentence_neighbors = sentence_neighbor_ids(contexts)
    paragraphs = _ContextIndex(contexts, "paragraph")
    sentences = _ContextIndex(contexts, "sentence")
    occurrences = []
    first_words = {}
    counts = {}

    for section in snapshot.sections:
        if (
                snapshot.canonical_text[
                    section.start_offset:section.end_offset]
                != section.text):
            raise CorpusValidationError(
                f"Section offsets do not reproduce {section.section_id}.")

        previous_local_end = 0
        for raw_token in tokenizer.tokenize(section.text):
            if (
                    raw_token.start_offset < previous_local_end
                    or raw_token.end_offset <= raw_token.start_offset
                    or section.text[
                        raw_token.start_offset:raw_token.end_offset]
                    != raw_token.surface):
                raise CorpusValidationError(
                    f"Tokenizer offsets failed in {section.section_id}.")
            previous_local_end = raw_token.end_offset
            for token in _lexical_subspans(
                    raw_token,
                    snapshot.source_language_key):
                surface = unicodedata.normalize("NFC", token.surface)
                normalized = normalize_word(
                    surface,
                    snapshot.source_language_key)
                global_start = section.start_offset + token.start_offset
                global_end = section.start_offset + token.end_offset
                paragraph = paragraphs.containing(
                    global_start,
                    global_end,
                    section.section_id)
                sentence = sentences.containing(
                    global_start,
                    global_end,
                    section.section_id)
                token_index = len(occurrences) + 1
                occurrence = TokenOccurrence(
                    occurrence_id=f"token:{token_index:09d}",
                    token_index=token_index,
                    surface=surface,
                    normalized=normalized,
                    section_id=section.section_id,
                    start_offset=global_start,
                    end_offset=global_end,
                    paragraph_id=paragraph.context_id,
                    sentence_id=sentence.context_id,
                    confidence=token.confidence)
                occurrences.append(occurrence)
                counts[normalized] = counts.get(normalized, 0) + 1
                first_words.setdefault(normalized, occurrence)
        if progress_callback is not None:
            progress_callback(
                section.order,
                len(snapshot.sections),
                section.title)

    unique_words = []
    for rank, (normalized, occurrence) in enumerate(
            first_words.items(),
            start=1):
        previous_sentence_id, next_sentence_id = sentence_neighbors[
            occurrence.sentence_id]
        unique_words.append(UniqueWord(
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
            previous_sentence_id=previous_sentence_id,
            next_sentence_id=next_sentence_id))

    return CorpusBuild(
        snapshot=snapshot,
        config=config,
        tokenizer=tokenizer.identity,
        contexts=contexts,
        occurrences=tuple(occurrences),
        unique_words=tuple(unique_words))


def with_occurrence_counts(unique_words, counts):
    """Update counts in externally constructed word records."""
    return tuple(
        replace(
            word,
            occurrence_count=counts.get(
                word.normalized,
                word.occurrence_count))
        for word in unique_words)
