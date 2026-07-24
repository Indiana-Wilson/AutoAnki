"""Offset-safe token processing and first-occurrence vocabulary ranking."""

from bisect import bisect_right
from dataclasses import replace
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
        snapshot.sections)
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
            for token in _han_subspans(raw_token):
                surface = unicodedata.normalize("NFC", token.surface)
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
                    normalized=surface,
                    section_id=section.section_id,
                    start_offset=global_start,
                    end_offset=global_end,
                    paragraph_id=paragraph.context_id,
                    sentence_id=sentence.context_id,
                    confidence=token.confidence)
                occurrences.append(occurrence)
                counts[surface] = counts.get(surface, 0) + 1
                first_words.setdefault(surface, occurrence)
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
