"""Fail-closed validation for source snapshots and vocabulary builds."""

import hashlib
import re
import unicodedata

from corpus_pipeline.chunks import make_chunks
from corpus_pipeline.contexts import (
    SECTION_SEPARATOR,
    TITLE_SEPARATOR,
    sentence_neighbor_ids,
)
from corpus_pipeline.models import CorpusValidationError
from corpus_pipeline.processing import is_han_component, is_han_word


_FORBIDDEN_CLEAN_PATTERNS = (
    ("template opening", re.compile(r"{{")),
    ("template closing", re.compile(r"}}")),
    ("wiki-link opening", re.compile(r"\[\[")),
    ("wiki-link closing", re.compile(r"\]\]")),
    ("web address", re.compile(r"https?://", re.IGNORECASE)),
    ("category markup", re.compile(r"\bCategory:", re.IGNORECASE)),
    ("MediaWiki directive", re.compile(r"__[A-Z]+__")),
    ("HTML/XML tag", re.compile(r"</?[A-Za-z][^>]*>")),
)


def _fail(message):
    raise CorpusValidationError(message)


def audit_snapshot(
        snapshot,
        *,
        expected_section_count=None,
        expected_section_ids=None,
        require_no_latin=False):
    """Validate completeness, provenance hashes, and cleaned text offsets."""
    if expected_section_count is not None:
        if len(snapshot.sections) != expected_section_count:
            _fail(
                f"Expected {expected_section_count} sections; found "
                f"{len(snapshot.sections)}.")
    if expected_section_ids is not None:
        actual_ids = tuple(
            section.section_id
            for section in snapshot.sections)
        if actual_ids != tuple(expected_section_ids):
            _fail("Corpus section IDs are missing, duplicated, or reordered.")

    page_keys = set()
    for page in snapshot.pages:
        if page.page_key in page_keys:
            _fail(f"Duplicate source page key: {page.page_key}")
        page_keys.add(page.page_key)
        actual_hash = hashlib.sha256(
            page.raw_wikitext.encode("utf-8")).hexdigest()
        if actual_hash != page.raw_sha256:
            _fail(f"Raw source hash failed for {page.page_key}.")

    section_ids = set()
    rebuilt_parts = []
    previous_end = 0
    for expected_order, section in enumerate(snapshot.sections, start=1):
        if section.order != expected_order:
            _fail("Section order is not contiguous.")
        if section.section_id in section_ids:
            _fail(f"Duplicate section ID: {section.section_id}")
        if not section.text.strip():
            _fail(f"Empty section: {section.section_id}")
        if (
                snapshot.include_section_titles
                and not section.text.startswith(
                    f"{section.title}{TITLE_SEPARATOR}")):
            _fail(
                f"Included title is missing from {section.section_id}.")
        if section.source_page_key not in page_keys:
            _fail(
                f"Unknown source page for {section.section_id}: "
                f"{section.source_page_key}")
        if expected_order > 1:
            previous_end += len(SECTION_SEPARATOR)
            rebuilt_parts.append(SECTION_SEPARATOR)
        if section.start_offset != previous_end:
            _fail(f"Incorrect start offset for {section.section_id}.")
        if section.end_offset != (
                section.start_offset + len(section.text)):
            _fail(f"Incorrect end offset for {section.section_id}.")
        if snapshot.canonical_text[
                section.start_offset:section.end_offset] != section.text:
            _fail(f"Section slice mismatch for {section.section_id}.")

        for description, pattern in _FORBIDDEN_CLEAN_PATTERNS:
            if pattern.search(section.text):
                _fail(
                    f"Cleaned {section.section_id} still contains "
                    f"{description}.")
        if require_no_latin and re.search(r"[A-Za-z]", section.text):
            _fail(
                f"Cleaned {section.section_id} contains Latin website text.")
        rebuilt_parts.append(section.text)
        previous_end = section.end_offset
        section_ids.add(section.section_id)

    if "".join(rebuilt_parts) != snapshot.canonical_text:
        _fail("Canonical text is not the ordered section concatenation.")
    return True


def audit_build(build):
    """Validate exact offsets, source order, first-seen ranks, and chunks."""
    canonical_text = build.snapshot.canonical_text
    if not build.occurrences:
        _fail("Vocabulary build contains no token occurrences.")
    if not build.unique_words:
        _fail("Vocabulary build contains no unique words.")
    sections = {
        section.section_id: section
        for section in build.snapshot.sections
    }
    contexts = {}
    previous_context_end = {}
    for context in build.contexts:
        if context.context_id in contexts:
            _fail(f"Duplicate context ID: {context.context_id}")
        section = sections.get(context.section_id)
        if section is None:
            _fail(f"Unknown context section: {context.context_id}")
        if context.kind not in {"paragraph", "sentence"}:
            _fail(f"Unknown context kind: {context.context_id}")
        if (
                context.start_offset < section.start_offset
                or context.end_offset <= context.start_offset
                or context.end_offset > section.end_offset):
            _fail(f"Context is outside its section: {context.context_id}")
        if canonical_text[
                context.start_offset:context.end_offset] != context.text:
            _fail(f"Context offset mismatch: {context.context_id}")
        order_key = (context.section_id, context.kind)
        if context.start_offset < previous_context_end.get(order_key, -1):
            _fail(
                f"{context.kind.title()} contexts overlap or reorder in "
                f"{context.section_id}.")
        previous_context_end[order_key] = context.end_offset
        contexts[context.context_id] = context
    sentence_neighbors = sentence_neighbor_ids(build.contexts)

    first_occurrences = {}
    counts = {}
    previous_end = -1
    occurrence_ids = set()
    covered_offsets = bytearray(len(canonical_text))
    for expected_index, occurrence in enumerate(
            build.occurrences,
            start=1):
        if occurrence.token_index != expected_index:
            _fail("Token indexes are not contiguous.")
        expected_id = f"token:{expected_index:09d}"
        if occurrence.occurrence_id != expected_id:
            _fail("Token occurrence IDs are not contiguous.")
        if occurrence.occurrence_id in occurrence_ids:
            _fail(f"Duplicate token occurrence: {occurrence.occurrence_id}")
        if occurrence.start_offset < previous_end:
            _fail("Token occurrences overlap or are not in source order.")
        if occurrence.end_offset <= occurrence.start_offset:
            _fail(f"Empty token span: {occurrence.occurrence_id}")
        section = sections.get(occurrence.section_id)
        if (
                section is None
                or occurrence.start_offset < section.start_offset
                or occurrence.end_offset > section.end_offset):
            _fail(f"Token is outside its section: {occurrence.occurrence_id}")
        if canonical_text[
                occurrence.start_offset:occurrence.end_offset] != (
                    occurrence.surface):
            _fail(f"Token offset mismatch: {occurrence.occurrence_id}")
        if (
                occurrence.normalized
                != unicodedata.normalize("NFC", occurrence.surface)):
            _fail(f"Token normalization mismatch: {occurrence.occurrence_id}")
        if not is_han_word(occurrence.surface):
            _fail(f"Non-Han token occurrence: {occurrence.occurrence_id}")
        paragraph = contexts.get(occurrence.paragraph_id)
        sentence = contexts.get(occurrence.sentence_id)
        if paragraph is None or sentence is None:
            _fail(
                f"Missing token context: {occurrence.occurrence_id}")
        if (
                paragraph.kind != "paragraph"
                or sentence.kind != "sentence"
                or paragraph.section_id != occurrence.section_id
                or sentence.section_id != occurrence.section_id):
            _fail(
                f"Token context identity mismatch: "
                f"{occurrence.occurrence_id}")
        if not (
                paragraph.start_offset <= occurrence.start_offset
                and occurrence.end_offset <= paragraph.end_offset):
            _fail(
                f"Paragraph does not contain {occurrence.occurrence_id}.")
        if not (
                sentence.start_offset <= occurrence.start_offset
                and occurrence.end_offset <= sentence.end_offset):
            _fail(
                f"Sentence does not contain {occurrence.occurrence_id}.")
        first_occurrences.setdefault(
            occurrence.normalized,
            occurrence)
        counts[occurrence.normalized] = (
            counts.get(occurrence.normalized, 0) + 1)
        occurrence_ids.add(occurrence.occurrence_id)
        covered_offsets[
            occurrence.start_offset:occurrence.end_offset
        ] = b"\x01" * (
            occurrence.end_offset - occurrence.start_offset)
        previous_end = occurrence.end_offset

    for offset, character in enumerate(canonical_text):
        if is_han_component(character) and not covered_offsets[offset]:
            _fail(
                f"Han/variation code point at offset {offset} is not "
                "covered by a token occurrence.")

    if len(first_occurrences) != len(build.unique_words):
        _fail("Unique-word count does not match first occurrences.")
    if tuple(first_occurrences) != tuple(
            word.normalized
            for word in build.unique_words):
        _fail("Unique words are not ordered by first occurrence.")
    seen_words = set()
    for expected_rank, word in enumerate(build.unique_words, start=1):
        if word.rank != expected_rank:
            _fail("Unique-word ranks are not contiguous.")
        if word.normalized in seen_words:
            _fail(f"Duplicate unique word: {word.normalized}")
        first = first_occurrences.get(word.normalized)
        if first is None or first.occurrence_id != word.first_occurrence_id:
            _fail(f"First occurrence mismatch for {word.surface}.")
        if (
                word.surface != first.surface
                or word.normalized != first.normalized
                or word.section_id != first.section_id
                or word.start_offset != first.start_offset
                or word.end_offset != first.end_offset
                or word.paragraph_id != first.paragraph_id
                or word.sentence_id != first.sentence_id):
            _fail(f"Unique-word location mismatch for {word.surface}.")
        expected_neighbors = sentence_neighbors.get(word.sentence_id)
        if (
                expected_neighbors is None
                or word.previous_sentence_id != expected_neighbors[0]
                or word.next_sentence_id != expected_neighbors[1]):
            _fail(f"Sentence-neighbor mismatch for {word.surface}.")
        if counts[word.normalized] != word.occurrence_count:
            _fail(f"Occurrence count mismatch for {word.surface}.")
        seen_words.add(word.normalized)

    chunks = make_chunks(
        build.unique_words,
        build.config.chunk_size)
    expected_rank = 1
    for chunk in chunks:
        if chunk.start_rank != expected_rank:
            _fail("Chunk ranges skip or overlap ranks.")
        if len(chunk.words) > build.config.chunk_size:
            _fail("A chunk exceeds the configured size.")
        if chunk.end_rank - chunk.start_rank + 1 != len(chunk.words):
            _fail("A chunk filename range disagrees with its contents.")
        expected_rank = chunk.end_rank + 1
    if expected_rank != len(build.unique_words) + 1:
        _fail("Chunk ranges do not cover every unique word.")
    return True


def audit_vocabulary_view(view):
    """Validate the source/context/first-occurrence data used for generation."""
    canonical_text = view.snapshot.canonical_text
    if not view.unique_words:
        _fail("Vocabulary view contains no unique words.")
    sections = {
        section.section_id: section
        for section in view.snapshot.sections
    }
    contexts = {}
    previous_context_end = {}
    for context in view.contexts:
        if context.context_id in contexts:
            _fail(f"Duplicate context ID: {context.context_id}")
        section = sections.get(context.section_id)
        if section is None:
            _fail(f"Unknown context section: {context.context_id}")
        if context.kind not in {"paragraph", "sentence"}:
            _fail(f"Unknown context kind: {context.context_id}")
        if (
                context.start_offset < section.start_offset
                or context.end_offset <= context.start_offset
                or context.end_offset > section.end_offset):
            _fail(f"Context is outside its section: {context.context_id}")
        if canonical_text[
                context.start_offset:context.end_offset] != context.text:
            _fail(f"Context offset mismatch: {context.context_id}")
        order_key = (context.section_id, context.kind)
        if context.start_offset < previous_context_end.get(order_key, -1):
            _fail(
                f"{context.kind.title()} contexts overlap or reorder in "
                f"{context.section_id}.")
        previous_context_end[order_key] = context.end_offset
        contexts[context.context_id] = context

    sentence_neighbors = sentence_neighbor_ids(view.contexts)
    seen_words = set()
    previous_first_end = -1
    for expected_rank, word in enumerate(view.unique_words, start=1):
        if word.rank != expected_rank:
            _fail("Unique-word ranks are not contiguous.")
        if word.normalized in seen_words:
            _fail(f"Duplicate unique word: {word.normalized}")
        if (
                not isinstance(word.occurrence_count, int)
                or word.occurrence_count < 1):
            _fail(f"Invalid occurrence count for {word.surface}.")
        if word.start_offset < previous_first_end:
            _fail("Unique words are not ordered by first occurrence.")
        if word.end_offset <= word.start_offset:
            _fail(f"Empty unique-word span: {word.surface}")
        section = sections.get(word.section_id)
        paragraph = contexts.get(word.paragraph_id)
        sentence = contexts.get(word.sentence_id)
        if (
                section is None
                or paragraph is None
                or sentence is None
                or paragraph.kind != "paragraph"
                or sentence.kind != "sentence"
                or paragraph.section_id != word.section_id
                or sentence.section_id != word.section_id):
            _fail(f"Unique-word context mismatch for {word.surface}.")
        if not (
                section.start_offset <= word.start_offset
                and word.end_offset <= section.end_offset
                and paragraph.start_offset <= word.start_offset
                and word.end_offset <= paragraph.end_offset
                and sentence.start_offset <= word.start_offset
                and word.end_offset <= sentence.end_offset):
            _fail(f"Unique word is outside its context: {word.surface}.")
        if canonical_text[
                word.start_offset:word.end_offset] != word.surface:
            _fail(f"Unique-word offset mismatch for {word.surface}.")
        if (
                word.normalized
                != unicodedata.normalize("NFC", word.surface)):
            _fail(f"Unique-word normalization mismatch for {word.surface}.")
        if not is_han_word(word.surface):
            _fail(f"Non-Han unique word: {word.surface}.")
        expected_neighbors = sentence_neighbors.get(word.sentence_id)
        if (
                expected_neighbors is None
                or word.previous_sentence_id != expected_neighbors[0]
                or word.next_sentence_id != expected_neighbors[1]):
            _fail(f"Sentence-neighbor mismatch for {word.surface}.")
        seen_words.add(word.normalized)
        previous_first_end = word.end_offset

    chunks = make_chunks(
        view.unique_words,
        view.config.chunk_size)
    if sum(len(chunk.words) for chunk in chunks) != len(view.unique_words):
        _fail("Chunk ranges do not cover every unique word.")
    return True
