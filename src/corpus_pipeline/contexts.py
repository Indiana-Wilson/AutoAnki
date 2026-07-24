"""Canonical text assembly and future-proof context span extraction."""

from dataclasses import replace
import re
import unicodedata

from corpus_pipeline.models import (
    CorpusSnapshot,
    ContextSpan,
    CorpusValidationError,
    TextSection,
)


SECTION_SEPARATOR = "\n\n"
TITLE_SEPARATOR = "\n\n"
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_END = re.compile(r"[。！？!?]+[」』】）》”’]*")


def _trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def assemble_sections(sections):
    """NFC-normalize sections and assign global canonical-text offsets."""
    aligned = []
    parts = []
    offset = 0
    seen_ids = set()

    for expected_order, section in enumerate(sections, start=1):
        if section.section_id in seen_ids:
            raise CorpusValidationError(
                f"Duplicate section ID: {section.section_id}")
        if section.order != expected_order:
            raise CorpusValidationError(
                "Sections must be supplied in contiguous source order.")

        text = unicodedata.normalize("NFC", section.text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            raise CorpusValidationError(
                f"Section {section.section_id} contains no text.")

        if parts:
            parts.append(SECTION_SEPARATOR)
            offset += len(SECTION_SEPARATOR)
        start = offset
        parts.append(text)
        offset += len(text)
        aligned.append(replace(
            section,
            text=text,
            start_offset=start,
            end_offset=offset))
        seen_ids.add(section.section_id)

    return tuple(aligned), "".join(parts)


def with_section_titles(snapshot: CorpusSnapshot, enabled):
    """Return a snapshot whose canonical sections include/exclude titles."""
    if not isinstance(enabled, bool):
        raise ValueError("Section-title inclusion must be true or false.")
    if snapshot.include_section_titles == enabled:
        return snapshot

    sections = []
    for section in snapshot.sections:
        prefix = f"{section.title}{TITLE_SEPARATOR}"
        if enabled:
            text = prefix + section.text
        else:
            if not section.text.startswith(prefix):
                raise CorpusValidationError(
                    f"Section-title prefix is missing from "
                    f"{section.section_id}.")
            text = section.text[len(prefix):]
        sections.append(replace(
            section,
            text=text,
            start_offset=0,
            end_offset=0,
        ))
    aligned, canonical_text = assemble_sections(sections)
    return replace(
        snapshot,
        sections=aligned,
        canonical_text=canonical_text,
        include_section_titles=enabled,
    )


def _paragraph_spans(section):
    local_text = section.text
    boundaries = [0]
    boundaries.extend(match.end() for match in _PARAGRAPH_BREAK.finditer(
        local_text))
    ends = [match.start() for match in _PARAGRAPH_BREAK.finditer(
        local_text)]
    ends.append(len(local_text))

    spans = []
    for start, end in zip(boundaries, ends, strict=True):
        start, end = _trim_span(local_text, start, end)
        if start < end:
            spans.append((start, end))
    return spans


def _sentence_spans(text, paragraph_start, paragraph_end):
    spans = []
    cursor = paragraph_start
    for match in _SENTENCE_END.finditer(
            text,
            paragraph_start,
            paragraph_end):
        end = match.end()
        start, trimmed_end = _trim_span(text, cursor, end)
        if start < trimmed_end:
            spans.append((start, trimmed_end))
        cursor = end
    start, end = _trim_span(text, cursor, paragraph_end)
    if start < end:
        spans.append((start, end))
    return spans


def build_contexts(canonical_text, sections):
    """Return exact paragraph and sentence spans in canonical text."""
    contexts = []
    for section in sections:
        paragraph_number = 0
        sentence_number = 0
        for local_start, local_end in _paragraph_spans(section):
            paragraph_number += 1
            global_start = section.start_offset + local_start
            global_end = section.start_offset + local_end
            paragraph_id = (
                f"{section.section_id}:paragraph:{paragraph_number:04d}")
            paragraph_text = canonical_text[global_start:global_end]
            contexts.append(ContextSpan(
                context_id=paragraph_id,
                kind="paragraph",
                section_id=section.section_id,
                start_offset=global_start,
                end_offset=global_end,
                text=paragraph_text))

            for sentence_start, sentence_end in _sentence_spans(
                    section.text,
                    local_start,
                    local_end):
                sentence_number += 1
                sentence_global_start = (
                    section.start_offset + sentence_start)
                sentence_global_end = (
                    section.start_offset + sentence_end)
                sentence_id = (
                    f"{section.section_id}:sentence:"
                    f"{sentence_number:05d}")
                contexts.append(ContextSpan(
                    context_id=sentence_id,
                    kind="sentence",
                    section_id=section.section_id,
                    start_offset=sentence_global_start,
                    end_offset=sentence_global_end,
                    text=canonical_text[
                        sentence_global_start:sentence_global_end]))
    return tuple(contexts)


def sentence_neighbor_ids(contexts):
    """Map every sentence ID to its preceding and succeeding sentence IDs."""
    sentences = tuple(
        context
        for context in contexts
        if context.kind == "sentence"
    )
    return {
        sentence.context_id: (
            sentences[index - 1].context_id if index else None,
            (
                sentences[index + 1].context_id
                if index + 1 < len(sentences)
                else None
            ),
        )
        for index, sentence in enumerate(sentences)
    }
