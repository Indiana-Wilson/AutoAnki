"""Deterministic 500-word inspection chunk construction."""

from dataclasses import dataclass

from corpus_pipeline.models import CorpusBuild, CorpusValidationError


@dataclass(frozen=True)
class WordChunk:
    start_rank: int
    end_rank: int
    words: tuple

    @property
    def stem(self):
        return f"{self.start_rank}-{self.end_rank}"


def make_chunks(unique_words, chunk_size=500):
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")
    chunks = []
    for start in range(0, len(unique_words), chunk_size):
        words = tuple(unique_words[start:start + chunk_size])
        if not words:
            continue
        expected_start = start + 1
        if any(
                word.rank != expected_start + offset
                for offset, word in enumerate(words)):
            raise CorpusValidationError(
                "Unique-word ranks are not contiguous.")
        chunks.append(WordChunk(
            start_rank=words[0].rank,
            end_rank=words[-1].rank,
            words=words))
    return tuple(chunks)


def chunk_text(chunk):
    return "".join(f"{word.surface}\n" for word in chunk.words)


def chunk_records(build: CorpusBuild, chunk):
    contexts = {
        context.context_id: context
        for context in build.contexts
    }
    sections = {
        section.section_id: section
        for section in build.snapshot.sections
    }
    records = []
    for word in chunk.words:
        section = sections[word.section_id]
        paragraph = contexts[word.paragraph_id]
        sentence = contexts[word.sentence_id]
        previous_sentence = (
            contexts[word.previous_sentence_id]
            if word.previous_sentence_id is not None
            else None
        )
        next_sentence = (
            contexts[word.next_sentence_id]
            if word.next_sentence_id is not None
            else None
        )
        record = word.to_dict()
        record.update({
            "section_title": section.title,
            "first_occurrence_is_section_title": (
                build.snapshot.include_section_titles
                and section.start_offset <= word.start_offset
                and word.end_offset
                <= section.start_offset + len(section.title)
            ),
            "previous_sentence_text": (
                previous_sentence.text
                if previous_sentence is not None
                else None
            ),
            "sentence_text": sentence.text,
            "next_sentence_text": (
                next_sentence.text
                if next_sentence is not None
                else None
            ),
            "paragraph_text": paragraph.text,
        })
        records.append(record)
    return records
