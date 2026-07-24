"""Strict validators that can be injected into a source-generation runner."""

import unicodedata

import pipeline_store
import process_text


def make_pipeline_response_validator(pipeline):
    """Validate schema, card semantics, and membership in the source chunk."""
    pipeline_store.validate_pipelines((pipeline,))
    term_field = pipeline_store.get_language(
        pipeline.language_key).term_field

    def validate(raw_text, chunk):
        result = process_text.validate_generated_response(
            raw_text,
            pipeline)
        rank_by_term = {
            unicodedata.normalize("NFC", word.surface): word.rank
            for word in chunk.words
        }
        unexpected = sorted({
            card[term_field]
            for card in result["cards"]
            if unicodedata.normalize(
                "NFC",
                card[term_field]) not in rank_by_term
        })
        if unexpected:
            raise process_text.GeneratedCardValidationError(
                "The response contains terms outside this source chunk: "
                + ", ".join(unexpected))
        returned_terms = {
            unicodedata.normalize("NFC", card[term_field])
            for card in result["cards"]
        }
        missing = [
            word.surface
            for word in chunk.words
            if word.normalized not in returned_terms
        ]
        if missing:
            raise process_text.GeneratedCardValidationError(
                "The response omitted requested source terms: "
                + ", ".join(missing))
        # Concurrency cannot disturb source order: combined output reads chunks
        # by index, and cards within a chunk are normalized back to rank order.
        result["cards"].sort(key=lambda card: rank_by_term[
            unicodedata.normalize("NFC", card[term_field])])
        return result

    return validate
