"""Archived local-model token-occurrence metadata tests."""

from dataclasses import replace
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline.models import TokenSpan, TokenizerIdentity
from corpus_pipeline.storage import write_build
from source_generation import (
    ContextOccurrenceSpan,
    GenerationChunk,
    SourceGenerationConfig,
    load_processed_source,
    plan_source_generation,
    render_chunk_input,
)
from tests.test_source_generation import make_source


class WangBiOpeningTokenizer:
    """Tokenize 常道 as one word while retaining both standalone 道 tokens."""

    @property
    def identity(self):
        return TokenizerIdentity(
            backend="context-occurrence-test",
            backend_version="1",
            model="wang-bi-opening",
            model_revision="1")

    def tokenize(self, text):
        expected = "道可道，非常道。"
        if text != expected:
            raise AssertionError(f"Unexpected fixture text: {text!r}")
        return (
            TokenSpan("道", 0, 1),
            TokenSpan("可", 1, 2),
            TokenSpan("道", 2, 3),
            TokenSpan("非", 4, 5),
            TokenSpan("常道", 5, 7),
        )


def make_saved_opening(directory):
    source = make_source(
        ("道可道，非常道。",),
        tokenizer=WangBiOpeningTokenizer())
    run_path = write_build(source.build, directory)
    return load_processed_source(
        source.key,
        run_path=run_path,
        corpus_root=directory)


def make_v10_plan(source):
    return plan_source_generation(
        source,
        SourceGenerationConfig(
            source_key=source.key,
            request_protocol="v10"))


class ContextOccurrenceSpanTests(unittest.TestCase):
    def test_span_is_immutable_validated_and_round_trips_in_a_chunk(self):
        span = ContextOccurrenceSpan(
            start_offset=0,
            end_offset=1,
            surface="道")
        with self.assertRaises(ValueError):
            ContextOccurrenceSpan(
                start_offset=True,
                end_offset=1,
                surface="道")
        with self.assertRaises(ValueError):
            ContextOccurrenceSpan(
                start_offset=1,
                end_offset=1,
                surface="道")
        with self.assertRaises(ValueError):
            ContextOccurrenceSpan(
                start_offset=0,
                end_offset=1,
                surface="")

        with tempfile.TemporaryDirectory() as directory:
            chunk = make_v10_plan(
                make_saved_opening(directory)).chunks[0]
        restored = GenerationChunk.from_dict(chunk.to_dict())

        self.assertEqual(
            restored.words[0].context_occurrences,
            (span, ContextOccurrenceSpan(2, 3, "道")))
        self.assertEqual(restored, chunk)

    def test_legacy_word_serialization_omits_empty_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_saved_opening(directory)
            plan = plan_source_generation(
                source,
                SourceGenerationConfig(
                    source_key=source.key,
                    request_protocol="v9"))

        word_json = plan.chunks[0].to_dict()["words"][0]
        self.assertNotIn("context_occurrences", word_json)


class ContextOccurrencePlanningTests(unittest.TestCase):
    def test_streams_only_complete_tokens_matching_the_normalized_word(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = make_v10_plan(make_saved_opening(directory))

        words = {
            word.surface: word
            for word in plan.chunks[0].words
        }
        self.assertEqual(
            words["道"].context_occurrences,
            (
                ContextOccurrenceSpan(0, 1, "道"),
                ContextOccurrenceSpan(2, 3, "道"),
            ))
        # The final literal 道 is part of the independently tokenized 常道,
        # so it must never be presented as a third standalone 道 occurrence.
        self.assertEqual(
            words["常道"].context_occurrences,
            (ContextOccurrenceSpan(5, 7, "常道"),))

    def test_absent_artifact_preserves_hand_built_source_behavior(self):
        source = make_source(
            ("道可道，非常道。",),
            tokenizer=WangBiOpeningTokenizer())

        plan = make_v10_plan(source)

        self.assertTrue(all(
            not word.context_occurrences
            for word in plan.chunks[0].words))

    def test_stream_verifies_occurrence_bytes_against_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_saved_opening(directory)
            occurrence_path = source.run_path / "occurrences.jsonl"
            occurrence_path.write_bytes(
                occurrence_path.read_bytes() + b"\n")

            with self.assertRaisesRegex(ValueError, "artifact hash failed"):
                make_v10_plan(source)


class ContextOccurrencePayloadTests(unittest.TestCase):
    def test_only_v10_renders_relative_spans_and_actual_surfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            chunk = make_v10_plan(
                make_saved_opening(directory)).chunks[0]

        v10 = json.loads(render_chunk_input(
            chunk,
            protocol_version=10))
        word = next(
            item
            for item in v10["words"]
            if item["term"] == "道")
        self.assertEqual(
            word["context_occurrences"],
            [
                {"span": [0, 1], "surface": "道"},
                {"span": [2, 3], "surface": "道"},
            ])

        v9_with_metadata = render_chunk_input(
            chunk,
            protocol_version=9)
        without_metadata = replace(
            chunk,
            words=tuple(
                replace(word, context_occurrences=())
                for word in chunk.words))
        self.assertEqual(
            v9_with_metadata,
            render_chunk_input(
                without_metadata,
                protocol_version=9))
        self.assertNotIn(
            "context_occurrences",
            chunk.request_payload()["words"][0])


if __name__ == "__main__":
    unittest.main()
