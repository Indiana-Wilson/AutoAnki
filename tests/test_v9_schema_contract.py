import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
import pipeline_store
from source_generation import (
    ContextUnit,
    GenerationChunk,
    GenerationWord,
    build_source_request_contract,
    render_chunk_input,
    source_request_uses_compact_source_results,
    source_request_uses_grouped_source_results,
)
from source_generation.models import OccurrenceLocator
from source_generation.planning import estimate_text_tokens
from source_generation.requests import (
    normalise_source_request_contract,
)
from tests.test_v8_schema_contract import (
    context_pipeline,
    detailed_context_pipeline,
    grouped_chunk,
)


def located_chunk():
    locator = OccurrenceLocator(
        before="道可",
        target="道",
        after="，非",
        literal_match_ordinal=2,
        marked_excerpt="道可⟪TARGET⟫道⟪/TARGET⟫，非")
    return GenerationChunk(
        chunk_id="000001-r1-r2",
        index=1,
        total=1,
        start_rank=1,
        end_rank=2,
        words=(
            GenerationWord(
                rank=1,
                surface="道",
                normalized="道",
                section_id="section-1",
                sentence_id="sentence-1",
                context_id="context-1",
                start_offset=2,
                end_offset=3,
                occurrence_locator=locator),
            GenerationWord(
                rank=2,
                surface="可",
                normalized="可",
                section_id="section-1",
                sentence_id="sentence-1",
                context_id="context-1",
                start_offset=1,
                end_offset=2),
        ),
        contexts=(
            ContextUnit(
                context_id="context-1",
                mode="sentence",
                start_offset=0,
                end_offset=8,
                text="道可道，非常道",
                section_ids=("section-1",),
                sentence_ids=("sentence-1",),
                word_ranks=(1, 2)),
        ))


class CompactV9SchemaTests(unittest.TestCase):
    def test_fixed_schema_uses_rank_and_context_arrays(self):
        pipeline = context_pipeline()

        response_format = (
            process_text.build_compact_source_response_format(pipeline))
        schema = response_format["schema"]

        self.assertEqual(
            schema["required"],
            ["term_results", "source_context_translations"])
        self.assertFalse(schema["additionalProperties"])
        results = schema["properties"]["term_results"]
        self.assertEqual(results["type"], "array")
        result = results["items"]
        self.assertEqual(
            result["required"],
            ["rank", "contextual_sense", "additional_senses"])
        self.assertEqual(
            result["properties"]["rank"],
            {"type": "integer"})
        self.assertFalse(result["additionalProperties"])

        contextual = result["properties"]["contextual_sense"]
        self.assertEqual(
            contextual["required"],
            [
                "Translation (English)",
                "Dictionary Meaning (English)",
            ])
        self.assertNotIn("Classical Chinese", contextual["properties"])
        self.assertNotIn("Sentences", contextual["properties"])
        self.assertFalse(contextual["additionalProperties"])

        additional = result["properties"]["additional_senses"]["items"]
        self.assertNotIn("Classical Chinese", additional["properties"])
        for field_name in (
                "Sentences",
                "Sentence Translations (English)"):
            collection = additional["properties"][field_name]
            self.assertEqual(collection["minItems"], 3)
            self.assertEqual(collection["maxItems"], 3)
            self.assertEqual(collection["items"], {"type": "string"})
        self.assertEqual(
            set(additional["required"]),
            set(additional["properties"]))
        self.assertFalse(additional["additionalProperties"])

        translations = schema["properties"][
            "source_context_translations"]
        self.assertEqual(translations["type"], "array")
        self.assertEqual(
            translations["items"]["required"],
            ["context_id", "translation"])
        self.assertEqual(
            translations["items"]["properties"],
            {
                "context_id": {"type": "string"},
                "translation": {"type": "string"},
            })
        self.assertFalse(
            translations["items"]["additionalProperties"])
        encoded = json.dumps(response_format, ensure_ascii=False)
        self.assertNotIn("ctx:opening", encoded)
        self.assertNotIn('"道"', encoded)
        self.assertNotIn('"2"', encoded)
        self.assertLessEqual(len(response_format["name"]), 64)

    def test_fixed_schema_is_identical_for_every_chunk(self):
        pipeline = detailed_context_pipeline()

        first = process_text.build_compact_source_response_format(
            pipeline)
        second = process_text.build_compact_source_response_format(
            pipeline)

        self.assertEqual(first, second)
        self.assertIn(
            "Pronunciation (English)",
            first["schema"]["properties"]["term_results"]["items"][
                "properties"]["contextual_sense"]["properties"])


class VersionNineContractTests(unittest.TestCase):
    def test_explicit_contract_uses_compact_v9_protocol(self):
        pipeline = context_pipeline()
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=9)

        self.assertEqual(contract["schema_version"], 9)
        self.assertEqual(
            contract["response_format"],
            process_text.build_compact_source_response_format(
                pipeline,
                source_lexical_only=True,
                include_generated_examples=False))
        self.assertEqual(contract["response_formats_by_chunk"], {})
        self.assertEqual(contract["reasoning"], {"effort": "low"})
        self.assertRegex(
            contract["prompt_cache_key"],
            r"^autoanki-source-v9-[0-9a-f]{32}$")
        self.assertLessEqual(len(contract["prompt_cache_key"]), 64)
        self.assertTrue(
            source_request_uses_compact_source_results(contract))
        self.assertFalse(
            source_request_uses_grouped_source_results(contract))
        prompt = contract["composed_prompt"]
        self.assertIn(
            '"term_results" and "source_context_translations"',
            " ".join(prompt.split()))
        self.assertIn("Copy a required numeric \"rank\" exactly", prompt)
        self.assertIn("Do not perform or return lexical analysis", prompt)
        self.assertNotIn("GROUPED V8 TARGET RULE", prompt)
        self.assertTrue(
            prompt.endswith("Here is the source batch JSON:\n"))
        self.assertLessEqual(estimate_text_tokens(prompt), 1_750)

    def test_v9_reasoning_and_cache_key_are_stable_and_frozen(self):
        pipeline = context_pipeline()
        chunk = grouped_chunk()

        low = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=9,
            reasoning_effort="low")
        none = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=9,
            reasoning_effort="none")

        self.assertEqual(none["reasoning"], {"effort": "none"})
        self.assertEqual(
            low["prompt_cache_key"],
            none["prompt_cache_key"])
        self.assertEqual(
            normalise_source_request_contract(none),
            none)
        with self.assertRaisesRegex(ValueError, "none.*low"):
            build_source_request_contract(
                pipeline,
                protocol_version=9,
                reasoning_effort="medium")

    def test_protocol_eight_rollback_keeps_exact_chunk_schema(self):
        pipeline = context_pipeline()
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=8)

        self.assertEqual(contract["schema_version"], 8)
        self.assertNotIn("prompt_cache_key", contract)
        self.assertEqual(contract["reasoning"], {"effort": "low"})
        self.assertEqual(
            set(contract["response_formats_by_chunk"]),
            {chunk.chunk_id})
        self.assertEqual(
            contract["response_formats_by_chunk"][chunk.chunk_id],
            process_text.build_grouped_source_response_format(
                pipeline,
                chunk,
                source_lexical_only=True,
                include_generated_examples=False))
        self.assertTrue(
            source_request_uses_grouped_source_results(contract))
        self.assertFalse(
            source_request_uses_compact_source_results(contract))
        self.assertNotIn("exactly three", contract["composed_prompt"])
        self.assertNotIn("<strong>", contract["composed_prompt"])
        with self.assertRaisesRegex(ValueError, "v8.*low"):
            build_source_request_contract(
                pipeline,
                protocol_version=8,
                reasoning_effort="none")

    def test_v9_normalizer_rejects_per_chunk_formats(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=9)
        contract["response_formats_by_chunk"] = {
            "unexpected": contract["response_format"],
        }

        with self.assertRaisesRegex(ValueError, "Only grouped v8"):
            normalise_source_request_contract(contract)

    def test_detailed_prompt_preserves_language_and_field_semantics(self):
        pipeline = detailed_context_pipeline()
        settings = pipeline_store.get_language_settings(
            pipeline,
            pipeline.language_key)
        settings = replace(
            settings,
            cards=tuple(
                replace(card, enabled=True)
                for card in settings.cards))
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,),
            active_language_key=pipeline.language_key)
        contract = build_source_request_contract(
            pipeline,
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=9)
        prompt = contract["composed_prompt"]

        for text in (
                "received Laozi text transmitted with Wang Bi",
                'For "Translation (English)"',
                'For "Dictionary Meaning (English)"',
                'For "Pronunciation (English)"',
                'For "Part of Speech (English)"',
                'For "Register (English)"',
                'For "Nuance (English)"'):
            self.assertIn(text, prompt)
        self.assertLessEqual(estimate_text_tokens(prompt), 2_000)


class VersionNinePayloadTests(unittest.TestCase):
    def test_v8_and_older_payload_bytes_are_unchanged(self):
        chunk = located_chunk()
        expected = json.dumps(
            chunk.request_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"))

        self.assertEqual(render_chunk_input(chunk), expected)
        self.assertEqual(
            render_chunk_input(chunk, protocol_version=8),
            expected)
        self.assertEqual(
            render_chunk_input(chunk, protocol_version=3),
            expected)

    def test_v9_payload_keeps_only_compact_rank_addressing(self):
        payload = json.loads(render_chunk_input(
            located_chunk(),
            protocol_version=9))

        self.assertEqual(set(payload), {"words", "contexts"})
        self.assertEqual(
            payload["contexts"],
            [{"context_id": "context-1", "text": "道可道，非常道"}])
        self.assertEqual(
            payload["words"][0],
            {
                "rank": 1,
                "term": "道",
                "context_id": "context-1",
                "marked_excerpt": (
                    "道可⟪TARGET⟫道⟪/TARGET⟫，非"),
            })
        self.assertEqual(
            payload["words"][1],
            {
                "rank": 2,
                "term": "可",
                "context_id": "context-1",
            })
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("chunk_id", encoded)
        self.assertNotIn("rank_range", encoded)
        self.assertNotIn("occurrence_span", encoded)
        self.assertNotIn("literal_match_ordinal", encoded)

    def test_v9_does_not_add_term_specific_quality_hints(self):
        original = located_chunk()
        opening = GenerationChunk(
            chunk_id=original.chunk_id,
            index=original.index,
            total=original.total,
            start_rank=original.start_rank,
            end_rank=original.end_rank,
            words=original.words,
            contexts=(
                ContextUnit(
                    context_id="context-1",
                    mode="sentence",
                    start_offset=0,
                    end_offset=20,
                    text="道可道，非常道，名可名，非常名；",
                    section_ids=("section-1",),
                    sentence_ids=("sentence-1",),
                    word_ranks=(1, 2)),
            ))

        payload = json.loads(render_chunk_input(
            opening,
            protocol_version=9))

        self.assertNotIn("quality_hint", payload["words"][0])
        self.assertNotIn("quality_hint", payload["words"][1])


if __name__ == "__main__":
    unittest.main()
