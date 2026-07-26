"""Focused tests for the rollback-safe compact v10 source protocol."""

import hashlib
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
from source_generation import (
    ContextUnit,
    GenerationChunk,
    SOURCE_REQUEST_CONTRACT_SCHEMA_VERSION,
    build_source_request_contract,
    render_chunk_input,
    source_request_uses_compact_source_results,
)
from source_generation.requests import normalise_source_request_contract
from tests.test_v8_schema_contract import context_pipeline, grouped_chunk
from tests.test_v9_schema_contract import located_chunk


def opening_chunk():
    original = located_chunk()
    return GenerationChunk(
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


def sentence_item_schema(response_format):
    return response_format["schema"]["properties"][
        "term_results"]["items"]["properties"][
            "additional_senses"]["items"]["properties"][
                "Sentences"]["items"]


def root_array_schema(response_format, field_name):
    return response_format["schema"]["properties"][field_name]


class CompactV10SchemaTests(unittest.TestCase):
    def test_v10_sentences_are_plain_strings_without_model_emphasis_schema(self):
        pipeline = context_pipeline()

        v9 = process_text.build_compact_source_response_format(
            pipeline,
            protocol_version=9)
        v10 = process_text.build_compact_source_response_format(
            pipeline,
            protocol_version=10)

        self.assertEqual(sentence_item_schema(v9), {"type": "string"})
        self.assertEqual(
            sentence_item_schema(v10),
            {"type": "string"})
        self.assertTrue(v9["name"].endswith("_compact_source_v9"))
        self.assertTrue(v10["name"].endswith("_compact_source_v10"))

    def test_v10_bounds_additional_senses_but_v9_schema_stays_unbounded(self):
        pipeline = context_pipeline()

        v9 = process_text.build_compact_source_response_format(
            pipeline,
            protocol_version=9)
        v10 = process_text.build_compact_source_response_format(
            pipeline,
            protocol_version=10)
        v9_additional = root_array_schema(
            v9,
            "term_results")["items"]["properties"]["additional_senses"]
        v10_additional = root_array_schema(
            v10,
            "term_results")["items"]["properties"]["additional_senses"]

        self.assertNotIn("minItems", v9_additional)
        self.assertNotIn("maxItems", v9_additional)
        self.assertEqual(v10_additional["minItems"], 0)
        self.assertEqual(v10_additional["maxItems"], 8)

    def test_per_chunk_v10_schema_has_exact_root_array_bounds(self):
        chunk = grouped_chunk()

        response_format = process_text.build_compact_source_response_format(
            context_pipeline(),
            protocol_version=10,
            chunk=chunk)
        term_results = root_array_schema(
            response_format,
            "term_results")
        translations = root_array_schema(
            response_format,
            "source_context_translations")

        self.assertEqual(
            (term_results["minItems"], term_results["maxItems"]),
            (len(chunk.words), len(chunk.words)))
        self.assertEqual(
            (translations["minItems"], translations["maxItems"]),
            (len(chunk.contexts), len(chunk.contexts)))

    def test_translation_memory_allows_zero_but_never_extra_translations(self):
        chunk = grouped_chunk()

        response_format = process_text.build_compact_source_response_format(
            context_pipeline(),
            protocol_version=10,
            chunk=chunk,
            translation_memory_enabled=True)
        translations = root_array_schema(
            response_format,
            "source_context_translations")

        self.assertEqual(translations["minItems"], 0)
        self.assertEqual(
            translations["maxItems"],
            len(chunk.contexts))

    def test_repeating_one_context_until_truncation_is_schema_impossible(self):
        one_context = opening_chunk()
        response_format = process_text.build_compact_source_response_format(
            context_pipeline(),
            protocol_version=10,
            chunk=one_context)
        translations = root_array_schema(
            response_format,
            "source_context_translations")
        repeated_output = [
            {
                "context_id": "context-1",
                "translation": "The Way that can be spoken...",
            },
            {
                "context_id": "context-1",
                "translation": "The Way that can be spoken...",
            },
        ]

        self.assertEqual(translations["maxItems"], 1)
        self.assertGreater(
            len(repeated_output),
            translations["maxItems"])

    def test_compact_schema_rejects_unknown_protocols(self):
        with self.assertRaisesRegex(ValueError, "9 or 10"):
            process_text.build_compact_source_response_format(
                context_pipeline(),
                protocol_version=11)


class VersionTenContractTests(unittest.TestCase):
    def test_programmatic_default_builds_a_v10_contract(self):
        pipeline = context_pipeline()
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True)

        self.assertEqual(SOURCE_REQUEST_CONTRACT_SCHEMA_VERSION, 10)
        self.assertEqual(contract["schema_version"], 10)
        self.assertEqual(
            contract["response_format"],
            process_text.build_compact_source_response_format(
                pipeline,
                protocol_version=10))
        self.assertEqual(
            set(contract["response_formats_by_chunk"]),
            {chunk.chunk_id})
        self.assertEqual(
            contract["response_formats_by_chunk"][chunk.chunk_id],
            process_text.build_compact_source_response_format(
                pipeline,
                protocol_version=10,
                chunk=chunk))
        self.assertRegex(
            contract["prompt_cache_key"],
            r"^autoanki-source-v10-[0-9a-f]{32}$")
        self.assertTrue(
            source_request_uses_compact_source_results(contract))

    def test_old_v10_contract_with_empty_per_chunk_map_still_loads(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=10)
        contract["response_formats_by_chunk"] = {}

        normalised = normalise_source_request_contract(contract)

        self.assertEqual(normalised["response_formats_by_chunk"], {})
        self.assertEqual(
            normalised["response_format"],
            contract["response_format"])

    def test_v10_contract_translation_memory_relaxes_only_context_minimum(self):
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=10,
            translation_memory_enabled=True)
        response_format = contract[
            "response_formats_by_chunk"][chunk.chunk_id]
        term_results = root_array_schema(response_format, "term_results")
        translations = root_array_schema(
            response_format,
            "source_context_translations")

        self.assertEqual(
            (term_results["minItems"], term_results["maxItems"]),
            (len(chunk.words), len(chunk.words)))
        self.assertEqual(translations["minItems"], 0)
        self.assertEqual(
            translations["maxItems"],
            len(chunk.contexts))

    def test_v10_prompt_requests_plain_unformatted_sentences(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=10)
        prompt = contract["composed_prompt"]

        for forbidden in (
                "<strong>",
                "</strong>",
                "quality_hint",
                "highlight"):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("three plain strings", prompt)
        self.assertIn("complete lexical item", prompt)
        self.assertIn("ordinary inflected form", prompt)
        self.assertIn("visually similar or homophonous", prompt)
        self.assertIn("never invent component senses", prompt)
        self.assertTrue(
            prompt.endswith("Here is the source batch JSON:\n"))

    def test_saved_v9_contract_remains_supported_and_normalises_unchanged(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=9)

        self.assertEqual(contract["schema_version"], 9)
        self.assertRegex(
            contract["prompt_cache_key"],
            r"^autoanki-source-v9-[0-9a-f]{32}$")
        self.assertEqual(
            normalise_source_request_contract(contract),
            contract)
        self.assertTrue(
            source_request_uses_compact_source_results(contract))

    def test_v9_request_material_matches_its_pre_v10_fingerprints(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=9)
        material = {
            "prompt": contract["composed_prompt"],
            "schema": json.dumps(
                contract["response_format"],
                ensure_ascii=False,
                separators=(",", ":")),
            "contract": json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":")),
            "payload": render_chunk_input(
                located_chunk(),
                protocol_version=9),
        }

        self.assertEqual(
            {
                key: hashlib.sha256(
                    value.encode("utf-8")).hexdigest()
                for key, value in material.items()
            },
            {
                "prompt": (
                    "ba36a30bc806eca11e0282d19943b0843556bec5eede95f112e418"
                    "abc7abd28e"),
                "schema": (
                    "ba829e73d899ffb44dbc022d4757c10d4e31053fac348d912bb7338"
                    "0bd1b31ed"),
                "contract": (
                    "e940f5b772fda44ba97ba5895b3ee04e22fa31b52244906d6bb48c"
                    "259c8eb648"),
                "payload": (
                    "3f569818c47eedc9bfc4337da31da0bf14f7f0a08c3ffc67859e8ed"
                    "cca4f1351"),
            })


class VersionTenPayloadTests(unittest.TestCase):
    def test_compact_payloads_omit_term_specific_quality_hints(self):
        chunk = opening_chunk()

        v9 = json.loads(render_chunk_input(
            chunk,
            protocol_version=9))
        v10 = json.loads(render_chunk_input(
            chunk,
            protocol_version=10))

        self.assertNotIn("quality_hint", v9["words"][0])
        self.assertNotIn("quality_hint", v9["words"][1])
        self.assertNotIn("quality_hint", v10["words"][0])
        self.assertNotIn("quality_hint", v10["words"][1])
        self.assertEqual(v10["contexts"], v9["contexts"])
        self.assertEqual(v10["words"], v9["words"])


if __name__ == "__main__":
    unittest.main()
