import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from source_generation.requests import (
    normalise_source_request_contract,
    source_request_contract_digest,
)


def _leaf():
    return {"type": "string"}


def _rank_result_schema():
    return {
        "type": "object",
        "properties": {
            "additional_senses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Sentence Translations (English)": {
                            "type": "array",
                            "items": _leaf(),
                        },
                        "Sentences": {
                            "type": "array",
                            "items": _leaf(),
                        },
                        "optional_note": _leaf(),
                    },
                    "required": [
                        "Sentences",
                        "Sentence Translations (English)",
                    ],
                },
            },
            "contextual_sense": {
                "type": "object",
                "properties": {
                    "Dictionary Meaning (English)": _leaf(),
                    "Translation (English)": _leaf(),
                },
                "required": [
                    "Translation (English)",
                    "Dictionary Meaning (English)",
                ],
            },
            "optional_note": _leaf(),
        },
        "required": ["contextual_sense", "additional_senses"],
    }


def _strict_format(name):
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "aaa_optional": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "alpha": _leaf(),
                                "optional": _leaf(),
                                "zeta": _leaf(),
                            },
                            "required": ["zeta", "alpha"],
                        },
                    ],
                },
                "source_context_translations": {
                    "type": "object",
                    "properties": {
                        "ctx:a": _leaf(),
                        "ctx:z": _leaf(),
                    },
                    "required": ["ctx:z", "ctx:a"],
                },
                "term_results": {
                    "type": "object",
                    "properties": {
                        "10": _rank_result_schema(),
                        "2": _rank_result_schema(),
                    },
                    "required": ["2", "10"],
                },
                "zzz_optional": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "first": _leaf(),
                            "second": _leaf(),
                        },
                        "required": ["second", "first"],
                    },
                },
            },
            "required": [
                "term_results",
                "source_context_translations",
            ],
        },
    }


def _version_eight_contract():
    return {
        "schema_version": 8,
        "model": "test-model",
        "composed_prompt": "Test prompt.",
        "response_format": _strict_format("generic_source_format"),
        "reasoning": {"effort": "none"},
        "tools": [],
        "max_tool_calls": 0,
        "use_source_for_example_sentences": True,
        "require_sentence_translations": True,
        "max_output_tokens_by_chunk": {"chunk-a": 8192},
        "response_formats_by_chunk": {
            "chunk-a": _strict_format("chunk_source_format"),
        },
    }


class SchemaPropertyOrderTests(unittest.TestCase):
    def assert_required_property_order(self, response_format):
        schema = response_format["schema"]
        self.assertEqual(
            list(schema["properties"]),
            [
                "term_results",
                "source_context_translations",
                "aaa_optional",
                "zzz_optional",
            ])

        term_results = schema["properties"]["term_results"]
        self.assertEqual(
            list(term_results["properties"]),
            ["2", "10"])
        result = term_results["properties"]["2"]
        self.assertEqual(
            list(result["properties"]),
            [
                "contextual_sense",
                "additional_senses",
                "optional_note",
            ])
        contextual = result["properties"]["contextual_sense"]
        self.assertEqual(
            list(contextual["properties"]),
            [
                "Translation (English)",
                "Dictionary Meaning (English)",
            ])
        additional_item = result["properties"][
            "additional_senses"]["items"]
        self.assertEqual(
            list(additional_item["properties"]),
            [
                "Sentences",
                "Sentence Translations (English)",
                "optional_note",
            ])

        translations = schema["properties"][
            "source_context_translations"]
        self.assertEqual(
            list(translations["properties"]),
            ["ctx:z", "ctx:a"])
        any_of = schema["properties"]["aaa_optional"]["anyOf"][0]
        self.assertEqual(
            list(any_of["properties"]),
            ["zeta", "alpha", "optional"])
        array_item = schema["properties"]["zzz_optional"]["items"]
        self.assertEqual(
            list(array_item["properties"]),
            ["second", "first"])

    def test_sorted_persisted_v8_contract_restores_every_schema_order(self):
        contract = _version_eight_contract()
        persisted = json.loads(json.dumps(contract, sort_keys=True))
        self.assertEqual(
            list(
                persisted["response_formats_by_chunk"]["chunk-a"]
                ["schema"]["properties"]),
            [
                "aaa_optional",
                "source_context_translations",
                "term_results",
                "zzz_optional",
            ])

        normalised = normalise_source_request_contract(persisted)

        self.assert_required_property_order(
            normalised["response_format"])
        self.assert_required_property_order(
            normalised["response_formats_by_chunk"]["chunk-a"])
        self.assert_required_property_order(
            normalise_source_request_contract(normalised)[
                "response_formats_by_chunk"]["chunk-a"])

    def test_property_order_does_not_change_contract_digest(self):
        contract = _version_eight_contract()
        persisted = json.loads(json.dumps(contract, sort_keys=True))

        self.assertEqual(
            source_request_contract_digest(contract),
            source_request_contract_digest(persisted))
        self.assertEqual(
            source_request_contract_digest(contract),
            source_request_contract_digest(
                normalise_source_request_contract(persisted)))

    def test_legacy_contract_also_restores_required_property_order(self):
        legacy = {
            "schema_version": 1,
            "model": "legacy-model",
            "composed_prompt": "Legacy prompt.",
            "response_format": _strict_format("legacy_source_format"),
            "reasoning": {"effort": "none"},
            "max_output_tokens_by_chunk": {},
        }
        persisted = json.loads(json.dumps(legacy, sort_keys=True))

        normalised = normalise_source_request_contract(persisted)

        self.assert_required_property_order(
            normalised["response_format"])
        self.assertEqual(normalised["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
