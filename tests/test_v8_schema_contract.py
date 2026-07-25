import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation import (
    build_source_request_contract,
    source_request_contract_digest,
    source_request_uses_grouped_source_results,
    source_request_uses_occurrence_locators,
    source_request_uses_split_contextual_cards,
)
from source_generation.requests import (
    normalise_source_request_contract,
    source_request_uses_obsolete_context_translation_protocol,
)


def context_pipeline():
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "classical_chinese_wang_bi")
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key == "context")
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=(
            pipeline_store.FieldSetting("translation", "english"),
            pipeline_store.FieldSetting(
                "dictionary_meaning",
                "english"),
        ))
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese_wang_bi")


def detailed_context_pipeline():
    pipeline = context_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "classical_chinese_wang_bi")
    settings = replace(
        settings,
        shared_fields=tuple(
            pipeline_store.FieldSetting(field_key, "english")
            for field_key in (
                "translation",
                "dictionary_meaning",
                "pronunciation",
                "part_of_speech",
                "register",
                "nuance",
            )))
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese_wang_bi")


def grouped_chunk():
    return SimpleNamespace(
        chunk_id="000001-r2-r9",
        words=(
            SimpleNamespace(
                rank=2,
                surface="道",
                context_id="ctx:opening"),
            SimpleNamespace(
                rank=9,
                surface="之",
                context_id="ctx:shared"),
        ),
        contexts=(
            SimpleNamespace(context_id="ctx:opening"),
            SimpleNamespace(context_id="ctx:shared"),
        ))


def wang_bi_beginning_chunk():
    return SimpleNamespace(
        chunk_id="000001-r10-r10",
        words=(
            SimpleNamespace(
                rank=10,
                surface="始",
                context_id="ctx:beginning",
                start_offset=24,
                end_offset=25),
        ),
        contexts=(
            SimpleNamespace(
                context_id="ctx:beginning",
                start_offset=18,
                text="無名，天地之始，有名，萬物之母。"),
        ))


class GroupedSourceResponseSchemaTests(unittest.TestCase):
    def test_schema_keys_every_rank_and_context_exactly(self):
        response_format = process_text.build_grouped_source_response_format(
            context_pipeline(),
            grouped_chunk())
        schema = response_format["schema"]

        self.assertEqual(
            schema["required"],
            ["term_results", "source_context_translations"])
        self.assertFalse(schema["additionalProperties"])
        term_results = schema["properties"]["term_results"]
        self.assertEqual(term_results["required"], ["2", "9"])
        self.assertEqual(set(term_results["properties"]), {"2", "9"})
        self.assertFalse(term_results["additionalProperties"])

        result = term_results["properties"]["2"]
        self.assertEqual(
            result["required"],
            ["contextual_sense", "additional_senses"])
        contextual = result["properties"]["contextual_sense"]
        self.assertEqual(
            contextual["required"],
            [
                "Translation (English)",
                "Dictionary Meaning (English)",
            ])
        self.assertNotIn("Classical Chinese", contextual["properties"])
        self.assertNotIn("Sentences", contextual["properties"])
        self.assertNotIn(
            "Sentence Translations (English)",
            contextual["properties"])
        self.assertIn(
            "exact complete requested term '道'",
            contextual["properties"][
                "Translation (English)"]["description"])
        self.assertFalse(contextual["additionalProperties"])

        additional = result["properties"]["additional_senses"]["items"]
        self.assertIn(
            "exact complete requested term '道'",
            additional["description"])
        self.assertNotIn("Classical Chinese", additional["properties"])
        self.assertEqual(
            additional["properties"]["Sentences"]["minItems"],
            4)
        self.assertEqual(
            additional["properties"]["Sentences"]["maxItems"],
            4)
        self.assertEqual(
            additional["properties"][
                "Sentence Translations (English)"]["minItems"],
            4)
        self.assertEqual(
            additional["properties"][
                "Sentence Translations (English)"]["maxItems"],
            4)
        self.assertIn(
            "never copy source-language text",
            additional["properties"][
                "Sentence Translations (English)"]["items"]["description"])
        self.assertIn(
            "exact complete requested term '道' exactly once",
            additional["properties"]["Sentences"]["items"]["description"])
        self.assertIn(
            "including inside a compound",
            additional["properties"]["Sentences"]["items"]["description"])
        self.assertEqual(
            additional["properties"]["Sentences"]["items"]["pattern"],
            "^[^道]*<strong>道</strong>[^道]*$")
        self.assertEqual(
            set(additional["required"]),
            set(additional["properties"]))
        self.assertFalse(additional["additionalProperties"])

        translations = schema["properties"][
            "source_context_translations"]
        self.assertEqual(
            translations["required"],
            ["ctx:opening", "ctx:shared"])
        self.assertEqual(
            set(translations["properties"]),
            {"ctx:opening", "ctx:shared"})
        self.assertTrue(all(
            value["type"] == "string"
            and "complete natural English translation" in value["description"]
            for value in translations["properties"].values()))
        self.assertFalse(translations["additionalProperties"])
        self.assertNotIn("items", translations)
        self.assertTrue(response_format["strict"])
        self.assertLessEqual(len(response_format["name"]), 64)
        self.assertNotIn('"term"', json.dumps(schema))

    def test_schema_rejects_duplicate_ranks_and_unknown_contexts(self):
        chunk = grouped_chunk()
        with self.assertRaisesRegex(ValueError, "rank 2 appears"):
            process_text.build_grouped_source_response_format(
                context_pipeline(),
                replace_namespace(
                    chunk,
                    words=(
                        chunk.words[0],
                        SimpleNamespace(
                            rank=2,
                            context_id="ctx:shared"),
                    )))

        with self.assertRaisesRegex(ValueError, "reference a context"):
            process_text.build_grouped_source_response_format(
                context_pipeline(),
                replace_namespace(
                    chunk,
                    words=(
                        SimpleNamespace(
                            rank=2,
                            surface="道",
                            context_id="ctx:missing"),
                    )))

    def test_schema_freezes_unambiguous_wang_bi_beginning_fields(self):
        response_format = process_text.build_grouped_source_response_format(
            detailed_context_pipeline(),
            wang_bi_beginning_chunk())

        contextual = response_format["schema"]["properties"][
            "term_results"]["properties"]["10"]["properties"][
                "contextual_sense"]["properties"]

        self.assertEqual(
            contextual["Translation (English)"]["enum"],
            ["beginning; origin"])
        self.assertEqual(
            contextual["Dictionary Meaning (English)"]["enum"],
            ["The starting point or origin of something."])
        self.assertEqual(
            contextual["Pronunciation (English)"]["enum"],
            ["shǐ"])
        self.assertEqual(
            contextual["Part of Speech (English)"]["enum"],
            ["noun"])
        self.assertIn(
            "origin of heaven and earth",
            contextual["Nuance (English)"]["enum"][0])


class VersionEightSourceContractTests(unittest.TestCase):
    def test_context_contract_freezes_one_grouped_schema_per_chunk(self):
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")

        self.assertEqual(contract["schema_version"], 8)
        self.assertIn("response_format", contract)
        self.assertEqual(
            set(contract["response_formats_by_chunk"]),
            {chunk.chunk_id})
        self.assertEqual(
            contract["response_formats_by_chunk"][chunk.chunk_id],
            process_text.build_grouped_source_response_format(
                context_pipeline(),
                chunk))
        self.assertGreaterEqual(
            contract["max_output_tokens_by_chunk"][chunk.chunk_id],
            8_192)
        self.assertTrue(
            source_request_uses_grouped_source_results(contract))
        self.assertFalse(
            source_request_uses_split_contextual_cards(contract))
        self.assertTrue(
            source_request_uses_occurrence_locators(contract))
        self.assertFalse(
            source_request_uses_obsolete_context_translation_protocol(
                contract))
        altered_contract = json.loads(json.dumps(contract))
        altered_contract["response_formats_by_chunk"][
            chunk.chunk_id]["name"] = "different_grouped_source_schema"
        self.assertNotEqual(
            source_request_contract_digest(contract),
            source_request_contract_digest(altered_contract))

        prompt = contract["composed_prompt"]
        self.assertIn(
            '"term_results" and "source_context_translations"',
            prompt)
        self.assertIn(
            '"contextual_sense"',
            prompt)
        self.assertIn(
            '"additional_senses"',
            prompt)
        self.assertIn(
            "input rank 7 becomes property",
            prompt)
        self.assertIn(
            "This source-specific rule overrides any general instruction",
            " ".join(prompt.split()))
        self.assertIn(
            "Perform an explicit coverage audit for every rank",
            prompt)
        self.assertIn(
            "practical method/course/teaching",
            " ".join(prompt.split()))
        self.assertIn(
            "FINAL SENSE-COVERAGE CHECK—MANDATORY",
            prompt)
        self.assertIn(
            "omitting the common road/path or practical method/course sense",
            " ".join(prompt.split()))
        self.assertIn(
            "Never copy, paraphrase, or broaden",
            " ".join(prompt.split()))
        self.assertIn(
            "FINAL TRANSLATION-LANGUAGE CHECK—MANDATORY",
            prompt)
        self.assertIn(
            "非 + 常道 or 非 + 常名",
            prompt)
        self.assertIn(
            'negative copula or negative predicate "is not"',
            " ".join(prompt.split()))
        self.assertIn(
            "Never define, pronounce, highlight, or exemplify only one "
            "character",
            " ".join(prompt.split()))
        self.assertIn(
            "Never emit <em>",
            prompt)
        self.assertIn(
            "Returning \"constant/enduring\", cháng, or adjective",
            " ".join(prompt.split()))
        self.assertIn(
            'Return an empty "additional_senses" array for 天地',
            " ".join(prompt.split()))
        self.assertIn(
            "Do not return an empty array for 可",
            " ".join(prompt.split()))
        self.assertIn(
            "after removing the one <strong>...</strong> span",
            " ".join(prompt.split()))
        self.assertIn(
            "This exact-once v8 rule overrides the general",
            " ".join(prompt.split()))
        self.assertIn(
            "GROUPED V8 TARGET RULE—THIS REPLACES THE GENERAL "
            "REPEATED-USAGE RULE",
            prompt)
        self.assertNotIn(
            "leave that other occurrence completely untagged",
            " ".join(prompt.split()))
        self.assertIn(
            "a 道 card must not use 大道",
            " ".join(prompt.split()))
        self.assertIn(
            'A construction such as 未有名 ("had no name")',
            " ".join(prompt.split()))
        self.assertIn(
            '以有觀<strong>無</strong> means "use being to contemplate '
            'nonbeing"',
            " ".join(prompt.split()))
        self.assertIn(
            "use 遂<strong>之</strong>郊, not the unidiomatic",
            " ".join(prompt.split()))
        self.assertIn(
            "do not use the compound 是非 anywhere",
            " ".join(prompt.split()))
        self.assertIn(
            "do not preface it with a clause such as 既告之",
            " ".join(prompt.split()))
        self.assertIn(
            '"Part of Speech (English)" MUST be "negative existential verb '
            '/ negator"',
            " ".join(prompt.split()))
        self.assertIn(
            "Avoid ambiguous 可行, 可守, 可取, 可任, and 可嘉",
            " ".join(prompt.split()))
        self.assertIn(
            "never 王<strong>之</strong>於東",
            " ".join(prompt.split()))
        self.assertIn(
            "never 春<strong>始</strong>生",
            " ".join(prompt.split()))
        self.assertIn(
            "never the subjectless and ambiguous "
            "出於此<strong>道</strong>",
            " ".join(prompt.split()))
        self.assertIn(
            "never the incoherent 告而聽<strong>之</strong>",
            " ".join(prompt.split()))
        self.assertIn(
            "never 吾欲<strong>之</strong>遠",
            " ".join(prompt.split()))
        self.assertIn(
            'For 始 in 天地之始, "contextual_sense" MUST be the noun',
            " ".join(prompt.split()))
        self.assertIn(
            "never 子欲<strong>之</strong>何方",
            " ".join(prompt.split()))
        self.assertIn(
            'never "His fame was widely heard"',
            " ".join(prompt.split()))
        self.assertTrue(
            prompt.endswith("Here is the source batch JSON:\n"))
        self.assertLess(
            prompt.index("TERM RESULTS"),
            prompt.index("The supplied source batch is a JSON object."))

    def test_non_context_contract_keeps_generic_format_and_empty_map(self):
        chunk = grouped_chunk()

        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(chunk,),
            use_source_for_example_sentences=False,
            protocol_version=8,
            reasoning_effort="low")

        self.assertEqual(contract["schema_version"], 8)
        self.assertEqual(contract["response_formats_by_chunk"], {})
        self.assertIn(
            "cards",
            contract["response_format"]["schema"]["properties"])
        self.assertFalse(
            source_request_uses_grouped_source_results(contract))
        self.assertGreaterEqual(
            contract["max_output_tokens_by_chunk"][chunk.chunk_id],
            8_192)

    def test_normalizer_requires_exact_grouped_chunk_format_keys(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        contract["response_formats_by_chunk"] = {}

        with self.assertRaisesRegex(ValueError, "exactly match"):
            normalise_source_request_contract(contract)

    def test_normalizer_bounds_nested_names_and_preserves_v7_shape(self):
        contract = build_source_request_contract(
            context_pipeline(),
            chunks=(grouped_chunk(),),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        chunk_format = next(iter(
            contract["response_formats_by_chunk"].values()))
        chunk_format["name"] = "grouped_" + ("x" * 100)

        normalised = normalise_source_request_contract(contract)

        self.assertLessEqual(
            len(next(iter(
                normalised[
                    "response_formats_by_chunk"].values()))["name"]),
            64)

        legacy = dict(normalised)
        legacy["schema_version"] = 7
        del legacy["response_formats_by_chunk"]
        legacy = normalise_source_request_contract(legacy)
        self.assertFalse(
            source_request_uses_grouped_source_results(legacy))
        self.assertTrue(
            source_request_uses_split_contextual_cards(legacy))
        self.assertTrue(
            source_request_uses_occurrence_locators(legacy))
        self.assertTrue(
            source_request_uses_obsolete_context_translation_protocol(
                legacy))


def replace_namespace(namespace, **changes):
    values = dict(vars(namespace))
    values.update(changes)
    return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
