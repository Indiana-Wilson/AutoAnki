"""Focused tests for v8 rank-grouped source-response validation."""

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
from source_generation import (
    ContextMode,
    SourceGenerationConfig,
    plan_source_generation,
)
from source_generation.validation import (
    inspect_pipeline_response,
    make_pipeline_response_validator,
)
from tests.test_source_generation import (
    context_classical_pipeline,
    detailed_context_classical_pipeline,
    make_source,
)
from tests.test_v8_schema_contract import detailed_context_pipeline


class GroupedSourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = context_classical_pipeline()
        source = make_source(section_texts=("道可道。",))
        self.chunk = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_stagger_ms=0),
        ).chunks[0]
        (
            self.term_field,
            self.sentences_field,
            self.sentence_translations_field,
            self.translation_field,
        ) = process_text.get_response_field_names(self.pipeline)

    def examples(self, term="道"):
        return {
            self.sentences_field: [
                f"<strong>{term}</strong>一。",
                f"<strong>{term}</strong>二。",
                f"<strong>{term}</strong>三。",
                f"<strong>{term}</strong>四。",
            ],
            self.sentence_translations_field: [
                "First.",
                "Second.",
                "Third.",
                "Fourth.",
            ],
        }

    def payload(self):
        first, second = self.chunk.words
        return {
            process_text.SOURCE_TERM_RESULTS_KEY: {
                str(first.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                        self.translation_field: "way",
                    },
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [{
                        **self.examples(first.surface),
                        self.translation_field: "road",
                    }],
                },
                str(second.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                        self.translation_field: "can",
                    },
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
                },
            },
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                self.chunk.contexts[0].context_id: (
                    "The way that can be spoken."),
            },
        }

    def inspect(self, payload):
        return inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_grouped_source_results=True,
        )

    def test_normalizes_rank_groups_and_injects_exact_terms_and_contexts(self):
        # Reverse the model-controlled object order. Canonical output must
        # still follow immutable source ranks, with each rank's additions
        # immediately after its contextual sense.
        payload = self.payload()
        payload[process_text.SOURCE_TERM_RESULTS_KEY] = dict(reversed(
            tuple(payload[process_text.SOURCE_TERM_RESULTS_KEY].items())))

        report = self.inspect(payload)

        self.assertTrue(report["valid"], report["problems"])
        cards = report["canonical_response"]["cards"]
        self.assertEqual(
            [card[self.term_field] for card in cards],
            ["道", "道", "可"],
        )
        self.assertEqual(
            cards[0][self.sentence_translations_field],
            "The way that can be spoken.",
        )
        self.assertEqual(
            cards[2][self.sentence_translations_field],
            "The way that can be spoken.",
        )
        self.assertIn("<strong>道</strong>可道。", cards[0][
            self.sentences_field])
        self.assertEqual(
            cards[0][self.sentences_field].count(
                "<strong>道</strong>"),
            1,
        )
        self.assertIn("道<strong>可</strong>道。", cards[2][
            self.sentences_field])
        self.assertEqual(
            cards[1][self.sentences_field].split("|"),
            self.examples()["Sentences"],
        )

        validator = make_pipeline_response_validator(
            self.pipeline,
            use_grouped_source_results=True,
        )
        self.assertEqual(
            validator(
                json.dumps(payload, ensure_ascii=False),
                self.chunk),
            report["canonical_response"],
        )

    def test_rejects_non_exact_root_rank_and_group_shapes(self):
        cases = {}

        extra_root = self.payload()
        extra_root["cards"] = []
        cases["root"] = (
            extra_root,
            "invalid_grouped_source_response_root",
        )

        non_object_results = self.payload()
        non_object_results[process_text.SOURCE_TERM_RESULTS_KEY] = []
        cases["term results"] = (
            non_object_results,
            "invalid_grouped_term_results",
        )

        wrong_ranks = self.payload()
        results = wrong_ranks[process_text.SOURCE_TERM_RESULTS_KEY]
        results["999"] = results.pop(str(self.chunk.words[1].rank))
        cases["ranks"] = (
            wrong_ranks,
            "invalid_grouped_source_ranks",
        )

        extra_group_field = self.payload()
        first_group = next(iter(
            extra_group_field[
                process_text.SOURCE_TERM_RESULTS_KEY].values()))
        first_group["term"] = "道"
        cases["group keys"] = (
            extra_group_field,
            "invalid_grouped_term_result_fields",
        )

        invalid_contextual = self.payload()
        first_group = next(iter(
            invalid_contextual[
                process_text.SOURCE_TERM_RESULTS_KEY].values()))
        first_group[process_text.SOURCE_CONTEXTUAL_SENSE_KEY] = []
        cases["contextual type"] = (
            invalid_contextual,
            "invalid_grouped_source_sense",
        )

        invalid_additions = self.payload()
        first_group = next(iter(
            invalid_additions[
                process_text.SOURCE_TERM_RESULTS_KEY].values()))
        first_group[process_text.SOURCE_ADDITIONAL_SENSES_KEY] = {}
        cases["additional type"] = (
            invalid_additions,
            "invalid_grouped_additional_senses",
        )

        repeated_term = self.payload()
        first_group = next(iter(
            repeated_term[
                process_text.SOURCE_TERM_RESULTS_KEY].values()))
        first_group[
            process_text.SOURCE_CONTEXTUAL_SENSE_KEY][
                self.term_field] = "model-supplied"
        cases["echoed term"] = (
            repeated_term,
            "grouped_source_sense_repeats_term",
        )

        for label, (payload, expected_code) in cases.items():
            with self.subTest(label=label):
                report = self.inspect(payload)
                self.assertFalse(report["valid"])
                self.assertIn(
                    expected_code,
                    {problem["code"] for problem in report["problems"]},
                )
                problem = next(
                    problem
                    for problem in report["problems"]
                    if problem["code"] == expected_code
                )
                self.assertFalse(problem["overrideable"])
                self.assertFalse(report["can_manually_accept"])

    def test_context_translation_keys_must_match_exactly(self):
        payload = self.payload()
        translations = payload[
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY]
        translations["unrequested-context"] = translations.pop(
            self.chunk.contexts[0].context_id)

        report = self.inspect(payload)

        self.assertIn(
            "invalid_grouped_source_context_ids",
            {problem["code"] for problem in report["problems"]},
        )
        self.assertFalse(report["can_manually_accept"])

    def test_unambiguous_contextual_constant_is_hard_validated(self):
        pipeline = detailed_context_pipeline()
        source = make_source(
            section_texts=("無名，天地之始，有名，萬物之母。",),
            source_language_key="classical_chinese_wang_bi")
        excluded = tuple(
            word.surface
            for word in source.build.unique_words
            if word.surface != "始")
        chunk = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_stagger_ms=0,
                excluded_words=excluded),
        ).chunks[0]
        word = chunk.words[0]
        fields = process_text.get_response_field_names(pipeline)
        (
            _term_field,
            _sentences_field,
            _sentence_translations_field,
            translation_field,
            definition_field,
            pronunciation_field,
            part_of_speech_field,
            register_field,
            nuance_field,
        ) = fields
        contextual = {
            translation_field: "beginning; origin",
            definition_field: "The starting point or origin of something.",
            pronunciation_field: "shǐ",
            part_of_speech_field: "verb",
            register_field: "Classical Chinese",
            nuance_field: (
                "Here it names the origin of heaven and earth, not the "
                "verbal act of beginning."),
        }
        payload = {
            process_text.SOURCE_TERM_RESULTS_KEY: {
                str(word.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: contextual,
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
                },
            },
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                chunk.contexts[0].context_id: (
                    "Nameless, it is the beginning of heaven and earth; "
                    "named, it is the mother of the myriad things."),
            },
        }

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            pipeline,
            chunk,
            use_grouped_source_results=True,
        )

        problem = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "contextual_source_field_mismatch")
        self.assertFalse(problem["overrideable"])
        self.assertEqual(problem["expected"], "noun")
        self.assertEqual(problem["actual"], "verb")
        self.assertEqual(
            problem["path"],
            (
                f'$.term_results["{word.rank}"].contextual_sense'
                f'[{json.dumps(part_of_speech_field)}]'
            ),
        )
        self.assertFalse(report["can_manually_accept"])

    def test_root_translation_content_is_hard_validated_once_per_context(self):
        cases = {
            "blank": ("", "blank_source_context_translation"),
            "pipe": (
                "First fragment.|Second fragment.",
                "sentence_translation_count_mismatch",
            ),
            "escaped pipe": (
                "First fragment. &#124; Second fragment.",
                "sentence_translation_count_mismatch",
            ),
            "html": (
                "The <strong>marked</strong> passage.",
                "source_context_translation_contains_html",
            ),
            "source copy": (
                self.chunk.contexts[0].text,
                "source_context_translation_not_english",
            ),
            "source script": (
                "此道不可言。",
                "source_context_translation_not_english",
            ),
        }
        context_id = self.chunk.contexts[0].context_id

        for label, (translation, expected_code) in cases.items():
            with self.subTest(label=label):
                payload = self.payload()
                payload[
                    process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY][
                        context_id] = translation
                report = self.inspect(payload)
                matching = [
                    problem
                    for problem in report["problems"]
                    if problem["code"] == expected_code
                ]
                # Both requested words share this context, but the root value
                # is audited once rather than producing one error per card.
                self.assertEqual(len(matching), 1)
                self.assertFalse(matching[0]["overrideable"])
                self.assertFalse(report["can_manually_accept"])

    def test_malformed_sentence_array_is_reported_without_crashing(self):
        payload = self.payload()
        first_group = next(iter(
            payload[process_text.SOURCE_TERM_RESULTS_KEY].values()))
        first_group[
            process_text.SOURCE_ADDITIONAL_SENSES_KEY][0][
                self.sentences_field][2] = {"not": "text"}

        report = self.inspect(payload)

        self.assertIn(
            "sentence_collection_item_not_text",
            {problem["code"] for problem in report["problems"]},
        )
        self.assertFalse(report["structurally_valid"])

    def test_group_indices_preserve_duplicate_sense_and_example_checks(self):
        pipeline = detailed_context_classical_pipeline()
        (
            _term_field,
            sentences_field,
            sentence_translations_field,
            translation_field,
            definition_field,
        ) = process_text.get_response_field_names(pipeline)
        first, second = self.chunk.words
        examples = self.examples(first.surface)
        examples = {
            sentences_field: examples[self.sentences_field],
            sentence_translations_field: examples[
                self.sentence_translations_field],
        }
        payload = {
            process_text.SOURCE_TERM_RESULTS_KEY: {
                str(first.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                        translation_field: "say",
                        definition_field: "to utter words",
                    },
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [
                        {
                            **deepcopy(examples),
                            translation_field: "say",
                            definition_field: "to utter words",
                        },
                        {
                            **deepcopy(examples),
                            translation_field: "road",
                            definition_field: "a path for travel",
                        },
                    ],
                },
                str(second.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                        translation_field: "can",
                        definition_field: "to be able to",
                    },
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
                },
            },
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                self.chunk.contexts[0].context_id: (
                    "The way that can be spoken."),
            },
        }

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            pipeline,
            self.chunk,
            use_grouped_source_results=True,
        )
        codes = {problem["code"] for problem in report["problems"]}

        self.assertIn("duplicate_source_sense", codes)
        self.assertIn("duplicate_additional_sense_examples", codes)
        self.assertFalse(report["can_manually_accept"])


if __name__ == "__main__":
    unittest.main()
