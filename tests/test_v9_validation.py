"""Focused tests for compact v9 source-response validation."""

from copy import deepcopy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


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
    make_source,
)
from tests.test_v8_schema_contract import detailed_context_pipeline


class CompactSourceValidationTests(unittest.TestCase):
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
            ],
            self.sentence_translations_field: [
                "First.",
                "Second.",
                "Third.",
            ],
        }

    def payload(self):
        first, second = self.chunk.words
        return {
            process_text.SOURCE_TERM_RESULTS_KEY: [{
                process_text.SOURCE_RANK_FIELD_NAME: first.rank,
                process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                    self.translation_field: "way",
                },
                process_text.SOURCE_ADDITIONAL_SENSES_KEY: [{
                    **self.examples(first.surface),
                    self.translation_field: "road",
                }],
            }, {
                process_text.SOURCE_RANK_FIELD_NAME: second.rank,
                process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                    self.translation_field: "can",
                },
                process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
            }],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [{
                process_text.SOURCE_CONTEXT_ID_FIELD_NAME: (
                    self.chunk.contexts[0].context_id),
                process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: (
                    "The way that can be spoken."),
            }],
        }

    def inspect(self, payload):
        return inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
        )

    @staticmethod
    def problem(report, code):
        return next(
            problem
            for problem in report["problems"]
            if problem["code"] == code
        )

    def test_normalizes_arrays_and_injects_terms_from_immutable_ranks(self):
        payload = self.payload()

        report = self.inspect(payload)

        self.assertTrue(report["valid"], report["problems"])
        self.assertFalse(report["full_retry_required"])
        cards = report["canonical_response"]["cards"]
        self.assertEqual(
            [card[self.term_field] for card in cards],
            ["道", "道", "可"],
        )
        self.assertNotIn(self.sentences_field, cards[0])
        self.assertNotIn(self.sentence_translations_field, cards[0])
        self.assertEqual(
            cards[1][self.sentences_field].split("|"),
            self.examples()["Sentences"],
        )
        self.assertEqual(
            report["canonical_response"]["source_contexts"][0][
                "english_translation"],
            "The way that can be spoken.",
        )

        validator = make_pipeline_response_validator(
            self.pipeline,
            use_compact_source_results=True,
        )
        self.assertEqual(
            validator(
                json.dumps(payload, ensure_ascii=False),
                self.chunk),
            report["canonical_response"],
        )

    def test_root_rank_and_item_integrity(self):
        cases = {}

        extra_root = self.payload()
        extra_root["cards"] = []
        cases["root fields"] = (
            extra_root,
            "invalid_compact_source_response_root",
            True,
        )

        non_array = self.payload()
        non_array[process_text.SOURCE_TERM_RESULTS_KEY] = {}
        cases["term-results type"] = (
            non_array,
            "invalid_compact_term_results",
            True,
        )

        scalar_item = self.payload()
        scalar_item[process_text.SOURCE_TERM_RESULTS_KEY][0] = []
        cases["item type"] = (
            scalar_item,
            "invalid_compact_term_result",
            True,
        )

        extra_item_field = self.payload()
        extra_item_field[process_text.SOURCE_TERM_RESULTS_KEY][0][
            "term"] = "model supplied"
        cases["item fields"] = (
            extra_item_field,
            "invalid_compact_term_result_fields",
            False,
        )

        bool_rank = self.payload()
        bool_rank[process_text.SOURCE_TERM_RESULTS_KEY][0][
            process_text.SOURCE_RANK_FIELD_NAME] = True
        cases["boolean rank"] = (
            bool_rank,
            "invalid_compact_source_rank",
            True,
        )

        zero_rank = self.payload()
        zero_rank[process_text.SOURCE_TERM_RESULTS_KEY][0][
            process_text.SOURCE_RANK_FIELD_NAME] = 0
        cases["zero rank"] = (
            zero_rank,
            "invalid_compact_source_rank",
            True,
        )

        unexpected_rank = self.payload()
        unexpected_rank[process_text.SOURCE_TERM_RESULTS_KEY][0][
            process_text.SOURCE_RANK_FIELD_NAME] = 999
        cases["unexpected rank"] = (
            unexpected_rank,
            "unexpected_compact_source_rank",
            True,
        )

        descending = self.payload()
        descending[process_text.SOURCE_TERM_RESULTS_KEY].reverse()
        cases["rank order"] = (
            descending,
            "compact_source_ranks_not_ascending",
            True,
        )

        for label, (payload, expected_code, full_retry) in cases.items():
            with self.subTest(label=label):
                report = self.inspect(payload)
                problem = self.problem(report, expected_code)
                self.assertEqual(
                    problem.get("full_retry_required", False),
                    full_retry,
                )
                self.assertFalse(problem["overrideable"])

    def test_rank_duplicates_and_omissions_have_selective_targets(self):
        payload = self.payload()
        first_rank = self.chunk.words[0].rank
        second_rank = self.chunk.words[1].rank
        payload[process_text.SOURCE_TERM_RESULTS_KEY][1][
            process_text.SOURCE_RANK_FIELD_NAME] = first_rank

        report = self.inspect(payload)

        duplicate = self.problem(
            report,
            "duplicate_compact_source_rank")
        self.assertEqual(duplicate["source_rank"], first_rank)
        self.assertEqual(
            duplicate["repair_target"],
            {
                "kind": "source_rank",
                "source_rank": first_rank,
            },
        )
        missing = self.problem(report, "missing_compact_source_rank")
        self.assertEqual(missing["source_rank"], second_rank)
        self.assertEqual(
            missing["repair_target"]["kind"],
            "source_rank",
        )

    def test_context_items_require_exact_unique_requested_ids(self):
        context_id = self.chunk.contexts[0].context_id

        wrong_fields = self.payload()
        wrong_fields[
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY][0][
                "extra"] = "no"
        report = self.inspect(wrong_fields)
        problem = self.problem(
            report,
            "invalid_compact_source_context_translation_fields")
        self.assertEqual(problem["context_id"], context_id)
        self.assertEqual(
            problem["repair_target"],
            {
                "kind": "context",
                "context_id": context_id,
            },
        )

        duplicate = self.payload()
        duplicate[
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY].append(
                deepcopy(duplicate[
                    process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY][0]))
        report = self.inspect(duplicate)
        problem = self.problem(
            report,
            "duplicate_compact_source_context_translation")
        self.assertEqual(problem["context_id"], context_id)
        self.assertEqual(problem["repair_target"]["kind"], "context")

        unexpected = self.payload()
        unexpected[
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY][0][
                process_text.SOURCE_CONTEXT_ID_FIELD_NAME] = "not-requested"
        report = self.inspect(unexpected)
        problem = self.problem(
            report,
            "unexpected_compact_source_context_id")
        self.assertTrue(problem["full_retry_required"])
        self.assertNotIn("repair_target", problem)

        missing = self.payload()
        missing[process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY] = []
        report = self.inspect(missing)
        problem = self.problem(
            report,
            "missing_source_context_translation")
        self.assertEqual(problem["context_id"], context_id)
        self.assertEqual(problem["repair_target"]["kind"], "context")

    def test_context_translation_semantics_are_scoped_and_use_raw_paths(self):
        context_id = self.chunk.contexts[0].context_id
        cases = {
            "blank": ("", "blank_source_context_translation"),
            "pipe": (
                "First fragment.|Second fragment.",
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
        }

        for label, (translation, expected_code) in cases.items():
            with self.subTest(label=label):
                payload = self.payload()
                payload[
                    process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY][0][
                        process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME] = (
                            translation)
                report = self.inspect(payload)
                problem = self.problem(report, expected_code)
                self.assertEqual(problem["context_id"], context_id)
                self.assertEqual(problem["repair_target"]["kind"], "context")
                self.assertEqual(
                    problem["path"],
                    "$.source_context_translations[0].translation",
                )

    def test_generic_and_semantic_card_problems_inherit_rank_and_raw_path(self):
        payload = self.payload()
        first_rank = self.chunk.words[0].rank
        contextual = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][0][
                process_text.SOURCE_CONTEXTUAL_SENSE_KEY]
        contextual[self.translation_field] = 42

        report = self.inspect(payload)

        problem = self.problem(report, "field_not_string")
        self.assertEqual(problem["source_rank"], first_rank)
        self.assertEqual(problem["repair_target"]["kind"], "source_rank")
        self.assertEqual(
            problem["path"],
            (
                "$.term_results[0].contextual_sense"
                f"[{json.dumps(self.translation_field)}]"
            ),
        )

        payload = self.payload()
        additional = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][0][
                process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        additional[self.sentences_field][2] += "|invalid"
        report = self.inspect(payload)
        problem = self.problem(
            report,
            "sentence_collection_item_contains_delimiter")
        self.assertEqual(problem["source_rank"], first_rank)
        self.assertEqual(
            problem["path"],
            (
                "$.term_results[0].additional_senses[0]"
                f"[{json.dumps(self.sentences_field)}][2]"
            ),
        )

    def test_contextual_constants_and_exact_examples_reuse_v8_semantics(self):
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
            sentences_field,
            sentence_translations_field,
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
        additional = dict(contextual)
        additional.update({
            translation_field: "to begin",
            definition_field: "To cause an action or process to start.",
            part_of_speech_field: "verb",
            sentences_field: [
                "始而<strong>始</strong>。",
                "<strong>始</strong>二。",
                "<strong>始</strong>三。",
            ],
            sentence_translations_field: [
                "First.",
                "Second.",
                "Third.",
            ],
        })
        payload = {
            process_text.SOURCE_TERM_RESULTS_KEY: [{
                process_text.SOURCE_RANK_FIELD_NAME: word.rank,
                process_text.SOURCE_CONTEXTUAL_SENSE_KEY: contextual,
                process_text.SOURCE_ADDITIONAL_SENSES_KEY: [additional],
            }],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [{
                process_text.SOURCE_CONTEXT_ID_FIELD_NAME: (
                    chunk.contexts[0].context_id),
                process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: (
                    "Nameless, it is the beginning of heaven and earth; "
                    "named, it is the mother of the myriad things."),
            }],
        }

        report = inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            pipeline,
            chunk,
            use_compact_source_results=True,
        )

        constant = self.problem(
            report,
            "contextual_source_field_mismatch")
        self.assertEqual(constant["source_rank"], word.rank)
        self.assertEqual(constant["expected"], "noun")
        self.assertEqual(
            constant["path"],
            (
                "$.term_results[0].contextual_sense"
                f"[{json.dumps(part_of_speech_field)}]"
            ),
        )
    def test_audited_wang_bi_readings_are_immutable_in_v9(self):
        pipeline = detailed_context_pipeline()
        cases = (
            (
                "見",
                (
                    "不尚賢，使民不爭；不貴難得之貨，使民不為盜；"
                    "不見可欲，使民心不亂。"
                ),
                {
                    "Translation (English)": "to display; to show",
                    "Pronunciation (English)": "xiàn",
                    "Part of Speech (English)": "verb",
                },
            ),
            (
                "夫",
                "夫唯弗居，是以不去。",
                {
                    "Translation (English)": "now; indeed; as for",
                    "Pronunciation (English)": "fú",
                    "Part of Speech (English)": "discourse particle",
                },
            ),
            (
                "智者",
                "使夫智者不敢為也。",
                {
                    "Translation (English)": "wise or clever people",
                    "Pronunciation (English)": "zhì zhě",
                    "Part of Speech (English)": "noun phrase",
                },
            ),
        )

        for index, (term, text, expected) in enumerate(cases):
            context_id = f"context-{index}"
            constants = process_text.compact_contextual_field_constants(
                pipeline,
                SimpleNamespace(
                    surface=term,
                    context_id=context_id),
                {
                    context_id: SimpleNamespace(text=text),
                })
            self.assertEqual(constants, expected)

    def test_grouped_v8_report_shape_is_unchanged(self):
        compact = self.payload()
        grouped = {
            process_text.SOURCE_TERM_RESULTS_KEY: {
                str(item[process_text.SOURCE_RANK_FIELD_NAME]): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: item[
                        process_text.SOURCE_CONTEXTUAL_SENSE_KEY],
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: item[
                        process_text.SOURCE_ADDITIONAL_SENSES_KEY],
                }
                for item in compact[process_text.SOURCE_TERM_RESULTS_KEY]
            },
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                item[process_text.SOURCE_CONTEXT_ID_FIELD_NAME]: item[
                    process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME]
                for item in compact[
                    process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY]
            },
        }
        grouped[
            process_text.SOURCE_TERM_RESULTS_KEY][
                str(self.chunk.words[0].rank)][
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY][
                        self.translation_field] = 42

        report = inspect_pipeline_response(
            json.dumps(grouped, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_grouped_source_results=True,
        )

        problem = self.problem(report, "field_not_string")
        self.assertTrue(problem["path"].startswith('$.term_results["'))
        self.assertNotIn("source_rank", problem)
        self.assertNotIn("repair_target", problem)
        self.assertNotIn("full_retry_required", report)


if __name__ == "__main__":
    unittest.main()
