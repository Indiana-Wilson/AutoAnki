"""Regression tests for raw JSON paths in grouped v8 validation reports."""

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
from source_generation.validation import inspect_pipeline_response
from tests.test_source_generation import (
    context_classical_pipeline,
    detailed_context_classical_pipeline,
    make_source,
)


class GroupedSourceRawPathTests(unittest.TestCase):
    def setUp(self):
        source = make_source(section_texts=("道可道。",))
        self.chunk = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_stagger_ms=0),
        ).chunks[0]

    @staticmethod
    def examples(
            term,
            sentences_field,
            sentence_translations_field):
        return {
            sentences_field: [
                f"<strong>{term}</strong>一。",
                f"<strong>{term}</strong>二。",
                f"<strong>{term}</strong>三。",
                f"<strong>{term}</strong>四。",
            ],
            sentence_translations_field: [
                "First.",
                "Second.",
                "Third.",
                "Fourth.",
            ],
        }

    def payload(self, pipeline):
        fields = process_text.get_response_field_names(pipeline)
        term_field = fields[0]
        sentences_field = fields[1]
        sentence_translations_field = fields[2]
        lexical_fields = fields[3:]
        first, second = self.chunk.words

        def lexical(prefix):
            return {
                field: f"{prefix} {index}"
                for index, field in enumerate(
                    lexical_fields,
                    start=1)
            }

        return {
            process_text.SOURCE_TERM_RESULTS_KEY: {
                str(first.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: (
                        lexical("contextual first")),
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [{
                        **self.examples(
                            first.surface,
                            sentences_field,
                            sentence_translations_field),
                        **lexical("additional first"),
                    }],
                },
                str(second.rank): {
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: (
                        lexical("contextual second")),
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
                },
            },
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: {
                self.chunk.contexts[0].context_id: (
                    "The way that can be spoken."),
            },
        }, {
            "term": term_field,
            "sentences": sentences_field,
            "sentence_translations": sentence_translations_field,
            "lexical": lexical_fields,
        }

    def inspect(self, payload, pipeline):
        return inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            pipeline,
            self.chunk,
            use_grouped_source_results=True,
        )

    @staticmethod
    def problem(report, code):
        return next(
            problem
            for problem in report["problems"]
            if problem["code"] == code
        )

    def test_generic_card_and_field_paths_point_into_raw_rank_group(self):
        pipeline = context_classical_pipeline()
        payload, fields = self.payload(pipeline)
        first_rank = str(self.chunk.words[0].rank)
        contextual = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank][
                process_text.SOURCE_CONTEXTUAL_SENSE_KEY]
        contextual[fields["lexical"][0]] = 42

        report = self.inspect(payload, pipeline)

        problem = self.problem(report, "field_not_string")
        self.assertEqual(problem["card_index"], 0)
        self.assertEqual(
            problem["path"],
            (
                f'$.term_results["{first_rank}"].contextual_sense'
                f'[{json.dumps(fields["lexical"][0], ensure_ascii=False)}]'
            ),
        )

        # A malformed additional object makes canonical validation return
        # early; its card-level path still needs remapping.
        payload, fields = self.payload(pipeline)
        additional = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank][
                process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        additional.pop(fields["sentence_translations"])
        report = self.inspect(payload, pipeline)

        problem = self.problem(report, "invalid_card_fields")
        self.assertEqual(problem["card_index"], 1)
        self.assertEqual(
            problem["path"],
            f'$.term_results["{first_rank}"].additional_senses[0]',
        )

    def test_array_item_preflight_path_points_into_raw_additional_sense(self):
        pipeline = context_classical_pipeline()
        payload, fields = self.payload(pipeline)
        first_rank = str(self.chunk.words[0].rank)
        additional = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank][
                process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        additional[fields["sentences"]][2] += "|invalid"

        report = self.inspect(payload, pipeline)

        problem = self.problem(
            report,
            "sentence_collection_item_contains_delimiter")
        self.assertEqual(problem["card_index"], 1)
        self.assertEqual(
            problem["path"],
            (
                f'$.term_results["{first_rank}"].additional_senses[0]'
                f'[{json.dumps(fields["sentences"])}][2]'
            ),
        )

    def test_later_duplicate_checks_use_raw_additional_sense_paths(self):
        pipeline = detailed_context_classical_pipeline()
        payload, fields = self.payload(pipeline)
        first_rank = str(self.chunk.words[0].rank)
        group = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank]
        contextual = group[process_text.SOURCE_CONTEXTUAL_SENSE_KEY]
        first_additional = group[
            process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        translation_field, definition_field = fields["lexical"]

        contextual.update({
            translation_field: "say",
            definition_field: "to utter words",
        })
        first_additional.update({
            translation_field: "say",
            definition_field: "to utter words",
        })
        second_additional = dict(first_additional)
        second_additional[translation_field] = "road"
        second_additional[definition_field] = "a path for travel"
        group[process_text.SOURCE_ADDITIONAL_SENSES_KEY].append(
            second_additional)

        report = self.inspect(payload, pipeline)

        duplicate_sense = self.problem(report, "duplicate_source_sense")
        self.assertEqual(duplicate_sense["card_index"], 1)
        self.assertEqual(
            duplicate_sense["path"],
            f'$.term_results["{first_rank}"].additional_senses[0]',
        )
        duplicate_examples = self.problem(
            report,
            "duplicate_additional_sense_examples")
        self.assertEqual(duplicate_examples["card_index"], 2)
        self.assertEqual(
            duplicate_examples["path"],
            (
                f'$.term_results["{first_rank}"].additional_senses[1]'
                f'[{json.dumps(fields["sentences"])}]'
            ),
        )
        self.assertFalse(any(
            problem["path"].startswith("$.cards[")
            for problem in report["problems"]
        ))

    def test_unemphasized_term_occurrence_reports_exact_raw_array_item(self):
        pipeline = context_classical_pipeline()
        payload, fields = self.payload(pipeline)
        first_rank = str(self.chunk.words[0].rank)
        additional = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank][
                process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        additional[fields["sentences"]][0] = (
            "是道可辨，未必皆<strong>道</strong>。")

        report = self.inspect(payload, pipeline)

        problem = self.problem(
            report,
            "unemphasized_source_term_occurrence")
        self.assertTrue(problem["overrideable"])
        self.assertEqual(problem["card_index"], 1)
        self.assertEqual(
            problem["path"],
            (
                f'$.term_results["{first_rank}"].additional_senses[0]'
                f'[{json.dumps(fields["sentences"])}][0]'
            ),
        )
        self.assertEqual(
            problem["actual"]["unemphasized_character_offsets"],
            [1],
        )

    def test_repeated_emphasized_term_reports_exact_raw_array_item(self):
        pipeline = context_classical_pipeline()
        payload, fields = self.payload(pipeline)
        first_rank = str(self.chunk.words[0].rank)
        additional = payload[
            process_text.SOURCE_TERM_RESULTS_KEY][first_rank][
                process_text.SOURCE_ADDITIONAL_SENSES_KEY][0]
        additional[fields["sentences"]][0] = (
            "<strong>道</strong>可<strong>道</strong>。")

        report = self.inspect(payload, pipeline)

        problem = self.problem(
            report,
            "repeated_emphasized_source_term")
        self.assertTrue(problem["overrideable"])
        self.assertEqual(problem["card_index"], 1)
        self.assertEqual(
            problem["path"],
            (
                f'$.term_results["{first_rank}"].additional_senses[0]'
                f'[{json.dumps(fields["sentences"])}][0]'
            ),
        )
        self.assertEqual(
            problem["actual"]["emphasized_occurrence_count"],
            2,
        )

    def test_legacy_card_paths_are_unchanged(self):
        pipeline = context_classical_pipeline()
        _payload, fields = self.payload(pipeline)
        raw = {
            "cards": [{
                fields["term"]: "道",
                fields["sentences"]: 42,
                fields["sentence_translations"]: "Example.",
                fields["lexical"][0]: "way",
            }],
        }

        report = inspect_pipeline_response(
            json.dumps(raw, ensure_ascii=False),
            pipeline,
            self.chunk,
        )

        problem = self.problem(report, "field_not_string")
        self.assertEqual(
            problem["path"],
            f'$.cards[0][{json.dumps(fields["sentences"])}]',
        )


if __name__ == "__main__":
    unittest.main()
