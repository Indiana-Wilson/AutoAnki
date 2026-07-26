"""Archived local-model occurrence-to-sense accounting tests."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import process_text
from source_generation.validation import inspect_pipeline_response
from tests.test_v10_schema_contract import (
    context_pipeline,
    repeated_occurrence_chunk,
)


class OccurrenceSenseAccountingTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = context_pipeline()
        self.chunk = repeated_occurrence_chunk()

    @staticmethod
    def additional_sense():
        return {
            "Translation (English)": "path",
            "Dictionary Meaning (English)": "A route or course.",
            "Sentences": [
                "此道甚遠。",
                "循道而行。",
                "前道已開。",
                "本道可通。",
            ],
            "Sentence Translations (English)": [
                "This road is very long.",
                "Proceed along the road.",
                "The road ahead is already open.",
                "This road is passable.",
            ],
        }

    def payload(self):
        return {
            "term_results": [{
                "rank": 1,
                "contextual_sense": {
                    "Translation (English)": "speak",
                    "Dictionary Meaning (English)": (
                        "To express something in words."),
                },
                "additional_senses": [self.additional_sense()],
                # The fixture's audited selection is the second listed 道.
                "occurrence_sense_indices": [1, 0],
            }, {
                "rank": 2,
                "contextual_sense": {
                    "Translation (English)": "can",
                    "Dictionary Meaning (English)": (
                        "To be able or permitted to do something."),
                },
                "additional_senses": [],
                "occurrence_sense_indices": [0],
            }],
            "source_context_translations": [{
                "context_id": "context-1",
                "translation": (
                    "The Way that can be spoken is not the constant Way; "
                    "the name that can be named is not the constant name."),
            }],
        }

    def inspect(self, payload, *, enabled=True):
        return inspect_pipeline_response(
            json.dumps(payload, ensure_ascii=False),
            self.pipeline,
            self.chunk,
            use_compact_source_results=True,
            use_local_example_emphasis=True,
            use_occurrence_sense_indices=enabled)

    @staticmethod
    def problem(report, code):
        return next(
            problem
            for problem in report["problems"]
            if problem["code"] == code)

    def test_exhaustive_map_validates_without_changing_source_card_behavior(self):
        report = self.inspect(self.payload())

        self.assertTrue(report["valid"], report["problems"])
        cards = report["canonical_response"]["cards"]
        self.assertEqual(len(cards), 3)
        self.assertIn("<strong>道</strong>", cards[0]["Sentences"])
        self.assertEqual(
            len(cards[1]["Sentences"].split("|")),
            4)
        self.assertTrue(all(
            sentence.count("<strong>道</strong>") == 1
            for sentence in cards[1]["Sentences"].split("|")))
        self.assertNotIn(
            process_text.SOURCE_OCCURRENCE_SENSE_INDICES_KEY,
            cards[0])

    def test_same_sense_repetition_may_reuse_zero(self):
        payload = self.payload()
        payload["term_results"][0]["occurrence_sense_indices"] = [0, 0]

        report = self.inspect(payload)

        self.assertTrue(report["valid"], report["problems"])

    def test_wrong_length_is_rank_scoped(self):
        payload = self.payload()
        payload["term_results"][0]["occurrence_sense_indices"] = [0]

        report = self.inspect(payload)
        problem = self.problem(
            report,
            "context_occurrence_sense_count_mismatch")

        self.assertEqual(problem["source_rank"], 1)
        self.assertEqual(
            problem["repair_target"],
            {"kind": "source_rank", "source_rank": 1})

    def test_unavailable_and_selected_indices_are_rejected(self):
        unavailable = self.payload()
        unavailable["term_results"][0][
            "occurrence_sense_indices"] = [2, 0]
        unavailable_report = self.inspect(unavailable)
        self.assertEqual(
            self.problem(
                unavailable_report,
                "unavailable_context_occurrence_sense_index")[
                    "source_rank"],
            1)

        selected = self.payload()
        selected["term_results"][0][
            "occurrence_sense_indices"] = [0, 1]
        selected_report = self.inspect(selected)
        self.assertEqual(
            self.problem(
                selected_report,
                "selected_context_occurrence_not_contextual")[
                    "source_rank"],
            1)

    def test_old_v10_shape_remains_valid_when_frozen_schema_disables_feature(
            self):
        payload = deepcopy(self.payload())
        for result in payload["term_results"]:
            result.pop("occurrence_sense_indices")

        report = self.inspect(payload, enabled=False)

        self.assertTrue(report["valid"], report["problems"])


if __name__ == "__main__":
    unittest.main()
