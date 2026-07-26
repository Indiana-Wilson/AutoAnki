# Archived with the abandoned local-model integration.
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "evaluate_local_source_v10.py")
SPEC = importlib.util.spec_from_file_location(
    "evaluate_local_source_v10",
    SCRIPT_PATH)
evaluate_local_source_v10 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_local_source_v10)


def summary(*, stages=()):
    value = {
        "counts": [1, 3, 10],
        "stages": list(stages),
    }
    evaluate_local_source_v10._set_progressive_summary_state(value)
    return value


class ProgressiveSummaryStateTests(unittest.TestCase):
    def test_new_summary_is_explicitly_incomplete_and_not_green(self):
        value = summary()

        self.assertEqual(value["completion_state"], "incomplete")
        self.assertFalse(value["complete"])
        self.assertIsNone(value["all_passed"])
        self.assertEqual(value["requested_stage_count"], 3)
        self.assertEqual(value["completed_stage_count"], 0)

    def test_partial_success_remains_nonterminal_and_not_green(self):
        value = summary(stages=({
            "requested_unique_count": 1,
            "passed": True,
        },))

        self.assertEqual(value["completion_state"], "running")
        self.assertFalse(value["complete"])
        self.assertIsNone(value["all_passed"])
        self.assertEqual(value["completed_stage_count"], 1)

        evaluate_local_source_v10._set_progressive_summary_state(
            value,
            terminal=True)

        self.assertEqual(value["completion_state"], "incomplete")
        self.assertFalse(value["complete"])
        self.assertFalse(value["all_passed"])

    def test_only_all_requested_successes_finish_green(self):
        value = summary(stages=tuple(
            {
                "requested_unique_count": count,
                "passed": True,
            }
            for count in (1, 3, 10)))

        evaluate_local_source_v10._set_progressive_summary_state(
            value,
            terminal=True)

        self.assertEqual(value["completion_state"], "completed")
        self.assertTrue(value["complete"])
        self.assertTrue(value["all_passed"])

    def test_failed_or_interrupted_runs_cannot_finish_green(self):
        failed = summary(stages=({
            "requested_unique_count": 1,
            "passed": False,
        },))
        evaluate_local_source_v10._set_progressive_summary_state(
            failed,
            terminal=True)

        self.assertEqual(failed["completion_state"], "failed")
        self.assertFalse(failed["complete"])
        self.assertFalse(failed["all_passed"])

        interrupted = summary(stages=({
            "requested_unique_count": 1,
            "passed": True,
        },))
        evaluate_local_source_v10._set_progressive_summary_state(
            interrupted,
            interruption=KeyboardInterrupt())

        self.assertEqual(
            interrupted["completion_state"],
            "interrupted")
        self.assertFalse(interrupted["complete"])
        self.assertFalse(interrupted["all_passed"])
        self.assertEqual(
            interrupted["interruption"]["type"],
            "KeyboardInterrupt")


class EvaluatorDisplayNameTests(unittest.TestCase):
    def test_catalogue_display_name_is_used_for_request_and_deck(self):
        backend = SimpleNamespace(
            catalogue=lambda: {
                "sources": [{
                    "key": "daodejing_wang_bi",
                    "name": "Daodejing [Wang Bi]",
                }],
            })
        source = SimpleNamespace(
            key="daodejing_wang_bi",
            title="道德經",
            run_path="/fixture/run",
            build=SimpleNamespace(
                snapshot=SimpleNamespace(
                    source_language_key="classical_chinese_wang_bi")))
        display_name = (
            evaluate_local_source_v10._catalogue_source_display_name(
                backend,
                source))
        args = SimpleNamespace(
            source_examples=True,
            context_mode="sentence",
            model="ollama/qwen3:14b",
            automatic_repair=True,
            max_automatic_repairs=None,
            transient_retries=None)

        request = evaluate_local_source_v10._request_for_stage(
            args,
            source,
            SimpleNamespace(),
            {
                "unique_count": 1,
                "prefix_token_limit": 1,
            },
            chunk_size=5,
            source_display_name=display_name)

        self.assertEqual(display_name, "Daodejing [Wang Bi]")
        self.assertEqual(
            request["source_name"],
            "Daodejing [Wang Bi]")
        self.assertEqual(
            request["output_deck_name"],
            "Vocabulary from Daodejing [Wang Bi]")


class EvaluatorOccurrenceReviewTests(unittest.TestCase):
    def test_repeated_occurrence_decisions_are_exposed_without_guessing(self):
        inspection = {
            "request": {
                "words": [{
                    "rank": 1,
                    "surface": "道",
                    "context_occurrences": [
                        {"span": [0, 1]},
                        {"span": [2, 3]},
                    ],
                }],
            },
            "attempts": [{
                "raw_text": json.dumps({
                    "term_results": [{
                        "rank": 1,
                        "occurrence_sense_indices": [0, 0],
                    }],
                }),
            }],
        }

        result = (
            evaluate_local_source_v10
            ._occurrence_accounting_summary(inspection))

        self.assertEqual(result, [{
            "rank": 1,
            "term": "道",
            "occurrence_count": 2,
            "sense_indices": [0, 0],
            "distinct_mapped_sense_count": 1,
            "manual_same_sense_review": True,
        }])


if __name__ == "__main__":
    unittest.main()
