import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import sentinel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui


class SourceJobsGuiTests(unittest.TestCase):
    def test_jobs_and_inspection_readers_have_bounded_fallback_sizes(self):
        # Grid weights fill the actual notebook viewport. These values are
        # merely natural-size fallbacks and must not recreate the giant
        # scrolling page that previously hid the action controls.
        self.assertGreaterEqual(
            gui.SOURCE_JOBS_TREE_VISIBLE_ROWS,
            8)
        self.assertLessEqual(
            gui.SOURCE_JOBS_TREE_VISIBLE_ROWS,
            14)
        self.assertGreaterEqual(
            gui.SOURCE_JOB_INSPECTION_VISIBLE_LINES,
            10)
        self.assertLessEqual(
            gui.SOURCE_JOB_INSPECTION_VISIBLE_LINES,
            18)

    def test_problem_reviewer_geometry_never_exceeds_the_display(self):
        for screen_width, screen_height in (
                (640, 480),
                (800, 600),
                (1366, 768),
                (1920, 1080)):
            with self.subTest(
                    screen_width=screen_width,
                    screen_height=screen_height):
                width, height, x, y = (
                    gui.source_validation_dialog_geometry(
                        screen_width,
                        screen_height))

                self.assertGreater(width, 0)
                self.assertGreater(height, 0)
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + width, screen_width)
                self.assertLessEqual(y + height, screen_height)

    def test_rows_group_each_deck_stage_with_its_own_job(self):
        rows = (
            {
                "job_id": "job-a::chunk-1",
                "parent_job_id": "job-a",
                "chunk_label": "1/1",
            },
            {
                "job_id": "job-b::chunk-1",
                "parent_job_id": "job-b",
                "chunk_label": "1/1",
            },
            {
                "job_id": "job-a::finalize",
                "parent_job_id": "job-a",
                "chunk_label": "Deck",
            },
            {
                "job_id": "job-b::finalize",
                "parent_job_id": "job-b",
                "chunk_label": "Deck",
            },
        )

        grouped = gui.group_source_job_rows(rows)

        self.assertEqual(
            tuple(parent_id for parent_id, _job_rows in grouped),
            ("job-a", "job-b"))
        self.assertEqual(
            tuple(row["job_id"] for row in grouped[0][1]),
            ("job-a::chunk-1", "job-a::finalize"))
        self.assertEqual(
            tuple(row["job_id"] for row in grouped[1][1]),
            ("job-b::chunk-1", "job-b::finalize"))

    def test_refresh_inserts_explicit_group_boundaries(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_jobs_loader = MagicMock(return_value={"jobs": (
            {
                "job_id": "job-a::chunk-1",
                "parent_job_id": "job-a",
                "source_name": "Source A",
                "chunk_label": "1/1",
                "worker": "worker-a",
                "status": "succeeded",
                "attempts": 1,
                "detail": "Accepted.",
            },
            {
                "job_id": "job-b::chunk-1",
                "parent_job_id": "job-b",
                "source_name": "Source B",
                "chunk_label": "1/1",
                "worker": "worker-b",
                "status": "succeeded",
                "attempts": 1,
                "detail": "Accepted.",
            },
            {
                "job_id": "job-a::finalize",
                "parent_job_id": "job-a",
                "source_name": "Source A",
                "chunk_label": "Deck",
                "worker": "package/import",
                "status": "completed",
                "attempts": 1,
                "detail": "Imported.",
            },
            {
                "job_id": "job-b::finalize",
                "parent_job_id": "job-b",
                "source_name": "Source B",
                "chunk_label": "Deck",
                "worker": "package/import",
                "status": "completed",
                "attempts": 1,
                "detail": "Imported.",
            },
        )})
        app.source_jobs_tree = MagicMock()
        app.source_jobs_tree.get_children.return_value = ()
        app.source_jobs_tree.selection.return_value = ()
        app.source_job_by_tree_id = {}
        app.source_action_status = MagicMock()
        app._resize_source_job_columns = MagicMock()
        app._source_job_selection_changed = MagicMock()

        app._refresh_source_jobs()

        inserts = app.source_jobs_tree.insert.call_args_list
        self.assertEqual(len(inserts), 6)
        self.assertEqual(inserts[0].args[:2], ("", gui.tk.END))
        self.assertEqual(inserts[1].args[0], "source_job_group_0")
        self.assertEqual(inserts[1].kwargs["values"][1], "1/1")
        self.assertEqual(inserts[2].args[0], "source_job_group_0")
        self.assertEqual(inserts[2].kwargs["values"][1], "Deck")
        self.assertEqual(inserts[3].args[:2], ("", gui.tk.END))
        self.assertEqual(inserts[4].args[0], "source_job_group_1")
        self.assertEqual(inserts[5].args[0], "source_job_group_1")

    def test_columns_fit_the_viewport_and_leave_detail_remainder(self):
        rows = ({
            "job": "Job 2026-07-24 13:31:53",
            "source": "The maximally long source title",
            "chunk": "31/31",
            "worker": "autoanki-source_12",
            "status": "needs attention",
            "attempts": "12",
            "detail": "This content must not steal width from fixed columns.",
        },)
        measure = lambda value: len(str(value)) * 7

        widths = gui.source_job_column_widths(
            rows,
            measure,
            available_width=1400)

        self.assertEqual(sum(widths.values()), 1396)
        self.assertGreaterEqual(
            widths["detail"],
            gui.SOURCE_JOB_DETAIL_MIN_WIDTH)
        for column, minimum in gui.SOURCE_JOB_COLUMN_MIN_WIDTHS.items():
            self.assertGreaterEqual(widths[column], minimum)

    def test_problem_columns_fit_and_keep_review_state_readable(self):
        measure = lambda value: len(str(value)) * 9

        widths = gui.validation_problem_column_widths(
            measure,
            available_width=1500)

        self.assertEqual(sum(widths.values()), 1496)
        self.assertGreaterEqual(
            widths["decision"],
            measure("Review state") + 28)
        self.assertGreater(widths["problem"], widths["location"])

    def test_inspection_separates_request_from_response(self):
        request_text, response_text = gui.source_job_inspection_texts({
            "request": {
                "chunk_id": "chunk-1",
                "words": [{"term": "道"}],
            },
            "request_contract": {
                "model": "gpt-test",
            },
            "status": {
                "status": "invalid_response",
            },
            "attempts": [{
                "attempt": 1,
                "raw_text": "{\"cards\": []}",
                "error": {
                    "message": "A required card was missing.",
                },
            }],
        })

        self.assertIn(
            "SAVED INPUT FOR THIS REQUEST CHUNK",
            request_text)
        self.assertIn("chunk-1", request_text)
        self.assertNotIn("CURRENT CHUNK STATUS", request_text)
        self.assertIn("CURRENT CHUNK STATUS", response_text)
        self.assertIn("SAVED RESPONSE ATTEMPTS", response_text)
        self.assertIn(
            "A required card was missing.",
            response_text)

    def test_inspection_scope_says_one_chunk_not_whole_job(self):
        scope = gui.source_job_inspection_scope({
            "job_id": "job-a::chunk-2",
            "parent_job_id": "job-a",
            "source": "Fixture",
            "chunk": "2/31",
        })
        deck_scope = gui.source_job_inspection_scope({
            "job_id": "job-a::finalize",
            "parent_job_id": "job-a",
            "source": "Fixture",
            "chunk": "Deck",
        })

        self.assertIn("one saved OpenAI request chunk", scope)
        self.assertIn("not the whole job", scope)
        self.assertIn("chunk 2/31", scope)
        self.assertIn("one Deck packaging/import stage", deck_scope)
        self.assertIn("not an OpenAI request", deck_scope)

    def test_backend_request_view_stays_in_request_tab(self):
        request_text, response_text = gui.source_job_inspection_texts({
            "scope": {"summary": "One batch."},
            "request_view": {
                "title": "Request · batch 2 of 4",
                "batch_input": "Whan Aprill",
            },
            "response_view": {
                "title": "Response · attempt 1",
                "summary": "Rejected.",
            },
        })

        self.assertIn("Request · batch 2 of 4", request_text)
        self.assertIn("Whan Aprill", request_text)
        self.assertNotIn("Request · batch 2 of 4", response_text)
        self.assertIn("Response · attempt 1", response_text)
        self.assertNotIn("One batch.", response_text)

    def test_validation_problem_details_are_human_readable(self):
        problem = {
            "title": "Example sentence count is wrong",
            "message": "Exactly three examples are required.",
            "location": "Card 3 (“wyrd”) → Sentences",
            "path": '$.cards[2]["Sentences"]',
            "expected": 4,
            "actual": 2,
            "suggestion": "Add two examples or accept after review.",
            "overrideable": True,
            "accepted": False,
        }

        details = gui.format_validation_problem_details(problem)

        self.assertIn("Card 3 (“wyrd”) → Sentences", details)
        self.assertIn("WHAT IS WRONG", details)
        self.assertIn("EXPECTED", details)
        self.assertIn("\n4\n", details)
        self.assertIn("ACTUAL", details)
        self.assertIn("\n2\n", details)
        self.assertIn('$.cards[2]["Sentences"]', details)
        self.assertIn("may be accepted", details)

    def test_problem_view_points_to_exact_affected_card(self):
        inspection = {
            "response_view": {
                "latest_attempt": {
                    "raw_text": (
                        '{"cards": ['
                        '{"Word": "whan", "Translation": "when"},'
                        '{"Word": "wyrd", "Translation": "fate"}'
                        "]}"
                    ),
                },
            },
        }

        affected = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 1, "scope": "field"})

        self.assertIn('"Word": "wyrd"', affected)
        self.assertIn('"Translation": "fate"', affected)
        self.assertNotIn('"Word": "whan"', affected)

    def test_problem_view_maps_v6_split_card_indices(self):
        inspection = {
            "response_view": {
                "latest_attempt": {
                    "raw_text": (
                        '{"contextual_cards": ['
                        '{"Word": "whan", "Meaning": "when"},'
                        '{"Word": "wyrd", "Meaning": "fate"}'
                        '], "additional_sense_cards": ['
                        '{"Word": "whan", "Meaning": "at what time"},'
                        '{"Word": "wyrd", "Meaning": "personal destiny"}'
                        '], "source_context_translations": []}'
                    ),
                },
            },
        }

        contextual = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 1, "scope": "field"})
        additional = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 2, "scope": "field"})

        self.assertIn('"Word": "wyrd"', contextual)
        self.assertIn('"Meaning": "fate"', contextual)
        self.assertNotIn('"personal destiny"', contextual)
        self.assertIn('"Word": "whan"', additional)
        self.assertIn('"Meaning": "at what time"', additional)
        self.assertNotIn('"Meaning": "when"', additional)

    def test_problem_view_maps_v8_rank_group_indices(self):
        inspection = {
            "response_view": {
                "latest_attempt": {
                    "raw_text": (
                        '{"term_results": {'
                        '"9": {"contextual_sense": {"Meaning": "of"}, '
                        '"additional_senses": []}, '
                        '"2": {"contextual_sense": {"Meaning": "Way"}, '
                        '"additional_senses": ['
                        '{"Meaning": "road"}, {"Meaning": "speak"}]}} , '
                        '"source_context_translations": {"ctx": "Text"}}'
                    ),
                },
            },
        }

        contextual = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 0, "scope": "field"})
        second_additional = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 2, "scope": "field"})
        later_rank = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 3, "scope": "field"})

        self.assertIn('"rank": "2"', contextual)
        self.assertIn('"role": "contextual_sense"', contextual)
        self.assertIn('"Meaning": "Way"', contextual)
        self.assertIn('"additional_sense_number": 2', second_additional)
        self.assertIn('"Meaning": "speak"', second_additional)
        self.assertIn('"rank": "9"', later_rank)
        self.assertIn('"Meaning": "of"', later_rank)

    def test_problem_view_maps_v9_compact_rank_array_indices(self):
        inspection = {
            "response_view": {
                "latest_attempt": {
                    "raw_text": (
                        '{"term_results": ['
                        '{"rank": 2, '
                        '"contextual_sense": {"Meaning": "Way"}, '
                        '"additional_senses": ['
                        '{"Meaning": "road"}, {"Meaning": "speak"}]}, '
                        '{"rank": 9, '
                        '"contextual_sense": {"Meaning": "of"}, '
                        '"additional_senses": []}], '
                        '"source_context_translations": []}'
                    ),
                },
            },
        }

        contextual = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 0, "scope": "field"})
        second_additional = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 2, "scope": "field"})
        later_rank = gui.validation_problem_affected_section(
            inspection,
            {"card_index": 3, "scope": "field"})

        self.assertIn('"rank": 2', contextual)
        self.assertIn('"role": "contextual_sense"', contextual)
        self.assertIn('"Meaning": "Way"', contextual)
        self.assertIn('"additional_sense_number": 2', second_additional)
        self.assertIn('"Meaning": "speak"', second_additional)
        self.assertIn('"rank": 9', later_rank)
        self.assertIn('"Meaning": "of"', later_rank)

    def test_validation_summary_distinguishes_locked_and_reviewable(self):
        locked = gui.validation_report_summary({
            "problem_count": 1,
            "accepted_problem_count": 0,
            "remaining_problem_count": 1,
            "syntax_valid": True,
            "structurally_valid": False,
        })
        reviewable = gui.validation_report_summary({
            "problem_count": 2,
            "accepted_problem_count": 0,
            "remaining_problem_count": 2,
            "syntax_valid": True,
            "structurally_valid": True,
            "can_complete_with_manual_acceptance": True,
        })

        self.assertIn("cannot be overridden", locked)
        self.assertIn("card structure is unsafe", locked)
        self.assertIn("may be accepted after review", reviewable)

    def test_source_prefix_toggle_is_wired_and_invalidates_authorization(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_limit_to_prefix = MagicMock()
        app.source_limit_to_prefix.get.return_value = True
        app.source_prefix_token_entry = MagicMock()
        app.source_paid_authorized = MagicMock()
        app.source_zero_notice_shown = True
        app._schedule_source_estimate = MagicMock()

        app._source_prefix_setting_changed()

        app.source_prefix_token_entry.configure.assert_called_once_with(
            state=gui.tk.NORMAL)
        app.source_paid_authorized.set.assert_called_once_with(False)
        self.assertFalse(app.source_zero_notice_shown)
        app._schedule_source_estimate.assert_called_once_with()

        app.source_limit_to_prefix.get.return_value = False
        app._sync_source_prefix_control()
        app.source_prefix_token_entry.configure.assert_called_with(
            state=gui.tk.DISABLED)

    def test_source_request_includes_prefix_only_when_enabled(self):
        def variable(value):
            result = MagicMock()
            result.get.return_value = value
            return result

        app = object.__new__(gui.AutoAnkiApp)
        app._selected_source_option = MagicMock(
            return_value=gui.SourceUiOption(
                key="fixture",
                name="Fixture",
                source_language_key="middle_english"))
        app.source_chunk_size = variable("30")
        app.source_concurrency = variable("8")
        app.source_request_stagger_ms = variable("100")
        app.source_context_label = variable(
            gui.SOURCE_CONTEXT_LABELS["sentence"])
        app.source_use_source_examples = variable(False)
        app.source_limit_to_prefix = variable(True)
        app.source_prefix_token_limit = variable("100")
        app.source_language_label = variable("")
        app.source_allow_web_search = variable(False)
        app.get_pipeline_configs = MagicMock(return_value=())
        app._selected_anki_exclusions = MagicMock(return_value=())

        request = app._source_request()

        self.assertEqual(request["source_prefix_token_limit"], 100)
        self.assertIsInstance(request["source_prefix_token_limit"], int)

        app.source_limit_to_prefix.get.return_value = False
        app.source_prefix_token_limit.get.return_value = "not a number"
        request = app._source_request()
        self.assertIsNone(request["source_prefix_token_limit"])

    def test_prefix_accounting_is_visible_in_estimate(self):
        _price, detail = gui.format_source_estimate({
            "source_prefix_token_limit": 100,
            "source_prefix_token_count": 100,
            "prefix_unique_candidate_count": 63,
            "candidate_count": 60,
            "request_count": 2,
        })

        self.assertIn("first 100 running-word occurrences", detail)
        self.assertIn("63 unique candidates", detail)
        self.assertIn("60 new words", detail)

    def test_manual_acceptance_dispatches_only_explicit_problem_ids(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_validation_problem_context = {
            "record": {"job_id": "job-a::chunk-1"},
            "report": {"remaining_problem_count": 2},
        }
        app.source_validation_problem_dialog = sentinel.dialog
        app.source_validation_reason = MagicMock()
        app.source_validation_reason.get.return_value = "Reviewed in context"
        app.source_validation_accept_callback = sentinel.accept_callback
        app._dispatch_source_action = MagicMock()
        problems = (
            {"problem_id": "problem-a"},
            {"problem_id": "problem-b"},
        )

        with patch.object(
                gui.messagebox,
                "askyesno",
                return_value=True) as confirmation:
            app._accept_validation_problems(problems)

        confirmation.assert_called_once()
        app._dispatch_source_action.assert_called_once_with(
            "validation_accept",
            sentinel.accept_callback,
            {
                "job_id": "job-a::chunk-1",
                "problem_ids": ("problem-a", "problem-b"),
                "reason": "Reviewed in context",
            })


if __name__ == "__main__":
    unittest.main()
