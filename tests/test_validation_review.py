import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation import (
    ContextMode,
    GenerationChunk,
    GenerationJobRunner,
    GenerationPlan,
    GenerationWord,
    SourceGenerationBackend,
    SourceGenerationConfig,
    inspect_pipeline_response,
)
import source_workflow


def one_field_pipeline(language_key="classical_chinese"):
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        language_key)
    fields = (
        pipeline_store.FieldSetting("translation", "english"),
    )
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key == "word_to_meaning",
                fields=fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key=language_key)


def one_chunk_plan(words):
    config = SourceGenerationConfig(
        source_key="fixture",
        chunk_size=len(words),
        context_mode=ContextMode.NONE,
        concurrency=1,
        request_stagger_ms=0,
        max_transient_retries=0)
    generation_words = tuple(
        GenerationWord(
            rank=index,
            surface=surface,
            normalized=normalized,
            section_id="section-1",
            sentence_id="sentence-1",
            context_id=None)
        for index, (surface, normalized) in enumerate(words, start=1))
    chunk = GenerationChunk(
        chunk_id=f"000001-r1-r{len(words)}",
        index=1,
        total=1,
        start_rank=1,
        end_rank=len(words),
        words=generation_words,
        contexts=())
    return GenerationPlan(
        plan_id="fixture-plan",
        source_key="fixture",
        source_title="Fixture",
        source_build_id="fixture-build",
        source_run_path="/fixture/build",
        config=config,
        original_word_count=len(words),
        excluded_word_count=0,
        chunks=(chunk,))


class DetailedValidationTests(unittest.TestCase):
    def test_syntax_and_field_type_problems_are_precise_and_not_overrideable(
            self):
        pipeline = one_field_pipeline()
        syntax = process_text.inspect_generated_response(
            '{"cards": [',
            pipeline)

        self.assertFalse(syntax["syntax_valid"])
        self.assertEqual(syntax["problems"][0]["code"], "invalid_json")
        self.assertIn("line", syntax["problems"][0]["location"])
        self.assertFalse(syntax["problems"][0]["overrideable"])

        term_field, meaning_field = process_text.get_response_field_names(
            pipeline)
        structural = process_text.inspect_generated_response(
            json.dumps({
                "cards": [{
                    term_field: "甲",
                    meaning_field: None,
                }],
            }),
            pipeline)

        problem = next(
            problem
            for problem in structural["problems"]
            if problem["code"] == "field_not_string")
        self.assertEqual(problem["card_number"], 1)
        self.assertEqual(problem["field_name"], meaning_field)
        self.assertIn(meaning_field, problem["path"])
        self.assertFalse(problem["overrideable"])
        self.assertLessEqual(len(json.dumps(problem["actual"])), 600)

    def test_content_report_is_deterministic_and_override_packaging_is_explicit(
            self):
        pipeline = one_field_pipeline()
        term_field, meaning_field = process_text.get_response_field_names(
            pipeline)
        raw = json.dumps({
            "cards": [{
                term_field: "甲",
                meaning_field: "",
            }],
        })

        first = process_text.inspect_generated_response(raw, pipeline)
        second = process_text.inspect_generated_response(raw, pipeline)

        self.assertEqual(
            [problem["problem_id"] for problem in first["problems"]],
            [problem["problem_id"] for problem in second["problems"]])
        self.assertTrue(first["structurally_valid"])
        self.assertTrue(all(
            problem["overrideable"]
            for problem in first["problems"]))
        with self.assertRaises(
                process_text.GeneratedCardValidationError):
            process_text.validate_generated_cards(raw, pipeline)
        self.assertEqual(
            len(process_text.validate_generated_cards(
                raw,
                pipeline,
                allow_accepted_content_problems=True)),
            1)

    def test_source_membership_is_casefolded_and_unknown_terms_sort_last(self):
        pipeline = one_field_pipeline("middle_english")
        chunk = one_chunk_plan((("Whan", "whan"), ("Aprill", "aprill"))).chunks[0]
        term_field, meaning_field = process_text.get_response_field_names(
            pipeline)
        raw = json.dumps({
            "cards": [
                {term_field: "Other", meaning_field: "extra"},
                {term_field: "APRILL", meaning_field: "month"},
                {term_field: "whan", meaning_field: "when"},
            ],
        })

        report = inspect_pipeline_response(raw, pipeline, chunk)

        self.assertFalse(any(
            problem["code"] == "missing_source_term"
            for problem in report["problems"]))
        self.assertEqual(
            [
                card[term_field]
                for card in report["canonical_response"]["cards"]
            ],
            ["whan", "APRILL", "Other"])
        outside = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "term_outside_source_chunk")
        self.assertEqual(outside["card_number"], 1)
        self.assertTrue(outside["overrideable"])


class ManualValidationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=root / "corpora",
            jobs_root=root / "jobs")
        self.pipeline = one_field_pipeline()
        self.plan = one_chunk_plan((("甲", "甲"),))
        self.package_creator = MagicMock(
            side_effect=self._create_package)
        self.package_importer = MagicMock(return_value=True)
        self.controller = source_workflow.SourceWorkflowController(
            backend=self.backend,
            package_creator=self.package_creator,
            package_importer=self.package_importer)

    @staticmethod
    def _create_package(_combined, **kwargs):
        kwargs["output_path"].write_bytes(b"apkg")
        return kwargs["output_path"], 1

    def _run_invalid(self, raw):
        snapshot = self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
            })
        runner = GenerationJobRunner(
            self.backend.jobs,
            snapshot.job_id,
            lambda _chunk: raw,
            source_workflow.make_pipeline_response_validator(
                self.pipeline),
            concurrency=1,
            request_stagger_ms=0)
        runner.run()
        return (
            snapshot.job_id,
            f"{snapshot.job_id}::{self.plan.chunks[0].chunk_id}")

    def test_selected_acceptance_is_audited_and_only_last_one_completes(self):
        term_field, meaning_field = process_text.get_response_field_names(
            self.pipeline)
        job_id, row_id = self._run_invalid(json.dumps({
            "cards": [{
                term_field: "甲",
                meaning_field: "",
            }],
        }))
        inspected = self.controller.inspect({"job_id": row_id})
        validation = inspected["validation"]
        self.assertTrue(validation["available"])
        self.assertGreaterEqual(validation["problem_count"], 2)
        selected = validation["problems"][0]["problem_id"]

        partial = self.controller.accept_validation_problems({
            "job_id": row_id,
            "problem_ids": [selected],
            "reason": "The empty reverse side is intentional.",
        })

        self.assertFalse(partial["chunk_completed"])
        self.assertGreater(partial["remaining_problem_count"], 0)
        self.package_creator.assert_not_called()
        chunk_row = next(
            row
            for row in self.controller.job_rows()["jobs"]
            if row["job_id"] == row_id)
        self.assertIn("manually accepted", chunk_row["detail"])
        self.assertIn(
            f"{partial['remaining_problem_count']:,} remaining",
            chunk_row["detail"])
        self.assertTrue(chunk_row["has_validation_error"])
        reread = self.controller.inspect({"job_id": row_id})
        self.assertTrue(next(
            problem["accepted"]
            for problem in reread["validation"]["problems"]
            if problem["problem_id"] == selected))

        remaining_ids = [
            problem["problem_id"]
            for problem in reread["validation"]["problems"]
            if not problem["accepted"]
        ]
        completed = self.controller.accept_validation_problems({
            "job_id": row_id,
            "problem_ids": remaining_ids,
        })

        self.assertTrue(completed["chunk_completed"])
        self.assertEqual(completed["remaining_problem_count"], 0)
        self.assertEqual(
            self.backend.jobs.chunk_status(
                job_id,
                self.plan.chunks[0].chunk_id)["status"],
            "succeeded")
        self.assertTrue(
            self.package_creator.call_args.kwargs[
                "allow_accepted_content_problems"])
        self.package_importer.assert_called_once()
        audit = self.backend.jobs.load_manual_validation(
            job_id,
            self.plan.chunks[0].chunk_id)
        self.assertEqual(len(audit["history"]), 2)
        self.assertEqual(
            len(audit["accepted_problems"]),
            validation["problem_count"])

    def test_syntax_problem_is_shown_but_cannot_be_accepted(self):
        _job_id, row_id = self._run_invalid('{"cards": [')
        inspected = self.controller.inspect({"job_id": row_id})
        problem = inspected["validation"]["problems"][0]

        self.assertEqual(problem["code"], "invalid_json")
        self.assertFalse(problem["overrideable"])
        with self.assertRaisesRegex(
                ValueError,
                "Structural validation problems"):
            self.controller.accept_validation_problems({
                "job_id": row_id,
                "problem_ids": [problem["problem_id"]],
            })


if __name__ == "__main__":
    unittest.main()
