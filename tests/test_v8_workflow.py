import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
from source_generation import (
    ContextMode,
    ContextUnit,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    SourceGenerationBackend,
    SourceGenerationConfig,
    build_source_request_contract,
)
import source_workflow


def context_pipeline():
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "classical_chinese")
    fields = (
        pipeline_store.FieldSetting("translation", "english"),
    )
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key == "context",
                fields=fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese")


def context_plan():
    chunk = GenerationChunk(
        chunk_id="000001-r1-r1",
        index=1,
        total=1,
        start_rank=1,
        end_rank=1,
        words=(
            GenerationWord(
                rank=1,
                surface="甲",
                normalized="甲",
                section_id="section-1",
                sentence_id="sentence-1",
                context_id="context-1",
                start_offset=0,
                end_offset=1),
        ),
        contexts=(
            ContextUnit(
                context_id="context-1",
                mode="sentence",
                start_offset=0,
                end_offset=3,
                text="甲乙。",
                section_ids=("section-1",),
                sentence_ids=("sentence-1",),
                word_ranks=(1,)),
        ))
    config = SourceGenerationConfig(
        source_key="fixture",
        chunk_size=1,
        context_mode=ContextMode.SENTENCE,
        concurrency=1,
        request_stagger_ms=0,
        max_transient_retries=0,
        request_protocol="v8",
        reasoning_effort="low")
    return GenerationPlan(
        plan_id="fixture-v8-plan",
        source_key="fixture",
        source_title="Fixture",
        source_build_id="fixture-build",
        source_run_path="/fixture/build",
        config=config,
        original_word_count=1,
        excluded_word_count=0,
        chunks=(chunk,))


def grouped_contract(pipeline, plan):
    # These tests exercise the frozen contract and workflow boundary, not
    # prompt-component file discovery.
    with patch(
            "source_generation.requests.build_source_prompt",
            return_value="frozen v8 prompt\n"):
        return build_source_request_contract(
            pipeline,
            chunks=plan.chunks,
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")


class V8WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=self.root / "corpora",
            jobs_root=self.root / "jobs")
        self.pipeline = context_pipeline()
        self.plan = context_plan()
        self.chunk = self.plan.chunks[0]
        self.contract = grouped_contract(self.pipeline, self.plan)

    def controller(self):
        return source_workflow.SourceWorkflowController(
            backend=self.backend)

    def create_job(self, contract=None):
        return self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
            },
            request_contract=contract or self.contract)

    def test_paid_request_uses_exact_frozen_format_for_grouped_chunk(self):
        response = SimpleNamespace(
            status="completed",
            output_text='{"term_results": {}}',
            output=())
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=MagicMock(return_value=response)))
        request = self.controller()._paid_request_callable(
            self.pipeline,
            client,
            self.contract)

        paid_response = request(self.chunk)

        self.assertEqual(
            paid_response.raw_text,
            '{"term_results": {}}')

        supplied_format = client.responses.create.call_args.kwargs[
            "text"]["format"]
        expected_format = self.contract[
            "response_formats_by_chunk"][self.chunk.chunk_id]
        self.assertIs(supplied_format, expected_format)
        self.assertNotEqual(
            supplied_format,
            self.contract["response_format"])

    def test_legacy_request_keeps_its_generic_frozen_format(self):
        legacy_contract = dict(self.contract)
        legacy_contract["schema_version"] = 7
        legacy_contract.pop("response_formats_by_chunk")
        response = SimpleNamespace(
            status="completed",
            output_text='{"contextual_cards": []}',
            output=())
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=MagicMock(return_value=response)))
        request = self.controller()._paid_request_callable(
            self.pipeline,
            client,
            legacy_contract)

        request(self.chunk)

        self.assertIs(
            client.responses.create.call_args.kwargs["text"]["format"],
            legacy_contract["response_format"])

    def test_grouped_request_hard_fails_when_chunk_format_is_missing(self):
        broken_contract = dict(self.contract)
        broken_contract["response_formats_by_chunk"] = {}
        client = SimpleNamespace(
            responses=SimpleNamespace(create=MagicMock()))
        request = self.controller()._paid_request_callable(
            self.pipeline,
            client,
            broken_contract)

        with patch.object(
                source_workflow,
                "source_request_uses_grouped_source_results",
                return_value=True):
            with self.assertRaisesRegex(
                    ValueError,
                    "no response format for chunk 000001-r1-r1"):
                request(self.chunk)

        client.responses.create.assert_not_called()

    def test_inspection_shows_the_chunks_exact_grouped_format(self):
        job = self.create_job()
        row_id = self.backend._row_id(
            job.job_id,
            self.chunk.chunk_id)

        inspection = self.controller().inspect({
            "job_id": row_id,
        })

        self.assertEqual(
            inspection["request_view"]["response_format"],
            self.contract[
                "response_formats_by_chunk"][self.chunk.chunk_id])
        self.assertNotEqual(
            inspection["request_view"]["response_format"],
            self.contract["response_format"])

    def test_runner_and_inspection_enable_grouped_validation(self):
        job = self.create_job()
        controller = self.controller()
        validator = MagicMock(name="validator")
        with (
                patch.object(
                    source_workflow,
                    "make_pipeline_response_validator",
                    return_value=validator) as make_validator,
                patch.object(
                    source_workflow,
                    "GenerationJobRunner") as runner_class):
            runner_class.return_value.run.return_value = "run-result"

            result = controller._run_job(
                job.job_id,
                self.pipeline,
                MagicMock())

        self.assertEqual(result, "run-result")
        self.assertTrue(
            make_validator.call_args.kwargs[
                "use_grouped_source_results"])
        runner_class.assert_called_once()

        raw_inspection = {
            "status": {
                "latest_attempt_path": "attempts/1",
            },
            "attempts": [{
                "attempt": 1,
                "raw_text": "{}",
            }],
        }
        report = {
            "problems": [],
            "valid": True,
            "structurally_valid": True,
        }
        with (
                patch.object(
                    source_workflow,
                    "inspect_pipeline_response",
                    return_value=report) as inspect_response,
                patch.object(
                    source_workflow,
                    "public_validation_report",
                    return_value={"problems": []})):
            controller._inspect_chunk_validation(
                job.job_id,
                self.chunk.chunk_id,
                inspection=raw_inspection,
                pipeline=self.pipeline)

        self.assertTrue(
            inspect_response.call_args.kwargs[
                "use_grouped_source_results"])

    def test_retry_guard_directs_v7_jobs_to_v8_migration(self):
        legacy_contract = dict(self.contract)
        legacy_contract["schema_version"] = 7
        legacy_contract.pop("response_formats_by_chunk")
        job = self.create_job(legacy_contract)
        row_id = self.backend._row_id(
            job.job_id,
            self.chunk.chunk_id)

        with self.assertRaisesRegex(
                ValueError,
                "obsolete v4-v7 source-context protocol.*migrate to v8"):
            self.controller().retry({
                "job_ids": (row_id,),
                "paid_confirmed": True,
            })


if __name__ == "__main__":
    unittest.main()
