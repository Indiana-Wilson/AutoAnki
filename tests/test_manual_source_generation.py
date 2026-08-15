import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import manual_source_generation
import pipeline_store
import source_workflow
from source_generation import (
    ContextMode,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    SourceGenerationConfig,
)
from source_generation.jobs import GenerationJobStore


def one_chunk_plan():
    chunk = GenerationChunk(
        chunk_id="000001-r1-r1",
        index=1,
        total=1,
        start_rank=1,
        end_rank=1,
        words=(GenerationWord(
            rank=1,
            surface="猴",
            normalized="猴",
            section_id="section-1",
            sentence_id="sentence-1",
            context_id=None),),
        contexts=())
    return GenerationPlan(
        plan_id="manual-fixture-plan",
        source_key="fixture",
        source_title="Fixture",
        source_build_id="fixture-build",
        source_run_path="/fixture/run",
        config=SourceGenerationConfig(
            source_key="fixture",
            chunk_size=1,
            context_mode=ContextMode.NONE,
            request_protocol="v10"),
        original_word_count=1,
        excluded_word_count=0,
        chunks=(chunk,))


def strict_fixture_validator(raw_text, _chunk):
    value = json.loads(raw_text)
    if value != {"cards": [{"term": "猴"}]}:
        raise ValueError("invalid manual response")
    return value


class ManualSourceGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = GenerationJobStore(self.temporary.name)
        self.job = self.store.create(
            one_chunk_plan(),
            request_metadata={"manual_offline_responses": True})
        self.backend = SimpleNamespace(jobs=self.store)
        self.chunk_id = self.store.chunk_ids(self.job.job_id)[0]
        self.raw_path = Path(self.temporary.name) / "response.json"

    @staticmethod
    def validation_context(validator=strict_fixture_validator):
        return None, None, {}, validator

    def test_valid_response_gets_one_attempt_and_repeat_is_idempotent(self):
        raw_text = json.dumps(
            {"cards": [{"term": "猴"}]},
            ensure_ascii=False)
        self.raw_path.write_text(raw_text, encoding="utf-8")

        with patch.object(
                manual_source_generation,
                "_job_validation_context",
                return_value=self.validation_context()):
            snapshot, created = (
                manual_source_generation.ingest_manual_response(
                    self.backend,
                    self.job.job_id,
                    self.chunk_id,
                    self.raw_path))
            repeated, repeated_created = (
                manual_source_generation.ingest_manual_response(
                    self.backend,
                    self.job.job_id,
                    self.chunk_id,
                    self.raw_path))

        self.assertTrue(created)
        self.assertFalse(repeated_created)
        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(repeated.overall_status, "completed")
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            self.chunk_id)
        self.assertEqual(inspection["status"]["attempts"], 1)
        self.assertEqual(inspection["status"]["worker"], "manual-offline")
        self.assertEqual(inspection["attempts"][0]["raw_text"], raw_text)
        self.assertEqual(
            inspection["attempts"][0]["validated"],
            {"cards": [{"term": "猴"}]})
        combined = json.loads(
            (self.job.path / "combined.json").read_text(encoding="utf-8"))
        self.assertTrue(combined["complete"])
        self.assertEqual(combined["cards"], [{"term": "猴"}])

    def test_invalid_draft_does_not_consume_an_attempt(self):
        self.raw_path.write_text('{"cards":[]}', encoding="utf-8")

        with patch.object(
                manual_source_generation,
                "_job_validation_context",
                return_value=self.validation_context()), self.assertRaisesRegex(
                    ValueError,
                    "invalid manual response"):
            manual_source_generation.ingest_manual_response(
                self.backend,
                self.job.job_id,
                self.chunk_id,
                self.raw_path)

        status = self.store.chunk_status(
            self.job.job_id,
            self.chunk_id)
        self.assertEqual(status["status"], "pending")
        self.assertEqual(status["attempts"], 0)

    def test_valid_revision_preserves_old_attempt_and_rebuilds_combined(self):
        first_text = json.dumps(
            {"cards": [{"term": "猴"}]},
            ensure_ascii=False)
        revised_text = json.dumps(
            {"cards": [{"term": "猿"}]},
            ensure_ascii=False)
        self.raw_path.write_text(first_text, encoding="utf-8")

        def revision_validator(raw_text, _chunk):
            return json.loads(raw_text)

        with patch.object(
                manual_source_generation,
                "_job_validation_context",
                return_value=self.validation_context(revision_validator)):
            manual_source_generation.ingest_manual_response(
                self.backend,
                self.job.job_id,
                self.chunk_id,
                self.raw_path)
            self.raw_path.write_text(revised_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replace-succeeded"):
                manual_source_generation.ingest_manual_response(
                    self.backend,
                    self.job.job_id,
                    self.chunk_id,
                    self.raw_path)
            snapshot, created = (
                manual_source_generation.ingest_manual_response(
                    self.backend,
                    self.job.job_id,
                    self.chunk_id,
                    self.raw_path,
                    replace_succeeded=True))

        self.assertTrue(created)
        self.assertEqual(snapshot.overall_status, "completed")
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            self.chunk_id)
        self.assertEqual(inspection["status"]["attempts"], 2)
        self.assertEqual(
            inspection["attempts"][0]["raw_text"],
            first_text)
        self.assertEqual(
            inspection["attempts"][1]["raw_text"],
            revised_text)
        combined = json.loads(
            (self.job.path / "combined.json").read_text(encoding="utf-8"))
        self.assertEqual(combined["cards"], [{"term": "猿"}])

    def test_missing_reports_predictable_response_path(self):
        snapshot, missing = manual_source_generation.missing_chunks(
            self.backend,
            self.job.job_id)

        self.assertEqual(snapshot.overall_status, "ready")
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["chunk_id"], self.chunk_id)
        self.assertEqual(missing[0]["terms"], ["猴"])
        self.assertTrue(
            missing[0]["response_path"].endswith(
                f"manual_responses/{self.chunk_id}.json"))

    def test_incomplete_job_cannot_reach_the_finalizer(self):
        with patch.object(
                manual_source_generation,
                "SourceWorkflowController",
                side_effect=AssertionError("finalizer instantiated")):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                manual_source_generation.finalize_job(
                    self.backend,
                    self.job.job_id,
                    confirm_import=True)

    def test_manual_job_is_never_resumed_through_a_paid_client(self):
        manual_job = self.store.create(
            one_chunk_plan(),
            request_metadata={"manual_offline_responses": True})
        factory = Mock(side_effect=AssertionError("client constructed"))
        workflow_backend = SimpleNamespace(
            jobs=self.store,
            exclusion_resolver=lambda *_args: ())
        controller = source_workflow.SourceWorkflowController(
            backend=workflow_backend,
            openai_client_factory=factory)

        targets = controller._safe_resume_targets()

        self.assertNotIn(manual_job.job_id, targets)
        with self.assertRaisesRegex(PermissionError, "manually authored"):
            controller._client_for_job(manual_job.job_id)
        factory.assert_not_called()

    def test_complete_job_uses_existing_finalizer_with_forbidden_client(self):
        raw_text = json.dumps(
            {"cards": [{"term": "猴"}]},
            ensure_ascii=False)
        self.raw_path.write_text(raw_text, encoding="utf-8")
        with patch.object(
                manual_source_generation,
                "_job_validation_context",
                return_value=self.validation_context()):
            manual_source_generation.ingest_manual_response(
                self.backend,
                self.job.job_id,
                self.chunk_id,
                self.raw_path)
        controller = Mock()
        controller._pipeline_for_job.return_value = "pipeline"
        controller._finalize.return_value = {"state": "imported"}

        with patch.object(
                manual_source_generation,
                "SourceWorkflowController",
                return_value=controller) as constructor:
            result = manual_source_generation.finalize_job(
                self.backend,
                self.job.job_id,
                confirm_import=True)

        self.assertEqual(result, {"state": "imported"})
        self.assertIs(
            constructor.call_args.kwargs["openai_client_factory"],
            manual_source_generation._forbid_openai_client)
        controller._require_enhanced_audio_ready.assert_called_once_with(
            "pipeline")
        controller._finalize.assert_called_once_with(
            self.job.job_id,
            "pipeline")

    def test_journey_request_uses_full_ming_v10_pipeline_without_exclusions(
            self):
        preferences = {
            "source_card_directions": [
                "context",
                "word_to_meaning",
                "meaning_to_word",
            ],
            "source_chunk_size": "30",
            "source_concurrency": "8",
            "source_request_stagger_ms": "100",
            "source_model_key": "gpt-5.4-mini",
            "source_protocol_key": "v10",
            "source_reasoning_key": "low",
            "source_execution_key": "standard",
            "source_automatic_repair": True,
            "source_context_key": "sentence",
            "source_use_source_examples": True,
            "source_include_context_nuance": True,
            "source_separate_decks": False,
        }

        request = manual_source_generation.build_journey_request(
            (pipeline_store.default_pipeline(),),
            preferences)

        self.assertEqual(request["source_key"], "journey_to_the_west")
        self.assertEqual(
            request["source_language_key"],
            "classical_chinese_ming")
        self.assertEqual(request["request_protocol"], "v10")
        self.assertIsNone(request["source_prefix_token_limit"])
        self.assertFalse(request["exclude_anki"])
        self.assertEqual(request["anki_exclusions"], ())
        self.assertEqual(
            request["pipeline"].language_key,
            "classical_chinese_ming")


if __name__ == "__main__":
    unittest.main()
