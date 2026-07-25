import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation import (
    ContextMode,
    SourceGenerationBackend,
    SourceGenerationConfig,
    build_source_request_contract,
    inspect_pipeline_response,
    plan_source_generation,
)
import source_workflow
from tests.test_source_generation import (
    context_classical_pipeline,
    make_source,
)


class FakeResponses:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-selective-1",
            model="gpt-5.4-mini",
            status="completed",
            service_tier="default",
            output_text=self.raw_text,
            output=(),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=20),
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=0),
            ),
        )


class CompactRepairWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=root / "corpora",
            jobs_root=root / "jobs")
        self.pipeline = context_classical_pipeline()
        source = make_source(section_texts=("道可道。",))
        self.plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                concurrency=1,
                request_stagger_ms=0),
        )
        self.chunk = self.plan.chunks[0]
        self.contract = build_source_request_contract(
            self.pipeline,
            chunks=self.plan.chunks,
            use_source_for_example_sentences=True,
            reasoning_effort="none")
        self.job = self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
            },
            request_contract=self.contract)

    def payload(self):
        translation_field = process_text.get_response_field_names(
            self.pipeline)[3]
        return {
            process_text.SOURCE_TERM_RESULTS_KEY: [
                {
                    process_text.SOURCE_RANK_FIELD_NAME: word.rank,
                    process_text.SOURCE_CONTEXTUAL_SENSE_KEY: {
                        translation_field: (
                            "way"
                            if word.surface == "道"
                            else "can"),
                    },
                    process_text.SOURCE_ADDITIONAL_SENSES_KEY: [],
                }
                for word in self.chunk.words
            ],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [{
                process_text.SOURCE_CONTEXT_ID_FIELD_NAME: (
                    self.chunk.contexts[0].context_id),
                process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: (
                    "The way that can be spoken."),
            }],
        }

    def test_retry_requests_only_missing_rank_and_persists_full_merge(self):
        complete = self.payload()
        missing_rank = self.chunk.words[1].rank
        base = {
            **complete,
            process_text.SOURCE_TERM_RESULTS_KEY: [
                item
                for item in complete[
                    process_text.SOURCE_TERM_RESULTS_KEY]
                if item[process_text.SOURCE_RANK_FIELD_NAME] != missing_rank
            ],
        }
        attempt, attempt_path = self.backend.jobs.begin_attempt(
            self.job.job_id,
            self.chunk.chunk_id)
        base_text = json.dumps(base, ensure_ascii=False)
        self.backend.jobs.write_raw(attempt_path, base_text)
        report = inspect_pipeline_response(
            base_text,
            self.pipeline,
            self.chunk,
            use_compact_source_results=True)
        error = process_text.GeneratedCardValidationError(
            "Missing compact rank.")
        error.validation_report = report
        error_record = self.backend.jobs.write_error(
            attempt_path,
            error,
            transient=False)
        self.backend.jobs._set_chunk_status(
            self.job.job_id,
            self.chunk.chunk_id,
            status="invalid_response",
            last_error=error_record)
        self.backend.jobs.reset_for_manual_retry(
            self.job.job_id,
            (self.chunk.chunk_id,))

        repair = {
            process_text.SOURCE_TERM_RESULTS_KEY: [
                item
                for item in complete[
                    process_text.SOURCE_TERM_RESULTS_KEY]
                if item[process_text.SOURCE_RANK_FIELD_NAME] == missing_rank
            ],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: complete[
                process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY],
        }
        responses = FakeResponses(
            json.dumps(repair, ensure_ascii=False))
        client = SimpleNamespace(responses=responses)
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        snapshot = controller._run_standard_retry_targets(
            self.job.job_id,
            (self.chunk.chunk_id,),
            self.pipeline,
            client)

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 1)
        request_payload = json.loads(
            responses.calls[0]["input"][
                len(self.contract["composed_prompt"]):])
        self.assertEqual(
            [word["rank"] for word in request_payload["words"]],
            [missing_rank])
        latest = self.backend.jobs.latest_attempt_path(
            self.job.job_id,
            self.chunk.chunk_id)
        self.assertTrue((latest / "repair_scope.json").is_file())
        self.assertEqual(
            json.loads((latest / "response.json").read_text(
                encoding="utf-8"))["raw_text"],
            json.dumps(repair, ensure_ascii=False))
        merged = json.loads(
            (latest / "raw.txt").read_text(encoding="utf-8"))
        self.assertEqual(
            [
                item["rank"]
                for item in merged[
                    process_text.SOURCE_TERM_RESULTS_KEY]
            ],
            [word.rank for word in self.chunk.words])
        self.assertEqual(
            self.backend.jobs.usage_summary(
                self.job.job_id)["input_tokens"],
            100)


if __name__ == "__main__":
    unittest.main()
