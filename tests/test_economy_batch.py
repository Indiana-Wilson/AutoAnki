import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
from source_generation import (
    SourceGenerationBackend,
    build_source_request_contract,
)
import source_workflow
from tests.test_source_workflow import one_word_plan, source_pipeline


class FakeFiles:
    def __init__(self, result_text=None):
        self.result_text = result_text
        self.create_calls = []
        self.content_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="file-economy-input")

    def content(self, file_id):
        self.content_calls.append(file_id)
        if self.result_text is None:
            raise AssertionError("No remote result download was expected.")
        return SimpleNamespace(text=self.result_text)


class FakeBatches:
    def __init__(
            self,
            *,
            create_error=None,
            created_batch=None,
            listed=(),
            retrieved_batch=None):
        self.create_error = create_error
        self.created_batch = created_batch
        self.listed = tuple(listed)
        self.retrieved_batch = retrieved_batch
        self.create_calls = []
        self.list_calls = []
        self.retrieve_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return SimpleNamespace(data=list(self.listed))

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        if self.created_batch is not None:
            return self.created_batch
        return SimpleNamespace(
            id="batch-economy-1",
            status="validating",
            endpoint=kwargs["endpoint"],
            input_file_id=kwargs["input_file_id"],
            output_file_id=None,
            error_file_id=None,
            metadata=kwargs["metadata"],
            request_counts=SimpleNamespace(
                total=0,
                completed=0,
                failed=0))

    def retrieve(self, batch_id):
        self.retrieve_calls.append(batch_id)
        if self.retrieved_batch is None:
            raise AssertionError("No Batch retrieval was expected.")
        return self.retrieved_batch


class FakeClient:
    def __init__(self, *, files=None, batches=None):
        self.files = files or FakeFiles()
        self.batches = batches or FakeBatches()


class EconomyBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=self.root / "corpora",
            jobs_root=self.root / "jobs")
        self.pipeline = source_pipeline()
        base_plan = one_word_plan()
        self.plan = replace(
            base_plan,
            plan_id="economy-plan",
            config=replace(
                base_plan.config,
                execution_mode="economy",
                request_protocol="v9",
                reasoning_effort="none"))
        self.contract = build_source_request_contract(
            self.pipeline,
            chunks=self.plan.chunks,
            protocol_version=9,
            reasoning_effort="none")
        self.snapshot = self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
                "request_contract": self.contract,
            },
            request_contract=self.contract)
        self.controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

    @property
    def job_id(self):
        return self.snapshot.job_id

    @property
    def chunk_id(self):
        return self.plan.chunks[0].chunk_id

    def valid_output(self):
        return json.dumps({
            "cards": [{
                "Classical Chinese": "甲",
                "Translation (English)": "first",
            }],
        })

    def batch_result_line(self, *, custom_id="chunk-00001"):
        body = {
            "id": "resp-economy-1",
            "status": "completed",
            "model": "gpt-5.4-mini",
            "service_tier": "default",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": self.valid_output(),
                    "annotations": [],
                }],
            }],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 20,
                },
                "output_tokens": 40,
                "output_tokens_details": {
                    "reasoning_tokens": 5,
                },
                "total_tokens": 140,
            },
        }
        return json.dumps({
            "id": "batch-request-1",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": "request-1",
                "body": body,
            },
            "error": None,
        }) + "\n"

    def submitted_client(self, *, result_text=None):
        completed = SimpleNamespace(
            id="batch-economy-1",
            status="completed",
            endpoint="/v1/responses",
            input_file_id="file-economy-input",
            output_file_id="file-economy-output",
            error_file_id=None,
            metadata={
                "autoanki_job_id": self.job_id[:64],
                "autoanki_batch": "1",
            },
            request_counts=SimpleNamespace(
                total=1,
                completed=1,
                failed=0))
        return FakeClient(
            files=FakeFiles(result_text),
            batches=FakeBatches(retrieved_batch=completed))

    def test_submit_and_collect_raw_responses_body_with_exact_usage(self):
        client = FakeClient(files=FakeFiles(self.batch_result_line()))

        submitted = self.controller._run_job(
            self.job_id,
            self.pipeline,
            client)

        self.assertEqual(submitted.overall_status, "ready")
        state = self.controller._workflow(
            self.job_id)["economy_batch"]
        self.assertEqual(state["stage"], "submitted")
        self.assertEqual(state["request_count"], 1)
        input_path = Path(state["input_path"])
        self.assertEqual(
            state["input_sha256"],
            source_workflow.hashlib.sha256(
                input_path.read_bytes()).hexdigest())
        self.assertEqual(len(client.files.create_calls), 1)
        self.assertEqual(len(client.batches.create_calls), 1)

        client.batches.retrieved_batch = SimpleNamespace(
            id="batch-economy-1",
            status="completed",
            endpoint="/v1/responses",
            input_file_id="file-economy-input",
            output_file_id="file-economy-output",
            error_file_id=None,
            metadata=state["metadata"],
            request_counts=SimpleNamespace(
                total=1,
                completed=1,
                failed=0))
        completed = self.controller._run_job(
            self.job_id,
            self.pipeline,
            client)

        self.assertEqual(completed.overall_status, "completed")
        inspection = self.backend.jobs.inspect_chunk(
            self.job_id,
            self.chunk_id)
        self.assertEqual(
            inspection["attempts"][0]["raw_text"],
            self.valid_output())
        self.assertEqual(
            self.backend.jobs.usage_summary(self.job_id),
            {
                "attempt_count": 1,
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "cache_write_input_tokens": 0,
                "uncached_input_tokens": 80,
                "output_tokens": 40,
                "reasoning_tokens": 5,
                "visible_output_tokens": 35,
                "total_tokens": 140,
                "web_search_calls": 0,
            })

    def test_ambiguous_create_is_reconciled_without_a_second_create(self):
        first = FakeClient(
            batches=FakeBatches(
                create_error=ConnectionError("response lost")))

        with self.assertRaises(ConnectionError):
            self.controller._run_job(
                self.job_id,
                self.pipeline,
                first)

        state = self.controller._workflow(
            self.job_id)["economy_batch"]
        self.assertEqual(state["stage"], "uploaded")
        self.assertEqual(state["status"], "submission_unknown")
        first_headers = first.batches.create_calls[0][
            "extra_headers"]
        self.assertEqual(
            first_headers["Idempotency-Key"],
            state["idempotency_key"])

        matching_batch = SimpleNamespace(
            id="batch-reconciled",
            status="in_progress",
            endpoint="/v1/responses",
            input_file_id=state["input_file_id"],
            output_file_id=None,
            error_file_id=None,
            metadata=state["metadata"],
            request_counts=SimpleNamespace(
                total=1,
                completed=0,
                failed=0))
        second = FakeClient(
            batches=FakeBatches(listed=(matching_batch,)))

        resumed = self.controller._run_job(
            self.job_id,
            self.pipeline,
            second)

        self.assertEqual(resumed.overall_status, "ready")
        self.assertEqual(len(second.batches.create_calls), 0)
        recovered_state = self.controller._workflow(
            self.job_id)["economy_batch"]
        self.assertEqual(
            recovered_state["batch_id"],
            "batch-reconciled")
        self.assertEqual(recovered_state["stage"], "submitted")

    def test_orphaned_prepared_input_is_reused_only_when_hash_matches(self):
        client = FakeClient()
        with (
                patch.object(
                    self.controller,
                    "_update_workflow",
                    side_effect=RuntimeError("simulated crash")),
                self.assertRaises(RuntimeError)):
            self.controller._submit_economy_batch(
                self.job_id,
                client,
                self.contract,
                (self.chunk_id,))
        self.assertEqual(len(client.files.create_calls), 0)
        input_path = (
            self.snapshot.path
            / "economy"
            / "batch_0001"
            / "input.jsonl")
        retained = input_path.read_bytes()

        self.controller._submit_economy_batch(
            self.job_id,
            client,
            self.contract,
            (self.chunk_id,))

        self.assertEqual(input_path.read_bytes(), retained)
        self.assertEqual(len(client.files.create_calls), 1)

    def test_collection_prefers_retained_file_and_rejects_unknown_ids(self):
        initial = FakeClient()
        self.controller._run_job(
            self.job_id,
            self.pipeline,
            initial)
        state = self.controller._workflow(
            self.job_id)["economy_batch"]
        output_path = Path(state["input_path"]).parent / "output.jsonl"
        output_path.write_text(
            self.batch_result_line(custom_id="not-ours"),
            encoding="utf-8")
        collecting = self.submitted_client()

        with self.assertRaisesRegex(
                ValueError,
                "unknown Economy Batch custom_id"):
            self.controller._run_job(
                self.job_id,
                self.pipeline,
                collecting)

        self.assertEqual(collecting.files.content_calls, [])
        self.assertEqual(
            self.backend.jobs.chunk_status(
                self.job_id,
                self.chunk_id)["attempts"],
            0)

    def test_input_limits_are_enforced_before_upload(self):
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "50,000"):
            self.controller._submit_economy_batch(
                self.job_id,
                client,
                self.contract,
                tuple("chunk" for _ in range(50_001)))
        self.assertEqual(client.files.create_calls, [])

        with (
                patch.object(
                    source_workflow,
                    "_ECONOMY_MAX_INPUT_BYTES",
                    10),
                self.assertRaisesRegex(ValueError, "200 MB")):
            self.controller._submit_economy_batch(
                self.job_id,
                client,
                self.contract,
                (self.chunk_id,))
        self.assertEqual(client.files.create_calls, [])

    def test_inconsistent_created_batch_is_retained_and_blocks_resubmit(self):
        invalid = SimpleNamespace(
            id="batch-wrong-endpoint",
            status="validating",
            endpoint="/v1/chat/completions",
            input_file_id="file-economy-input",
            output_file_id=None,
            error_file_id=None,
            metadata={},
            request_counts=None)
        client = FakeClient(
            batches=FakeBatches(created_batch=invalid))

        with self.assertRaisesRegex(ValueError, "unexpected endpoint"):
            self.controller._run_job(
                self.job_id,
                self.pipeline,
                client)

        state = self.controller._workflow(
            self.job_id)["economy_batch"]
        self.assertTrue(state["submission_blocked"])
        self.assertEqual(
            state["observed_batch"]["batch_id"],
            "batch-wrong-endpoint")
        with self.assertRaisesRegex(
                RuntimeError,
                "must be inspected"):
            self.controller._run_job(
                self.job_id,
                self.pipeline,
                client)
        self.assertEqual(len(client.batches.create_calls), 1)


if __name__ == "__main__":
    unittest.main()
