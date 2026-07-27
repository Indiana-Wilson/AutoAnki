import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pipeline_store
import process_text
from source_generation import (
    ContextMode,
    GenerationChunk,
    PaidDispatchControl,
    SourceGenerationBackend,
    SourceGenerationConfig,
    build_source_request_contract,
    plan_source_generation,
    render_chunk_input,
)
from source_generation.translation_memory import (
    SourceContextTranslationMemory,
)
from source_generation.repair import (
    CompactRepairScope,
    build_compact_repair_chunk,
    merge_compact_repair,
)
import source_workflow
from tests.test_source_generation import (
    detailed_context_classical_pipeline,
    make_source,
)


class SequentialResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raw_text = self.outputs.pop(0)
        number = len(self.calls)
        return SimpleNamespace(
            id=f"resp-auto-{number}",
            model=kwargs["model"],
            status="completed",
            service_tier="default",
            output_text=raw_text,
            output=(),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=0,
                    cache_write_tokens=0),
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=0),
            ),
        )


class AutomaticExampleRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=root / "corpora",
            jobs_root=root / "jobs",
            translation_memory=SourceContextTranslationMemory(
                root / "translation-memory"))
        self.pipeline = detailed_context_classical_pipeline()
        source = make_source(section_texts=("知道。",))
        self.plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                concurrency=1,
                request_stagger_ms=0,
                request_protocol="v10",
                reasoning_effort="none",
                automatic_repair=True,
                max_automatic_repairs=3,
                retain_excluded_source_contexts=True))
        self.chunk = self.plan.chunks[0]
        self.contract = build_source_request_contract(
            self.pipeline,
            chunks=self.plan.chunks,
            use_source_for_example_sentences=True,
            protocol_version=10,
            reasoning_effort="none",
            translation_memory_enabled=True)
        # Exercise compatibility with saved pre-refactor v10 jobs whose
        # frozen schema still requested generated examples for additional
        # senses. New source-sentence contracts deliberately omit them.
        self.contract["response_format"] = (
            process_text.build_compact_source_response_format(
                self.pipeline,
                protocol_version=10))
        self.contract["response_formats_by_chunk"] = {
            chunk.chunk_id: (
                process_text.build_compact_source_response_format(
                    self.pipeline,
                    protocol_version=10,
                    chunk=chunk,
                    translation_memory_enabled=True))
            for chunk in self.plan.chunks
        }
        self.job = self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
                "automatic_repair": True,
                "max_automatic_repairs": 3,
                "translation_memory_by_chunk": {
                    self.chunk.chunk_id: {},
                },
            },
            request_contract=self.contract)

    def payload(self):
        results = []
        for word in self.chunk.words:
            result = {
                "rank": word.rank,
                "contextual_sense": {
                    "Translation (English)": (
                        "know"
                        if word.surface == "知"
                        else "way"),
                    "Dictionary Meaning (English)": (
                        "The contextual meaning."),
                },
                "additional_senses": [],
            }
            if word.surface == "知":
                result["additional_senses"] = [{
                    "Translation (English)": "understand",
                    "Dictionary Meaning (English)": (
                        "To comprehend something."),
                    "Sentences": [
                        "明白此理。",
                        "知其所以。",
                        "知而後行。",
                    ],
                    "Sentence Translations (English)": [
                        "Understand this principle.",
                        "Know the reason for it.",
                        "Know, and then act.",
                    ],
                }]
            results.append(result)
        return {
            "term_results": results,
            "source_context_translations": [{
                "context_id": self.chunk.contexts[0].context_id,
                "translation": "To know the Way.",
            }],
        }

    def test_new_v10_dispatches_bounded_schema_but_legacy_chunk_stays_fixed(
            self):
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        current = controller._request_options(
            self.contract,
            self.chunk,
            {})
        legacy_mapping = self.chunk.to_dict()
        legacy_mapping.pop("source_context_translation_ids")
        legacy_chunk = GenerationChunk.from_dict(legacy_mapping)
        legacy = controller._request_options(
            self.contract,
            legacy_chunk,
            {})

        self.assertEqual(
            current["text"]["format"],
            self.contract["response_formats_by_chunk"][
                self.chunk.chunk_id])
        term_results = current["text"]["format"]["schema"]["properties"][
            "term_results"]
        self.assertEqual(
            (term_results["minItems"], term_results["maxItems"]),
            (len(self.chunk.words), len(self.chunk.words)))
        self.assertEqual(
            legacy["text"]["format"],
            self.contract["response_format"])

    def test_all_known_context_only_dispatch_has_zero_term_bounds(self):
        source = make_source(section_texts=("知道。",))
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_protocol="v10",
                reasoning_effort="none",
                excluded_words=("知", "道"),
                retain_excluded_source_contexts=True))
        chunk = plan.chunks[0]
        contract = build_source_request_contract(
            self.pipeline,
            chunks=plan.chunks,
            use_source_for_example_sentences=True,
            protocol_version=10,
            reasoning_effort="none")

        options = source_workflow.SourceWorkflowController(
            backend=self.backend)._request_options(
                contract,
                chunk,
                {})
        term_results = options["text"]["format"]["schema"]["properties"][
            "term_results"]

        self.assertEqual(
            (term_results["minItems"], term_results["maxItems"]),
            (0, 0))
        self.assertEqual(
            options["text"]["format"],
            contract["response_formats_by_chunk"][chunk.chunk_id])

    def test_non_owner_lexical_repair_keeps_context_as_input_only(self):
        source = make_source(section_texts=("知道。",))
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=1,
                context_mode=ContextMode.SENTENCE,
                request_protocol="v10",
                reasoning_effort="none",
                retain_excluded_source_contexts=True))
        chunk = plan.chunks[1]
        context_id = chunk.contexts[0].context_id
        self.assertEqual(chunk.requested_source_context_ids, ())
        scope = CompactRepairScope(
            replace_ranks=(chunk.words[0].rank,),
            replace_context_ids=(),
            request_ranks=(chunk.words[0].rank,),
            request_context_ids=(context_id,))

        repair_chunk = build_compact_repair_chunk(
            chunk,
            scope)
        response_format = process_text.build_compact_source_response_format(
            self.pipeline,
            protocol_version=10,
            chunk=repair_chunk,
            source_lexical_only=True,
            include_generated_examples=True)
        translations = response_format["schema"]["properties"][
            "source_context_translations"]
        request_payload = json.loads(render_chunk_input(
            repair_chunk,
            protocol_version=10))
        term_result = self.payload()["term_results"][0]
        term_result["rank"] = chunk.words[0].rank
        merged = json.loads(merge_compact_repair(
            json.dumps({
                "term_results": [],
                "source_context_translations": [],
            }),
            json.dumps({
                "term_results": [term_result],
                "source_context_translations": [],
            }),
            chunk,
            scope))

        self.assertEqual(
            request_payload["contexts"][0]["context_id"],
            context_id)
        self.assertEqual(
            request_payload["source_context_translation_ids"],
            [])
        self.assertEqual(
            (translations["minItems"], translations["maxItems"]),
            (0, 0))
        self.assertEqual(
            merged["source_context_translations"],
            [])

    def test_selective_lexical_repair_preserves_omitted_memory_translation(
            self):
        context_id = self.chunk.requested_source_context_ids[0]
        base = self.payload()
        missing = base["term_results"].pop(0)
        base["source_context_translations"] = []
        rank = missing["rank"]
        scope = CompactRepairScope(
            replace_ranks=(rank,),
            replace_context_ids=(),
            request_ranks=(rank,),
            request_context_ids=(
                next(
                    word.context_id
                    for word in self.chunk.words
                    if word.rank == rank),))

        merged = json.loads(merge_compact_repair(
            json.dumps(base, ensure_ascii=False),
            json.dumps({
                "term_results": [missing],
                "source_context_translations": [],
            }, ensure_ascii=False),
            self.chunk,
            scope,
            remembered_context_ids={context_id}))

        self.assertEqual(
            [item["rank"] for item in merged["term_results"]],
            [word.rank for word in self.chunk.words])
        self.assertEqual(
            merged["source_context_translations"],
            [])

    def test_context_only_invalid_translation_uses_selective_repair(self):
        source = make_source(section_texts=("知道。",))
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_protocol="v10",
                reasoning_effort="none",
                automatic_repair=True,
                max_automatic_repairs=3,
                retain_excluded_source_contexts=True))
        original_chunk = plan.chunks[0]
        chunk = replace(
            original_chunk,
            words=(),
            context_anchors=original_chunk.words)
        plan = replace(
            plan,
            chunks=(chunk,))
        contract = build_source_request_contract(
            self.pipeline,
            chunks=plan.chunks,
            use_source_for_example_sentences=True,
            protocol_version=10,
            reasoning_effort="none",
            translation_memory_enabled=True)
        job = self.backend.jobs.create(
            plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
                "automatic_repair": True,
                "max_automatic_repairs": 3,
                "translation_memory_by_chunk": {
                    chunk.chunk_id: {},
                },
            },
            request_contract=contract)
        context_id = chunk.requested_source_context_ids[0]
        responses = SequentialResponses((
            json.dumps({
                "term_results": [],
                "source_context_translations": [{
                    "context_id": context_id,
                    "translation": "知道。",
                }],
            }, ensure_ascii=False),
            json.dumps({
                "term_results": [],
                "source_context_translations": [{
                    "context_id": context_id,
                    "translation": "To know the Way.",
                }],
            }, ensure_ascii=False),
        ))

        snapshot = source_workflow.SourceWorkflowController(
            backend=self.backend)._run_job(
                job.job_id,
                self.pipeline,
                SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 2)
        latest = self.backend.jobs.latest_attempt_path(
            job.job_id,
            chunk.chunk_id)
        scope = json.loads(
            (latest / "repair_scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["replace_ranks"], [])
        self.assertEqual(
            scope["replace_context_ids"],
            [context_id])

    def test_automatic_repair_uses_tiny_pair_call_and_preserves_other_fields(
            self):
        invalid = self.payload()
        repair = {
            "repairs": {
                "p001": {
                    "sentence": "知此理。",
                    "translation": "Understand this principle.",
                },
            },
        }
        responses = SequentialResponses((
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(repair, ensure_ascii=False),
        ))
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        snapshot = controller._run_job(
            self.job.job_id,
            self.pipeline,
            SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 2)
        self.assertLess(
            len(responses.calls[1]["input"]),
            len(responses.calls[0]["input"]))
        latest = self.backend.jobs.latest_attempt_path(
            self.job.job_id,
            self.chunk.chunk_id)
        scope = json.loads(
            (latest / "repair_scope.json").read_text(
                encoding="utf-8"))
        self.assertEqual(
            scope["kind"],
            "compact_v10_example_pair_repair")
        self.assertTrue(scope["automatic"])
        self.assertEqual(scope["dispatch_ordinal"], 1)
        merged = json.loads(
            (latest / "raw.txt").read_text(encoding="utf-8"))
        repaired_sense = next(
            result["additional_senses"][0]
            for result in merged["term_results"]
            if result["additional_senses"])
        self.assertEqual(repaired_sense["Sentences"][0], "知此理。")
        self.assertEqual(
            repaired_sense["Sentences"][1:],
            invalid["term_results"][0]["additional_senses"][0][
                "Sentences"][1:])
        self.assertEqual(
            self.backend.jobs.usage_summary(
                self.job.job_id)["attempt_count"],
            2)

    def test_pause_before_automatic_repair_preserves_retry_allowance(self):
        class PauseAfterFirstDispatch(PaidDispatchControl):
            def __init__(inner_self):
                super().__init__()
                inner_self.completed = 0

            def dispatch(inner_self, operation):
                result = super().dispatch(operation)
                inner_self.completed += 1
                if inner_self.completed == 1:
                    inner_self.pause()
                return result

        responses = SequentialResponses((
            json.dumps(self.payload(), ensure_ascii=False),
        ))
        control = PauseAfterFirstDispatch()
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend,
            paid_dispatch_control=control)

        snapshot = controller._run_job(
            self.job.job_id,
            self.pipeline,
            SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "ready")
        self.assertEqual(len(responses.calls), 1)
        status = self.backend.jobs.chunk_status(
            self.job.job_id,
            self.chunk.chunk_id)
        self.assertEqual(status["status"], "pending")
        latest = self.backend.jobs.latest_attempt_path(
            self.job.job_id,
            self.chunk.chunk_id)
        self.assertFalse(
            (latest / "automatic_repair_dispatch.json").exists())
        self.assertEqual(
            controller._automatic_repair_dispatch_count(
                self.job.job_id,
                self.chunk.chunk_id),
            0)

    def test_repeated_exact_chinese_term_is_accepted_without_repair(
            self):
        invalid = self.payload()
        invalid["term_results"][0]["additional_senses"][0]["Sentences"][0] = (
            "知之則知。")
        responses = SequentialResponses((
            json.dumps(invalid, ensure_ascii=False),
        ))
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        snapshot = controller._run_job(
            self.job.job_id,
            self.pipeline,
            SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 1)
        latest = self.backend.jobs.latest_attempt_path(
            self.job.job_id,
            self.chunk.chunk_id)
        self.assertFalse((latest / "repair_scope.json").exists())

    def test_exact_translation_memory_is_sent_once_and_provider_omits_it(self):
        context = self.chunk.contexts[0]
        self.backend.translation_memory.commit(
            self.pipeline.language_key,
            context.text,
            "To know the Way.",
            provenance={
                "provenance_id": "validated-fixture",
                "origin": "provider",
                "fully_validated": True,
                "manual_acceptance": False,
            })
        memory = self.backend.translation_memory.lookup_chunks(
            self.pipeline.language_key,
            self.plan.chunks)
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        options = controller._request_options(
            self.contract,
            self.chunk,
            memory)
        request_payload = json.loads(
            options["input"][
                len(self.contract["composed_prompt"]):])
        self.assertEqual(
            request_payload["source_context_translation_memory"],
            {
                context.context_id: "To know the Way.",
            })

        provider = self.payload()
        provider["term_results"][0]["additional_senses"][0][
            "Sentences"][0] = "知此理。"
        provider["source_context_translations"] = []
        validator = controller._response_validator(
            self.pipeline,
            self.contract,
            memory)

        validated = validator(
            json.dumps(provider, ensure_ascii=False),
            self.chunk)

        contextual_cards = [
            card
            for card in validated["cards"]
            if (
                isinstance(card.get("Sentences"), str)
                and "|" not in card["Sentences"])
        ]
        self.assertFalse(contextual_cards)
        self.assertEqual(
            validated["source_contexts"][0]["english_translation"],
            "To know the Way.")

    def test_missing_rank_uses_bounded_automatic_selective_repair(self):
        invalid = self.payload()
        invalid["term_results"][0]["additional_senses"][0][
            "Sentences"][0] = "知此理。"
        missing_result = invalid["term_results"].pop()
        repair = {
            "term_results": [missing_result],
            "source_context_translations": [],
        }
        responses = SequentialResponses((
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(repair, ensure_ascii=False),
        ))
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        snapshot = controller._run_job(
            self.job.job_id,
            self.pipeline,
            SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 2)
        latest = self.backend.jobs.latest_attempt_path(
            self.job.job_id,
            self.chunk.chunk_id)
        scope = json.loads(
            (latest / "repair_scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["kind"], "compact_v10_selective_repair")
        self.assertTrue(scope["automatic"])
        self.assertEqual(scope["dispatch_ordinal"], 1)

    def test_lexical_free_source_translation_uses_automatic_context_repair(
            self):
        contract = build_source_request_contract(
            self.pipeline,
            chunks=self.plan.chunks,
            use_source_for_example_sentences=True,
            protocol_version=10,
            reasoning_effort="none")
        job = self.backend.jobs.create(
            self.plan,
            request_metadata={
                "pipeline": pipeline_store.pipeline_to_mapping(
                    self.pipeline),
                "automatic_repair": True,
                "max_automatic_repairs": 3,
                "translation_memory_by_chunk": {
                    self.chunk.chunk_id: {},
                },
            },
            request_contract=contract)
        term_results = [
            {
                "rank": word.rank,
                "contextual_sense": {},
                "additional_senses": [],
            }
            for word in self.chunk.words
        ]
        context_id = self.chunk.contexts[0].context_id
        invalid = {
            "term_results": term_results,
            "source_context_translations": [{
                "context_id": context_id,
                "translation": "",
            }],
        }
        repair = {
            "term_results": term_results,
            "source_context_translations": [{
                "context_id": context_id,
                "translation": "To know the Way.",
            }],
        }
        responses = SequentialResponses((
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(repair, ensure_ascii=False),
        ))
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        snapshot = controller._run_job(
            job.job_id,
            self.pipeline,
            SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(responses.calls), 2)
        latest = self.backend.jobs.latest_attempt_path(
            job.job_id,
            self.chunk.chunk_id)
        scope = json.loads(
            (latest / "repair_scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["replace_context_ids"], [context_id])
        self.assertTrue(scope["automatic"])

    def test_automatic_pair_repair_stops_when_three_dispatches_are_retained(
            self):
        invalid = self.payload()
        responses = SequentialResponses((
            json.dumps(invalid, ensure_ascii=False),
        ))
        controller = source_workflow.SourceWorkflowController(
            backend=self.backend)

        with patch.object(
                controller,
                "_automatic_repair_dispatch_count",
                return_value=3):
            snapshot = controller._run_job(
                self.job.job_id,
                self.pipeline,
                SimpleNamespace(responses=responses))

        self.assertEqual(snapshot.overall_status, "completed_with_failures")
        self.assertEqual(len(responses.calls), 1)


if __name__ == "__main__":
    unittest.main()
