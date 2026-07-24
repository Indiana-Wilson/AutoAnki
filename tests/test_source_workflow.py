import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anki_integration
import codex_source_retrieval
import pipeline_store
from source_generation import (
    build_source_request_contract,
    ContextMode,
    GenerationChunk,
    GenerationPlan,
    GenerationWord,
    SourceGenerationBackend,
    SourceGenerationConfig,
)
import source_workflow


def source_pipeline():
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
                enabled=card.direction_key == "word_to_meaning",
                fields=fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese")


def one_word_plan():
    config = SourceGenerationConfig(
        source_key="fixture",
        chunk_size=1,
        context_mode=ContextMode.NONE,
        concurrency=1,
        request_stagger_ms=0,
        max_transient_retries=0)
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
                context_id=None),
        ),
        contexts=())
    return GenerationPlan(
        plan_id="fixture-plan",
        source_key="fixture",
        source_title="Fixture",
        source_build_id="fixture-build",
        source_run_path="/fixture/build",
        config=config,
        original_word_count=1,
        excluded_word_count=0,
        chunks=(chunk,))


def empty_plan():
    return replace(
        one_word_plan(),
        plan_id="empty-plan",
        original_word_count=0,
        chunks=())


class FakeResponses:
    def __init__(self, output_texts):
        self.output_texts = list(output_texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output_text=self.output_texts.pop(0),
            output=())


class FakeOpenAIClient:
    def __init__(self, output_texts):
        self.responses = FakeResponses(output_texts)


class PreparedFixture:
    title = "Prepared Fixture"
    unique_word_count = 2
    section_count = 1

    def to_dict(self):
        return {
            "source_key": "custom-prepared",
            "title": self.title,
            "unique_word_count": self.unique_word_count,
            "section_count": self.section_count,
        }


class SourceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = SourceGenerationBackend(
            corpus_root=self.root / "corpora",
            jobs_root=self.root / "jobs")
        self.pipeline = source_pipeline()

    def install_plan(self, plan):
        def create_job(request):
            return self.backend.jobs.create(
                plan,
                request_metadata={
                    "pipeline": pipeline_store.pipeline_to_mapping(
                        request["pipeline"]),
                })

        self.backend.create_job = create_job

    def controller(self, **kwargs):
        return source_workflow.SourceWorkflowController(
            backend=self.backend,
            **kwargs)

    def valid_output(self):
        return json.dumps({
            "cards": [{
                "Classical Chinese": "甲",
                "Translation (English)": "first",
            }],
        })

    def test_gui_hooks_expose_preview_and_manual_filter_callbacks(self):
        controller = self.controller()
        hooks = controller.gui_hooks()

        self.assertIs(hooks["source_preview_loader"].__self__, controller)
        self.assertIs(
            hooks["manual_input_filter_callback"].__self__,
            controller)

    def test_paid_request_passes_bounded_optional_web_search_to_api(self):
        client = FakeOpenAIClient((self.valid_output(),))
        chunk = one_word_plan().chunks[0]
        contract = build_source_request_contract(
            self.pipeline,
            chunks=(chunk,),
            allow_web_search=True)
        request = self.controller()._paid_request_callable(
            self.pipeline,
            client,
            contract)

        request(chunk)

        call = client.responses.calls[0]
        self.assertEqual(call["tools"], [{"type": "web_search"}])
        self.assertEqual(call["tool_choice"], "auto")
        self.assertEqual(call["max_tool_calls"], 1)

    def test_empty_plan_makes_no_client_request_package_or_import(self):
        self.install_plan(empty_plan())
        client_factory = MagicMock()
        package_creator = MagicMock()
        package_importer = MagicMock()
        controller = self.controller(
            openai_client_factory=client_factory,
            package_creator=package_creator,
            package_importer=package_importer)

        result = controller.generate({
            "pipeline": self.pipeline,
            "paid_confirmed": True,
        })

        self.assertIn("No new vocabulary", result["message"])
        client_factory.assert_not_called()
        package_creator.assert_not_called()
        package_importer.assert_not_called()
        self.assertEqual(
            controller._workflow(result["job_id"])["state"],
            "nothing_to_generate")

    def test_fresh_exclusion_to_zero_bypasses_stale_paid_authorization_safely(
            self):
        self.backend._plan = MagicMock(side_effect=(
            (None, one_word_plan()),
            (None, empty_plan()),
        ))
        request = {
            "source_key": "fixture",
            "pipeline": self.pipeline,
            "exclude_anki": True,
            "anki_exclusion": {
                "deck": "Known",
                "model": "Vocabulary",
                "field": "Word",
            },
        }
        estimate = self.backend.estimate(request)
        client_factory = MagicMock()
        package_creator = MagicMock()
        package_importer = MagicMock()
        controller = self.controller(
            openai_client_factory=client_factory,
            package_creator=package_creator,
            package_importer=package_importer)

        with patch.object(
                controller,
                "_resolve_anki_exclusion",
                return_value=frozenset({"甲"})) as resolver:
            result = controller.generate({
                **request,
                "estimate": estimate,
                "paid_confirmed": True,
            })

        self.assertTrue(result["no_new_vocabulary"])
        self.assertIn("No new vocabulary", result["message"])
        self.assertTrue(resolver.call_args.kwargs["force_refresh"])
        client_factory.assert_not_called()
        package_creator.assert_not_called()
        package_importer.assert_not_called()
        self.assertEqual(
            controller._workflow(result["job_id"])["state"],
            "nothing_to_generate")

    def test_one_paid_request_is_persisted_packaged_and_imported_in_place(self):
        self.install_plan(one_word_plan())
        client = FakeOpenAIClient((self.valid_output(),))

        def create_package(_combined, **kwargs):
            kwargs["output_path"].write_bytes(b"apkg")
            self.assertEqual(kwargs["source_title"], "Fixture")
            return kwargs["output_path"], 1

        package_creator = MagicMock(side_effect=create_package)
        package_importer = MagicMock(return_value=True)
        controller = self.controller(
            openai_client_factory=lambda _key: client,
            package_creator=package_creator,
            package_importer=package_importer)

        with patch.object(
                source_workflow.process_text,
                "get_api_key",
                return_value="test-key"):
            result = controller.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
            })

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], source_workflow.SOURCE_MODEL)
        self.assertEqual(call["reasoning"], {"effort": "none"})
        self.assertIn('"term":"甲"', call["input"])
        self.assertGreaterEqual(call["max_output_tokens"], 1_024)
        self.assertLessEqual(
            call["max_output_tokens"],
            source_workflow.MODEL_MAX_OUTPUT_TOKENS)
        package_creator.assert_called_once()
        package_importer.assert_called_once()
        workflow = controller._workflow(result["job_id"])
        self.assertEqual(workflow["state"], "imported")
        self.assertEqual(workflow["deck_name"], "Vocabulary from Fixture")
        rows = controller.job_rows()["jobs"]
        self.assertEqual(
            next(
                row["status"]
                for row in rows
                if row["chunk_label"] == "Deck"),
            "completed")

    def test_invalid_output_waits_for_explicit_retry_and_reuses_successes(self):
        self.install_plan(one_word_plan())
        client = FakeOpenAIClient((
            json.dumps({"cards": []}),
            self.valid_output(),
        ))

        def create_package(_combined, **kwargs):
            kwargs["output_path"].write_bytes(b"apkg")
            return kwargs["output_path"], 1

        package_importer = MagicMock(return_value=True)
        controller = self.controller(
            openai_client_factory=lambda _key: client,
            package_creator=create_package,
            package_importer=package_importer)

        with patch.object(
                source_workflow.process_text,
                "get_api_key",
                return_value="test-key"):
            first = controller.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
            })

            self.assertEqual(first["status"], "completed_with_failures")
            self.assertTrue(first["requires_attention"])
            self.assertEqual(len(client.responses.calls), 1)
            package_importer.assert_not_called()
            failed_row = next(
                row
                for row in controller.job_rows()["jobs"]
                if row["status"] == "invalid_response")
            inspection = controller.inspect({
                "job_id": failed_row["job_id"],
            })
            self.assertIn(
                '"cards": []',
                inspection["attempts"][0]["raw_text"])

            with patch.object(
                    source_workflow,
                    "build_source_request_contract",
                    side_effect=AssertionError(
                        "A retry rebuilt its frozen request.")):
                controller.retry({
                    "job_ids": (failed_row["job_id"],),
                    "paid_confirmed": True,
                })

        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(
            client.responses.calls[0],
            client.responses.calls[1])
        inspection = controller.inspect({
            "job_id": failed_row["job_id"],
        })
        self.assertEqual(
            inspection["request_contract_origin"],
            "legacy_reconstructed")
        self.assertEqual(
            inspection["request_contract"]["reasoning"],
            {"effort": "none"})
        package_importer.assert_called_once()
        self.assertEqual(
            controller._workflow(first["job_id"])["state"],
            "imported")

    def test_import_failure_is_recoverable_without_another_openai_call(self):
        self.install_plan(one_word_plan())
        client = FakeOpenAIClient((self.valid_output(),))

        def create_package(_combined, **kwargs):
            kwargs["output_path"].write_bytes(b"apkg")
            return kwargs["output_path"], 1

        importer = MagicMock(
            side_effect=(RuntimeError("Anki unavailable"), True))
        controller = self.controller(
            openai_client_factory=lambda _key: client,
            package_creator=create_package,
            package_importer=importer)
        with (
                patch.object(
                    source_workflow.process_text,
                    "get_api_key",
                    return_value="test-key"),
                self.assertRaisesRegex(RuntimeError, "Anki unavailable")):
            controller.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
            })

        finalize_row = next(
            row
            for row in controller.job_rows()["jobs"]
            if row["chunk_label"] == "Deck")
        self.assertEqual(finalize_row["status"], "failed")

        controller.retry({
            "job_ids": (finalize_row["job_id"],),
            "paid_confirmed": True,
        })

        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(importer.call_count, 2)
        self.assertEqual(
            controller._workflow(
                finalize_row["parent_job_id"])["state"],
            "imported")

    def test_completed_chunks_with_missed_finalization_are_recoverable(self):
        self.install_plan(one_word_plan())
        client = FakeOpenAIClient((self.valid_output(),))

        def create_package(_combined, **kwargs):
            kwargs["output_path"].write_bytes(b"apkg")
            return kwargs["output_path"], 1

        package_importer = MagicMock(return_value=True)
        controller = self.controller(
            openai_client_factory=lambda _key: client,
            package_creator=create_package,
            package_importer=package_importer)
        with (
                patch.object(
                    source_workflow.process_text,
                    "get_api_key",
                    return_value="test-key"),
                patch.object(
                    controller,
                    "_finalize_if_complete",
                    return_value=None)):
            generated = controller.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
            })

        finalize_row = next(
            row
            for row in controller.job_rows()["jobs"]
            if row["chunk_label"] == "Deck")
        self.assertEqual(finalize_row["status"], "failed")
        self.assertIn(
            "without repeating OpenAI",
            finalize_row["detail"])

        controller.retry({
            "job_ids": (finalize_row["job_id"],),
            "paid_confirmed": True,
        })

        self.assertEqual(len(client.responses.calls), 1)
        package_importer.assert_called_once()
        self.assertEqual(
            controller._workflow(generated["job_id"])["state"],
            "imported")

    def test_cross_process_finalization_lease_prevents_duplicate_import(self):
        self.install_plan(one_word_plan())
        seed = self.controller(
            openai_client_factory=lambda _key: FakeOpenAIClient((
                self.valid_output(),
            )))
        with (
                patch.object(
                    source_workflow.process_text,
                    "get_api_key",
                    return_value="test-key"),
                patch.object(
                    seed,
                    "_finalize_if_complete",
                    return_value=None)):
            generated = seed.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
            })

        package_started = threading.Event()
        release_package = threading.Event()
        first_failures = []

        def create_package(_combined, **kwargs):
            package_started.set()
            if not release_package.wait(timeout=2):
                raise TimeoutError("Test did not release packaging.")
            kwargs["output_path"].write_bytes(b"apkg")
            return kwargs["output_path"], 1

        first_importer = MagicMock(return_value=True)
        first = self.controller(
            package_creator=create_package,
            package_importer=first_importer)
        second_creator = MagicMock()
        second_importer = MagicMock()
        second = self.controller(
            package_creator=second_creator,
            package_importer=second_importer)

        def finalize_first():
            try:
                first._finalize(
                    generated["job_id"],
                    self.pipeline)
            except Exception as error:
                first_failures.append(error)

        worker = threading.Thread(target=finalize_first)
        worker.start()
        self.assertTrue(package_started.wait(timeout=2))

        competing = second._finalize(
            generated["job_id"],
            self.pipeline)

        self.assertTrue(competing["finalization_in_progress"])
        second_creator.assert_not_called()
        second_importer.assert_not_called()
        rows = second.job_rows()["jobs"]
        self.assertEqual(
            next(
                row["status"]
                for row in rows
                if row["chunk_label"] == "Deck"),
            "running")

        release_package.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(first_failures, [])
        first_importer.assert_called_once()

        reread = second._finalize(
            generated["job_id"],
            self.pipeline)
        self.assertEqual(reread["state"], "imported")
        second_creator.assert_not_called()
        second_importer.assert_not_called()

    def test_anki_option_discovery_and_exclusion_are_read_only_and_cached(self):
        client = MagicMock()
        client.invoke.side_effect = [
            [1],
            [{"modelName": "Vocabulary"}],
            ["Word", "Meaning"],
            [11],
            [{
                "fields": {
                    "Word": {"value": "<b>甲</b>"},
                },
            }],
        ]
        controller = self.controller(
            anki_client_factory=lambda: client,
            clock=lambda: 10)
        with (
                patch.object(
                    anki_integration,
                    "ensure_anki_running"),
                patch.object(
                    anki_integration,
                    "wait_for_collection_ready")):
            note_types = controller.anki_options({
                "action": "note_types",
                "deck": "Known",
            })
            details = controller.anki_options({
                "action": "note_type_details",
                "note_type": "Vocabulary",
            })
            specification = SimpleNamespace(
                deck_name="Known",
                note_type="Vocabulary",
                field_name="Word",
                card_template_name=None)
            first = controller._resolve_anki_exclusion(
                specification,
                None)
            second = controller._resolve_anki_exclusion(
                specification,
                None)

        self.assertEqual(note_types, {"note_types": ("Vocabulary",)})
        self.assertEqual(details["fields"], ("Word", "Meaning"))
        self.assertEqual(first, frozenset({"甲"}))
        self.assertIs(first, second)
        mutating_actions = {
            "changeDeck",
            "deleteDecks",
            "importPackage",
            "storeMediaFile",
        }
        self.assertFalse(
            mutating_actions & {
                call.args[0]
                for call in client.invoke.call_args_list
            })

    def test_anki_exclusion_force_refresh_bypasses_the_estimate_cache(self):
        controller = self.controller(clock=lambda: 10)
        specification = SimpleNamespace(
            deck_name="Known",
            note_type="Vocabulary",
            field_name="Word",
            card_template_name=None)
        vocabularies = (
            frozenset({"甲"}),
            frozenset({"甲", "乙"}),
        )
        with (
                patch.object(
                    controller,
                    "_ready_anki_client",
                    return_value=MagicMock()),
                patch.object(
                    anki_integration,
                    "read_existing_vocabulary",
                    side_effect=vocabularies) as reader):
            first = controller._resolve_anki_exclusion(
                specification,
                None)
            cached = controller._resolve_anki_exclusion(
                specification,
                None)
            refreshed = controller._resolve_anki_exclusion(
                specification,
                None,
                force_refresh=True)

        self.assertIs(first, cached)
        self.assertEqual(refreshed, frozenset({"甲", "乙"}))
        self.assertEqual(reader.call_count, 2)

    def test_concurrent_estimates_share_one_anki_collection_read(self):
        controller = self.controller(clock=lambda: 10)
        specification = SimpleNamespace(
            deck_name="Known",
            note_type="Vocabulary",
            field_name="Word",
            card_template_name=None)
        read_started = threading.Event()
        release_read = threading.Event()
        results = []
        failures = []

        def read_vocabulary(_client, _source):
            read_started.set()
            if not release_read.wait(timeout=2):
                raise TimeoutError("Test did not release the Anki read.")
            return frozenset({"甲"})

        def resolve():
            try:
                results.append(controller._resolve_anki_exclusion(
                    specification,
                    None))
            except Exception as error:
                failures.append(error)

        with (
                patch.object(
                    controller,
                    "_ready_anki_client",
                    return_value=MagicMock()),
                patch.object(
                    anki_integration,
                    "read_existing_vocabulary",
                    side_effect=read_vocabulary) as reader):
            first = threading.Thread(target=resolve)
            second = threading.Thread(target=resolve)
            first.start()
            self.assertTrue(read_started.wait(timeout=2))
            second.start()
            release_read.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(failures, [])
        self.assertEqual(results, [frozenset({"甲"}), frozenset({"甲"})])
        self.assertEqual(reader.call_count, 1)

    def test_manual_filter_removes_only_exact_nonblank_lines_in_order(self):
        controller = self.controller()
        with patch.object(
                controller,
                "_resolve_anki_exclusion",
                return_value=frozenset({"甲", "épanouir"})) as resolver:
            result = controller.filter_manual_input({
                "text": "甲\n甲乙\n\n  e\u0301panouir  \n丙",
                "anki_exclusion": {
                    "deck": "Known",
                    "model": "Vocabulary",
                    "field": "Word",
                    "card_template_name": "Recognition",
                },
            })

        self.assertEqual(result["filtered_text"], "甲乙\n\n丙")
        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(result["remaining_count"], 2)
        self.assertEqual(result["original_count"], 4)
        self.assertTrue(resolver.call_args.kwargs["force_refresh"])

    def test_manual_filter_accepts_the_gui_input_text_contract(self):
        controller = self.controller()
        with patch.object(
                controller,
                "_resolve_anki_exclusion",
                return_value=frozenset({"known"})):
            result = controller.filter_manual_input({
                "input_text": "known\nnew",
                "anki_exclusion": {
                    "deck": "Known",
                    "model": "Vocabulary",
                    "field": "Word",
                },
            })

        self.assertEqual(result["filtered_text"], "new")
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["remaining_count"], 1)

    def test_manual_filter_unions_multiple_deck_note_field_sources(self):
        controller = self.controller()

        def resolve(specification, _source, *, force_refresh=False):
            self.assertTrue(force_refresh)
            return {
                "First": frozenset({"known-one"}),
                "Second": frozenset({"known-two"}),
            }[specification.deck_name]

        with patch.object(
                controller,
                "_resolve_anki_exclusion",
                side_effect=resolve) as resolver:
            result = controller.filter_manual_input({
                "text": "known-one\nnew\nknown-two",
                "anki_exclusions": [
                    {
                        "deck": "First",
                        "model": "Vocabulary",
                        "field": "Word",
                    },
                    {
                        "deck": "Second",
                        "model": "Vocabulary",
                        "field": "Term",
                    },
                ],
            })

        self.assertEqual(result["filtered_text"], "new")
        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(resolver.call_count, 2)

    def test_source_generation_forces_a_fresh_anki_read_before_job_creation(
            self):
        self.install_plan(empty_plan())
        controller = self.controller()
        with patch.object(
                controller,
                "_resolve_anki_exclusion",
                return_value=frozenset({"甲"})) as resolver:
            result = controller.generate({
                "pipeline": self.pipeline,
                "paid_confirmed": True,
                "exclude_anki": True,
                "anki_exclusion": {
                    "deck": "Known",
                    "model": "Vocabulary",
                    "field": "Word",
                },
            })

        self.assertTrue(result["no_new_vocabulary"])
        self.assertTrue(resolver.call_args.kwargs["force_refresh"])

    def test_retry_attention_ignores_unselected_failed_jobs(self):
        metadata = {
            "pipeline": pipeline_store.pipeline_to_mapping(
                self.pipeline),
        }
        selected = self.backend.jobs.create(
            one_word_plan(),
            request_metadata=metadata)
        unrelated = self.backend.jobs.create(
            one_word_plan(),
            request_metadata=metadata)
        selected_chunk = self.backend.jobs.chunk_ids(
            selected.job_id)[0]
        unrelated_chunk = self.backend.jobs.chunk_ids(
            unrelated.job_id)[0]
        self.backend.jobs._set_chunk_status(
            selected.job_id,
            selected_chunk,
            status="invalid_response")
        self.backend.jobs._set_chunk_status(
            unrelated.job_id,
            unrelated_chunk,
            status="invalid_response")
        client = FakeOpenAIClient((self.valid_output(),))

        def create_package(_combined, **kwargs):
            kwargs["output_path"].write_bytes(b"apkg")
            return kwargs["output_path"], 1

        controller = self.controller(
            openai_client_factory=lambda _key: client,
            package_creator=create_package,
            package_importer=lambda _path: True)
        with patch.object(
                source_workflow.process_text,
                "get_api_key",
                return_value="test-key"):
            result = controller.retry({
                "job_ids": (
                    self.backend._row_id(
                        selected.job_id,
                        selected_chunk),
                ),
                "paid_confirmed": True,
            })

        self.assertFalse(result["requires_attention"])
        self.assertEqual(
            self.backend.jobs.chunk_status(
                unrelated.job_id,
                unrelated_chunk)["status"],
            "invalid_response")

    def test_codex_retrieval_combines_files_then_only_prepares_words(self):
        first = self.root / "one.txt"
        second = self.root / "two.txt"
        first.write_text("甲。", encoding="utf-8")
        second.write_text("乙。", encoding="utf-8")
        job_path = self.root / "codex-job"
        (job_path / "retrieved").mkdir(parents=True)
        result = codex_source_retrieval.CodexRetrievalResult(
            job_id="codex-job",
            job_path=job_path,
            title="Fixture",
            source_urls=("https://example.invalid",),
            files=(first, second),
            summary="retrieved",
            warnings=())
        preparer = MagicMock(return_value=PreparedFixture())
        controller = self.controller(
            source_preparer=preparer,
            retrieval_job_creator=lambda _request: job_path,
            retrieval_job_runner=lambda _path: result)
        tokenizer = MagicMock()
        with patch.object(
                controller,
                "_document_tokenizer",
                return_value=tokenizer):
            output = controller.retrieve_with_codex({
                "description": "Retrieve fixture",
                "source_title": "Fixture",
                "language_key": (
                    "classical_chinese_warring_states"),
                "retrieval_authorized": True,
                "generate_cards": False,
            })

        prepared_path = Path(preparer.call_args.args[0])
        self.assertTrue(prepared_path.is_file())
        self.assertIn(
            "\f",
            prepared_path.read_text(encoding="utf-8"))
        self.assertEqual(
            preparer.call_args.kwargs["tokenizer"],
            tokenizer)
        self.assertIn("retrieved and prepared", output["message"])


if __name__ == "__main__":
    unittest.main()
