import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from corpus_pipeline.contexts import assemble_sections
from corpus_pipeline.models import (
    BuildConfig,
    CorpusSnapshot,
    SourcePage,
    TextSection,
    TokenSpan,
    TokenizerIdentity,
)
from corpus_pipeline.processing import build_vocabulary
from corpus_pipeline.storage import build_id, write_build
from source_generation import (
    ContextMode,
    GenerationJobRunner,
    GenerationJobStore,
    LoadedSource,
    OutputDetail,
    RetryPolicy,
    SOURCE_REQUEST_MODEL,
    SOURCE_REQUEST_REASONING,
    SourceGenerationConfig,
    SourceGenerationBackend,
    build_source_request_contract,
    estimate_plan_cost,
    load_processed_source,
    plan_source_generation,
)
from source_generation.validation import make_pipeline_response_validator
from source_generation.requests import load_source_batch_instructions
from source_generation.jobs import _RequestStartGate
import pipeline_store
import process_text


class CharacterTokenizer:
    @property
    def identity(self):
        return TokenizerIdentity(
            backend="source-generation-test",
            backend_version="1",
            model="characters",
            model_revision="1")

    def tokenize(self, text):
        return tuple(
            TokenSpan(character, index, index + 1)
            for index, character in enumerate(text)
            if "\u3400" <= character <= "\u9fff")


def make_source(section_texts=("甲乙。丙丁。戊己。",)):
    raw = "\n".join(section_texts)
    page = SourcePage(
        page_key="fixture-001",
        order=1,
        title="Fixture",
        url="https://example.invalid/fixture",
        revision_id=1,
        revision_timestamp="2026-01-01T00:00:00Z",
        revision_sha1="abc",
        retrieved_at="2026-01-02T00:00:00Z",
        raw_wikitext=raw,
        raw_sha256=__import__("hashlib").sha256(
            raw.encode("utf-8")).hexdigest())
    sections, canonical = assemble_sections(tuple(
        TextSection(
            section_id=f"section-{index:03d}",
            order=index,
            title=f"Section {index}",
            source_page_key=page.page_key,
            text=text)
        for index, text in enumerate(section_texts, start=1)))
    snapshot = CorpusSnapshot(
        spec_key="fixture_source",
        edition="Fixture Source",
        source_language_key="classical_chinese",
        pages=(page,),
        sections=sections,
        canonical_text=canonical,
        cleaner_version="fixture-cleaner")
    build = build_vocabulary(
        snapshot,
        CharacterTokenizer(),
        BuildConfig())
    return LoadedSource(
        key="fixture_source",
        title="Fixture Source",
        build_id=build_id(build),
        run_path=Path("/fixture/run") / build_id(build),
        build=build)


def make_plan(
        *,
        chunk_size=500,
        context_mode=ContextMode.SENTENCE,
        excluded_words=()):
    source = make_source()
    return plan_source_generation(
        source,
        SourceGenerationConfig(
            source_key=source.key,
            chunk_size=chunk_size,
            context_mode=context_mode,
            concurrency=3,
            request_stagger_ms=0,
            max_transient_retries=2,
            excluded_words=tuple(excluded_words)))


class SourcePlanningTests(unittest.TestCase):
    def test_default_request_size_and_worker_count_are_30_and_8(self):
        config = SourceGenerationConfig(source_key="fixture_source")

        self.assertEqual(config.chunk_size, 30)
        self.assertEqual(config.concurrency, 8)

    def test_source_batch_instructions_are_an_editable_prompt_component(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch"
            path.write_text("Custom source batch instructions", encoding="utf-8")
            component = pipeline_store.PromptOption(
                key="source/batch",
                name="Source Batch",
                path=path)
            with patch.object(
                    pipeline_store,
                    "prompt_component_map",
                    return_value={"source/batch": component}):
                text = load_source_batch_instructions()

        self.assertEqual(text, "Custom source batch instructions")

    def test_config_mapping_accepts_gui_aliases_and_retains_anki_spec(self):
        config = SourceGenerationConfig.from_mapping({
            "source": "fixture_source",
            "chunk_size": 50,
            "context_mode": "current+/-1",
            "concurrency": 8,
            "exclude_anki": {
                "deck": "Known words",
                "note_type": "Vocabulary",
                "field": "Word",
                "card_template_name": "Recognition",
            },
        })

        self.assertEqual(config.source_key, "fixture_source")
        self.assertEqual(config.context_mode, ContextMode.SENTENCE_NEIGHBORS)
        self.assertEqual(config.exclude_anki.deck_name, "Known words")
        self.assertEqual(config.exclude_anki.field_name, "Word")
        self.assertEqual(
            config.exclude_anki.card_template_name,
            "Recognition")

    def test_config_accepts_multiple_anki_exclusion_sources(self):
        config = SourceGenerationConfig.from_mapping({
            "source": "fixture_source",
            "anki_exclusions": [
                {
                    "deck": "Recognition",
                    "model": "Vocabulary",
                    "field": "Word",
                },
                {
                    "deck": "Production",
                    "model": "Vocabulary",
                    "field": "Answer",
                },
            ],
        })

        self.assertEqual(
            tuple(
                specification.deck_name
                for specification in config.anki_exclusions),
            ("Recognition", "Production"))

    def test_variable_chunk_size_repartitions_existing_unique_words(self):
        source = make_source()
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=2,
                context_mode=ContextMode.NONE))

        self.assertEqual(len(source.build.unique_words), 6)
        self.assertEqual(len(plan.chunks), 3)
        self.assertEqual(
            tuple(len(chunk.words) for chunk in plan.chunks),
            (2, 2, 2))
        self.assertEqual(
            tuple(word.rank for chunk in plan.chunks for word in chunk.words),
            (1, 2, 3, 4, 5, 6))

    def test_sentence_context_is_stored_once_and_words_point_to_it(self):
        plan = make_plan(context_mode=ContextMode.SENTENCE)
        chunk = plan.chunks[0]

        self.assertEqual(
            tuple(context.text for context in chunk.contexts),
            ("甲乙。", "丙丁。", "戊己。"))
        self.assertEqual(len({
            word.context_id
            for word in chunk.words[:2]
        }), 1)
        payload = chunk.request_payload()
        self.assertEqual(
            json.dumps(payload, ensure_ascii=False).count("甲乙。"),
            1)
        self.assertNotIn(
            "text",
            payload["words"][0])

    def test_overlapping_neighbor_windows_merge_into_one_context(self):
        plan = make_plan(
            context_mode=ContextMode.SENTENCE_NEIGHBORS)
        chunk = plan.chunks[0]

        self.assertEqual(len(chunk.contexts), 1)
        self.assertEqual(chunk.contexts[0].text, "甲乙。丙丁。戊己。")
        self.assertEqual(
            chunk.contexts[0].word_ranks,
            (1, 2, 3, 4, 5, 6))
        self.assertEqual(
            {word.context_id for word in chunk.words},
            {chunk.contexts[0].context_id})

    def test_neighbor_context_clamps_cleanly_at_source_boundaries(self):
        plan = make_plan(
            chunk_size=1,
            context_mode=ContextMode.SENTENCE_NEIGHBORS)

        self.assertEqual(
            plan.chunks[0].contexts[0].text,
            "甲乙。丙丁。")
        self.assertEqual(
            plan.chunks[-1].contexts[0].text,
            "丙丁。戊己。")

    def test_chunk_span_is_one_interval_from_first_to_last_sentence(self):
        plan = make_plan(
            chunk_size=4,
            context_mode=ContextMode.CHUNK_SPAN)

        self.assertEqual(len(plan.chunks), 2)
        self.assertEqual(len(plan.chunks[0].contexts), 1)
        self.assertEqual(
            plan.chunks[0].contexts[0].text,
            "甲乙。丙丁。")
        self.assertEqual(
            plan.chunks[0].contexts[0].word_ranks,
            (1, 2, 3, 4))
        self.assertEqual(
            {word.context_id for word in plan.chunks[0].words},
            {plan.chunks[0].contexts[0].context_id})
        self.assertEqual(len(plan.chunks[1].contexts), 1)
        self.assertEqual(
            plan.chunks[1].contexts[0].text,
            "戊己。")

    def test_chunk_span_keeps_sentences_without_selected_new_words(self):
        plan = make_plan(
            chunk_size=2,
            context_mode=ContextMode.CHUNK_SPAN,
            excluded_words=("乙", "丙", "丁", "己"))

        self.assertEqual(
            tuple(word.surface for word in plan.chunks[0].words),
            ("甲", "戊"))
        self.assertEqual(
            plan.chunks[0].contexts[0].text,
            "甲乙。丙丁。戊己。")
        self.assertEqual(
            plan.chunks[0].contexts[0].sentence_ids,
            (
                "section-001:sentence:00001",
                "section-001:sentence:00002",
                "section-001:sentence:00003",
            ))

    def test_no_context_has_null_references_and_no_context_payload(self):
        chunk = make_plan(
            context_mode=ContextMode.NONE).chunks[0]

        self.assertFalse(chunk.contexts)
        self.assertTrue(all(
            word.context_id is None
            for word in chunk.words))
        self.assertEqual(chunk.request_payload()["contexts"], [])

    def test_exclusions_preserve_first_occurrence_order_and_source_rank(self):
        plan = make_plan(
            chunk_size=2,
            excluded_words=("乙", "丁"))

        self.assertEqual(plan.original_word_count, 6)
        self.assertEqual(plan.excluded_word_count, 2)
        self.assertEqual(
            tuple(word.rank for chunk in plan.chunks for word in chunk.words),
            (1, 3, 5, 6))
        self.assertEqual(len(plan.chunks), 2)

    def test_zero_remaining_words_produces_no_requests(self):
        plan = make_plan(
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"))

        self.assertEqual(plan.word_count, 0)
        self.assertFalse(plan.chunks)

    def test_plan_identity_ignores_worker_count_but_tracks_request_shape(self):
        source = make_source()
        first = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                concurrency=1,
                chunk_size=3))
        second = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                concurrency=12,
                chunk_size=3))
        changed = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                concurrency=1,
                chunk_size=2))

        self.assertEqual(first.plan_id, second.plan_id)
        self.assertNotEqual(first.plan_id, changed.plan_id)

    def test_cost_estimate_tracks_fields_context_and_repeated_prompts(self):
        source = make_source()
        small_chunks = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=2,
                context_mode=ContextMode.SENTENCE))
        one_chunk = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=6,
                context_mode=ContextMode.SENTENCE))
        simple = OutputDetail(("translation",))
        detailed = OutputDetail(
            (
                "translation",
                "dictionary_meaning",
                "pronunciation",
                "part_of_speech",
                "register",
                "nuance",
            ),
            include_example_sentences=True)

        small_estimate = estimate_plan_cost(
            small_chunks,
            simple,
            prompt_text="A repeated prompt " * 20)
        one_estimate = estimate_plan_cost(
            one_chunk,
            simple,
            prompt_text="A repeated prompt " * 20)
        detailed_estimate = estimate_plan_cost(
            one_chunk,
            detailed,
            prompt_text="A repeated prompt " * 20)

        self.assertEqual(small_estimate.request_count, 3)
        self.assertEqual(one_estimate.request_count, 1)
        self.assertGreater(
            small_estimate.prompt_tokens,
            one_estimate.prompt_tokens)
        self.assertGreater(
            small_estimate.estimated_input_usd,
            one_estimate.estimated_input_usd)
        self.assertGreater(
            detailed_estimate.estimated_output_usd,
            one_estimate.estimated_output_usd)
        self.assertEqual(
            detailed_estimate.assumptions["token_estimator"],
            "deterministic character-class approximation; no API call")

    def test_web_search_estimate_is_bounded_to_one_search_per_request(self):
        plan = make_plan(chunk_size=2)

        without_search = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")
        with_search = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt",
            web_search_enabled=True)

        self.assertEqual(
            with_search.assumptions["web_search_max_calls_per_request"],
            1)
        self.assertGreater(
            with_search.estimated_cost_high_usd,
            without_search.estimated_cost_high_usd)
        self.assertGreater(
            with_search.estimated_cost_high_aud,
            with_search.estimated_cost_high_usd)

    def test_load_processed_source_follows_and_audits_latest_build(self):
        source = make_source()
        with tempfile.TemporaryDirectory() as directory:
            written = write_build(source.build, directory)
            loaded = load_processed_source(
                source.key,
                corpus_root=directory)

        self.assertEqual(loaded.build_id, written.name)
        self.assertEqual(
            tuple(word.surface for word in loaded.build.unique_words),
            ("甲", "乙", "丙", "丁", "戊", "己"))


def simple_classical_pipeline():
    pipeline = pipeline_store.default_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "classical_chinese")
    selected_fields = (
        pipeline_store.FieldSetting("translation", "english"),
    )
    settings = replace(
        settings,
        cards=tuple(
            replace(
                card,
                enabled=card.direction_key == "word_to_meaning",
                fields=selected_fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=selected_fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese")


class SourceResponseValidationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = simple_classical_pipeline()
        self.chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.NONE).chunks[0]
        self.validator = make_pipeline_response_validator(self.pipeline)
        self.term_field, self.meaning_field = (
            process_text.get_response_field_names(self.pipeline))

    def card(self, term, meaning):
        return {
            self.term_field: term,
            self.meaning_field: meaning,
        }

    def test_missing_requested_term_is_invalid(self):
        raw = json.dumps({
            "cards": [
                self.card("甲", "first"),
            ],
        })

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "omitted.*乙"):
            self.validator(raw, self.chunk)

    def test_multiple_senses_are_allowed_and_source_order_is_restored(self):
        raw = json.dumps({
            "cards": [
                self.card("乙", "second"),
                self.card("甲", "first sense"),
                self.card("甲", "another sense"),
            ],
        })

        validated = self.validator(raw, self.chunk)

        self.assertEqual(
            [card[self.term_field] for card in validated["cards"]],
            ["甲", "甲", "乙"])
        self.assertEqual(
            [
                card[self.meaning_field]
                for card in validated["cards"][:2]
            ],
            ["first sense", "another sense"])

    def test_term_outside_requested_chunk_is_invalid(self):
        raw = json.dumps({
            "cards": [
                self.card("甲", "first"),
                self.card("乙", "second"),
                self.card("丙", "not requested"),
            ],
        })

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "outside.*丙"):
            self.validator(raw, self.chunk)


class SourceBackendAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.corpus_root = root / "corpora"
        self.jobs_root = root / "jobs"
        source = make_source()
        local_snapshot = replace(
            source.build.snapshot,
            edition="Local document: My Book")
        local_build = replace(
            source.build,
            snapshot=local_snapshot)
        write_build(local_build, self.corpus_root)
        self.exclusion_calls = []

        def exclusion_resolver(spec, loaded):
            self.exclusion_calls.append((spec, loaded))
            return ("甲",)

        self.backend = SourceGenerationBackend(
            corpus_root=self.corpus_root,
            jobs_root=self.jobs_root,
            exclusion_resolver=exclusion_resolver)
        self.request = {
            "source_key": "fixture_source",
            "chunk_size": 2,
            "context_mode": "sentence",
            "concurrency": 2,
            "request_stagger_ms": 100,
            "pipeline": simple_classical_pipeline(),
            "exclude_anki": True,
            "anki_exclusion": {
                "deck": "Known",
                "model": "Vocabulary",
                "field": "Classical Chinese",
                "card_template_name": "Recognition",
            },
            "output_deck_name": "Vocabulary from My Book",
        }

    def test_catalogue_and_estimator_return_gui_mapping_contract(self):
        catalogue = self.backend.catalogue()
        estimate = self.backend.estimate(self.request)

        self.assertEqual(len(catalogue["sources"]), 1)
        source = catalogue["sources"][0]
        self.assertEqual(source["key"], "fixture_source")
        self.assertEqual(source["name"], "My Book")
        self.assertEqual(source["word_count"], 6)
        self.assertEqual(
            source["source_language_key"],
            "classical_chinese")
        self.assertEqual(estimate["candidate_count"], 5)
        self.assertEqual(estimate["excluded_candidate_count"], 1)
        self.assertIn("estimated_cost_low_usd", estimate)
        self.assertIn("estimated_cost_high_usd", estimate)
        self.assertIn("input_tokens", estimate)
        self.assertEqual(
            self.exclusion_calls[0][0].card_template_name,
            "Recognition")

    def test_estimator_unions_multiple_anki_exclusion_sources(self):
        def exclusion_resolver(specification, _loaded):
            return {
                "First": ("甲",),
                "Second": ("乙",),
            }[specification.deck_name]

        self.backend.exclusion_resolver = exclusion_resolver
        request = {
            **self.request,
            "anki_exclusion": None,
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
        }

        estimate = self.backend.estimate(request)

        self.assertEqual(estimate["candidate_count"], 4)
        self.assertEqual(estimate["excluded_candidate_count"], 2)

    def test_preview_returns_ordered_retained_sentence_metadata(self):
        page = self.backend.preview({
            "source_key": "fixture_source",
            "offset": 1,
            "limit": 2,
        })

        self.assertEqual(page["source_key"], "fixture_source")
        self.assertEqual(page["total"], 6)
        self.assertEqual(page["offset"], 1)
        self.assertEqual(
            tuple(item["rank"] for item in page["items"]),
            (2, 3))
        self.assertEqual(
            tuple(item["term"] for item in page["items"]),
            ("乙", "丙"))
        self.assertEqual(
            page["items"][0]["current_sentence"],
            "甲乙。")
        self.assertEqual(
            page["items"][1]["previous_sentence"],
            "甲乙。")
        self.assertEqual(
            page["items"][1]["next_sentence"],
            "戊己。")
        self.assertEqual(
            page["items"][0]["section_title"],
            "Section 1")

    def test_preview_is_bounded_and_never_resolves_anki_exclusion(self):
        page = self.backend.preview({
            "source_key": "fixture_source",
            "offset": 6,
            "limit": 500,
        })

        self.assertEqual(page["items"], [])
        self.assertEqual(self.exclusion_calls, [])
        with self.assertRaisesRegex(ValueError, "beyond"):
            self.backend.preview({
                "source_key": "fixture_source",
                "offset": 7,
                "limit": 1,
            })
        with self.assertRaisesRegex(ValueError, "10,000"):
            self.backend.preview({
                "source_key": "fixture_source",
                "offset": 0,
                "limit": 10001,
            })

    def test_create_job_requires_confirmation_and_rows_are_inspectable(self):
        with self.assertRaises(PermissionError):
            self.backend.create_job(self.request)

        job = self.backend.create_job({
            **self.request,
            "paid_confirmed": True,
            "estimate": self.backend.estimate(self.request),
        })
        rows = self.backend.job_rows()["jobs"]

        self.assertEqual(job.overall_status, "ready")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        pending_targets = self.backend.manual_retry_targets({
            "job_ids": (rows[0]["job_id"],),
            "paid_confirmed": True,
        })
        self.assertEqual(
            pending_targets[job.job_id],
            (rows[0]["chunk_id"],))
        inspected = self.backend.inspect({
            "job_id": rows[0]["job_id"],
        })
        self.assertEqual(
            inspected["request"]["chunk_id"],
            rows[0]["chunk_id"])
        contract = inspected["request_contract"]
        self.assertEqual(contract["model"], SOURCE_REQUEST_MODEL)
        self.assertEqual(
            contract["reasoning"],
            SOURCE_REQUEST_REASONING)
        self.assertTrue(contract["response_format"]["strict"])
        self.assertEqual(
            json.loads(
                (job.path / "request_contract.json").read_text(
                    encoding="utf-8")),
            contract)
        self.assertEqual(
            set(contract["max_output_tokens_by_chunk"]),
            set(self.backend.jobs.chunk_ids(job.job_id)))

    def test_create_job_rejects_stale_prompt_authorization(self):
        estimate = self.backend.estimate(self.request)
        _loaded, plan = self.backend._plan(self.request)
        changed_contract = build_source_request_contract(
            self.request["pipeline"],
            chunks=plan.chunks)
        changed_contract["composed_prompt"] += (
            "\nA prompt edit made after authorization.\n")

        with (
                patch(
                    "source_generation.adapters."
                    "build_source_request_contract",
                    return_value=changed_contract),
                self.assertRaisesRegex(
                    PermissionError,
                    "changed after authorization")):
            self.backend.create_job({
                **self.request,
                "paid_confirmed": True,
                "estimate": estimate,
            })

        self.assertEqual(self.backend.jobs.list(), ())

    def test_create_job_rejects_stale_execution_policy_authorization(self):
        estimate = self.backend.estimate(self.request)

        with self.assertRaisesRegex(
                PermissionError,
                "changed after authorization"):
            self.backend.create_job({
                **self.request,
                "concurrency": 7,
                "paid_confirmed": True,
                "estimate": estimate,
            })

        self.assertEqual(self.backend.jobs.list(), ())

    def test_request_contract_freezes_exact_prompt_schema_and_reasoning(self):
        pipeline = simple_classical_pipeline()

        contract = build_source_request_contract(pipeline)

        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(contract["model"], "gpt-5.4-mini")
        self.assertEqual(contract["reasoning"], {"effort": "none"})
        self.assertIn(
            "Here is the source batch JSON:",
            contract["composed_prompt"])
        self.assertIn(
            "never skip a supplied source term",
            contract["composed_prompt"])
        self.assertEqual(
            contract["response_format"]["type"],
            "json_schema")
        self.assertTrue(contract["response_format"]["strict"])
        self.assertEqual(
            contract["max_output_tokens_by_chunk"],
            {})
        self.assertEqual(contract["tools"], [])
        self.assertEqual(contract["max_tool_calls"], 0)

    def test_web_search_contract_freezes_tool_limit_and_prompt_guidance(self):
        pipeline = simple_classical_pipeline()

        contract = build_source_request_contract(
            pipeline,
            allow_web_search=True)

        self.assertEqual(contract["tools"], [{"type": "web_search"}])
        self.assertEqual(contract["max_tool_calls"], 1)
        prompt_text = " ".join(
            contract["composed_prompt"].lower().split())
        self.assertIn(
            "use web search only when",
            prompt_text)
        self.assertIn(
            "one consolidated search action",
            prompt_text)
        self.assertIn(
            "all of those unclear terms",
            prompt_text)

    def test_manual_retry_target_resolution_requires_confirmation(self):
        job = self.backend.create_job({
            **self.request,
            "paid_confirmed": True,
            "estimate": self.backend.estimate(self.request),
        })
        chunk_id = self.backend.jobs.chunk_ids(job.job_id)[0]
        self.backend.jobs._set_chunk_status(
            job.job_id,
            chunk_id,
            status="invalid_response")
        row_id = self.backend._row_id(job.job_id, chunk_id)

        with self.assertRaises(PermissionError):
            self.backend.manual_retry_targets({
                "job_ids": (row_id,),
            })
        result = self.backend.manual_retry_targets({
            "job_ids": (row_id,),
            "paid_confirmed": True,
        })

        self.assertEqual(result[job.job_id], (chunk_id,))
        self.assertEqual(
            self.backend.jobs.chunk_status(
                job.job_id,
                chunk_id)["status"],
            "pending")


class StatusError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.headers = headers


def valid_response(chunk):
    return {
        "cards": [
            {
                "term": word.surface,
                "definition": f"Definition {word.surface}",
            }
            for word in chunk.words
        ],
    }


def validate_response(raw_text, _chunk):
    parsed = json.loads(raw_text)
    if set(parsed) != {"cards"} or not isinstance(parsed["cards"], list):
        raise ValueError("Invalid cards response.")
    return parsed


class SourceJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = GenerationJobStore(self.temporary.name)
        self.plan = make_plan(
            chunk_size=2,
            context_mode=ContextMode.NONE)
        self.job = self.store.create(
            self.plan,
            request_metadata={
                "paid_confirmed": True,
                "pipeline_id": "fixture",
            })

    def runner(self, request, validator=validate_response, **kwargs):
        return GenerationJobRunner(
            self.store,
            self.job.job_id,
            request,
            validator,
            request_stagger_ms=0,
            sleeper=lambda _seconds: None,
            random_source=lambda: 0,
            **kwargs)

    def test_successful_concurrent_run_persists_each_and_combines_in_order(self):
        barrier = threading.Barrier(3)

        def request(chunk):
            barrier.wait(timeout=2)
            return valid_response(chunk)

        snapshot = self.runner(
            request,
            concurrency=3).run()

        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(snapshot.counts, {"succeeded": 3})
        combined = json.loads(
            (snapshot.path / "combined.json").read_text(encoding="utf-8"))
        self.assertTrue(combined["complete"])
        self.assertEqual(
            [card["term"] for card in combined["cards"]],
            ["甲", "乙", "丙", "丁", "戊", "己"])
        self.assertTrue(all(
            (snapshot.path / "chunks" / chunk_id
             / "attempts" / "0001" / "raw.txt").is_file()
            for chunk_id in self.store.chunk_ids(snapshot.job_id)))

    def test_transient_connection_errors_retry_automatically_and_resume(self):
        calls = 0
        delays = []
        now = [0.0]

        def sleep(seconds):
            delays.append(seconds)
            now[0] += seconds

        def request(chunk):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("temporary disconnect")
            return valid_response(chunk)

        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        runner = GenerationJobRunner(
            self.store,
            self.job.job_id,
            request,
            validate_response,
            concurrency=1,
            request_stagger_ms=0,
            retry_policy=RetryPolicy(
                max_retries=2,
                initial_backoff_seconds=0.25,
                maximum_backoff_seconds=1),
            sleeper=sleep,
            clock=lambda: now[0])
        runner.random_source = lambda: 0
        snapshot = runner.run((first_chunk,))

        self.assertEqual(calls, 3)
        self.assertEqual(delays, [0.25, 0.5])
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "succeeded")
        self.assertEqual(snapshot.overall_status, "ready")
        self.assertEqual(
            len(self.store.inspect_chunk(
                self.job.job_id,
                first_chunk)["attempts"]),
            3)

    def test_rate_limit_reset_headers_and_positive_jitter_are_respected(self):
        policy = RetryPolicy(
            max_retries=1,
            initial_backoff_seconds=0.5,
            maximum_backoff_seconds=2,
            jitter_fraction=0.20)
        error = StatusError(429, {
            "retry-after-ms": "250",
            "x-ratelimit-reset-requests": "1m2.5s",
            "x-ratelimit-reset-tokens": "3s",
        })

        self.assertEqual(
            policy.delay(1, error, random_unit=0),
            62.5)
        self.assertEqual(
            policy.delay(1, error, random_unit=0.5),
            68.75)

    def test_rate_limit_defer_can_extend_a_sleeping_global_gate(self):
        now = [0.0]
        delays = []
        first_sleep_started = threading.Event()
        release_first_sleep = threading.Event()
        defer_finished = threading.Event()

        def sleep(seconds):
            delays.append(seconds)
            if len(delays) == 1:
                first_sleep_started.set()
                self.assertTrue(
                    release_first_sleep.wait(timeout=2),
                    "Test did not release the first gate wait.")
            now[0] += seconds

        gate = _RequestStartGate(
            1.0,
            clock=lambda: now[0],
            sleeper=sleep)
        gate.next_start = 1.0
        waiting_worker = threading.Thread(target=gate.wait)
        waiting_worker.start()
        self.assertTrue(first_sleep_started.wait(timeout=2))

        def apply_cooldown():
            gate.defer(5.0)
            defer_finished.set()

        cooldown_worker = threading.Thread(target=apply_cooldown)
        cooldown_worker.start()
        self.assertTrue(
            defer_finished.wait(timeout=2),
            "A sleeping gate held the lock and blocked the shared cooldown.")
        release_first_sleep.set()
        waiting_worker.join(timeout=2)
        cooldown_worker.join(timeout=2)

        self.assertFalse(waiting_worker.is_alive())
        self.assertFalse(cooldown_worker.is_alive())
        self.assertEqual(delays, [1.0, 4.0])
        self.assertEqual(gate.next_start, 6.0)

    def test_rate_limit_and_server_status_are_transient_only(self):
        statuses = [429, 503]
        calls = 0

        def request(chunk):
            nonlocal calls
            calls += 1
            if statuses:
                raise StatusError(statuses.pop(0))
            return valid_response(chunk)

        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        self.runner(
            request,
            concurrency=1,
            retry_policy=RetryPolicy(
                max_retries=2,
                initial_backoff_seconds=0,
                maximum_backoff_seconds=0)).run((first_chunk,))

        self.assertEqual(calls, 3)
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "succeeded")

    def test_invalid_response_is_not_retried_without_manual_authorization(self):
        calls = 0
        allow_valid = False

        def request(chunk):
            nonlocal calls
            calls += 1
            return valid_response(chunk) if allow_valid else {"wrong": []}

        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        runner = self.runner(request, concurrency=1)
        runner.run((first_chunk,))
        runner.run((first_chunk,))

        self.assertEqual(calls, 1)
        status = self.store.chunk_status(
            self.job.job_id,
            first_chunk)
        self.assertEqual(status["status"], "invalid_response")
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertIn('"wrong"', inspection["attempts"][0]["raw_text"])

        allow_valid = True
        runner.retry_one(first_chunk)

        self.assertEqual(calls, 2)
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "succeeded")
        self.assertEqual(
            len(self.store.inspect_chunk(
                self.job.job_id,
                first_chunk)["attempts"]),
            2)

    def test_exhausted_connection_failure_requires_manual_retry(self):
        calls = 0
        working = False

        def request(chunk):
            nonlocal calls
            calls += 1
            if not working:
                raise TimeoutError("timeout")
            return valid_response(chunk)

        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        runner = self.runner(
            request,
            concurrency=1,
            retry_policy=RetryPolicy(
                max_retries=1,
                initial_backoff_seconds=0,
                maximum_backoff_seconds=0))
        runner.run((first_chunk,))
        runner.run((first_chunk,))
        self.assertEqual(calls, 2)
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "connection_failed")

        working = True
        runner.retry_all()
        self.assertEqual(calls, 3)

    def test_cancel_before_run_makes_no_paid_request_and_can_be_retried(self):
        calls = []

        def request(chunk):
            calls.append(chunk.chunk_id)
            return valid_response(chunk)

        runner = self.runner(request, concurrency=3)
        runner.cancel()
        snapshot = runner.run()

        self.assertFalse(calls)
        self.assertEqual(snapshot.overall_status, "cancelled")
        self.assertEqual(snapshot.counts, {"cancelled": 3})

        retried = runner.retry_all()
        self.assertEqual(len(calls), 3)
        self.assertEqual(retried.overall_status, "completed")

    def test_interrupted_running_status_is_recovered_without_losing_attempt(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        self.store.write_error(
            attempt_path,
            RuntimeError("process stopped"),
            transient=True)

        self.runner(
            lambda chunk: valid_response(chunk),
            concurrency=1).run((first_chunk,))

        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertEqual(
            inspection["status"]["status"],
            "succeeded")
        self.assertEqual(len(inspection["attempts"]), 2)
        self.assertEqual(
            inspection["attempts"][0]["error"]["message"],
            "process stopped")

    def test_crash_after_validated_write_never_repeats_paid_request(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        chunk = self.store.load_chunk(
            self.job.job_id,
            first_chunk)
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        retained = valid_response(chunk)
        self.store.write_raw(
            attempt_path,
            json.dumps(retained))
        self.store.write_validated(
            attempt_path,
            retained)
        calls = []

        snapshot = self.runner(
            lambda requested: calls.append(requested),
            concurrency=1).run((first_chunk,))

        self.assertEqual(calls, [])
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "succeeded")
        self.assertEqual(
            len(self.store.inspect_chunk(
                self.job.job_id,
                first_chunk)["attempts"]),
            1)
        self.assertEqual(snapshot.overall_status, "ready")

    def test_recovery_revalidates_raw_instead_of_trusting_valid_json_artifact(
            self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        chunk = self.store.load_chunk(
            self.job.job_id,
            first_chunk)
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        retained = valid_response(chunk)
        self.store.write_raw(
            attempt_path,
            json.dumps(retained))
        # Parseable JSON alone is not proof that the pipeline validator
        # accepted it. Simulate corruption/tampering after the raw write.
        self.store.write_validated(
            attempt_path,
            {"wrong": []})
        calls = []

        self.runner(
            lambda requested: calls.append(requested),
            concurrency=1).run((first_chunk,))

        self.assertEqual(calls, [])
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertEqual(inspection["status"]["status"], "succeeded")
        self.assertEqual(
            inspection["attempts"][0]["validated"],
            retained)

    def test_validated_artifact_without_raw_is_not_trusted_or_retried(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        self.store.write_validated(
            attempt_path,
            {"cards": []})
        calls = []

        self.runner(
            lambda requested: calls.append(requested),
            concurrency=1).run((first_chunk,))

        self.assertEqual(calls, [])
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertEqual(inspection["status"]["status"], "failed")
        self.assertIn(
            "semantic validity",
            inspection["attempts"][0]["recovery_error"]["message"])

    def test_crash_after_raw_write_revalidates_without_paid_request(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        chunk = self.store.load_chunk(
            self.job.job_id,
            first_chunk)
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        self.store.write_raw(
            attempt_path,
            json.dumps(valid_response(chunk)))
        calls = []

        self.runner(
            lambda requested: calls.append(requested),
            concurrency=1).run((first_chunk,))

        self.assertEqual(calls, [])
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertEqual(inspection["status"]["status"], "succeeded")
        self.assertEqual(len(inspection["attempts"]), 1)
        self.assertIn("validated", inspection["attempts"][0])

    def test_invalid_crash_retained_raw_stops_without_paid_retry(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        _attempt, attempt_path = self.store.begin_attempt(
            self.job.job_id,
            first_chunk)
        self.store.write_raw(
            attempt_path,
            '{"wrong":[]}')
        calls = []

        self.runner(
            lambda requested: calls.append(requested),
            concurrency=1).run((first_chunk,))

        self.assertEqual(calls, [])
        inspection = self.store.inspect_chunk(
            self.job.job_id,
            first_chunk)
        self.assertEqual(
            inspection["status"]["status"],
            "invalid_response")
        self.assertIn(
            "recovery_error",
            inspection["attempts"][0])

    def test_live_cross_process_lease_is_not_recovered_as_interrupted(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        other_process_store = GenerationJobStore(self.temporary.name)

        with self.store.chunk_lease(
                self.job.job_id,
                first_chunk) as acquired:
            self.assertTrue(acquired)
            self.store.begin_attempt(
                self.job.job_id,
                first_chunk)

            recovered = other_process_store.recover_interrupted(
                self.job.job_id)

            self.assertEqual(recovered, ())
            self.assertEqual(
                other_process_store.chunk_status(
                    self.job.job_id,
                    first_chunk)["status"],
                "running")

        recovered = other_process_store.recover_interrupted(
            self.job.job_id)
        self.assertEqual(recovered, (first_chunk,))
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "pending")

    def test_two_job_runners_cannot_pay_for_the_same_chunk(self):
        first_chunk = self.store.chunk_ids(self.job.job_id)[0]
        other_process_store = GenerationJobStore(self.temporary.name)
        request_started = threading.Event()
        release_request = threading.Event()
        calls = []
        failures = []

        def request(chunk):
            calls.append(chunk.chunk_id)
            request_started.set()
            if not release_request.wait(timeout=2):
                raise TimeoutError("Test did not release the paid request.")
            return valid_response(chunk)

        first_runner = self.runner(
            request,
            concurrency=1)
        second_runner = GenerationJobRunner(
            other_process_store,
            self.job.job_id,
            request,
            validate_response,
            concurrency=1,
            request_stagger_ms=0)

        def run_first():
            try:
                first_runner.run((first_chunk,))
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=run_first)
        worker.start()
        self.assertTrue(request_started.wait(timeout=2))

        second_runner.run((first_chunk,))
        self.assertEqual(calls, [first_chunk])

        release_request.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(calls, [first_chunk])
        self.assertEqual(
            self.store.chunk_status(
                self.job.job_id,
                first_chunk)["status"],
            "succeeded")

    def test_empty_plan_creates_nothing_to_generate_job(self):
        empty = make_plan(
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"))
        job = self.store.create(empty)
        calls = []
        runner = GenerationJobRunner(
            self.store,
            job.job_id,
            lambda chunk: calls.append(chunk),
            validate_response)

        snapshot = runner.run()

        self.assertEqual(snapshot.overall_status, "nothing_to_generate")
        self.assertFalse(calls)
