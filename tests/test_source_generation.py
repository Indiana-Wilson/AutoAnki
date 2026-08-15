import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


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
from corpus_pipeline.tokenizers import HistoricalEnglishTokenizer
from source_generation import (
    ContextMode,
    GenerationJobRunner,
    GenerationJobStore,
    LoadedSource,
    OutputDetail,
    PaidDispatchControl,
    PaidDispatchPaused,
    RetryPolicy,
    SOURCE_REQUEST_MODEL,
    SourceGenerationConfig,
    SourceGenerationBackend,
    build_source_request_contract,
    estimate_plan_cost,
    load_processed_source,
    plan_source_generation,
)
from source_generation.validation import (
    inspect_pipeline_response,
    make_pipeline_response_validator,
)
from source_generation.requests import (
    load_source_batch_instructions,
    normalise_source_request_contract,
    render_chunk_input,
    source_request_requires_sentence_translations,
    source_request_uses_obsolete_context_translation_protocol,
    source_request_uses_grouped_source_results,
    source_request_uses_occurrence_locators,
    source_request_uses_sentence_arrays,
    source_request_uses_split_contextual_cards,
)
from source_generation.planning import (
    BATCH_PRICING,
    STANDARD_PRICING,
    _make_occurrence_locator,
    estimate_text_tokens,
    price_source_usage,
)
from source_generation.jobs import _RequestStartGate
import pipeline_store
import process_text
import templates
import source_workflow


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


def make_source(
        section_texts=("甲乙。丙丁。戊己。",),
        *,
        source_language_key="classical_chinese",
        tokenizer=None):
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
        source_language_key=source_language_key,
        pages=(page,),
        sections=sections,
        canonical_text=canonical,
        cleaner_version="fixture-cleaner")
    build = build_vocabulary(
        snapshot,
        tokenizer or CharacterTokenizer(),
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
        excluded_words=(),
        request_protocol="v9",
        reasoning_effort="none",
        execution_mode="standard",
        model="gpt-5.4-mini",
        automatic_repair=False,
        retain_excluded_source_contexts=False):
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
            model=model,
            request_protocol=request_protocol,
            reasoning_effort=reasoning_effort,
            execution_mode=execution_mode,
            automatic_repair=automatic_repair,
            excluded_words=tuple(excluded_words),
            retain_excluded_source_contexts=(
                retain_excluded_source_contexts)))


class SourcePlanningTests(unittest.TestCase):
    def test_occurrence_locator_finds_an_overlapping_identical_spelling(self):
        context = SimpleNamespace(
            context_id="context-overlap",
            start_offset=0,
            text="佳佳佳")
        word = SimpleNamespace(
            rank=17,
            surface="佳佳",
            start_offset=1,
            end_offset=3)

        locator = _make_occurrence_locator(word, context)

        self.assertEqual(locator.target, "佳佳")
        self.assertEqual(locator.literal_match_ordinal, 2)
        self.assertEqual(locator.marked_excerpt, "佳⟪TARGET⟫佳佳⟪/TARGET⟫")

    def test_default_request_size_and_worker_count_are_30_and_8(self):
        config = SourceGenerationConfig(source_key="fixture_source")

        self.assertEqual(config.chunk_size, 30)
        self.assertEqual(config.concurrency, 8)
        self.assertEqual(config.request_protocol, "v10")
        self.assertEqual(config.reasoning_effort, "low")
        self.assertEqual(config.execution_mode, "standard")
        self.assertEqual(config.model, "gpt-5.4-mini")
        self.assertFalse(config.automatic_repair)

    def test_retired_models_and_repair_caps_are_rejected(self):
        with self.assertRaises(ValueError):
            SourceGenerationConfig(
                source_key="fixture_source",
                model="unknown")
        with self.assertRaises(ValueError):
            SourceGenerationConfig(
                source_key="fixture_source",
                model="gpt-5.6-luna")
        with self.assertRaises(ValueError):
            SourceGenerationConfig(
                source_key="fixture_source",
                execution_mode="economy",
                automatic_repair=True)
        with self.assertRaises(ValueError):
            SourceGenerationConfig(
                source_key="fixture_source",
                max_automatic_repairs=4)

    def test_request_protocol_reasoning_and_execution_settings_are_validated(
            self):
        legacy = SourceGenerationConfig.from_mapping({
            "source_key": "fixture_source",
            "protocol_version": 8,
            "reasoning_effort": "low",
            "processing_mode": "economy",
        })

        self.assertEqual(legacy.request_protocol, "v8")
        self.assertEqual(legacy.reasoning_effort, "low")
        self.assertEqual(legacy.execution_mode, "economy")
        for setting, value, message in (
                ("request_protocol", "v11", "protocol"),
                ("reasoning_effort", "medium", "reasoning"),
                ("execution_mode", "fast", "execution")):
            with self.subTest(setting=setting), self.assertRaisesRegex(
                    ValueError,
                    message):
                SourceGenerationConfig(
                    source_key="fixture_source",
                    **{setting: value})
        with self.assertRaisesRegex(ValueError, "requires reasoning"):
            SourceGenerationConfig(
                source_key="fixture_source",
                request_protocol="v8",
                reasoning_effort="none")
        with self.assertRaisesRegex(
                ValueError,
                "context mode other than None"):
            SourceGenerationConfig(
                source_key="fixture_source",
                context_mode=ContextMode.NONE,
                retain_excluded_source_contexts=True)

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
            "source_prefix_token_limit": 100,
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
        self.assertEqual(config.source_prefix_token_limit, 100)
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

    def test_source_prefix_limit_requires_a_positive_integer(self):
        for invalid in (True, 0, -1, 1.5, "100"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                        ValueError,
                        "Source prefix length"):
                    SourceGenerationConfig(
                        source_key="fixture_source",
                        source_prefix_token_limit=invalid)

    def test_source_prefix_counts_running_tokens_before_deduplication(self):
        source = make_source(("甲甲乙甲丙。",))

        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                source_prefix_token_limit=4,
                chunk_size=1,
                context_mode=ContextMode.NONE))

        self.assertEqual(plan.original_word_count, 3)
        self.assertEqual(plan.source_prefix_token_count, 4)
        self.assertEqual(plan.prefix_unique_word_count, 2)
        self.assertEqual(plan.word_count, 2)
        self.assertEqual(plan.excluded_word_count, 0)
        self.assertEqual(
            tuple(
                word.surface
                for chunk in plan.chunks
                for word in chunk.words),
            ("甲", "乙"))
        self.assertEqual(
            tuple(word.rank for chunk in plan.chunks for word in chunk.words),
            (1, 2))

    def test_source_prefix_clamps_at_end_and_then_applies_exclusions(self):
        source = make_source(("甲甲乙甲丙。",))

        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                source_prefix_token_limit=100,
                excluded_words=("乙",),
                context_mode=ContextMode.NONE))

        self.assertEqual(plan.source_prefix_token_count, 5)
        self.assertEqual(plan.prefix_unique_word_count, 3)
        self.assertEqual(plan.excluded_word_count, 1)
        self.assertEqual(
            tuple(
                word.surface
                for chunk in plan.chunks
                for word in chunk.words),
            ("甲", "丙"))

    def test_source_prefix_changes_plan_identity_at_the_token_boundary(self):
        source = make_source(("甲甲乙甲丙。",))
        first_two_tokens = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                source_prefix_token_limit=2))
        first_three_tokens = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                source_prefix_token_limit=3))

        self.assertNotEqual(
            first_two_tokens.plan_id,
            first_three_tokens.plan_id)
        self.assertEqual(first_two_tokens.word_count, 1)
        self.assertEqual(first_three_tokens.word_count, 2)

    def test_historical_english_exclusion_is_casefolded_like_candidates(self):
        source = make_source(
            ("Whan whan that Aprill.",),
            source_language_key="middle_english",
            tokenizer=(
                HistoricalEnglishTokenizer.for_middle_english()))
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                excluded_words=("WHAN",)))

        self.assertEqual(plan.original_word_count, 3)
        self.assertEqual(plan.excluded_word_count, 1)
        self.assertEqual(
            tuple(
                word.surface
                for chunk in plan.chunks
                for word in chunk.words),
            ("that", "Aprill"))

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
        self.assertEqual(
            payload["words"][0]["occurrence_span"],
            [0, 1])
        self.assertEqual(
            payload["words"][0]["occurrence_locator"],
            {
                "before": "",
                "target": "甲",
                "after": "乙。",
                "literal_match_ordinal": 1,
                "marked_excerpt": "⟪TARGET⟫甲⟪/TARGET⟫乙。",
            })
        self.assertEqual(
            payload["words"][1]["occurrence_span"],
            [1, 2])
        self.assertEqual(
            payload["words"][1]["occurrence_locator"],
            {
                "before": "甲",
                "target": "乙",
                "after": "。",
                "literal_match_ordinal": 1,
                "marked_excerpt": "甲⟪TARGET⟫乙⟪/TARGET⟫。",
            })
        legacy_word = replace(
            chunk.words[0],
            occurrence_locator=None)
        legacy_chunk = replace(
            chunk,
            words=(legacy_word,))
        self.assertIn(
            "occurrence_span",
            legacy_chunk.request_payload()["words"][0])
        self.assertNotIn(
            "occurrence_locator",
            legacy_chunk.request_payload()["words"][0])

    def test_context_translation_ownership_crosses_lexical_chunk_boundaries(
            self):
        plan = make_plan(
            chunk_size=1,
            request_protocol="v10",
            retain_excluded_source_contexts=True)

        context_ids = [
            chunk.contexts[0].context_id
            for chunk in plan.chunks
        ]
        translation_ids = [
            context_id
            for chunk in plan.chunks
            for context_id in chunk.requested_source_context_ids
        ]

        self.assertEqual(len(plan.chunks), 6)
        self.assertEqual(len(set(context_ids)), 3)
        self.assertEqual(len(translation_ids), 3)
        self.assertEqual(len(set(translation_ids)), 3)
        for context_id in set(context_ids):
            matching = [
                chunk
                for chunk in plan.chunks
                if chunk.contexts[0].context_id == context_id
            ]
            self.assertEqual(len(matching), 2)
            self.assertEqual(
                matching[0].requested_source_context_ids,
                (context_id,))
            self.assertEqual(
                matching[1].requested_source_context_ids,
                ())
            self.assertEqual(
                json.loads(render_chunk_input(
                    matching[1],
                    protocol_version=10))["contexts"][0]["context_id"],
                context_id)

        estimate = estimate_plan_cost(
            plan,
            OutputDetail(
                ("translation",),
                include_example_sentences=True),
            use_source_for_example_sentences=True,
            request_protocol="v10")
        self.assertEqual(estimate.request_count, 6)

    def test_excluded_context_only_chunks_translate_each_sentence_once(self):
        plan = make_plan(
            chunk_size=1,
            request_protocol="v10",
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"),
            retain_excluded_source_contexts=True)

        estimate = estimate_plan_cost(
            plan,
            OutputDetail((), include_example_sentences=False),
            use_source_for_example_sentences=True,
            request_protocol="v10")

        self.assertEqual(plan.word_count, 0)
        self.assertEqual(len(plan.chunks), 6)
        self.assertEqual(
            sum(
                len(chunk.requested_source_context_ids)
                for chunk in plan.chunks),
            3)
        self.assertEqual(estimate.request_count, 3)

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

    def test_excluded_words_still_anchor_requested_source_sentence_cards(self):
        plan = make_plan(
            chunk_size=2,
            request_protocol="v10",
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"),
            retain_excluded_source_contexts=True)

        self.assertEqual(plan.word_count, 0)
        self.assertEqual(len(plan.chunks), 3)
        self.assertTrue(all(
            not chunk.words
            for chunk in plan.chunks))
        self.assertEqual(
            tuple(
                anchor.surface
                for chunk in plan.chunks
                for anchor in chunk.context_anchors),
            ("甲", "乙", "丙", "丁", "戊", "己"))
        self.assertTrue(all(
            chunk.contexts
            for chunk in plan.chunks))
        self.assertTrue(all(
            chunk.request_payload()["words"] == []
            for chunk in plan.chunks))
        self.assertTrue(all(
            "context_anchors" not in render_chunk_input(
                chunk,
                protocol_version=10)
            for chunk in plan.chunks))

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
        self.assertEqual(
            detailed_estimate.assumptions["reasoning_effort"],
            "low")
        self.assertEqual(
            detailed_estimate.assumptions[
                "reasoning_output_reserve_multiplier"],
            1.20)

    def test_cost_estimate_counts_exact_per_chunk_response_schemas(self):
        plan = make_plan(
            chunk_size=2,
            request_protocol="v8",
            reasoning_effort="low")
        formats = {
            chunk.chunk_id: {
                "type": "json_schema",
                "strict": True,
                "name": f"schema_{chunk.index}",
                "schema": {
                    "type": "object",
                    "description": "rank keyed " * (100 + chunk.index),
                },
            }
            for chunk in plan.chunks
        }

        baseline = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")
        exact = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt",
            response_formats_by_chunk=formats)

        self.assertGreater(exact.schema_tokens, baseline.schema_tokens)
        self.assertEqual(
            set(exact.assumptions["schema_tokens_by_chunk"]),
            {chunk.chunk_id for chunk in plan.chunks})
        with self.assertRaisesRegex(
                ValueError,
                "do not match the generation plan"):
            estimate_plan_cost(
                plan,
                OutputDetail(("translation",)),
                response_formats_by_chunk={})

    def test_empty_v8_source_plan_accepts_its_empty_schema_map(self):
        plan = make_plan(
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"),
            request_protocol="v8",
            reasoning_effort="low")

        estimate = estimate_plan_cost(
            plan,
            OutputDetail(
                ("translation",),
                include_example_sentences=True),
            use_source_for_example_sentences=True,
            response_formats_by_chunk={})

        self.assertEqual(estimate.request_count, 0)
        self.assertEqual(estimate.estimated_input_tokens, 0)
        self.assertEqual(estimate.estimated_output_tokens, 0)
        self.assertEqual(
            estimate.assumptions["schema_estimation_mode"],
            "per_chunk_exact")

    def test_cost_estimate_counts_exact_fixed_v9_response_schema(self):
        plan = make_plan(chunk_size=2)
        response_format = {
            "type": "json_schema",
            "strict": True,
            "name": "fixed_source_v9",
            "schema": {
                "type": "object",
                "description": "fixed rank arrays " * 300,
            },
        }

        baseline = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")
        exact = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt",
            response_format=response_format)

        expected_per_request = estimate_text_tokens(json.dumps(
            response_format,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":")))
        self.assertEqual(
            exact.schema_tokens,
            expected_per_request * exact.request_count)
        self.assertGreater(exact.schema_tokens, baseline.schema_tokens)
        self.assertEqual(
            exact.assumptions["schema_estimation_mode"],
            "fixed_exact")
        with self.assertRaisesRegex(ValueError, "either one fixed"):
            estimate_plan_cost(
                plan,
                OutputDetail(("translation",)),
                response_format=response_format,
                response_formats_by_chunk={
                    chunk.chunk_id: response_format
                    for chunk in plan.chunks
                })

    def test_cost_estimate_uses_exact_protocol_payload_rendering(self):
        for protocol, reasoning in (
                ("v8", "low"),
                ("v9", "none"),
                ("v10", "none")):
            with self.subTest(protocol=protocol):
                plan = make_plan(
                    chunk_size=2,
                    request_protocol=protocol,
                    reasoning_effort=reasoning)
                estimate = estimate_plan_cost(
                    plan,
                    OutputDetail(("translation",)))
                expected = sum(
                    estimate_text_tokens(render_chunk_input(
                        chunk,
                        protocol_version=int(protocol[1:])))
                    for chunk in plan.chunks
                )

                self.assertEqual(estimate.payload_tokens, expected)

    def test_cost_estimate_prices_standard_and_economy_at_published_rates(
            self):
        standard_plan = make_plan(
            chunk_size=2,
            execution_mode="standard")
        economy_plan = make_plan(
            chunk_size=2,
            execution_mode="economy")

        standard = estimate_plan_cost(
            standard_plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")
        economy = estimate_plan_cost(
            economy_plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")

        self.assertEqual(STANDARD_PRICING.input_usd_per_million, 0.75)
        self.assertEqual(
            STANDARD_PRICING.cached_input_usd_per_million,
            0.075)
        self.assertEqual(STANDARD_PRICING.output_usd_per_million, 4.50)
        self.assertEqual(
            BATCH_PRICING.input_usd_per_million,
            STANDARD_PRICING.input_usd_per_million / 2)
        self.assertEqual(
            BATCH_PRICING.cached_input_usd_per_million,
            STANDARD_PRICING.cached_input_usd_per_million / 2)
        self.assertEqual(
            BATCH_PRICING.output_usd_per_million,
            STANDARD_PRICING.output_usd_per_million / 2)
        self.assertEqual(
            economy.estimated_input_tokens,
            standard.estimated_input_tokens)
        self.assertEqual(
            economy.estimated_output_tokens,
            standard.estimated_output_tokens)
        self.assertAlmostEqual(
            economy.estimated_total_usd,
            standard.estimated_total_usd / 2)
        self.assertAlmostEqual(
            economy.low_total_usd,
            standard.low_total_usd / 2)
        self.assertAlmostEqual(
            economy.high_total_usd,
            standard.high_total_usd / 2)

    def test_automatic_repair_reserve_is_bounded_and_not_in_headline(self):
        plan = make_plan(
            chunk_size=2,
            automatic_repair=True)

        estimate = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="Prompt")
        baseline = estimate_plan_cost(
            make_plan(chunk_size=2),
            OutputDetail(("translation",)),
            prompt_text="Prompt")

        self.assertEqual(
            estimate.assumptions["automatic_repair_max_requests"],
            estimate.request_count * 3)
        self.assertGreater(
            estimate.assumptions["automatic_repair_reserve_aud"],
            0)
        self.assertEqual(
            estimate.estimated_total_usd,
            baseline.estimated_total_usd)

    def test_cost_estimate_uses_cache_reads_centrally_and_cold_input_high(
            self):
        plan = make_plan(chunk_size=2)
        estimate = estimate_plan_cost(
            plan,
            OutputDetail(("translation",)),
            prompt_text="identical stable prompt prefix " * 1_000)

        self.assertGreater(estimate.estimated_cached_input_tokens, 0)
        self.assertEqual(
            estimate.estimated_input_tokens,
            (
                estimate.estimated_cached_input_tokens
                + estimate.estimated_uncached_input_tokens))
        self.assertAlmostEqual(
            estimate.estimated_input_usd,
            (
                estimate.estimated_cached_input_usd
                + estimate.estimated_uncached_input_usd))
        self.assertGreater(
            estimate.assumptions["cold_input_usd"],
            estimate.estimated_input_usd)
        self.assertEqual(
            estimate.assumptions["high_input_assumption"],
            "all input tokens are cold")

    def test_source_response_shape_translates_each_deduplicated_context_once(
            self):
        shared_source = make_source(("甲乙。",))
        separate_source = make_source(("甲。乙。",))
        config = SourceGenerationConfig(
            source_key="fixture_source",
            chunk_size=2,
            context_mode=ContextMode.SENTENCE)
        shared_plan = plan_source_generation(shared_source, config)
        separate_plan = plan_source_generation(separate_source, config)
        detail = OutputDetail(
            ("translation",),
            include_example_sentences=True)

        shared = estimate_plan_cost(
            shared_plan,
            detail,
            use_source_for_example_sentences=True)
        remembered = estimate_plan_cost(
            shared_plan,
            detail,
            use_source_for_example_sentences=True,
            request_protocol="v10",
            translation_memory_by_chunk={
                shared_plan.chunks[0].chunk_id: {
                    shared_plan.chunks[0].contexts[0].context_id: {
                        "translation": "A remembered translation.",
                    },
                },
            })
        separate = estimate_plan_cost(
            separate_plan,
            detail,
            use_source_for_example_sentences=True)
        with_nuance = estimate_plan_cost(
            shared_plan,
            detail,
            use_source_for_example_sentences=True,
            include_source_context_nuance=True)
        generated_shared = estimate_plan_cost(shared_plan, detail)
        generated_separate = estimate_plan_cost(separate_plan, detail)

        self.assertEqual(len(shared_plan.chunks[0].contexts), 1)
        self.assertEqual(len(separate_plan.chunks[0].contexts), 2)
        self.assertGreater(
            separate.estimated_output_tokens,
            shared.estimated_output_tokens)
        self.assertLess(
            remembered.estimated_output_tokens,
            shared.estimated_output_tokens)
        self.assertGreater(
            with_nuance.estimated_output_tokens,
            shared.estimated_output_tokens)
        self.assertEqual(
            remembered.assumptions["translation_memory_hit_count"],
            1)
        self.assertEqual(
            generated_separate.estimated_output_tokens,
            generated_shared.estimated_output_tokens)
        self.assertEqual(
            shared.assumptions["response_shape"],
            "source_sentences_plus_lexical_senses")
        shape = shared.assumptions["output_shape"]
        self.assertFalse(
            shape["contextual_senses_include_generated_examples"])
        self.assertEqual(
            shape["additional_senses_per_word"]["central"],
            1.0)
        self.assertTrue(
            shape["additional_senses_include_generated_examples"])
        self.assertFalse(shape["source_context_nuance"])
        self.assertTrue(
            with_nuance.assumptions[
                "output_shape"]["source_context_nuance"])
        self.assertEqual(
            shape["source_context_translations"],
            "one output translation per deduplicated context")

    def test_reasoning_reserve_is_scenario_specific_not_a_blanket_multiplier(
            self):
        without_reasoning = estimate_plan_cost(
            make_plan(
                chunk_size=2,
                reasoning_effort="none"),
            OutputDetail(("translation",)))
        with_low_reasoning = estimate_plan_cost(
            make_plan(
                chunk_size=2,
                reasoning_effort="low"),
            OutputDetail(("translation",)))

        self.assertEqual(
            without_reasoning.assumptions[
                "reasoning_output_reserve_multiplier"],
            1.0)
        self.assertEqual(
            with_low_reasoning.assumptions[
                "reasoning_output_reserve_multiplier"],
            1.20)
        self.assertNotEqual(
            with_low_reasoning.assumptions[
                "reasoning_output_reserve_multiplier"],
            1.50)
        self.assertGreater(
            with_low_reasoning.estimated_output_tokens,
            without_reasoning.estimated_output_tokens)

    def test_exact_usage_pricing_includes_cached_input_and_aud(self):
        usage = {
            "uncached_input_tokens": 1_000_000,
            "cached_input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        }

        standard = price_source_usage(
            usage,
            execution_mode="standard")
        economy = price_source_usage(
            usage,
            execution_mode="economy")

        self.assertEqual(standard["input_tokens"], 2_000_000)
        self.assertEqual(standard["total_tokens"], 3_000_000)
        self.assertEqual(standard["uncached_input_usd"], 0.75)
        self.assertEqual(standard["cached_input_usd"], 0.075)
        self.assertEqual(standard["output_usd"], 4.50)
        self.assertEqual(standard["total_usd"], 5.325)
        self.assertAlmostEqual(
            standard["total_aud"],
            standard["total_usd"] * standard["usd_to_aud_rate"])
        self.assertAlmostEqual(
            economy["total_usd"],
            standard["total_usd"] / 2)
        self.assertIn("published", standard["pricing_label"])
        self.assertIn("RBA", standard["aud_exchange_rate"])
        with_search = price_source_usage(
            {
                **usage,
                "web_search_calls": 2,
            },
            execution_mode="standard")
        self.assertEqual(with_search["web_search_usd"], 0.02)
        self.assertEqual(
            with_search["total_usd"],
            standard["total_usd"] + 0.02)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            price_source_usage({
                **usage,
                "cached_input_tokens": True,
            })
        with self.assertRaisesRegex(ValueError, "execution mode"):
            price_source_usage(usage, execution_mode="instant")

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


def context_classical_pipeline():
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
                enabled=card.direction_key == "context",
                fields=selected_fields)
            for card in settings.cards),
        share_field_settings=True,
        shared_fields=selected_fields)
    return pipeline_store.replace_active_language_settings(
        pipeline,
        settings,
        (settings,),
        active_language_key="classical_chinese")


def detailed_context_classical_pipeline():
    pipeline = context_classical_pipeline()
    settings = pipeline_store.get_language_settings(
        pipeline,
        "classical_chinese")
    selected_fields = (
        pipeline_store.FieldSetting("translation", "english"),
        pipeline_store.FieldSetting("dictionary_meaning", "english"),
    )
    settings = replace(
        settings,
        cards=tuple(
            replace(card, fields=selected_fields)
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

    def test_all_known_words_can_produce_a_source_sentence_without_lexical_cards(
            self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            request_protocol="v10",
            excluded_words=("甲", "乙", "丙", "丁", "戊", "己"),
            retain_excluded_source_contexts=True).chunks[0]
        response_format = process_text.build_compact_source_response_format(
            pipeline,
            protocol_version=10,
            chunk=chunk,
            source_lexical_only=True,
            include_generated_examples=False)
        term_schema = response_format["schema"]["properties"][
            "term_results"]
        self.assertEqual(term_schema["minItems"], 0)
        self.assertEqual(term_schema["maxItems"], 0)

        raw = json.dumps({
            "term_results": [],
            "source_context_translations": [
                {
                    "context_id": context.context_id,
                    "translation": "A complete source translation.",
                }
                for context in chunk.contexts
            ],
        }, ensure_ascii=False)
        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_compact_source_results=True,
            require_generated_examples=False)

        self.assertTrue(report["valid"], report["problems"])
        canonical = report["canonical_response"]
        self.assertEqual(canonical["cards"], [])
        self.assertEqual(
            canonical["source_contexts"][0]["ranked_terms"],
            [
                {"rank": 1, "term": "甲"},
                {"rank": 2, "term": "乙"},
            ])

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

    def test_source_context_is_inserted_only_for_each_first_sense(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE_NEIGHBORS).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = (
            process_text.get_response_field_names(pipeline))
        exact_contexts = {
            word.surface: next(
                context.text
                for context in chunk.contexts
                if context.context_id == word.context_id)
            for word in chunk.words
        }
        additional_examples = (
            "<strong>甲</strong>一。|<strong>甲</strong>二。|"
            "<strong>甲</strong>三。")
        additional_translations = (
            "First 甲 example.|Second 甲 example.|"
            "Third 甲 example.")
        raw = json.dumps({
            "cards": [
                {
                    term_field: "甲",
                    sentence_translations_field: "Exact 甲 source context.",
                    meaning_field: "contextual",
                },
                {
                    term_field: "甲",
                    sentence_field: additional_examples,
                    sentence_translations_field: additional_translations,
                    meaning_field: "other disjoint sense",
                },
                {
                    term_field: "乙",
                    sentence_translations_field: "Exact 乙 source context.",
                    meaning_field: "contextual",
                },
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)

        self.assertEqual(
            validated["cards"][0][sentence_field],
            process_text.encode_source_context_block(
                process_text.emphasize_term_in_sentences(
                    exact_contexts["甲"],
                    "甲",
                    pipeline.language_key)))
        self.assertNotIn(
            "|",
            validated["cards"][0][sentence_field])
        self.assertEqual(
            validated["cards"][1][sentence_field],
            additional_examples)
        self.assertEqual(
            validated["cards"][1][sentence_translations_field],
            additional_translations)
        self.assertEqual(
            validated["cards"][2][sentence_field],
            process_text.encode_source_context_block(
                process_text.emphasize_term_in_sentences(
                    exact_contexts["乙"],
                    "乙",
                    pipeline.language_key)))

    def test_v5_context_map_is_shared_and_arrays_are_canonicalized(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        context = chunk.contexts[0]
        raw = json.dumps({
            "source_context_translations": [{
                "context_id": context.context_id,
                "translation": "The complete shared source context.",
            }],
            "cards": [
                {
                    term_field: "甲",
                    meaning_field: "contextual",
                },
                {
                    term_field: "甲",
                    sentence_field: [
                        "<strong>甲</strong>一。",
                        "<strong>甲</strong>二。",
                        "<strong>甲</strong>三。",
                    ],
                    sentence_translations_field: [
                        "First.",
                        "Second.",
                        "Third.",
                    ],
                    meaning_field: "other sense",
                },
                {
                    term_field: "乙",
                    meaning_field: "contextual",
                },
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)["cards"]

        self.assertEqual(
            validated[0][sentence_translations_field],
            "The complete shared source context.")
        self.assertEqual(
            validated[2][sentence_translations_field],
            "The complete shared source context.")
        self.assertEqual(
            validated[1][sentence_field],
            (
                "<strong>甲</strong>一。|<strong>甲</strong>二。|"
                "<strong>甲</strong>三。"))
        self.assertEqual(
            validated[1][sentence_translations_field],
            "First.|Second.|Third.")

    def test_v6_split_cards_merge_contextual_first_in_source_rank_order(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True,
            use_split_source_context_cards=True)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        examples = [
            "<strong>甲</strong>一。",
            "<strong>甲</strong>二。",
            "<strong>甲</strong>三。",
        ]
        raw = json.dumps({
            "contextual_cards": [
                {term_field: "乙", meaning_field: "乙 contextual"},
                {term_field: "甲", meaning_field: "甲 contextual"},
            ],
            "additional_sense_cards": [{
                term_field: "甲",
                sentence_field: examples,
                sentence_translations_field: [
                    "First.", "Second.", "Third.",
                ],
                meaning_field: "甲 additional",
            }],
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "The complete shared source context.",
            }],
        }, ensure_ascii=False)

        cards = validator(raw, chunk)["cards"]

        self.assertEqual(
            [card[term_field] for card in cards],
            ["甲", "甲", "乙"])
        self.assertEqual(
            [card[meaning_field] for card in cards],
            ["甲 contextual", "甲 additional", "乙 contextual"])
        self.assertTrue(cards[0][sentence_field].startswith(
            templates.SOURCE_CONTEXT_BLOCK_PREFIX))
        self.assertEqual(
            cards[1][sentence_field],
            "|".join(examples))

    def test_v6_contextual_cards_require_exactly_one_requested_term(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        term_field, _sentences, _translations, meaning_field = (
            process_text.get_response_field_names(pipeline))
        context_map = [{
            "context_id": chunk.contexts[0].context_id,
            "translation": "The complete shared source context.",
        }]
        cases = {
            "missing_contextual_source_card": [
                {term_field: "甲", meaning_field: "first"},
            ],
            "duplicate_contextual_source_card": [
                {term_field: "甲", meaning_field: "first"},
                {term_field: "甲", meaning_field: "duplicate"},
                {term_field: "乙", meaning_field: "second"},
            ],
            "unexpected_contextual_source_term": [
                {term_field: "甲", meaning_field: "first"},
                {term_field: "乙", meaning_field: "second"},
                {term_field: "丙", meaning_field: "extra"},
            ],
        }

        for expected_code, contextual_cards in cases.items():
            with self.subTest(expected_code=expected_code):
                raw = json.dumps({
                    "contextual_cards": contextual_cards,
                    "additional_sense_cards": [],
                    "source_context_translations": context_map,
                }, ensure_ascii=False)
                report = inspect_pipeline_response(
                    raw,
                    pipeline,
                    chunk,
                    use_source_for_example_sentences=True,
                    sentence_collections_as_arrays=True,
                    use_source_context_translation_map=True,
                    use_split_source_context_cards=True)
                matching = [
                    problem
                    for problem in report["problems"]
                    if problem["code"] == expected_code
                ]
                self.assertTrue(matching)
                self.assertTrue(all(
                    not problem["overrideable"]
                    for problem in matching))
                self.assertFalse(report["can_manually_accept"])

    def test_v6_additional_sense_requires_both_four_item_arrays(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        term_field, _sentences, _translations, meaning_field = (
            process_text.get_response_field_names(pipeline))
        raw = json.dumps({
            "contextual_cards": [
                {term_field: "甲", meaning_field: "first"},
                {term_field: "乙", meaning_field: "second"},
            ],
            "additional_sense_cards": [{
                term_field: "甲",
                meaning_field: "additional",
            }],
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "The complete shared source context.",
            }],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True,
            use_split_source_context_cards=True)

        self.assertIn(
            "invalid_card_fields",
            {
                problem["code"]
                for problem in report["problems"]
            })
        self.assertFalse(report["can_manually_accept"])

    def test_v6_rejects_literal_pipe_inside_sentence_array_item(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "contextual_cards": [
                {term_field: "甲", meaning_field: "first"},
                {term_field: "乙", meaning_field: "second"},
            ],
            "additional_sense_cards": [{
                term_field: "甲",
                sentence_field: [
                    "<strong>甲</strong>一|續。",
                    "<strong>甲</strong>二。",
                    "<strong>甲</strong>三。",
                ],
                sentence_translations_field: [
                    "First.", "Second.", "Third.",
                ],
                meaning_field: "additional",
            }],
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "The complete shared source context.",
            }],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True,
            use_split_source_context_cards=True)

        problem = next(
            problem
            for problem in report["problems"]
            if (
                problem["code"]
                == "sentence_collection_item_contains_delimiter"))
        self.assertFalse(problem["overrideable"])
        self.assertFalse(report["can_manually_accept"])

    def test_v5_and_v6_require_complete_audited_occurrence_spans(self):
        pipeline = context_classical_pipeline()
        base_chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        chunk = replace(
            base_chunk,
            words=(
                replace(
                    base_chunk.words[0],
                    start_offset=None),
                *base_chunk.words[1:],
            ))
        term_field, _sentences, _translations, meaning_field = (
            process_text.get_response_field_names(pipeline))
        contextual_cards = [
            {term_field: "甲", meaning_field: "first"},
            {term_field: "乙", meaning_field: "second"},
        ]
        context_translations = [{
            "context_id": chunk.contexts[0].context_id,
            "translation": "The complete shared source context.",
        }]
        cases = (
            (
                "v5",
                {
                    "cards": contextual_cards,
                    "source_context_translations": context_translations,
                },
                False,
            ),
            (
                "v6",
                {
                    "contextual_cards": contextual_cards,
                    "additional_sense_cards": [],
                    "source_context_translations": context_translations,
                },
                True,
            ),
        )

        for version, payload, split_cards in cases:
            with self.subTest(version=version):
                report = inspect_pipeline_response(
                    json.dumps(payload, ensure_ascii=False),
                    pipeline,
                    chunk,
                    use_source_for_example_sentences=True,
                    sentence_collections_as_arrays=True,
                    use_source_context_translation_map=True,
                    use_split_source_context_cards=split_cards)

                problem = next(
                    problem
                    for problem in report["problems"]
                    if problem["code"] == "missing_source_occurrence_span")
                self.assertFalse(problem["overrideable"])
                self.assertFalse(report["can_manually_accept"])

    def test_v6_rejects_duplicate_examples_across_additional_senses(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        examples = [
            "<strong>甲</strong>一。",
            "<strong>甲</strong>二。",
            "<strong>甲</strong>三。",
        ]
        translations = ["First.", "Second.", "Third."]
        raw = json.dumps({
            "contextual_cards": [
                {term_field: "甲", meaning_field: "contextual"},
                {term_field: "乙", meaning_field: "second"},
            ],
            "additional_sense_cards": [
                {
                    term_field: "甲",
                    sentence_field: examples,
                    sentence_translations_field: translations,
                    meaning_field: "additional one",
                },
                {
                    term_field: "甲",
                    sentence_field: list(reversed(examples)),
                    sentence_translations_field: list(
                        reversed(translations)),
                    meaning_field: "additional two",
                },
            ],
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "The complete shared source context.",
            }],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True,
            use_split_source_context_cards=True)

        problem = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "duplicate_additional_sense_examples")
        self.assertFalse(problem["overrideable"])

    def test_split_response_rejects_duplicate_contextual_and_additional_sense(
            self):
        pipeline = detailed_context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        (
            term_field,
            sentence_field,
            sentence_translation_field,
            translation_field,
            definition_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            process_text.SOURCE_CONTEXTUAL_CARDS_KEY: [
                {
                    term_field: "甲",
                    translation_field: "to say!",
                    definition_field: "to speak / say!",
                },
                {
                    term_field: "乙",
                    translation_field: "second",
                    definition_field: "the item after the first",
                },
            ],
            process_text.SOURCE_ADDITIONAL_SENSE_CARDS_KEY: [{
                term_field: "甲",
                sentence_field: [
                    "<strong>甲</strong>一。",
                    "<strong>甲</strong>二。",
                    "<strong>甲</strong>三。",
                ],
                sentence_translation_field: [
                    "First.",
                    "Second.",
                    "Third.",
                ],
                translation_field: "to say",
                definition_field: "To speak / say.",
            }],
            process_text.SOURCE_CONTEXT_TRANSLATIONS_KEY: [
                {
                    process_text.SOURCE_CONTEXT_ID_FIELD_NAME: (
                        context.context_id),
                    process_text.SOURCE_CONTEXT_TRANSLATION_FIELD_NAME: (
                        "The complete retained context."),
                }
                for context in chunk.contexts
            ],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True,
            require_sentence_translations=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True,
            use_split_source_context_cards=True)

        problem = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "duplicate_source_sense")
        self.assertFalse(report["valid"])
        self.assertFalse(problem["overrideable"])
        self.assertEqual(
            problem["actual"]["duplicates_card"],
            1)
        self.assertFalse(report["can_manually_accept"])

    def test_v5_source_example_highlights_only_the_ranked_occurrence(self):
        pipeline = context_classical_pipeline()
        source = make_source(section_texts=("道可道。",))
        plan = plan_source_generation(
            source,
            SourceGenerationConfig(
                source_key=source.key,
                chunk_size=10,
                context_mode=ContextMode.SENTENCE,
                request_stagger_ms=0))
        chunk = plan.chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True)
        (
            term_field,
            sentence_field,
            _sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "The Way can be spoken as a way.",
            }],
            "cards": [
                {term_field: "道", meaning_field: "the ranked use"},
                {term_field: "可", meaning_field: "can"},
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)["cards"]
        source_example = validated[0][sentence_field]

        self.assertIn("<strong>道</strong>可道。", source_example)
        self.assertEqual(source_example.count("<strong>道</strong>"), 1)

    def test_v5_context_map_requires_each_requested_id_exactly_once(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True)
        term_field, _sentences, _translations, meaning_field = (
            process_text.get_response_field_names(pipeline))
        context_id = chunk.contexts[0].context_id
        cards = [
            {term_field: "甲", meaning_field: "first"},
            {term_field: "乙", meaning_field: "second"},
        ]

        invalid_maps = {
            "missing": [],
            "duplicate": [
                {"context_id": context_id, "translation": "First."},
                {"context_id": context_id, "translation": "Second."},
            ],
            "unexpected": [
                {"context_id": "not-requested", "translation": "Other."},
            ],
        }
        for case, context_map in invalid_maps.items():
            with self.subTest(case=case):
                raw = json.dumps({
                    "source_context_translations": context_map,
                    "cards": cards,
                }, ensure_ascii=False)
                with self.assertRaises(
                        process_text.GeneratedCardValidationError):
                    validator(raw, chunk)

    def test_v5_context_map_rejects_blank_pipe_and_html_translations(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        term_field, _sentences, _translations, meaning_field = (
            process_text.get_response_field_names(pipeline))
        cards = [
            {term_field: "甲", meaning_field: "first"},
            {term_field: "乙", meaning_field: "second"},
        ]
        cases = {
            "blank": ("", "blank_source_context_translation"),
            "pipe": (
                "First fragment.|Second fragment.",
                "sentence_translation_count_mismatch"),
            "html": (
                "A <strong>marked</strong> translation.",
                "source_context_translation_contains_html"),
        }

        for case, (translation, expected_code) in cases.items():
            with self.subTest(case=case):
                raw = json.dumps({
                    "source_context_translations": [{
                        "context_id": chunk.contexts[0].context_id,
                        "translation": translation,
                    }],
                    "cards": cards,
                }, ensure_ascii=False)
                report = inspect_pipeline_response(
                    raw,
                    pipeline,
                    chunk,
                    use_source_for_example_sentences=True,
                    sentence_collections_as_arrays=True,
                    use_source_context_translation_map=True)
                self.assertFalse(report["valid"])
                self.assertIn(
                    expected_code,
                    {
                        problem["code"]
                        for problem in report["problems"]
                    })

    def test_v5_rejects_pipe_strings_instead_of_fixed_arrays(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "Shared context.",
            }],
            "cards": [
                {term_field: "甲", meaning_field: "contextual"},
                {
                    term_field: "甲",
                    sentence_field: "一|二|三|四",
                    sentence_translations_field: "1|2|3|4",
                    meaning_field: "other",
                },
                {term_field: "乙", meaning_field: "contextual"},
            ],
        }, ensure_ascii=False)

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "JSON arrays"):
            validator(raw, chunk)

    def test_v5_does_not_silently_replace_model_context_examples(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "source_context_translations": [{
                "context_id": chunk.contexts[0].context_id,
                "translation": "Shared context.",
            }],
            "cards": [
                {
                    term_field: "甲",
                    sentence_field: [
                        "<strong>甲</strong>一。",
                        "<strong>甲</strong>二。",
                        "<strong>甲</strong>三。",
                    ],
                    sentence_translations_field: [
                        "One.", "Two.", "Three.", "Four.",
                    ],
                    meaning_field: "contextual",
                },
                {term_field: "乙", meaning_field: "contextual"},
            ],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True,
            sentence_collections_as_arrays=True,
            use_source_context_translation_map=True)

        problem_codes = {
            problem["code"]
            for problem in report["problems"]
        }
        self.assertIn(
            "contextual_card_contains_sentences",
            problem_codes)
        self.assertIn(
            "contextual_card_contains_translation",
            problem_codes)
        self.assertFalse(report["can_manually_accept"])

    def test_source_context_requires_one_matching_translation_block(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True)
        (
            term_field,
            _sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "cards": [
                {
                    term_field: "甲",
                    sentence_translations_field: (
                        "One.|Two.|Three.|Four."),
                    meaning_field: "first",
                },
                {
                    term_field: "乙",
                    sentence_translations_field: "Exact 乙 context.",
                    meaning_field: "second",
                },
            ],
        }, ensure_ascii=False)

        report = inspect_pipeline_response(
            raw,
            pipeline,
            chunk,
            use_source_for_example_sentences=True)
        mismatch = next(
            problem
            for problem in report["problems"]
            if problem["code"] == "sentence_translation_count_mismatch")
        self.assertFalse(mismatch["overrideable"])
        self.assertFalse(report["can_manually_accept"])
        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "needs one complete English translation"):
            validator(raw, chunk)

    def test_source_context_with_literal_pipe_remains_one_example(self):
        pipeline = context_classical_pipeline()
        base_chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        marker = templates.SOURCE_CONTEXT_ESCAPE_MARKER
        context = replace(
            base_chunk.contexts[0],
            text=f"甲曰 A|B，並記 {marker}。")
        chunk = replace(
            base_chunk,
            contexts=(
                context,
                *base_chunk.contexts[1:]),
            words=tuple(
                replace(
                    word,
                    start_offset=None,
                    end_offset=None)
                for word in base_chunk.words))
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = process_text.get_response_field_names(pipeline)
        raw = json.dumps({
            "cards": [
                {
                    term_field: "甲",
                    sentence_translations_field: (
                        "A complete translation containing no delimiter."),
                    meaning_field: "first",
                },
                {
                    term_field: "乙",
                    sentence_translations_field: "Exact 乙 context.",
                    meaning_field: "second",
                },
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)

        encoded = validated["cards"][0][sentence_field]
        self.assertTrue(encoded.startswith(
            templates.SOURCE_CONTEXT_BLOCK_PREFIX))
        self.assertNotIn(
            "|",
            encoded[len(templates.SOURCE_CONTEXT_BLOCK_PREFIX):])
        self.assertEqual(
            validated["cards"][0][sentence_translations_field],
            "A complete translation containing no delimiter.")

    def test_source_context_is_not_inserted_when_option_is_off(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE_NEIGHBORS).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=False)
        (
            term_field,
            sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = (
            process_text.get_response_field_names(pipeline))
        first_examples = (
            "<strong>甲</strong>一。|<strong>甲</strong>二。|"
            "<strong>甲</strong>三。")
        second_examples = (
            "<strong>乙</strong>一。|<strong>乙</strong>二。|"
            "<strong>乙</strong>三。")
        first_translations = "甲 one.|甲 two.|甲 three."
        second_translations = "乙 one.|乙 two.|乙 three."
        raw = json.dumps({
            "cards": [
                {
                    term_field: "甲",
                    sentence_field: first_examples,
                    sentence_translations_field: first_translations,
                    meaning_field: "first",
                },
                {
                    term_field: "乙",
                    sentence_field: second_examples,
                    sentence_translations_field: second_translations,
                    meaning_field: "second",
                },
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)

        self.assertEqual(
            validated["cards"][0][sentence_field],
            first_examples)
        self.assertEqual(
            validated["cards"][1][sentence_field],
            second_examples)
        self.assertNotEqual(
            validated["cards"][0][sentence_field],
            chunk.contexts[0].text)

    def test_additional_sense_cannot_omit_generated_examples(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True)
        (
            term_field,
            _sentence_field,
            sentence_translations_field,
            meaning_field,
        ) = (
            process_text.get_response_field_names(pipeline))
        raw = json.dumps({
            "cards": [
                {
                    term_field: "甲",
                    sentence_translations_field: "Exact 甲 context.",
                    meaning_field: "contextual",
                },
                {
                    term_field: "甲",
                    sentence_translations_field: (
                        "One.|Two.|Three.|Four."),
                    meaning_field: "other sense",
                },
                {
                    term_field: "乙",
                    sentence_translations_field: "Exact 乙 context.",
                    meaning_field: "contextual",
                },
            ],
        }, ensure_ascii=False)

        with self.assertRaisesRegex(
                process_text.GeneratedCardValidationError,
                "additional sense needs three"):
            validator(raw, chunk)


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
        self.assertEqual(source["token_occurrence_count"], 6)
        self.assertEqual(
            source["source_language_key"],
            "classical_chinese")
        self.assertEqual(estimate["candidate_count"], 5)
        self.assertEqual(estimate["excluded_candidate_count"], 1)
        self.assertIn("estimated_cost_low_usd", estimate)
        self.assertIn("estimated_cost_high_usd", estimate)
        self.assertIn("input_tokens", estimate)
        self.assertEqual(estimate["request_protocol"], "v10")
        self.assertEqual(estimate["reasoning_effort"], "low")
        self.assertEqual(estimate["execution_mode"], "standard")
        self.assertEqual(
            estimate["assumptions"]["schema_estimation_mode"],
            "fixed_exact")
        self.assertEqual(
            self.exclusion_calls[0][0].card_template_name,
            "Recognition")

    def test_estimator_reports_running_token_prefix_and_unique_candidates(self):
        request = {
            **self.request,
            "source_prefix_token_limit": 3,
        }

        estimate = self.backend.estimate(request)

        self.assertEqual(estimate["original_candidate_count"], 6)
        self.assertEqual(estimate["source_prefix_token_limit"], 3)
        self.assertEqual(estimate["source_prefix_token_count"], 3)
        self.assertEqual(estimate["prefix_unique_candidate_count"], 3)
        self.assertEqual(estimate["prefix_omitted_candidate_count"], 3)
        self.assertEqual(estimate["excluded_candidate_count"], 1)
        self.assertEqual(estimate["candidate_count"], 2)

        job = self.backend.create_job({
            **request,
            "paid_confirmed": True,
            "estimate": estimate,
        })
        manifest = json.loads(
            (job.path / "manifest.json").read_text(encoding="utf-8"))
        saved_plan = manifest["plan"]
        self.assertEqual(
            saved_plan["config"]["source_prefix_token_limit"],
            3)
        self.assertEqual(saved_plan["source_prefix_token_count"], 3)
        self.assertEqual(saved_plan["prefix_unique_word_count"], 3)

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

    def test_all_known_words_still_plan_source_sentence_translation_requests(
            self):
        self.backend.exclusion_resolver = lambda _spec, _loaded: (
            "甲", "乙", "丙", "丁", "戊", "己")
        request = {
            **self.request,
            "pipeline": context_classical_pipeline(),
            "use_source_for_example_sentences": True,
        }

        estimate = self.backend.estimate(request)

        self.assertEqual(estimate["candidate_count"], 0)
        self.assertEqual(estimate["excluded_candidate_count"], 6)
        self.assertEqual(estimate["request_count"], 3)
        job = self.backend.create_job({
            **request,
            "paid_confirmed": True,
            "estimate": estimate,
        })
        saved_chunks = [
            self.backend.jobs.load_chunk(job.job_id, chunk_id)
            for chunk_id in self.backend.jobs.chunk_ids(job.job_id)
        ]
        self.assertTrue(all(
            not chunk.words
            for chunk in saved_chunks))
        self.assertEqual(
            tuple(
                anchor.surface
                for chunk in saved_chunks
                for anchor in chunk.context_anchors),
            ("甲", "乙", "丙", "丁", "戊", "己"))

    def test_sentence_only_source_request_sends_contexts_without_lexical_shells(
            self):
        request = {
            **self.request,
            "chunk_size": 1,
            "pipeline": context_classical_pipeline(),
            "use_source_for_example_sentences": True,
            "exclude_anki": False,
            "anki_exclusion": None,
        }

        estimate = self.backend.estimate(request)
        job = self.backend.create_job({
            **request,
            "paid_confirmed": True,
            "estimate": estimate,
        })
        chunks = [
            self.backend.jobs.load_chunk(job.job_id, chunk_id)
            for chunk_id in self.backend.jobs.chunk_ids(job.job_id)
        ]

        self.assertEqual(estimate["candidate_count"], 0)
        self.assertEqual(estimate["source_sentence_card_count"], 3)
        self.assertEqual(estimate["request_count"], 3)
        self.assertTrue(
            estimate["assumptions"]["schema_cache_shared"])
        self.assertTrue(all(not chunk.words for chunk in chunks))
        self.assertEqual(
            [anchor.rank for chunk in chunks for anchor in chunk.context_anchors],
            [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(
            json.loads(render_chunk_input(
                chunk,
                protocol_version=10))["words"] == []
            for chunk in chunks))

    def test_fully_remembered_sentence_only_job_completes_without_provider(
            self):
        request = {
            **self.request,
            "chunk_size": 1,
            "pipeline": context_classical_pipeline(),
            "use_source_for_example_sentences": True,
            "exclude_anki": False,
            "anki_exclusion": None,
            "automatic_repair": True,
            "reasoning_effort": "none",
        }
        _loaded, initial_plan = self.backend._plan(request)
        for context in {
                context.context_id: context
                for chunk in initial_plan.chunks
                for context in chunk.contexts
        }.values():
            self.backend.translation_memory.commit(
                "classical_chinese",
                context.text,
                f"Remembered translation for {context.context_id}.",
                provenance={
                    "provenance_id": f"fixture-{context.context_id}",
                    "origin": "provider",
                    "fully_validated": True,
                    "manual_acceptance": False,
                })

        estimate = self.backend.estimate(request)
        job = self.backend.create_job({
            **request,
            "paid_confirmed": False,
            "estimate": estimate,
        })

        class NoProviderResponses:
            def create(self, **_kwargs):
                raise AssertionError("No provider request was expected.")

        snapshot = source_workflow.SourceWorkflowController(
            backend=self.backend)._run_job(
                job.job_id,
                context_classical_pipeline(),
                SimpleNamespace(responses=NoProviderResponses()))
        combined = json.loads(
            (snapshot.path / "combined.json").read_text(encoding="utf-8"))

        self.assertEqual(estimate["request_count"], 0)
        self.assertEqual(estimate["source_sentence_card_count"], 3)
        self.assertEqual(snapshot.overall_status, "completed")
        self.assertEqual(len(combined["source_contexts"]), 3)
        self.assertEqual(
            [context["word_ranks"] for context in combined["source_contexts"]],
            [[1, 2], [3, 4], [5, 6]])

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

    def test_preview_follows_a_new_latest_build_after_cache_warmup(self):
        first_page = self.backend.preview({
            "source_key": "fixture_source",
            "offset": 0,
            "limit": 10,
        })
        replacement = make_source(("庚辛。",))
        replacement_snapshot = replace(
            replacement.build.snapshot,
            edition="Local document: My Book")
        replacement_path = write_build(
            replace(
                replacement.build,
                snapshot=replacement_snapshot),
            self.corpus_root)

        second_page = self.backend.preview({
            "source_key": "fixture_source",
            "offset": 0,
            "limit": 10,
        })

        self.assertEqual(first_page["total"], 6)
        self.assertEqual(second_page["total"], 2)
        self.assertEqual(
            tuple(item["term"] for item in second_page["items"]),
            ("庚", "辛"))
        self.assertEqual(
            self.backend._load_source("fixture_source").run_path,
            replacement_path)

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
            {"effort": "low"})
        self.assertEqual(contract["schema_version"], 10)
        self.assertTrue(contract["response_format"]["strict"])
        self.assertEqual(
            json.loads(
                (job.path / "request_contract.json").read_text(
                    encoding="utf-8")),
            contract)
        self.assertEqual(
            set(contract["max_output_tokens_by_chunk"]),
            set(self.backend.jobs.chunk_ids(job.job_id)))

    def test_job_manifest_records_manual_offline_response_mode(self):
        estimate = self.backend.estimate(self.request)

        job = self.backend.create_job({
            **self.request,
            "paid_confirmed": False,
            "estimate": estimate,
            "manual_offline_responses": True,
        })
        manifest = json.loads(
            (job.path / "manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(
            manifest["request_metadata"]["manual_offline_responses"])
        self.assertFalse(
            manifest["request_metadata"]["paid_confirmed_at_creation"])

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

    def test_create_job_rejects_stale_protocol_and_repair_policy(
            self):
        estimate = self.backend.estimate(self.request)
        changed_settings = (
            {
                "request_protocol": "v8",
                "reasoning_effort": "low",
            },
            {"reasoning_effort": "none"},
            {"execution_mode": "economy"},
            {"automatic_repair": True},
            {"max_automatic_repairs": 2},
        )

        for settings in changed_settings:
            with (
                    self.subTest(settings=settings),
                    self.assertRaisesRegex(
                        PermissionError,
                        "changed after authorization")):
                self.backend.create_job({
                    **self.request,
                    **settings,
                    "paid_confirmed": True,
                    "estimate": estimate,
                })

        self.assertEqual(self.backend.jobs.list(), ())

    def test_request_contract_freezes_exact_prompt_schema_and_reasoning(self):
        pipeline = simple_classical_pipeline()

        contract = build_source_request_contract(pipeline)

        self.assertEqual(contract["schema_version"], 10)
        self.assertEqual(contract["model"], "gpt-5.4-mini")
        self.assertEqual(contract["reasoning"], {"effort": "low"})
        self.assertIn(
            "Here is the source batch JSON:",
            contract["composed_prompt"])
        self.assertIn(
            "never skip a supplied source term",
            contract["composed_prompt"])
        normalized_prompt = " ".join(
            contract["composed_prompt"].split())
        self.assertIn(
            "A context that unambiguously selects one sense does not remove "
            "this requirement to include other common disjoint senses",
            normalized_prompt)
        self.assertIn(
            "Within each response collection, preserve ascending source-rank "
            "order",
            normalized_prompt)
        self.assertEqual(
            contract["response_format"]["type"],
            "json_schema")
        self.assertTrue(contract["response_format"]["strict"])
        self.assertEqual(
            contract["max_output_tokens_by_chunk"],
            {})
        self.assertEqual(contract["response_formats_by_chunk"], {})
        self.assertEqual(contract["tools"], [])
        self.assertEqual(contract["max_tool_calls"], 0)

    def test_modern_english_source_contract_does_not_request_translations(self):
        pipeline = pipeline_store.default_pipeline()
        settings = pipeline_store.get_language_settings(pipeline, "english")
        settings = replace(
            settings,
            include_sentence_translations=False,
            cards=tuple(
                replace(
                    card,
                    enabled=card.direction_key in {
                        "context",
                        "word_to_meaning",
                    })
                for card in settings.cards))
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,),
            active_language_key="english")

        contract = build_source_request_contract(
            pipeline,
            use_source_for_example_sentences=True)
        prompt = contract["composed_prompt"]
        item_schema = contract["response_format"]["schema"]["properties"][
            "cards"]["items"]

        self.assertFalse(
            source_request_requires_sentence_translations(contract))
        self.assertNotIn(
            "Sentence Translations (English)",
            json.dumps(item_schema))
        self.assertIn(
            "all generated examples are already Modern English",
            prompt)
        self.assertNotIn(
            "repeat it as plain text",
            prompt)

    def test_modern_english_source_contract_requests_term_free_paraphrases(
            self):
        pipeline = pipeline_store.default_pipeline()
        settings = pipeline_store.get_language_settings(pipeline, "english")
        settings = replace(
            settings,
            cards=tuple(
                replace(
                    card,
                    enabled=card.direction_key in {
                        "context",
                        "word_to_meaning",
                    })
                for card in settings.cards))
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,),
            active_language_key="english")

        contract = build_source_request_contract(
            pipeline,
            use_source_for_example_sentences=True)

        self.assertTrue(
            source_request_requires_sentence_translations(contract))
        self.assertIn(
            "do not use the word or expression being defined",
            " ".join(contract["composed_prompt"].split()))

    def test_normalizer_keeps_legacy_none_reasoning_contracts_usable(self):
        contract = build_source_request_contract(
            simple_classical_pipeline())
        contract["reasoning"] = {"effort": "none"}

        normalized = normalise_source_request_contract(contract)

        self.assertEqual(normalized["reasoning"], {"effort": "none"})
        self.assertFalse(
            contract["use_source_for_example_sentences"])
        self.assertTrue(contract["require_sentence_translations"])
        self.assertTrue(
            contract["composed_prompt"].rstrip().endswith(
                "Here is the source batch JSON:"))

    def test_source_example_contract_freezes_prompt_and_omission_schema(self):
        pipeline = context_classical_pipeline()
        chunk = make_plan(
            chunk_size=1,
            context_mode=ContextMode.SENTENCE).chunks[0]

        contract = build_source_request_contract(
            pipeline,
            chunks=(chunk,),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")

        prompt_text = " ".join(contract["composed_prompt"].split())
        self.assertNotIn(
            '"Sentences" JSON array',
            contract["composed_prompt"])
        self.assertNotIn(
            "pipe-separated example sentences",
            contract["composed_prompt"])
        self.assertIn("structural association data only", prompt_text)
        self.assertIn(
            'empty "contextual_sense" object and an empty '
            '"additional_senses" array',
            prompt_text)
        self.assertIn(
            "The original sentence is already stored locally",
            prompt_text)
        self.assertIn(
            "Do not perform or return lexical analysis",
            prompt_text)
        self.assertNotIn("inspect every other occurrence", prompt_text)
        response_schema = contract["response_formats_by_chunk"][
            chunk.chunk_id]["schema"]
        self.assertEqual(
            set(response_schema["required"]),
            {
                "term_results",
                "source_context_translations",
            })
        self.assertNotIn("cards", response_schema["properties"])
        rank_key = str(chunk.words[0].rank)
        term_results_schema = response_schema["properties"][
            "term_results"]
        self.assertEqual(term_results_schema["required"], [rank_key])
        result_schema = term_results_schema["properties"][rank_key]
        contextual_item_schema = result_schema["properties"][
            "contextual_sense"]
        additional_item_schema = result_schema["properties"][
            "additional_senses"]["items"]
        self.assertNotIn(
            "Sentences",
            contextual_item_schema["properties"])
        self.assertNotIn(
            "Sentence Translations (English)",
            contextual_item_schema["properties"])
        self.assertNotIn(
            "Sentences",
            additional_item_schema["required"])
        self.assertNotIn(
            "Sentences",
            additional_item_schema["properties"])
        self.assertNotIn(
            "Sentence Translations (English)",
            additional_item_schema["properties"])
        context_schema = response_schema["properties"][
            "source_context_translations"]
        self.assertIn(
            "source_context_translations",
            response_schema["required"])
        self.assertEqual(
            context_schema["required"],
            [chunk.contexts[0].context_id])
        self.assertTrue(
            contract["use_source_for_example_sentences"])
        self.assertFalse(
            source_request_uses_split_contextual_cards(contract))
        self.assertTrue(
            source_request_uses_grouped_source_results(contract))
        self.assertTrue(
            source_request_uses_occurrence_locators(contract))
        self.assertIn(
            'Translate every context listed in '
            '"source_context_translation_ids" exactly once',
            prompt_text)
        self.assertNotIn('"occurrence_span"', contract["composed_prompt"])
        self.assertNotIn('"occurrence_locator"', contract["composed_prompt"])
        self.assertNotIn("grammatical role and lexical sense", prompt_text)
        self.assertTrue(
            contract["composed_prompt"].endswith(
                "Here is the source batch JSON:\n"))
        self.assertNotIn(
            "Here are the Classical Chinese words or expressions:",
            contract["composed_prompt"])

    def test_current_source_contract_uses_safe_output_token_floor(self):
        chunk = make_plan(
            chunk_size=1,
            context_mode=ContextMode.SENTENCE).chunks[0]

        contract = build_source_request_contract(
            context_classical_pipeline(),
            chunks=(chunk,),
            use_source_for_example_sentences=True)

        self.assertGreaterEqual(
            contract["max_output_tokens_by_chunk"][chunk.chunk_id],
            8_192)

    def test_version_three_contract_keeps_legacy_response_schema(self):
        pipeline = context_classical_pipeline()
        contract = build_source_request_contract(
            pipeline,
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        contract["schema_version"] = 3
        del contract["require_sentence_translations"]
        del contract["response_formats_by_chunk"]
        contract["response_format"] = process_text.build_response_format(
            pipeline,
            optional_fields=("Sentences",),
            require_sentence_translations=False)

        normalised = normalise_source_request_contract(contract)
        validator = make_pipeline_response_validator(
            pipeline,
            use_source_for_example_sentences=True,
            require_sentence_translations=(
                source_request_requires_sentence_translations(normalised)))
        chunk = make_plan(
            chunk_size=2,
            context_mode=ContextMode.SENTENCE).chunks[0]
        term_field, _sentence_field, meaning_field = (
            process_text.get_response_field_names(
                pipeline,
                include_sentence_translations=False))
        raw = json.dumps({
            "cards": [
                {term_field: "甲", meaning_field: "first"},
                {term_field: "乙", meaning_field: "second"},
            ],
        }, ensure_ascii=False)

        validated = validator(raw, chunk)

        self.assertFalse(
            source_request_requires_sentence_translations(normalised))
        self.assertFalse(
            source_request_uses_sentence_arrays(normalised))
        self.assertEqual(len(validated["cards"]), 2)

    def test_version_four_source_context_protocol_is_identified_as_obsolete(
            self):
        contract = build_source_request_contract(
            context_classical_pipeline(),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        contract["schema_version"] = 4
        del contract["response_formats_by_chunk"]

        normalised = normalise_source_request_contract(contract)

        self.assertTrue(
            source_request_uses_obsolete_context_translation_protocol(
                normalised))
        self.assertFalse(
            source_request_uses_sentence_arrays(normalised))

    def test_version_five_source_context_protocol_is_identified_as_obsolete(
            self):
        contract = build_source_request_contract(
            context_classical_pipeline(),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        contract["schema_version"] = 5
        del contract["response_formats_by_chunk"]

        normalised = normalise_source_request_contract(contract)

        self.assertTrue(
            source_request_uses_obsolete_context_translation_protocol(
                normalised))
        self.assertTrue(
            source_request_uses_sentence_arrays(normalised))
        self.assertFalse(
            source_request_uses_split_contextual_cards(normalised))

    def test_version_six_source_context_protocol_is_identified_as_obsolete(
            self):
        contract = build_source_request_contract(
            context_classical_pipeline(),
            use_source_for_example_sentences=True,
            protocol_version=8,
            reasoning_effort="low")
        contract["schema_version"] = 6
        del contract["response_formats_by_chunk"]

        normalised = normalise_source_request_contract(contract)

        self.assertTrue(
            source_request_uses_obsolete_context_translation_protocol(
                normalised))
        self.assertTrue(
            source_request_uses_sentence_arrays(normalised))
        self.assertTrue(
            source_request_uses_split_contextual_cards(normalised))
        self.assertFalse(
            source_request_uses_occurrence_locators(normalised))

    def test_legacy_overlong_schema_name_is_repaired_without_schema_change(
            self):
        contract = build_source_request_contract(
            context_classical_pipeline(),
            use_source_for_example_sentences=True)
        original_schema = contract["response_format"]["schema"]
        contract["response_format"]["name"] = (
            "autoanki_classical_chinese_wang_bi_"
            "context_optional_sentences_cards")

        repaired = normalise_source_request_contract(contract)

        self.assertLessEqual(
            len(repaired["response_format"]["name"]),
            64)
        self.assertEqual(
            repaired["response_format"]["schema"],
            original_schema)

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

    def test_emergency_pause_keeps_undispatched_chunk_pending(self):
        control = PaidDispatchControl(paused=True)
        paid_calls = []

        def request(chunk):
            return control.dispatch(
                lambda: paid_calls.append(chunk.chunk_id))

        chunk_id = self.store.chunk_ids(self.job.job_id)[0]
        snapshot = self.runner(
            request,
            concurrency=1).run((chunk_id,))

        self.assertEqual(paid_calls, [])
        status = self.store.chunk_status(
            self.job.job_id,
            chunk_id)
        self.assertEqual(status["status"], "pending")
        self.assertEqual(status["worker"], "paused")
        self.assertIsNone(status["completed_at"])
        self.assertEqual(
            status["last_error"]["type"],
            "PaidDispatchPaused")
        self.assertEqual(snapshot.overall_status, "ready")

    def test_emergency_pause_does_not_wait_for_an_inflight_call(self):
        control = PaidDispatchControl()
        entered = threading.Event()
        release = threading.Event()

        def operation():
            entered.set()
            self.assertTrue(release.wait(timeout=2))

        worker = threading.Thread(
            target=lambda: control.dispatch(operation),
            daemon=True)
        worker.start()
        self.assertTrue(entered.wait(timeout=2))

        state = control.pause()

        self.assertTrue(state["paused"])
        self.assertEqual(state["active_dispatches"], 1)
        with self.assertRaises(PaidDispatchPaused):
            control.dispatch(lambda: None)
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(
            control.status()["active_dispatches"],
            0)

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
