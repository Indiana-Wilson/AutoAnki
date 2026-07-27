import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui
import pipeline_store


def variable(value):
    result = MagicMock()
    result.get.return_value = value
    return result


def source_request_stub():
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
    app.source_limit_to_prefix = variable(False)
    app.source_language_label = variable("")
    app.source_allow_web_search = variable(False)
    app.get_pipeline_configs = MagicMock(return_value=())
    app._selected_anki_exclusions = MagicMock(return_value=())
    return app


class SourceGenerationModeGuiTests(unittest.TestCase):
    def test_model_options_do_not_offer_retired_luna(self):
        model_ids = tuple(
            key for key, _label in gui.SOURCE_MODEL_OPTIONS)

        self.assertNotIn("gpt-5.6-luna", model_ids)
        self.assertEqual(model_ids[0], "gpt-5.4-mini")

    def test_protocol_options_put_v10_before_both_rollbacks(self):
        self.assertEqual(
            tuple(key for key, _label in gui.SOURCE_PROTOCOL_OPTIONS),
            ("v10", "v9", "v8"))
        self.assertIn("local-first", gui.SOURCE_PROTOCOL_OPTIONS[0][1])
        self.assertIn("rollback", gui.SOURCE_PROTOCOL_OPTIONS[1][1])
        self.assertIn("rollback", gui.SOURCE_PROTOCOL_OPTIONS[2][1])

    def test_lightweight_source_request_stubs_receive_safe_v10_defaults(self):
        app = source_request_stub()

        request = app._source_request()

        self.assertEqual(request["request_protocol"], "v10")
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["execution_mode"], "standard")
        self.assertEqual(request["model"], "gpt-5.4-mini")
        self.assertFalse(request["automatic_repair"])

    def test_source_request_maps_explicit_economy_and_reasoning_choices(self):
        app = source_request_stub()
        app.source_request_protocol_label = variable(
            dict(gui.SOURCE_PROTOCOL_OPTIONS)["v9"])
        app.source_reasoning_label = variable(
            dict(gui.SOURCE_REASONING_OPTIONS)["low"])
        app.source_execution_label = variable(
            dict(gui.SOURCE_EXECUTION_OPTIONS)["economy"])

        request = app._source_request()

        self.assertEqual(request["request_protocol"], "v9")
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["execution_mode"], "economy")

    def test_source_card_menu_controls_pipeline_nuance_and_deck_split(self):
        app = source_request_stub()
        app.source_use_source_examples = variable(True)
        app.source_context_label = variable(
            gui.SOURCE_CONTEXT_LABELS["sentence_neighbors"])
        app.source_card_direction_variables = {
            "context": variable(True),
            "word_to_meaning": variable(True),
            "meaning_to_word": variable(False),
        }
        app.source_include_context_nuance = variable(True)
        app.source_separate_decks = variable(True)
        app.get_pipeline_configs = MagicMock(
            return_value=(pipeline_store.default_pipeline(),))

        request = app._source_request()

        self.assertEqual(
            request["source_card_directions"],
            ["context", "word_to_meaning"])
        self.assertEqual(request["context_mode"], "sentence")
        self.assertTrue(request["include_source_context_nuance"])
        self.assertTrue(request["separate_source_decks"])
        settings = pipeline_store.get_language_settings(
            request["pipeline"],
            "middle_english")
        self.assertFalse(settings.separate_target_decks)
        self.assertEqual(
            tuple(
                card.direction_key
                for card in pipeline_store.get_enabled_cards(
                    request["pipeline"])),
            ("context", "word_to_meaning"))

    def test_source_sentence_only_does_not_need_a_configured_definition_field(
            self):
        app = source_request_stub()
        app.source_use_source_examples = variable(True)
        app.source_card_direction_variables = {
            "context": variable(True),
            "word_to_meaning": variable(False),
            "meaning_to_word": variable(False),
        }
        app.source_include_context_nuance = variable(False)
        app.source_separate_decks = variable(True)
        pipeline = pipeline_store.default_pipeline()
        settings = pipeline_store.get_language_settings(
            pipeline,
            "middle_english")
        settings = replace(
            settings,
            share_field_settings=False,
            cards=tuple(
                replace(card, fields=())
                for card in settings.cards))
        pipeline = pipeline_store.replace_active_language_settings(
            pipeline,
            settings,
            (settings,),
            active_language_key="middle_english")
        app.get_pipeline_configs = MagicMock(return_value=(pipeline,))

        request = app._source_request()

        self.assertEqual(request["source_card_directions"], ["context"])
        self.assertEqual(
            pipeline_store.get_source_lexical_field_settings(
                request["pipeline"]),
            ())
        context_card = pipeline_store.get_enabled_cards(
            request["pipeline"])[0]
        request_settings = pipeline_store.get_language_settings(
            request["pipeline"],
            "middle_english")
        self.assertTrue(
            pipeline_store.get_effective_fields(
                request_settings,
                context_card))
        self.assertFalse(request_settings.separate_target_decks)

    def test_legacy_protocol_forces_low_reasoning_and_disables_selector(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_request_protocol_label = variable(
            dict(gui.SOURCE_PROTOCOL_OPTIONS)["v8"])
        app.source_reasoning_label = variable(
            dict(gui.SOURCE_REASONING_OPTIONS)["none"])
        app.source_reasoning_selector = MagicMock()
        app.source_paid_authorized = MagicMock()
        app._schedule_source_estimate = MagicMock()

        app._source_protocol_changed()

        app.source_reasoning_label.set.assert_called_once_with(
            dict(gui.SOURCE_REASONING_OPTIONS)["low"])
        app.source_reasoning_selector.configure.assert_called_once_with(
            state=gui.tk.DISABLED)
        app.source_paid_authorized.set.assert_called_once_with(False)
        app._schedule_source_estimate.assert_called_once_with()

    def test_switching_from_v8_restores_the_previous_reasoning_choice(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_request_protocol_label = variable(
            dict(gui.SOURCE_PROTOCOL_OPTIONS)["v8"])
        app.source_reasoning_label = variable(
            dict(gui.SOURCE_REASONING_OPTIONS)["none"])
        app.source_reasoning_selector = MagicMock()
        app.source_paid_authorized = MagicMock()
        app._schedule_source_estimate = MagicMock()

        app._source_protocol_changed()
        app.source_request_protocol_label.get.return_value = (
            dict(gui.SOURCE_PROTOCOL_OPTIONS)["v10"])
        app._source_protocol_changed()

        self.assertEqual(
            app.source_reasoning_label.set.call_args_list,
            [
                call(dict(gui.SOURCE_REASONING_OPTIONS)["low"]),
                call(dict(gui.SOURCE_REASONING_OPTIONS)["none"]),
            ])
        self.assertEqual(
            app.source_reasoning_selector.configure.call_args_list,
            [
                call(state=gui.tk.DISABLED),
                call(state="readonly"),
            ])

    def test_economy_disables_irrelevant_rate_controls_and_updates_action(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_execution_label = variable(
            dict(gui.SOURCE_EXECUTION_OPTIONS)["economy"])
        app.source_concurrency_entry = MagicMock()
        app.source_stagger_entry = MagicMock()
        app.source_rate_notice_label = MagicMock()
        app.source_generate_button = MagicMock()
        app.source_paid_authorized = MagicMock()
        app._schedule_source_estimate = MagicMock()

        app._source_execution_mode_changed()

        app.source_concurrency_entry.configure.assert_called_once_with(
            state=gui.tk.DISABLED)
        app.source_stagger_entry.configure.assert_called_once_with(
            state=gui.tk.DISABLED)
        app.source_generate_button.configure.assert_called_once_with(
            text="Submit Economy Batch")
        notice = app.source_rate_notice_label.configure.call_args.kwargs["text"]
        self.assertIn("asynchronous Batch", notice)
        self.assertIn("do not apply", notice)
        app.source_paid_authorized.set.assert_called_once_with(False)
        app._schedule_source_estimate.assert_called_once_with()

    def test_economy_request_ignores_invalid_disabled_rate_fields(self):
        app = source_request_stub()
        app.source_execution_label = variable(
            dict(gui.SOURCE_EXECUTION_OPTIONS)["economy"])
        app.source_concurrency = variable("")
        app.source_request_stagger_ms = variable("not a number")

        request = app._source_request()

        self.assertEqual(request["concurrency"], 1)
        self.assertEqual(request["request_stagger_ms"], 0)

    def test_estimate_shows_central_price_range_cache_and_pricing_mode(self):
        price, detail = gui.format_source_estimate({
            "estimated_cost_aud": 1.25,
            "estimated_cost_low_aud": 0.90,
            "estimated_cost_high_aud": 1.80,
            "candidate_count": 100,
            "request_count": 1,
            "input_tokens": 2_400,
            "output_tokens": 12_000,
            "estimated_cached_input_tokens": 1_200,
            "estimated_cache_write_input_tokens": 300,
            "translation_memory_hit_count": 4,
            "model": "gpt-5.4-mini",
            "automatic_repair": True,
            "max_automatic_repairs": 3,
            "pricing_label": "Batch API (50% token rates)",
            "assumptions": {
                "request_protocol": "v10",
                "reasoning_effort": "none",
                "execution_mode": "economy",
                "automatic_repair_reserve_aud": 2.50,
                "automatic_repair_max_requests": 12,
            },
        })

        self.assertEqual(price, "A$1.25 expected")
        self.assertIn("compact v10 local-first", detail)
        self.assertIn("gpt-5.4-mini", detail)
        self.assertIn("no reasoning", detail)
        self.assertIn("Economy Batch", detail)
        self.assertIn("modelled range A$0.90–A$1.80", detail)
        self.assertIn(
            "1,200 input tokens expected at cache-read rate",
            detail)
        self.assertIn(
            "300 input tokens expected at cache-write rate",
            detail)
        self.assertIn("4 exact source translations reused locally", detail)
        self.assertIn("automatic repair enabled", detail)
        self.assertIn(
            "worst-case optional repair reserve A$2.50",
            detail)
        self.assertIn("at most 12 calls", detail)
        self.assertIn("Batch API (50% token rates)", detail)
        self.assertNotIn("automatic transient retries", detail)

    def test_estimate_names_each_protocol_without_losing_rollback_status(self):
        for protocol, expected in (
                ("v10", "compact v10 local-first"),
                ("v9", "compact v9 rollback"),
                ("v8", "legacy v8 rollback")):
            with self.subTest(protocol=protocol):
                _price, detail = gui.format_source_estimate({
                    "assumptions": {
                        "request_protocol": protocol,
                        "reasoning_effort": "none",
                        "execution_mode": "standard",
                    },
                })

                self.assertIn(expected, detail)


if __name__ == "__main__":
    unittest.main()
