import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gui


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
    def test_lightweight_source_request_stubs_receive_safe_v9_defaults(self):
        app = source_request_stub()

        request = app._source_request()

        self.assertEqual(request["request_protocol"], "v9")
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["execution_mode"], "standard")

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

    def test_economy_disables_irrelevant_rate_controls_and_updates_action(self):
        app = object.__new__(gui.AutoAnkiApp)
        app.source_execution_label = variable(
            dict(gui.SOURCE_EXECUTION_OPTIONS)["economy"])
        app.source_concurrency_entry = MagicMock()
        app.source_stagger_entry = MagicMock()
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
        app.source_paid_authorized.set.assert_called_once_with(False)
        app._schedule_source_estimate.assert_called_once_with()

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
            "pricing_label": "Batch API (50% token rates)",
        })

        self.assertEqual(price, "A$1.25 expected")
        self.assertIn("modelled range A$0.90–A$1.80", detail)
        self.assertIn(
            "1,200 input tokens expected at cache-read rate",
            detail)
        self.assertIn("Batch API (50% token rates)", detail)


if __name__ == "__main__":
    unittest.main()
